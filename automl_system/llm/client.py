"""Multi-provider LLM chat wrapper with response caching and call counting.

Supported providers (set LLM_PROVIDER env var):
  groq       — Default. Uses Groq SDK.  Fast free-tier, no extra deps.
  cerebras   — Fastest inference. Needs: pip install openai  +  CEREBRAS_API_KEY
  together   — Together AI. Needs: pip install openai  +  TOGETHER_API_KEY
  ollama     — Local models. Needs: pip install openai  +  Ollama running locally
  openai     — OpenAI API. Needs: pip install openai  +  OPENAI_API_KEY
  (any other) — Generic OpenAI-compat via OPENAI_COMPAT_BASE_URL + OPENAI_COMPAT_API_KEY

"""

from __future__ import annotations

import hashlib
import logging
import random
import time
from collections import OrderedDict
from typing import Any

from groq import Groq

from automl_system import config

logger = logging.getLogger(__name__)

_DEFAULT_PROMPT_CHAR_CAP = 11000
_CACHE_MAX = 64


def _compact_prompt_text(text: str, max_chars: int) -> str:
    t = (text or "").strip()
    if len(t) <= max_chars:
        return t
    if max_chars < 400:
        return t[:max_chars]
    head = int(max_chars * 0.7)
    tail = max_chars - head - 48
    return (
        t[:head]
        + "\n\n[... prompt truncated for token budget ...]\n\n"
        + t[-max(0, tail):]
    )


def _is_payload_too_large_error(err: Exception) -> bool:
    msg = str(err).lower()
    return "413" in msg or "payload too large" in msg or "request too large" in msg


def _is_rate_limit_error(err: Exception) -> bool:
    msg = str(err).lower()
    return (
        "429" in msg
        or "rate_limit" in msg
        or "rate limit" in msg
        or "too many requests" in msg
        or "ratelimit" in msg
    )


# Module-level throttle: minimum seconds between successive LLM API calls.
# Groq free tier = ~30 RPM. 2.1 s/call ≈ 28 RPM — stays just under the limit
# and avoids the 3–43 s retry waits that 429s cause.
_MIN_CALL_INTERVAL_SEC: float = 2.1
_last_call_ts: float = 0.0


# ── Provider → (client object, canonical_flash_name, canonical_pro_name) ──

def _build_provider_client(provider: str) -> Any | None:
    """Return an API client for `provider` (Groq or OpenAI-compatible)."""
    import httpx, certifi
    _http = httpx.Client(verify=certifi.where())

    prov = provider.lower()
    if prov == "groq":
        key = config.GROQ_API_KEY
        if not key:
            logger.warning("GROQ_API_KEY not set.")
            return None
        return Groq(api_key=key, http_client=_http)

    if prov == "groq_fallback":
        key = config.GROQ_API_KEY_FALLBACK
        if not key:
            logger.warning("GROQ_API_KEY_FALLBACK not set.")
            return None
        return Groq(api_key=key, http_client=_http)

    # All other providers use the OpenAI package
    try:
        from openai import OpenAI as _OpenAI  # type: ignore[import]
    except ImportError:
        logger.warning(
            "Provider %r requires the 'openai' package: pip install openai", prov
        )
        return None

    if prov == "cerebras":
        key = config.CEREBRAS_API_KEY
        base = "https://api.cerebras.ai/v1"
    elif prov == "together":
        key = config.TOGETHER_API_KEY
        base = "https://api.together.xyz/v1"
    elif prov == "ollama":
        key = "ollama"
        base = "http://localhost:11434/v1"
    elif prov == "openai":
        key = config.OPENAI_API_KEY
        base = "https://api.openai.com/v1"
    else:
        key = config.OPENAI_COMPAT_API_KEY
        base = config.OPENAI_COMPAT_BASE_URL
        if not base:
            logger.warning("OPENAI_COMPAT_BASE_URL not set for provider %r", prov)
            return None

    if not key:
        logger.warning("%s API key not set.", prov.upper())
        return None
    return _OpenAI(api_key=key, base_url=base, http_client=_http)


def _resolve_model(logical_model: str, provider: str) -> str:
    """Map Groq logical model names to provider-specific names."""
    prov = provider.lower()
    is_flash = logical_model == config.LLM_FLASH
    is_pro = logical_model == config.LLM_PRO
    if prov in ("groq", "groq_fallback"):
        return logical_model
    if prov == "cerebras":
        return config.LLM_FLASH_CEREBRAS if is_flash else (config.LLM_PRO_CEREBRAS if is_pro else logical_model)
    if prov == "together":
        return config.LLM_FLASH_TOGETHER if is_flash else (config.LLM_PRO_TOGETHER if is_pro else logical_model)
    if prov == "ollama":
        return config.LLM_FLASH_OLLAMA if is_flash else (config.LLM_PRO_OLLAMA if is_pro else logical_model)
    if prov == "openai":
        return config.LLM_FLASH_OPENAI if is_flash else (config.LLM_PRO_OPENAI if is_pro else logical_model)
    return logical_model


class LLMClient:
    """
    One entry point: call(prompt, model=None, system_prompt=None, max_tokens=None, temperature=None) -> str.
    Caches last 64 (prompt_hash+model) -> response. Tracks call_count for non-cached calls.
    Provider is set globally via config.LLM_PROVIDER (default: groq).
    """

    _cache: OrderedDict[str, str] = OrderedDict()
    call_count: int = 0
    cache_hits: int = 0
    phase_label: str = ""

    def __init__(self, api_key: str | None = None) -> None:
        # api_key param kept for back-compat; provider selection uses config
        self._client = _build_provider_client(config.LLM_PROVIDER)
        if self._client is None:
            logger.warning(
                "LLM provider %r could not be initialised; calls will return ''.",
                config.LLM_PROVIDER,
            )

    @classmethod
    def reset_cache(cls) -> None:
        cls._cache.clear()

    @classmethod
    def _cache_key(cls, model_name: str, system_prompt: str | None, user_prompt: str) -> str:
        raw = f"{model_name}|{system_prompt or ''}|{user_prompt}"
        return hashlib.sha256(raw.encode("utf-8", errors="replace")).hexdigest()

    def call(
        self,
        prompt: str,
        model: str | None = None,
        system_prompt: str | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
        json_mode: bool = False,
    ) -> str:
        if not self._client:
            return ""

        provider = config.LLM_PROVIDER
        logical = model or config.LLM_FLASH
        model_name = _resolve_model(logical, provider)

        # Default temperature: low for fast, deterministic JSON
        temp = temperature if temperature is not None else config.LLM_TEMPERATURE_JSON

        user_prompt = _compact_prompt_text(prompt, _DEFAULT_PROMPT_CHAR_CAP)
        if len((prompt or "")) > len(user_prompt):
            logger.info(
                "LLM prompt compacted %s→%s chars", len(prompt or ""), len(user_prompt)
            )

        ck = self._cache_key(model_name, system_prompt, user_prompt)
        if ck in self._cache:
            LLMClient.cache_hits += 1
            self._cache.move_to_end(ck)
            return self._cache[ck]

        # Proactive throttle: avoid bursting into rate limit
        global _last_call_ts
        elapsed = time.time() - _last_call_ts
        if elapsed < _MIN_CALL_INTERVAL_SEC:
            time.sleep(_MIN_CALL_INTERVAL_SEC - elapsed)

        last_err: Exception | None = None
        for attempt in range(config.MAX_LLM_RETRIES):
            try:
                messages: list[dict[str, str]] = []
                if system_prompt:
                    messages.append({"role": "system", "content": system_prompt})
                messages.append({"role": "user", "content": user_prompt})
                LLMClient.call_count += 1
                _last_call_ts = time.time()

                call_kwargs: dict[str, Any] = dict(
                    model=model_name,
                    messages=messages,
                    temperature=temp,
                )
                if max_tokens:
                    call_kwargs["max_tokens"] = max_tokens
                # JSON mode: forces model to output raw valid JSON (no preamble, no fences).
                # Supported by Groq, OpenAI, Cerebras. Silently skipped if provider rejects it.
                if json_mode:
                    call_kwargs["response_format"] = {"type": "json_object"}

                resp = self._client.chat.completions.create(**call_kwargs)
                choice = resp.choices[0].message
                text = (choice.content or "").strip()

                self._cache[ck] = text
                if len(self._cache) > _CACHE_MAX:
                    self._cache.popitem(last=False)
                return text

            except Exception as e:
                last_err = e
                logger.warning(
                    "LLM call failed (attempt %s/%s) [%s]: %s",
                    attempt + 1,
                    config.MAX_LLM_RETRIES,
                    provider,
                    e,
                )
                if _is_payload_too_large_error(e) and len(user_prompt) > 1200:
                    user_prompt = _compact_prompt_text(
                        user_prompt, int(len(user_prompt) * 0.6)
                    )
                    logger.warning(
                        "Prompt shrunk to %s chars after payload-too-large", len(user_prompt)
                    )
                if attempt < config.MAX_LLM_RETRIES - 1:
                    if _is_rate_limit_error(e):
                        # Exponential backoff with jitter for 429 — Groq SDK may also
                        # retry internally, so our outer wait avoids a tight loop.
                        wait = min(
                            60.0,
                            config.LLM_RETRY_BACKOFF_SEC * (2 ** (attempt + 1))
                            + random.uniform(0.5, 2.0),
                        )
                        logger.info("Rate limit hit; backing off %.1f s", wait)
                    else:
                        wait = config.LLM_RETRY_BACKOFF_SEC * (attempt + 1)
                    time.sleep(wait)

        if last_err:
            logger.warning("LLM call exhausted retries on primary provider %r: %s", provider, last_err)

        # ── Fallback provider (if configured and different from primary) ──────
        fallback = config.LLM_FALLBACK_PROVIDER
        if fallback and fallback != provider:
            fallback_client = _build_provider_client(fallback)
            if fallback_client:
                logger.info("Trying fallback LLM provider %r", fallback)
                fallback_model = _resolve_model(logical, fallback)
                try:
                    messages_f: list[dict[str, str]] = []
                    if system_prompt:
                        messages_f.append({"role": "system", "content": system_prompt})
                    messages_f.append({"role": "user", "content": user_prompt})
                    call_kwargs_f: dict[str, Any] = dict(
                        model=fallback_model,
                        messages=messages_f,
                        temperature=temp,
                    )
                    if max_tokens:
                        call_kwargs_f["max_tokens"] = max_tokens
                    if json_mode:
                        call_kwargs_f["response_format"] = {"type": "json_object"}
                    LLMClient.call_count += 1
                    resp_f = fallback_client.chat.completions.create(**call_kwargs_f)
                    text_f = (resp_f.choices[0].message.content or "").strip()
                    self._cache[ck] = text_f
                    if len(self._cache) > _CACHE_MAX:
                        self._cache.popitem(last=False)
                    return text_f
                except Exception as fe:
                    logger.warning("Fallback provider %r also failed: %s", fallback, fe)

        return ""

    @classmethod
    def phase_call_stats_message(cls, phase_name: str, calls_at_phase_start: int) -> str:
        """Plain-English line for chatbot."""
        delta = cls.call_count - calls_at_phase_start
        return (
            f"LLM calls this phase ({phase_name}): **{delta}** | "
            f"Total so far: **{cls.call_count}** (cache hits: {cls.cache_hits})"
        )

