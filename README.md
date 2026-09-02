# LLM-Guided AutoML

An AutoML system that takes a raw tabular dataset and a plain-English goal, then runs the complete machine learning workflow through a seven-phase multi-agent pipeline built on [LangGraph](https://github.com/langchain-ai/langgraph) — and explains every decision it made.

Most AutoML frameworks are black boxes: they optimise a metric and hand back a model with no account of *why*. This system differs in three ways:

- **It reads your goal.** "Predict which patients are at risk of heart disease, prioritise catching every real case" changes which metric is selected and how the decision threshold is tuned. The goal is parsed, not ignored.
- **It only calls the LLM when it needs to.** A three-tier Decision Router resolves each preprocessing decision with a hard rule, a statistical check, or the LLM — in that order of preference. Across ten benchmark datasets, **71.7%** of decisions never reached the LLM.
- **Phases can see each other.** An episodic decision log persists across the whole run, so model selection knows what preprocessing did, and the final report can explain the entire chain in plain English.

## Results

Evaluated on ten public tabular datasets against AutoGluon, FLAML, an untuned Random Forest, and a linear baseline. Classification is F1 (weighted); regression is R². **Bold** = best.

| Dataset | Linear | Random Forest | FLAML | AutoGluon | **Ours** |
|---|---|---|---|---|---|
| Titanic | 0.8033 | 0.8038 | 0.8135 | 0.8022 | **0.8308** |
| Heart Disease | 0.8540 | 0.8806 | 0.8701 | 0.8751 | **0.8914** |
| Bank Marketing | 0.8070 | 0.8335 | 0.8359 | 0.8490 | **0.8616** |
| Customer Churn | 0.7838 | 0.7872 | 0.8001 | 0.7980 | **0.8050** |
| Credit Card Fraud | 0.9990 | 0.9992 | 0.9976 | 0.9989 | **0.9993** |
| Insurance | 0.8068 | 0.8818 | 0.9016 | 0.8991 | **0.9031** |
| House Prices | 0.6889 | 0.8746 | **0.9131** | 0.9100 | 0.8869 |
| Life Expectancy | 0.8192 | 0.9671 | **0.9725** | 0.9690 | 0.9724 |
| Diamonds | 0.9209 | 0.9823 | 0.9831 | **0.9846** | 0.9824 |
| Wine Quality | 0.3918 | 0.4556 | 0.4497 | 0.4692 | **0.4885** |

Highest score on **7 of 10** datasets — all five classification tasks and two of five regression tasks. AutoGluon and FLAML were each given a 90-second budget per dataset.

## Architecture

![System architecture of the seven-phase pipeline, showing the episodic decision log and the three-tier Decision Router](experiments/results/figures/fig0_system_architecture.png)

| Phase | Stage | Produces |
|---|---|---|
| 1 | Intent parsing & task classification | Target column, task type, resolved metric |
| 2 | Preprocessing planning (LLM) | High-level plan, carried forward as a prior |
| 3 | Preprocessing execution | Clean → split → EDA → feature selection → deep cleaning → feature engineering → verify |
| 4 | Execution planning | Model shortlist, search strategy, CV design, threshold |
| 5 | Model training & evaluation | Progressive elimination → Optuna TPE → ensembling → threshold tuning |
| 6 | Threshold check & failure diagnosis | Re-enters Phase 3 or 4 on failure, rather than restarting |
| 7 | Report generation | Stakeholder report, model recommendation, pipeline code |

Every phase reads from and writes to a shared **episodic decision log**, so no phase runs blind to what came before it. Phases 2–7 are a LangGraph `StateGraph`; when preprocessing output fails verification the graph rewinds to the failed sub-step, and when a run misses its performance threshold Phase 6 diagnoses the cause and re-enters only the affected phase.

### The three-tier Decision Router

Each column's imputation, outlier, encoding, scaling, and correlation decisions are assigned to the cheapest mechanism that can resolve them.

| Signal | Condition | Decision | Tier |
|---|---|---|---|
| Null rate | < 10% | Median / mode imputation | 1 — Rule |
| Null rate | 10–30% | Median / mode imputation | 2 — Statistical |
| Null rate | 30–70% | LLM decides | 3 — LLM |
| Null rate | > 70% | Flag as missing | 1 — Rule |
| Skewness | abs(skew) < 0.75 | No transform | 2 — Statistical |
| Skewness | abs(skew) in [0.75, 3.0) | LLM decides | 3 — LLM |
| Skewness | abs(skew) >= 3.0 | Log transform | 2 — Statistical |
| Correlation | r >= 0.97 | Drop lower-variance | 2 — Statistical |
| Cardinality | <= 10 unique | One-hot encoding | 1 — Rule |
| Cardinality | 11–100 unique | LLM decides | 3 — LLM |
| Cardinality | > 100 unique | Ordinal encoding | 1 — Rule |

Only Tier 3 reaches the LLM. Total LLM calls per run stay between 11 and 13 regardless of dataset size.

## Installation

Requires **Python 3.12+** and a running **MongoDB** instance (for the episodic decision log).

```bash
git clone https://github.com/krittika019/LLM-Based-AutoML.git
cd LLM-Based-AutoML

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

AutoGluon is intentionally left out of `requirements.txt` (~2 GB install). It is only needed to reproduce the benchmark comparison:

```bash
pip install autogluon.tabular
```

## Configuration

```bash
cp .env.example .env
```

Then fill in at minimum an LLM API key and your MongoDB URI:

```ini
LLM_PROVIDER=groq                 # groq | cerebras | together | openai | ollama
LLM_FALLBACK_PROVIDER=cerebras    # used automatically if the primary rate-limits
GROQ_API_KEY=your_key_here
MONGODB_URI=mongodb://localhost:27017
```

Two model tiers are used: a fast model for structured JSON and EDA narration, and a larger one for the two reasoning-heavy phases (failure diagnosis and report generation). Defaults are `llama-3.1-8b-instant` and `llama-3.3-70b-versatile`, overridable via `LLM_FLASH` and `LLM_PRO`.

## Usage

### Web interface

```bash
uvicorn server:app --reload --port 8000
```

Open http://localhost:8000. Upload a CSV, describe your goal in plain English, and watch the run stream phase by phase.

To develop the frontend with hot reload:

```bash
cd frontend
npm install
npm run dev          # production build: npm run build
```

The built frontend in `frontend/dist/` is committed so the FastAPI server can serve it without a Node runtime in production.

### API

| Method | Endpoint | Purpose |
|---|---|---|
| `POST` | `/upload` | Upload a CSV dataset |
| `POST` | `/run` | Start a pipeline run |
| `GET` | `/stream/{run_id}` | Server-sent events for live progress |
| `GET` | `/download/script/{run_id}` | Download the generated pipeline script |
| `GET` | `/download/report/{run_id}` | Download the stakeholder report |

## Reproducing the benchmark

```bash
# All ten datasets, every system
python experiments/benchmark.py

# A subset
python experiments/benchmark.py --datasets titanic,heart,wine

# Our pipeline only, or competitors only
python experiments/benchmark.py --only-ours
python experiments/benchmark.py --skip-automl

# Change the competitor time budget (default 90s)
python experiments/benchmark.py --automl-time 120

# Merge per-dataset results into benchmark_results.csv
python experiments/benchmark.py --merge
```

Results land in `experiments/results/`. Figures are regenerated with `python experiments/generate_figures.py`.

Datasets live in `experiments/datasets/`. `creditcard.csv` is excluded from version control (144 MB, over GitHub's per-file limit) — download it from [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) and place it there.

## Tests

```bash
pytest tests/
```

## Project structure

```
automl_system/
  agents/      intent parsing, task classification, preprocessing and pipeline
               planning, failure diagnosis, report generation
  graph/       LangGraph orchestrator — nodes, edges, and retry routing
  pipeline/    cleaning, EDA, feature selection/engineering, training,
               verification, pipeline-script generation
  memory/      episodic decision log (MongoDB-backed)
  utils/       decision router, dataset profiling, decision log, JSON repair
  llm/         multi-provider LLM client with automatic fallback
  state/       shared session state passed between phases
experiments/   benchmark harness, ablations, figure generation
frontend/      React + Vite UI
server.py      FastAPI backend
tests/         end-to-end tests
```

## Limitations

Tabular, supervised tasks only. Time series, image, and text data are not supported, nor are unsupervised or semi-supervised settings. Runtime is longer than AutoGluon or FLAML — the pipeline is sequential and also generates rationales and a natural-language report on every run, which those systems do not do.

## Citation

If you use this work, please cite:

```bibtex
@inproceedings{bisht2026llmautoml,
  title     = {An {LLM}-Guided Multi-Agent System for Automated and
               Interpretable Machine Learning on Tabular Data},
  author    = {Bisht, Krittika and Pawa, Manasvi and Saboo, Shreyansh
               and Sharma, Aditi},
  booktitle = {},
  year      = {2026}
}
```

## License

Released under the MIT License — see [LICENSE](LICENSE).
