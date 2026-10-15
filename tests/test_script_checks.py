from __future__ import annotations

from automl_system.pipeline.script_checks import static_check

VALID = '''
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

TRAIN_PATH = "train.csv"
TEST_PATH = "test.csv"
TARGET_COLUMN = "y"
PREDICTIONS_PATH = "predictions.csv"

# Tuned earlier with Optuna on the GPU — mentioning it in a comment is fine.
enc = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
pd.DataFrame({"prediction": [1]}).to_csv(PREDICTIONS_PATH, index=False)
'''


def test_valid_script_passes():
    assert static_check(VALID) == []


def test_syntax_error():
    issues = static_check(VALID + "\ndef broken(:\n")
    assert len(issues) == 1 and "SyntaxError" in issues[0]


def test_removed_keyword_argument_is_caught():
    issues = static_check(VALID.replace("sparse_output=False", "sparse=False"))
    assert any("OneHotEncoder() has no keyword argument 'sparse'" in i for i in issues)


def test_hallucinated_import_is_caught():
    issues = static_check(VALID.replace("import OneHotEncoder", "import OneHotEncoder, SmartAutoEncoder"))
    assert any("SmartAutoEncoder" in i for i in issues)


def test_module_attribute_keyword_is_caught():
    issues = static_check(VALID + "\ndf = pd.read_csv(TRAIN_PATH, seperator=',')\n")
    assert any("read_csv() has no keyword argument 'seperator'" in i for i in issues)


def test_contract_violations():
    code = VALID.replace('PREDICTIONS_PATH = "predictions.csv"\n', "")
    code += "\nfrom sklearn.model_selection import GridSearchCV\nparams = {'device': 'cuda'}\n"
    issues = static_check(code)
    assert any("PREDICTIONS_PATH" in i for i in issues)
    assert any("GridSearchCV" in i for i in issues)
    assert any("GPU" in i for i in issues)
