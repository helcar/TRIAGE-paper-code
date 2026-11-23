"""CLI construction tests: all nine experiment configurations, no GPU required."""
from pathlib import Path

from triage_repro.workflow import (
    extract_command, load_experiment, slug, tuning_command, evaluation_command
)


CONFIG = Path(__file__).resolve().parents[1] / "configs" / "workflow.example.yaml"


def test_all_nine_experiments_configured():
    from yaml import safe_load
    payload = safe_load(CONFIG.read_text(encoding="utf-8"))
    names = payload["experiments"]
    assert len(names) == 9
    assert {data["model"] for data in names.values()} == {
        "xception", "efficientnetb4", "vit"
    }
    assert {data["dataset"] for data in names.values()} == {
        "FF++ c23", "Celeb-DF-v2", "DFDC"
    }
    for name in names:
        _, exp = load_experiment(CONFIG, name)
        assert exp["layers"]
        assert len(set(exp["layers"])) == len(exp["layers"])


def test_extraction_requires_authorized_test_mode():
    for name in ("xception_ffpp", "effb4_celeb", "vit_dfdc"):
        _, exp = load_experiment(CONFIG, name)
        layer = exp["layers"][0]
        out = Path("somewhere") / slug(layer)
        train_cmd = extract_command(exp, layer, out, ("fit", "cal", "val"))
        test_cmd = extract_command(exp, layer, out, ("test",), authorize_test=True)
        assert "--authorize-test" not in train_cmd
        assert "--authorize-test" in test_cmd
        assert "test" not in train_cmd
        assert test_cmd.count("test") == 1


def test_tuning_excludes_test():
    _, exp = load_experiment(CONFIG, "effb4_ffpp")
    command = tuning_command(exp, Path("arrays"), Path("grid.json"), Path("grid.csv"))
    assert "--test" not in command
    assert "--fit" in command and "--cal" in command and "--val" in command
    assert command[command.index("--k") + 1:command.index("--m")] == [
        "64", "128", "256", "512"
    ]


def test_final_evaluator_test_gate():
    _, exp = load_experiment(CONFIG, "xception_ffpp")
    common = (exp, Path("arrays"), Path("grid.json"),
              Path("output.json"), Path("scores.csv"))
    gated = evaluation_command(*common, authorize_test=True)
    validation_only = evaluation_command(*common)
    assert "--authorize-test" in gated
    assert "--authorize-test" not in validation_only
    assert "--ffpp-official" in gated
