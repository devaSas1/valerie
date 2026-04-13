from valerie.config import load_experiment_config, load_model_config


def test_load_model_config():
    config = load_model_config("configs/models/dummy.yaml")
    assert config.backend == "dummy"
    assert config.device_preference == ["cpu"]


def test_load_experiment_config():
    config = load_experiment_config("configs/experiments/threat-vs-care.yaml")
    assert "neutral" in config.conditions
    assert len(config.tasks) == 2

