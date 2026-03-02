"""Tests for WLST experiment registry."""

from src.wlst.experiments import (
    CLASSICAL_BASELINES,
    WLST_EXPERIMENT_REGISTRY,
    WLSTExperimentConfig,
)


class TestWLSTExperimentRegistry:
    def test_registry_has_stage1_experiments(self):
        stage1 = {k: v for k, v in WLST_EXPERIMENT_REGISTRY.items() if v.stage == "stage1"}
        assert len(stage1) >= 4
        assert "W1_mlp_baseline" in stage1
        assert "W4_full_model" in stage1

    def test_registry_has_stage2_experiments(self):
        stage2 = {k: v for k, v in WLST_EXPERIMENT_REGISTRY.items() if v.stage == "stage2"}
        assert len(stage2) >= 3
        assert "W6_stage2_full_model" in stage2

    def test_w1_baseline_disables_all(self):
        config = WLST_EXPERIMENT_REGISTRY["W1_mlp_baseline"]
        assert not config.use_transformer
        assert not config.use_diffusion
        assert not config.use_temporal_encoding

    def test_w4_full_model_enables_all(self):
        config = WLST_EXPERIMENT_REGISTRY["W4_full_model"]
        assert config.use_transformer
        assert config.use_diffusion
        assert config.use_temporal_encoding

    def test_all_configs_have_names(self):
        for name, config in WLST_EXPERIMENT_REGISTRY.items():
            assert config.name == name
            assert config.description

    def test_config_serialization(self):
        config = WLST_EXPERIMENT_REGISTRY["W4_full_model"]
        d = config.to_dict()
        assert isinstance(d, dict)
        assert d["name"] == "W4_full_model"
        assert d["use_diffusion"] is True


class TestClassicalBaselines:
    def test_has_lr_and_xgboost(self):
        assert "logistic_regression" in CLASSICAL_BASELINES
        assert "xgboost" in CLASSICAL_BASELINES

    def test_lr_config(self):
        config = CLASSICAL_BASELINES["logistic_regression"]
        assert config.model_type == "logistic_regression"

    def test_xgboost_config(self):
        config = CLASSICAL_BASELINES["xgboost"]
        assert config.model_type == "xgboost"
