import importlib


def test_lite_checkpoint_score_uses_balanced_domain_r2():
    module = importlib.import_module(
        "mlops.scripts.train_and_quantize_standard_mlp_lite"
    )

    simulation = {"r2": 0.65, "mae": 8.0, "rmse": 12.0}
    epa = {"r2": 0.08, "mae": 4.0, "rmse": 6.0}

    score = module.validation_checkpoint_score(simulation, epa)

    assert score == 0.75 * 0.65 + 0.25 * 0.08


def test_lite_checkpoint_score_can_prefer_epa_gain_over_small_simulation_loss():
    module = importlib.import_module(
        "mlops.scripts.train_and_quantize_standard_mlp_lite"
    )

    simulation_a = {"r2": 0.65}
    epa_a = {"r2": 0.08}
    simulation_b = {"r2": 0.63}
    epa_b = {"r2": 0.16}

    assert module.validation_checkpoint_score(simulation_b, epa_b) > module.validation_checkpoint_score(
        simulation_a, epa_a
    )
