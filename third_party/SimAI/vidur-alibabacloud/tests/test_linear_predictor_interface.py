"""Regression test for the scheduler/predictor constructor contract."""

import inspect

import pandas as pd

from vidur.execution_time_predictor.linear_regression_execution_time_predictor import (
    LinearRegressionExecutionTimePredictor,
)
from vidur.execution_time_predictor.random_forrest_execution_time_predictor import (
    RandomForrestExecutionTimePredictor,
)


def test_sklearn_predictors_accept_simulation_config():
    for predictor in (
        LinearRegressionExecutionTimePredictor,
        RandomForrestExecutionTimePredictor,
    ):
        parameters = inspect.signature(predictor.__init__).parameters
        assert "simulation_config" in parameters


def test_attention_prediction_is_memoized_without_changing_value():
    class SumModel:
        calls = 0

        def predict(self, frame: pd.DataFrame):
            self.calls += 1
            return frame.sum(axis=1).to_numpy()

    predictor = LinearRegressionExecutionTimePredictor.__new__(
        LinearRegressionExecutionTimePredictor
    )
    model = SumModel()
    predictor._models = {"attn_prefill": model}
    predictor._predictions = {"attn_prefill": {}}

    first = predictor._get_or_predict_attention(
        "attn_prefill",
        (64, 512**2),
        ("kv_cache_size", "prefill_chunk_size_squared"),
    )
    second = predictor._get_or_predict_attention(
        "attn_prefill",
        (64, 512**2),
        ("kv_cache_size", "prefill_chunk_size_squared"),
    )

    assert first == second == 64 + 512**2
    assert model.calls == 1
