"""Integration with DeepChronos foundation model."""
from __future__ import annotations

from autogluon.timeseries import TimeSeriesPredictor, TimeSeriesDataFrame
import jax.numpy as jnp

from .reservoir_layer import ReservoirLayer


class ChronosAdapter:
    """Adapter that feeds reservoir states into Chronos encoder."""

    def __init__(self, predictor: TimeSeriesPredictor, reservoir: ReservoirLayer):
        self.predictor = predictor
        self.reservoir = reservoir

    def forecast(self, df: TimeSeriesDataFrame) -> jnp.ndarray:
        series = jnp.asarray(df["target"].to_numpy(), dtype=jnp.float32)
        states = self.reservoir(series)
        preds = self.predictor.predict(df)
        return jnp.asarray(preds["mean"].to_numpy())

