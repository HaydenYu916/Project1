"""Govee PWM→PPFD model adapter.

Wraps the sklearn Pipeline trained on Govee 9-position grid (features =
pwm_r, pwm_b, x_cm, y_cm, z_cm) so it can be plugged into the existing
MPPI flow that expects a `predict(r_pwm, b_pwm, key=None) -> ppfd` API.

The MPPI controller's PWMtoPPFDModel in led.py uses linear-per-ratio fits
loaded from a Shelly-style calibration CSV. That doesn't match the GPR
joblib package we trained for Govee, so this module gives a thin shim
that keeps the same surface but evaluates a sklearn model under the hood.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Iterable, Optional, Tuple

import joblib
import numpy as np

DEFAULT_MODEL_PATH = os.path.join(
    os.path.dirname(__file__), "..", "models", "pwmtoppfd", "best_model_package.joblib"
)


@dataclass
class GoveePosition:
    x_cm: float = 0.0
    y_cm: float = 0.0
    z_cm: float = 10.0


class GoveePWMtoPPFDModel:
    """Joblib-backed PWM→PPFD predictor for the Govee H6056 setup.

    Compatible-enough with led.PWMtoPPFDModel that mppi_v2 doesn't have to
    care about the underlying regressor type:

        model = GoveePWMtoPPFDModel().load()
        ppfd  = model.predict(r_pwm=30, b_pwm=20)              # uses default position
        model.set_position(x_cm=9.5, y_cm=0, z_cm=6)
        ppfd  = model.predict(r_pwm=30, b_pwm=20)              # at new position
    """

    def __init__(self, model_path: str = DEFAULT_MODEL_PATH,
                 position: Optional[GoveePosition] = None):
        self.model_path = model_path
        self.position = position or GoveePosition()
        self._pipeline = None
        self._meta = None

    def load(self) -> "GoveePWMtoPPFDModel":
        pkg = joblib.load(self.model_path)
        # joblib package shape from train_pwm_to_ppfd.py: dict with keys
        # 'pipeline', 'feature_columns', 'target_col'. Tolerate either.
        if isinstance(pkg, dict):
            self._pipeline = pkg.get("pipeline") or pkg.get("model")
            self._meta = {
                "feature_columns": pkg.get("feature_columns"),
                "target_col": pkg.get("target_col"),
            }
        else:
            self._pipeline = pkg
        if self._pipeline is None:
            raise RuntimeError(f"could not extract pipeline from {self.model_path}")
        return self

    def set_position(self, x_cm: float, y_cm: float, z_cm: float) -> "GoveePWMtoPPFDModel":
        self.position = GoveePosition(x_cm=float(x_cm), y_cm=float(y_cm), z_cm=float(z_cm))
        return self

    def _features(self, r_pwm: float, b_pwm: float) -> np.ndarray:
        return np.asarray([[float(r_pwm), float(b_pwm),
                            self.position.x_cm, self.position.y_cm, self.position.z_cm]],
                          dtype=float)

    def predict(self, *, r_pwm: float, b_pwm: float,
                key: Optional[str] = None,  # accepted for API parity, ignored
                position: Optional[GoveePosition] = None) -> float:
        if self._pipeline is None:
            self.load()
        if position is not None:
            old = self.position
            self.position = position
            try:
                return self.predict(r_pwm=r_pwm, b_pwm=b_pwm)
            finally:
                self.position = old
        X = self._features(r_pwm, b_pwm)
        y = self._pipeline.predict(X)
        return float(y[0])

    def predict_batch(self, r_pwm: Iterable[float],
                      b_pwm: Iterable[float]) -> np.ndarray:
        """Vectorized predict for an MPPI rollout."""
        if self._pipeline is None:
            self.load()
        r = np.asarray(list(r_pwm), dtype=float)
        b = np.asarray(list(b_pwm), dtype=float)
        if r.shape != b.shape:
            raise ValueError("r_pwm / b_pwm shape mismatch")
        n = r.size
        X = np.column_stack([
            r, b,
            np.full(n, self.position.x_cm),
            np.full(n, self.position.y_cm),
            np.full(n, self.position.z_cm),
        ])
        return np.asarray(self._pipeline.predict(X), dtype=float)


def load_default(position: Optional[GoveePosition] = None) -> GoveePWMtoPPFDModel:
    return GoveePWMtoPPFDModel(position=position).load()


if __name__ == "__main__":
    m = load_default()
    print(f"model: {m.model_path}")
    print(f"feature_columns: {m._meta and m._meta.get('feature_columns')}")
    for r, b in [(0, 0), (30, 0), (0, 30), (50, 50), (90, 0), (0, 90)]:
        print(f"  pwm_r={r:>2} pwm_b={b:>2} -> PPFD={m.predict(r_pwm=r, b_pwm=b):.2f}")
