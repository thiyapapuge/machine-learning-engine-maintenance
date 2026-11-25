"""Reusable preprocessing and evaluation utilities for the C-MAPSS FD002 project.

Everything that *learns* from data (operating-condition clustering, per-condition
mean/std) lives inside ``ConditionNormalizer`` so that it can be placed in a
scikit-learn ``Pipeline`` and re-fitted on each training fold. This is what
prevents preprocessing leakage during cross-validation.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import KMeans

SETTINGS = ["setting_1", "setting_2", "setting_3"]
SENSORS = [
    "T2", "T24", "T30", "T50", "P2", "P15", "P30", "Nf", "Nc", "epr", "Ps30",
    "phi", "NRf", "NRc", "BPR", "farB", "htBleed", "Nf_dmd", "PCNfR_dmd", "W31", "W32",
]
# Sensors that are exactly constant inside each operating condition (zero information).
CONSTANT_SENSORS = ["T2", "P2", "Nf_dmd", "PCNfR_dmd"]
# Sensors recorded with 2 decimals: only 2-3 distinct values per condition (quantised).
QUANTISED_SENSORS = ["P15", "epr", "farB"]

RUL_CAP = 125
RANDOM_STATE = 42


def load_raw(path) -> pd.DataFrame:
    """Load one of the provided CSV files and add a globally unique engine id."""
    df = pd.read_csv(path)
    # unit_id restarts at 1 in train AND test: unit 1 of train is NOT unit 1 of test.
    df["engine_uid"] = df["split"] + "_" + df["unit_id"].astype(str).str.zfill(3)
    return df.sort_values(["unit_id", "cycle"]).reset_index(drop=True)


class ConditionNormalizer(BaseEstimator, TransformerMixin):
    """Cluster the 3 operational settings into flight conditions (KMeans) and
    z-score each sensor *within* its condition, using statistics of the data
    passed to ``fit`` only.

    Input: DataFrame with SETTINGS + ``sensors`` columns.
    Output: DataFrame of normalised sensors (same index).
    """

    def __init__(self, sensors=None, n_conditions: int = 6, random_state: int = RANDOM_STATE):
        self.sensors = sensors
        self.n_conditions = n_conditions
        self.random_state = random_state

    def fit(self, X: pd.DataFrame, y=None):
        sensors = list(self.sensors) if self.sensors is not None else SENSORS
        self.kmeans_ = KMeans(self.n_conditions, n_init=10, random_state=self.random_state)
        cond = self.kmeans_.fit_predict(X[SETTINGS])
        grouped = X[sensors].groupby(cond)
        self.mean_ = grouped.mean()
        # std = 0 for sensors constant in a condition -> replace by 1 to avoid division by 0
        self.std_ = grouped.std().replace(0, 1.0).fillna(1.0)
        self.sensors_ = sensors
        return self

    def predict_condition(self, X: pd.DataFrame) -> np.ndarray:
        return self.kmeans_.predict(X[SETTINGS])

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        cond = self.predict_condition(X)
        mu = self.mean_.loc[cond].to_numpy()
        sd = self.std_.loc[cond].to_numpy()
        z = (X[self.sensors_].to_numpy() - mu) / sd
        return pd.DataFrame(z, columns=self.sensors_, index=X.index)

    def get_feature_names_out(self, input_features=None):
        return np.asarray(self.sensors_, dtype=object)


def nasa_score(y_true, y_pred) -> float:
    """PHM08 asymmetric scoring function (Saxena et al., 2008), summed over engines.
    Late predictions (d > 0, RUL over-estimated) are penalised more than early ones."""
    d = np.asarray(y_pred, dtype=float) - np.asarray(y_true, dtype=float)
    return float(np.sum(np.where(d < 0, np.exp(-d / 13.0) - 1.0, np.exp(d / 10.0) - 1.0)))


def regression_report(y_true, y_pred) -> dict:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    err = y_pred - y_true
    return {
        "RMSE": float(np.sqrt(np.mean(err ** 2))),
        "MAE": float(np.mean(np.abs(err))),
        "NASA score": nasa_score(y_true, y_pred),
        "n": int(len(y_true)),
    }


def last_cycle(df: pd.DataFrame) -> pd.DataFrame:
    """One row per engine: its last observed cycle (official C-MAPSS test protocol)."""
    return df.sort_values("cycle").groupby("engine_uid").tail(1).sort_values("engine_uid")


def random_truncation(df: pd.DataFrame, seed: int = RANDOM_STATE, min_cycles: int = 20) -> pd.DataFrame:
    """One row per engine, at a random cycle: mimics how the test set was built
    (trajectories cut at a random point before failure). Used on validation folds."""
    rng = np.random.default_rng(seed)
    rows = []
    for _, g in df.groupby("engine_uid", sort=True):
        g = g.sort_values("cycle")
        lo = min(min_cycles, len(g)) - 1
        rows.append(g.iloc[rng.integers(lo, len(g))])
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------------------
# Week 3 additions
# --------------------------------------------------------------------------------------
import time
from pathlib import Path
from sklearn.base import RegressorMixin, clone


def find_project_root(start=None) -> Path:
    """Walk up from the current folder until the folder containing src/cmapss.py is found.
    Makes the notebooks work whatever the working directory of Jupyter / VS Code is."""
    here = Path(start or Path.cwd()).resolve()
    for p in [here, *here.parents]:
        if (p / "src" / "cmapss.py").exists():
            return p
    raise FileNotFoundError("Project root (folder containing src/cmapss.py) not found from " + str(here))


class WindowFeatures(BaseEstimator, TransformerMixin):
    """Condition-normalise the sensors, then build *causal* history features per engine:
    current value, rolling mean and rolling slope over the last ``window`` cycles, plus the cycle number.

    Only the ConditionNormalizer is learned (in ``fit``). Rolling features use the past cycles of the
    same engine only, so a row's features never depend on the future or on other engines.
    Input X must contain engine_uid, cycle, SETTINGS and ``sensors``; rows may be in any order.
    """

    def __init__(self, sensors=None, window: int = 30, include_cycle: bool = True,
                 random_state: int = RANDOM_STATE):
        self.sensors = sensors
        self.window = window
        self.include_cycle = include_cycle
        self.random_state = random_state

    def fit(self, X, y=None):
        self.normalizer_ = ConditionNormalizer(sensors=self.sensors, random_state=self.random_state).fit(X)
        self.sensors_ = self.normalizer_.sensors_
        return self

    def transform(self, X):
        z = self.normalizer_.transform(X)
        z["engine_uid"], z["cycle"] = X["engine_uid"].to_numpy(), X["cycle"].to_numpy()
        z = z.sort_values(["engine_uid", "cycle"])
        g = z.groupby("engine_uid", sort=False)
        w = self.window
        roll = lambda s: s.rolling(w, min_periods=1)
        out = {"cycle": z["cycle"]}
        t = z["cycle"].astype(float)
        mean_t = g["cycle"].transform(lambda s: roll(s.astype(float)).mean())
        mean_t2 = g["cycle"].transform(lambda s: roll(s.astype(float) ** 2).mean())
        var_t = mean_t2 - mean_t ** 2
        for s in self.sensors_:
            m = g[s].transform(lambda v: roll(v).mean())
            mtx = (z[s] * t).groupby(z["engine_uid"], sort=False).transform(lambda v: roll(v).mean())
            slope = (mtx - mean_t * m) / var_t.where(var_t > 1e-9)
            out[f"{s}"] = z[s]
            out[f"{s}_mean{w}"] = m
            out[f"{s}_slope{w}"] = slope.fillna(0.0)
        if not self.include_cycle:
            out.pop("cycle")
        F = pd.DataFrame(out).loc[X.index]          # back to the input row order
        self.feature_names_ = list(F.columns)
        return F

    def get_feature_names_out(self, input_features=None):
        return np.asarray(self.feature_names_, dtype=object)


class SubsampledRegressor(BaseEstimator, RegressorMixin):
    """Fit ``estimator`` on one row out of ``step`` (rows are ordered engine/cycle, so this keeps
    every engine but one cycle in ``step``). Used for SVR, whose training cost grows ~n^2."""

    def __init__(self, estimator=None, step: int = 5):
        self.estimator = estimator
        self.step = step

    def fit(self, X, y):
        idx = np.arange(len(X))[:: self.step]
        Xs = X.iloc[idx] if hasattr(X, "iloc") else X[idx]
        ys = y.iloc[idx] if hasattr(y, "iloc") else np.asarray(y)[idx]
        self.estimator_ = clone(self.estimator).fit(Xs, ys)
        return self

    def predict(self, X):
        return self.estimator_.predict(X)


def cross_validate_model(model, X, y, folds, seed: int = RANDOM_STATE, target_col: str = "RUL_capped") -> pd.DataFrame:
    """GroupKFold evaluation. The whole pipeline (normalisation included) is cloned and re-fitted on
    each training fold. Predictions are made on the full validation trajectories (features are causal),
    then scored (a) on all rows and (b) on one random cut per engine, which mimics the test set."""
    rows = []
    for k, (tr_idx, va_idx) in enumerate(folds):
        t0 = time.perf_counter()
        m = clone(model).fit(X.iloc[tr_idx], y.iloc[tr_idx])
        fit_s = time.perf_counter() - t0
        va = X.iloc[va_idx]
        pred = pd.Series(np.clip(m.predict(va), 0, None), index=va.index)
        all_rows = regression_report(y.iloc[va_idx], pred)
        cut = random_truncation(va, seed=seed + k)
        one = regression_report(cut[target_col], pred.loc[cut.index])
        rows.append({"fold": k, "RMSE all rows": all_rows["RMSE"], "MAE all rows": all_rows["MAE"],
                     "RMSE random cut": one["RMSE"], "MAE random cut": one["MAE"],
                     "NASA score / engine (random cut)": one["NASA score"] / one["n"],
                     "fit time (s)": fit_s})
    return pd.DataFrame(rows)
