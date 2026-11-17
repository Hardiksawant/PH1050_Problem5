"""Part 2: Data Cleaning, Windowing, and Statistical Feature Extraction."""

import math
import numpy as np
import pandas as pd
from scipy import stats


def clean_sensor_series(raw_signal):
    """Detects missing values (NaN) and imputes via linear time interpolation."""
    n_missing = int(np.isnan(raw_signal).sum())
    cleaned = pd.Series(raw_signal).interpolate(method="linear").bfill().ffill().to_numpy()
    return cleaned, n_missing


def compute_window_features(signal, window_size=1000, label=0, condition="Condition_A"):
    """Cuts signal into non-overlapping windows of 1,000 samples and computes RMS, excess kurtosis, and crest factor."""
    n_windows = len(signal) // window_size
    records = []
    for w in range(n_windows):
        seg = signal[w * window_size : (w + 1) * window_size]
        rms = float(np.sqrt(np.mean(seg**2)))
        kurt = float(stats.kurtosis(seg, fisher=True, bias=False))
        crest = float(np.max(np.abs(seg)) / rms)
        records.append({
            "window_id": w,
            "condition": condition,
            "state": "Faulty" if label == 1 else "Healthy",
            "label": label,
            "rms": rms,
            "kurtosis": kurt,
            "crest_factor": crest,
        })
    return pd.DataFrame(records)


def welch_rms_test(df_healthy, df_faulty):
    """Performs two-sample Welch t-test on RMS between faulty and healthy windows."""
    h_rms = df_healthy["rms"].to_numpy()
    f_rms = df_faulty["rms"].to_numpy()
    t_stat, p_val = stats.ttest_ind(f_rms, h_rms, equal_var=False)
    diff = float(np.mean(f_rms) - np.mean(h_rms))
    n1, n0 = len(f_rms), len(h_rms)
    v1, v0 = float(np.var(f_rms, ddof=1)), float(np.var(h_rms, ddof=1))
    se = math.sqrt(v1 / n1 + v0 / n0)
    df = ((v1 / n1 + v0 / n0)**2) / (((v1 / n1)**2) / (n1 - 1) + ((v0 / n0)**2) / (n0 - 1))
    t_crit = float(stats.t.ppf(0.975, df=df))
    return {
        "mean_healthy": float(np.mean(h_rms)),
        "mean_faulty": float(np.mean(f_rms)),
        "diff": diff,
        "ci_95": (diff - t_crit * se, diff + t_crit * se),
        "t_stat": float(t_stat),
        "df": float(df),
        "p_val": float(p_val),
    }
