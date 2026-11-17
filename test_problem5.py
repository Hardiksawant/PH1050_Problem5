"""Unit Test Suite for PH1050 Problem 5 (Rubrics B2, B3, B4, B5, C4, D6)."""

import math
import sys
import pathlib
import unittest
import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from src.physics_model import (
    solve_euler,
    solve_rk4,
    exact_free_solution,
    steady_state_amplitude_exact,
    measure_period,
    measure_damping_rate,
)
from src.data_statistics import clean_sensor_series, compute_window_features, welch_rms_test
from src.ml_models import KNNClassifier, DecisionTreeClassifierScratch, chronological_train_test_split


class TestPH1050Problem5(unittest.TestCase):
    def test_b2_free_period_and_damped_decay(self):
        """Rubric B2: Free undamped period 2*pi/omega_0 and damped decay exp(-c*t/(2*m)) match analytic."""
        t, x, _ = solve_rk4(dt=5e-4, t_end=6.0, x0=1.0, v0=0.0, m=1.0, c=0.0, k=100.0, F0=0.0)
        T_sim = measure_period(t, x)
        T_exact = 2.0 * math.pi / 10.0
        self.assertAlmostEqual(T_sim, T_exact, places=4)

        td, xd, _ = solve_rk4(dt=1e-3, t_end=10.0, x0=1.0, v0=0.0, m=1.0, c=0.5, k=100.0, F0=0.0)
        gamma_sim = measure_damping_rate(td, xd)
        self.assertAlmostEqual(gamma_sim, 0.25, places=3)

    def test_b3_energy_decay_conservation(self):
        """Rubric B3: Mechanical energy decay and work-energy balance within tolerance 1e-5."""
        t, x, v = solve_rk4(dt=1e-3, t_end=8.0, x0=1.0, v0=0.0, m=1.0, c=0.5, k=100.0, F0=0.0)
        E = 0.5 * 1.0 * v**2 + 0.5 * 100.0 * x**2
        dt = t[1] - t[0]
        p_diss = 0.5 * v**2
        W_diss = np.zeros_like(E)
        W_diss[1:] = np.cumsum(0.5 * (p_diss[:-1] + p_diss[1:]) * dt)
        rel_err = np.max(np.abs((E + W_diss) - E[0]) / E[0])
        self.assertLess(rel_err, 1e-5)

    def test_b4_convergence_orders(self):
        """Rubric B4: Euler is 1st order (~1.0) and RK4 is 4th order (~4.0)."""
        dts = np.array([2e-3, 1e-3, 5e-4, 2.5e-4])
        x_ex, _ = exact_free_solution(np.array([1.0]), x0=1.0, v0=0.0, m=1.0, c=0.5, k=100.0)
        err_e, err_r = [], []
        for dt in dts:
            _, xe, _ = solve_euler(dt, 1.0, x0=1.0, v0=0.0, m=1.0, c=0.5, k=100.0)
            _, xr, _ = solve_rk4(dt, 1.0, x0=1.0, v0=0.0, m=1.0, c=0.5, k=100.0)
            err_e.append(abs(xe[-1] - x_ex[0]))
            err_r.append(abs(xr[-1] - x_ex[0]))
        slope_e = np.polyfit(np.log10(dts), np.log10(err_e), 1)[0]
        slope_r = np.polyfit(np.log10(dts), np.log10(err_r), 1)[0]
        self.assertTrue(0.90 <= slope_e <= 1.15)
        self.assertTrue(3.85 <= slope_r <= 4.15)

    def test_b5_c4_d6_data_and_ml_pipeline(self):
        """Rubrics B5, C4, D6: Kurtosis shift, Welch t-test on RMS, and 70/30 chronological ML split."""
        data_dir = pathlib.Path(__file__).resolve().parents[1] / "data"
        df_a = pd.read_csv(data_dir / "condition_A_1HP_windows.csv")
        df_h = df_a[df_a["label"] == 0]
        df_f = df_a[df_a["label"] == 1]
        self.assertEqual(len(df_h), 100)
        self.assertEqual(len(df_f), 100)
        self.assertGreater(df_f["kurtosis"].mean(), df_h["kurtosis"].mean() + 2.0)

        res = welch_rms_test(df_h, df_f)
        self.assertLess(res["p_val"], 1e-10)
        self.assertGreater(res["ci_95"][0], 0.0)

        tr, te = chronological_train_test_split(df_h, df_f, train_frac=0.70)
        self.assertEqual(len(tr), 140)
        self.assertEqual(len(te), 60)
        knn = KNNClassifier(k=3).fit(tr[["rms", "kurtosis", "crest_factor"]], tr["label"])
        preds = knn.predict(te[["rms", "kurtosis", "crest_factor"]])
        acc = float(np.mean(preds == te["label"].to_numpy()))
        self.assertGreater(acc, 0.90)


if __name__ == "__main__":
    unittest.main()
