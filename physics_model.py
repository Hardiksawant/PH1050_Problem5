"""Part 1: Physics Model (Simulate and Verify) for PH1050 Problem 5."""

import math
import numpy as np
from scipy import stats

M_DEFAULT = 1.0
K_DEFAULT = 100.0
C_DEFAULT = 0.5


def solve_euler(dt, t_end, x0=1.0, v0=0.0, m=M_DEFAULT, c=C_DEFAULT, k=K_DEFAULT, F0=0.0, omega=10.0):
    """Forward Euler solver for m*x'' + c*x' + k*x = F0*cos(omega*t)."""
    n_steps = int(round(t_end / dt))
    t = np.linspace(0.0, n_steps * dt, n_steps + 1)
    x = np.zeros(n_steps + 1)
    v = np.zeros(n_steps + 1)
    x[0], v[0] = x0, v0
    for i in range(n_steps):
        a = (F0 * math.cos(omega * t[i]) - c * v[i] - k * x[i]) / m
        x[i + 1] = x[i] + dt * v[i]
        v[i + 1] = v[i] + dt * a
    return t, x, v


def solve_rk4(dt, t_end, x0=1.0, v0=0.0, m=M_DEFAULT, c=C_DEFAULT, k=K_DEFAULT, F0=0.0, omega=10.0):
    """4th-Order Runge-Kutta (RK4) solver for m*x'' + c*x' + k*x = F0*cos(omega*t)."""
    n_steps = int(round(t_end / dt))
    t = np.linspace(0.0, n_steps * dt, n_steps + 1)
    x = np.zeros(n_steps + 1)
    v = np.zeros(n_steps + 1)
    x[0], v[0] = x0, v0
    for i in range(n_steps):
        ti, xi, vi = t[i], x[i], v[i]
        k1x = vi
        k1v = (F0 * math.cos(omega * ti) - c * vi - k * xi) / m

        t_half = ti + 0.5 * dt
        x2, v2 = xi + 0.5 * dt * k1x, vi + 0.5 * dt * k1v
        k2x, k2v = v2, (F0 * math.cos(omega * t_half) - c * v2 - k * x2) / m

        x3, v3 = xi + 0.5 * dt * k2x, vi + 0.5 * dt * k2v
        k3x, k3v = v3, (F0 * math.cos(omega * t_half) - c * v3 - k * x3) / m

        t_next = ti + dt
        x4, v4 = xi + dt * k3x, vi + dt * k3v
        k4x, k4v = v4, (F0 * math.cos(omega * t_next) - c * v4 - k * x4) / m

        x[i + 1] = xi + (dt / 6.0) * (k1x + 2.0 * k2x + 2.0 * k3x + k4x)
        v[i + 1] = vi + (dt / 6.0) * (k1v + 2.0 * k2v + 2.0 * k3v + k4v)
    return t, x, v


def exact_free_solution(t, x0=1.0, v0=0.0, m=M_DEFAULT, c=C_DEFAULT, k=K_DEFAULT):
    """Analytic solution for free undamped (c=0) or underdamped (0 < c < 2*sqrt(m*k)) oscillator."""
    w0 = math.sqrt(k / m)
    gamma = c / (2.0 * m)
    if gamma == 0.0:
        x = x0 * np.cos(w0 * t) + (v0 / w0) * np.sin(w0 * t)
        v = -x0 * w0 * np.sin(w0 * t) + v0 * np.cos(w0 * t)
        return x, v
    wd = math.sqrt(w0**2 - gamma**2)
    A = x0
    B = (v0 + gamma * x0) / wd
    exp_term = np.exp(-gamma * t)
    x = exp_term * (A * np.cos(wd * t) + B * np.sin(wd * t))
    v = exp_term * ((B * wd - A * gamma) * np.cos(wd * t) - (A * wd + B * gamma) * np.sin(wd * t))
    return x, v


def steady_state_amplitude_exact(omega, F0=10.0, m=M_DEFAULT, c=C_DEFAULT, k=K_DEFAULT):
    """Exact steady-state amplitude A(omega) = F0 / sqrt((k - m*omega^2)^2 + (c*omega)^2)."""
    return F0 / np.sqrt((k - m * (omega**2))**2 + (c * omega)**2)


def measure_period(t, x):
    """Measures mean oscillation period from upward zero crossings with linear interpolation."""
    crossings = []
    for i in range(len(x) - 1):
        if x[i] < 0.0 and x[i + 1] >= 0.0:
            tc = t[i] - x[i] * (t[i + 1] - t[i]) / (x[i + 1] - x[i])
            crossings.append(tc)
    return float(np.mean(np.diff(crossings)))


def measure_damping_rate(t, x):
    """Measures exponential decay constant gamma = c/(2m) from positive local peaks."""
    peaks = [i for i in range(1, len(x) - 1) if x[i] > x[i - 1] and x[i] > x[i + 1] and x[i] > 0]
    slope, _, _, _, _ = stats.linregress(t[peaks], np.log(x[peaks]))
    return float(-slope)
