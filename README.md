# PH1050 Final Group Project — Problem 5: Vibrations and Machine Health

**Course:** PH1050 — Foundation of Computational Physics (IIT Madras)  
**Instructor:** Dr. Ayana Ghosh  
**Research Question:** *Can we tell a healthy machine from a faulty one by how it shakes?*

## 1. Team Members & Individual Contributions (Group of 7)

| Member | Student Name | Primary Role & Technical Contributions |
| :--- | :--- | :--- |
| **1** | **Hardik Sawant (EP22B048)** | Physics formulation of the driven damped harmonic oscillator, analytic derivations, and free/damped verification (Rubric B2). |
| **2** | **K Ezhil (EP26B012)** | Mechanical energy dissipation derivation ($dE/dt = -c\dot{x}^2$), numerical power integral check, and resonance sweep analysis (Rubric B3). |
| **3** | **Lohitaksh K (PH26B007)** | Forward Euler and 4th-order Runge-Kutta (RK4) ODE solvers, step-size convergence study, and log-log error slope verification (Rubric B4). |
| **4** | **Preethika Saranya (EP26B037)** | Periodic spall impact fault modelling, unmodeled nonlinear Hertzian contact & cage-slip dataset generation, and missing-data interpolation. |
| **5** | **Reya PK (MM25B042)** | Windowed feature extraction (100 windows $\times$ 1,000 samples), excess kurtosis & crest factor analysis, and Welch's $t$-test on RMS (Rubrics B5, C4). |
| **6** | **Koushik Raghavan (EP26B040)** | Strict 70%/30% chronological train/test split, 5-fold cross-validation tuning for $k$-NN and Decision Tree classifiers vs. majority baseline. |
| **7** | **Anshul Bindal (PH26B002)** | Cross-load extrapolation testing (1 HP $\to$ 3 HP), scale-invariant shape feature comparison, unit testing suite, and repository documentation (Rubric D6). |

## 2. Dataset Provenance & Unmodeled Real-World Physics

In accordance with the project guidelines, this repository includes both:
1. **Pure Linear Physics Simulation (`data/pure_simulation_windows.csv`)**: Direct numerical integration of $m\ddot{x} + c\dot{x} + kx = F(t)$ with $m = 1.0\text{ kg}, k = 100.0\text{ N/m}, c = 0.5\text{ N}\cdot\text{s/m}$, driven by Gaussian white noise for the **Healthy** state and periodic short impact pulses ($T_{\text{kick}} = 125\Delta t$) plus noise for the **Faulty** state ($100$ windows $\times 1,000$ samples per state).
2. **Realistic Multi-Load Machine Benchmark Dataset (`data/condition_A_1HP_windows.csv` & `data/condition_B_3HP_windows.csv`)**: Modeled after the open [Case Western Reserve University (CWRU) Bearing Data Center](https://engineering.case.edu/bearingdatacenter) test rig (1797 RPM / 1 HP nominal load vs. 1720 RPM / 3 HP heavy load, $100$ windows of $1,000$ samples per state). To ensure our evaluation is a genuine physical test, the generated machine recordings include **four realistic effects not present in the linear ODE model**:
   - **Nonlinear Hertzian Contact Stiffness ($\beta x^3$)**: Cubic bearing raceway stiffening under radial load.
   - **Load-Dependent Shaft Rotational Unbalance**: Low-frequency rotational harmonic that shifts baseline RMS upward by $1.67\times$ when motor load increases from 1 HP to 3 HP.
   - **Cage-Slip Timing Jitter & Load-Zone Modulation**: Stochastic $\pm 7\%$ ball-pass impact timing jitter and log-normal impact amplitude modulation.
   - **Sensor Telemetry Dropouts (`NaN` values) & Switching Transients**: Exactly $50$ missing sensor readings ($28$ in Healthy, $22$ in Faulty, $0.025\%$ dropout rate) in Condition A, which are detected and imputed via linear time interpolation prior to windowing.

## 3. Summary of Key Quantitative Results (Rubric Mapping)

| Rubric | Verification Requirement | Quantitative Result |
| :--- | :--- | :--- |
| **B2** | Free period $2\pi/\omega_0$ & damped decay $e^{-ct/2m}$ | Analytic $T_0 = 0.628319\text{ s}$, RK4 $T_{0,\text{sim}} = 0.628319\text{ s}$ (rel. error $7.92e-14$); Analytic $\gamma = 0.2500\text{ s}^{-1}$, RK4 $\gamma_{\text{sim}} = 0.250006\text{ s}^{-1}$ (rel. error $2.25e-05$). |
| **B3** | Energy decay checked within stated tolerance | Stated tolerance $\varepsilon_{\text{tol}} = 10^{-5}$; RK4 work-energy conservation residual $\max|E(t)+W_{\text{diss}}(t)-E_0|/E_0 = 7.82e-07 < 10^{-5}$. |
| **B4** | Log-log convergence plot & measured orders | Measured log-log slope $p_{\text{Euler}} = 1.01$ (theoretical $1.00$) and $p_{\text{RK4}} = 4.04$ (theoretical $4.00$). |
| **B5** | Simulated vs. measured excess kurtosis | Pure Sim: Healthy $-0.129 \pm 0.381$ vs. Faulty $+3.819 \pm 0.873$; Machine Data (1 HP): Healthy $+0.022 \pm 0.202$ vs. Faulty $+8.021 \pm 2.917$. |
| **C4** | Mean RMS difference hypothesis test | Faulty RMS ($0.4116\text{ m/s}^2$) vs. Healthy RMS ($0.2517\text{ m/s}^2$): $\Delta\text{RMS} = +0.1599\text{ m/s}^2$, $95\%\text{ CI} [0.1495, 0.1703]$, Welch $t = 30.45$, $p = 4.12e-65$. |
| **D6** | 70/30 chronological ML & cross-load extrapolation | Same-load test (last 30%): kNN ($k=3$) & Decision Tree ($d=1$) achieve **96.67%** vs. 50.0% baseline. Cross-load (1 HP $\to$ 3 HP): RMS-only drops to **50.00%**, while Kurtosis/Shape models retain **95.50% / 94.50%**. |

## 4. Repository Structure & How to Run

```text
PH1050_Problem5_Repository/
├── README.md                          # Project overview, data provenance & instructions
├── requirements.txt                   # Python package dependencies
├── PH1050_Problem5_Notebook.ipynb     # Self-contained Google Colab Jupyter Notebook
├── src/
│   ├── physics_model.py               # Euler/RK4 ODE solvers, energy & resonance sweep
│   ├── data_statistics.py             # Missing-value interpolation, windowing & Welch t-test
│   └── ml_models.py                   # Chronological split, kNN, Decision Tree & CV
├── data/
│   ├── condition_A_1HP_windows.csv    # 200 windows (1,000 samples each) at 1 HP nominal load
│   ├── condition_B_3HP_windows.csv    # 200 windows (1,000 samples each) at 3 HP heavy load
│   └── pure_simulation_windows.csv    # 200 windows from idealized linear impact simulation
├── figures/                           # High-resolution publication plots (PNG)
└── tests/
    └── test_problem5.py               # Automated unit tests verifying rubrics B2-D6
```

### Quickstart Instructions
```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run automated verification unit tests
python3 -m unittest discover -s tests -v
```
