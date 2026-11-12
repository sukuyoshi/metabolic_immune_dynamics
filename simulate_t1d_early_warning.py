import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from src.metabolic_immune_dynamics import MetabolicImmuneSystem

def run_t1d_simulation():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    res_dir = os.path.join(base_dir, "results")
    os.makedirs(res_dir, exist_ok=True)
    
    system = MetabolicImmuneSystem(
        gamma_eff=0.35,
        rho_reg=0.20,
        kappa_kill=0.25,
        alpha_ag=15.0
    )
    
    t_span = (0.0, 365.0)
    t_eval = np.linspace(0.0, 365.0, 3651)
    
    y0_presymptomatic = [95.0, 9.5, 300.0, 10.0, 5.0, 0.08, 1.0, 0.01]
    
    def t1d_drift(t, state):
        trigger_ramp = 0.002 * (1.0 + t / 100.0)
        res = system.vector_field(t, state, mu_fat=0.05, u_meta=0.0, u_imm=0.0, d_meal=0.0)
        res[5] += trigger_ramp
        return res

    sol = solve_ivp(
        t1d_drift,
        t_span,
        y0_presymptomatic,
        t_eval=t_eval,
        method="RK45",
        rtol=1e-6,
        atol=1e-8
    )
    
    n_pts = len(sol.t)
    dt = sol.t[1] - sol.t[0]
    
    omega1_immune = np.zeros(n_pts)
    raychaudhuri_theta = np.zeros(n_pts)
    
    t_eff_arr = sol.y[5]
    t_reg_arr = sol.y[6]
    c_inf_arr = sol.y[7]
    
    window_pts = 60
    for k in range(n_pts):
        start_idx = max(0, k - window_pts + 1)
        if k - start_idx < 10:
            omega1_immune[k] = 0.12
            raychaudhuri_theta[k] = -0.25
            continue
            
        y_slice = np.vstack([
            t_eff_arr[start_idx : k + 1],
            t_reg_arr[start_idx : k + 1],
            c_inf_arr[start_idx : k + 1]
        ])
        
        y0 = y_slice[:, :-1]
        y1 = y_slice[:, 1:]
        
        reg_eye = 1e-4 * np.eye(3)
        cov0 = y0 @ y0.T + reg_eye
        cross_cov = y1 @ y0.T
        
        try:
            a_mat = cross_cov @ np.linalg.inv(cov0)
            eig_vals = np.linalg.eigvals(a_mat)
            cont_eigs = np.log(np.maximum(1e-6, np.abs(eig_vals))) / dt
            
            omega1_immune[k] = - float(np.max(cont_eigs))
            raychaudhuri_theta[k] = float(np.sum(cont_eigs))
        except Exception:
            omega1_immune[k] = omega1_immune[k - 1]
            raychaudhuri_theta[k] = raychaudhuri_theta[k - 1]

    smooth_span = 40
    kernel = np.ones(smooth_span) / smooth_span
    omega1_smooth = np.convolve(omega1_immune, kernel, mode="same")
    
    df_t1d = pd.DataFrame({
        "time_days": sol.t,
        "glucose": sol.y[0],
        "beta_func": sol.y[2],
        "beta_dead": sol.y[4],
        "t_effector": sol.y[5],
        "t_reg": sol.y[6],
        "omega1_immune": omega1_smooth,
        "raychaudhuri_theta": raychaudhuri_theta
    })
    
    csv_path = os.path.join(res_dir, "t1d_early_warning_simulation.csv")
    df_t1d.to_csv(csv_path, index=False)
    
    clinical_onset_day = None
    onset_mask = sol.y[0] > 180.0
    if np.any(onset_mask):
        clinical_onset_day = float(sol.t[np.argmax(onset_mask)])
        
    warning_day = None
    warning_mask = (omega1_smooth < 0.02) & (sol.t > 20.0)
    if np.any(warning_mask):
        warning_day = float(sol.t[np.argmax(warning_mask)])
        
    plt.figure(figsize=(14, 10))
    
    plt.subplot(2, 2, 1)
    plt.plot(sol.t, sol.y[0], "k-", linewidth=2.5, label="Fasting Blood Glucose")
    plt.axhline(100.0, color="green", linestyle=":", label="Normal Target (100 mg/dL)")
    plt.axhline(180.0, color="red", linestyle="--", label="Clinical Diagnosis (180 mg/dL)")
    if clinical_onset_day:
        plt.axvline(clinical_onset_day, color="red", linewidth=2.0, alpha=0.7, label=f"Clinical Onset (Day {clinical_onset_day:.1f})")
    if warning_day:
        plt.axvline(warning_day, color="blue", linewidth=2.0, linestyle="--", label=f"Omega_1 Alert (Day {warning_day:.1f})")
    plt.xlabel("Time (Days)", fontsize=12)
    plt.ylabel("Glucose G (mg/dL)", fontsize=12)
    plt.title("Clinical Glucose Manifestation in Autoimmune T1D", fontsize=13)
    plt.grid(True, alpha=0.3)
    plt.legend(loc="upper left", fontsize=9)
    
    plt.subplot(2, 2, 2)
    plt.plot(sol.t, sol.y[2], "b-", linewidth=2.5, label="Functional Beta Mass")
    plt.plot(sol.t, sol.y[4], "r-.", linewidth=2.0, label="Destroyed Beta Mass")
    if warning_day:
        plt.axvline(warning_day, color="blue", linewidth=2.0, linestyle="--")
    if clinical_onset_day:
        plt.axvline(clinical_onset_day, color="red", linewidth=2.0, alpha=0.7)
    plt.xlabel("Time (Days)", fontsize=12)
    plt.ylabel("Beta-Cell Mass (mg)", fontsize=12)
    plt.title("Sub-Clinical Autoimmune Beta-Cell Destruction", fontsize=13)
    plt.grid(True, alpha=0.3)
    plt.legend(loc="center left", fontsize=9)
    
    plt.subplot(2, 2, 3)
    plt.plot(sol.t, sol.y[5], "r-", linewidth=2.0, label="Auto-reactive Effector T (Teff)")
    plt.plot(sol.t, sol.y[6], "g-", linewidth=2.0, label="Regulatory T (Treg)")
    if warning_day:
        plt.axvline(warning_day, color="blue", linewidth=2.0, linestyle="--")
    plt.xlabel("Time (Days)", fontsize=12)
    plt.ylabel("Immune Population Density", fontsize=12)
    plt.title("Immune Repertoire Dynamics (Teff vs Treg)", fontsize=13)
    plt.grid(True, alpha=0.3)
    plt.legend(loc="upper left", fontsize=9)
    
    plt.subplot(2, 2, 4)
    plt.plot(sol.t, omega1_smooth, "m-", linewidth=2.5, label="Spectral Gap Omega_1 (Immune)")
    plt.axhline(0.0, color="black", linestyle="-", linewidth=1.0)
    plt.axhline(0.02, color="orange", linestyle=":", label="Early Warning Softening Threshold")
    if warning_day and clinical_onset_day:
        lead_days = clinical_onset_day - warning_day
        plt.axvspan(warning_day, clinical_onset_day, color="green", alpha=0.15, label=f"Early Warning Lead Time: {lead_days:.1f} Days")
    plt.xlabel("Time (Days)", fontsize=12)
    plt.ylabel("Turbulence Gap Omega_1", fontsize=12)
    plt.title("Precursor Phase Transition: Spectral Gap Softening", fontsize=13)
    plt.grid(True, alpha=0.3)
    plt.legend(loc="lower left", fontsize=9)
    
    plt.tight_layout()
    plot_path = os.path.join(res_dir, "t1d_gap_softening_plot.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()
    
    print("T1D Early Warning Simulation Completed.")
    print(f"Results saved to {csv_path} and {plot_path}")
    if warning_day and clinical_onset_day:
        print(f"Pre-symptomatic Early Warning Lead Time: {clinical_onset_day - warning_day:.1f} days before clinical onset.")

if __name__ == "__main__":
    run_t1d_simulation()
