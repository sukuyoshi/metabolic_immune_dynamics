import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from src.metabolic_immune_dynamics import MetabolicImmuneSystem

def run_simulation():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    res_dir = os.path.join(base_dir, "results")
    os.makedirs(res_dir, exist_ok=True)
    system = MetabolicImmuneSystem()
    
    t_span = (0.0, 180.0)
    t_eval = np.linspace(0.0, 180.0, 1801)
    
    y0_diabetic = [220.0, 12.0, 120.0, 180.0, 50.0, 0.02, 1.0, 0.02]
    
    def intervention_successful(t, state):
        if 14.0 <= t <= 70.0:
            mu_fat = max(0.05, 1.5 - (t - 14.0) / 35.0)
            u_meta = 4.0
            d_meal = -350.0
        elif t > 70.0:
            mu_fat = 0.05
            u_meta = 0.0
            d_meal = 0.0
        else:
            mu_fat = 1.5
            u_meta = 0.0
            d_meal = 0.0
        return system.vector_field(t, state, mu_fat=mu_fat, u_meta=u_meta, d_meal=d_meal)

    def intervention_insufficient(t, state):
        if 14.0 <= t <= 35.0:
            mu_fat = max(0.9, 1.5 - (t - 14.0) / 30.0)
            u_meta = 1.0
            d_meal = -150.0
        elif t > 35.0:
            mu_fat = 1.4
            u_meta = 0.0
            d_meal = 0.0
        else:
            mu_fat = 1.5
            u_meta = 0.0
            d_meal = 0.0
        return system.vector_field(t, state, mu_fat=mu_fat, u_meta=u_meta, d_meal=d_meal)

    sol_succ = solve_ivp(
        intervention_successful,
        t_span,
        y0_diabetic,
        t_eval=t_eval,
        method="RK45",
        rtol=1e-6,
        atol=1e-8
    )
    
    sol_fail = solve_ivp(
        intervention_insufficient,
        t_span,
        y0_diabetic,
        t_eval=t_eval,
        method="RK45",
        rtol=1e-6,
        atol=1e-8
    )
    
    df_res = pd.DataFrame({
        "time_days": sol_succ.t,
        "glucose_success": sol_succ.y[0],
        "insulin_success": sol_succ.y[1],
        "beta_func_success": sol_succ.y[2],
        "beta_dediff_success": sol_succ.y[3],
        "glucose_fail": sol_fail.y[0],
        "insulin_fail": sol_fail.y[1],
        "beta_func_fail": sol_fail.y[2],
        "beta_dediff_fail": sol_fail.y[3]
    })
    csv_path = os.path.join(res_dir, "t2d_remission_simulation.csv")
    df_res.to_csv(csv_path, index=False)
    
    plt.figure(figsize=(14, 10))
    
    plt.subplot(2, 2, 1)
    plt.plot(sol_succ.t, sol_succ.y[0], "b-", linewidth=2.5, label="Optimal Pulse (Remission)")
    plt.plot(sol_fail.t, sol_fail.y[0], "r--", linewidth=2.0, label="Insufficient Pulse (Relapse)")
    plt.axvspan(14.0, 70.0, color="blue", alpha=0.1, label="Active Window (Success 8w)")
    plt.axvspan(14.0, 35.0, color="red", alpha=0.1, label="Active Window (Fail 3w)")
    plt.axhline(100.0, color="green", linestyle=":", label="Normal Fasting (100 mg/dL)")
    plt.axhline(180.0, color="orange", linestyle=":", label="Diabetic Threshold (180 mg/dL)")
    plt.xlabel("Time (Days)", fontsize=12)
    plt.ylabel("Fasting Plasma Glucose (mg/dL)", fontsize=12)
    plt.title("Fasting Glucose Dynamics: Remission vs Relapse", fontsize=13)
    plt.grid(True, alpha=0.3)
    plt.legend(loc="upper right", fontsize=9)
    
    plt.subplot(2, 2, 2)
    plt.plot(sol_succ.t, sol_succ.y[2], "b-", linewidth=2.5, label="Functional Beta (Success)")
    plt.plot(sol_succ.t, sol_succ.y[3], "c-.", linewidth=2.0, label="Dedifferentiated (Success)")
    plt.plot(sol_fail.t, sol_fail.y[2], "r--", linewidth=2.0, label="Functional Beta (Fail)")
    plt.plot(sol_fail.t, sol_fail.y[3], "m:", linewidth=2.0, label="Dedifferentiated (Fail)")
    plt.xlabel("Time (Days)", fontsize=12)
    plt.ylabel("Beta-Cell Mass (mg)", fontsize=12)
    plt.title("Beta-Cell Redifferentiation and Mass Dynamics", fontsize=13)
    plt.grid(True, alpha=0.3)
    plt.legend(loc="center right", fontsize=9)
    
    plt.subplot(2, 2, 3)
    plt.plot(sol_succ.y[0], sol_succ.y[2], "b-", linewidth=2.5, label="Successful Steering Trajectory")
    plt.plot(sol_fail.y[0], sol_fail.y[2], "r--", linewidth=2.0, label="Failed Steering Trajectory")
    plt.scatter([sol_succ.y[0][0]], [sol_succ.y[2][0]], color="black", s=80, zorder=5, label="Diabetic Initial State")
    plt.scatter([sol_succ.y[0][-1]], [sol_succ.y[2][-1]], color="blue", marker="*", s=150, zorder=5, label="Healthy Attractor")
    plt.scatter([sol_fail.y[0][-1]], [sol_fail.y[2][-1]], color="red", marker="x", s=100, zorder=5, label="Relapsed Attractor")
    plt.xlabel("Glucose G (mg/dL)", fontsize=12)
    plt.ylabel("Functional Beta-Cell Mass (mg)", fontsize=12)
    plt.title("State-Space Phase Portrait (Attractor Steering)", fontsize=13)
    plt.grid(True, alpha=0.3)
    plt.legend(loc="upper right", fontsize=9)
    
    plt.subplot(2, 2, 4)
    storage_succ = 0.5 * ((sol_succ.y[0] - 100.0)/50.0)**2 + 0.5 * ((sol_succ.y[2] - 300.0)/100.0)**2
    storage_fail = 0.5 * ((sol_fail.y[0] - 100.0)/50.0)**2 + 0.5 * ((sol_fail.y[2] - 300.0)/100.0)**2
    plt.plot(sol_succ.t, storage_succ, "b-", linewidth=2.5, label="Storage Function S(X) (Success)")
    plt.plot(sol_fail.t, storage_fail, "r--", linewidth=2.0, label="Storage Function S(X) (Fail)")
    plt.axvspan(14.0, 70.0, color="blue", alpha=0.1)
    plt.xlabel("Time (Days)", fontsize=12)
    plt.ylabel("Willems Storage Function S(X)", fontsize=12)
    plt.title("Dissipative Storage Function Evolution", fontsize=13)
    plt.grid(True, alpha=0.3)
    plt.legend(loc="upper right", fontsize=9)
    
    plt.tight_layout()
    plot_path = os.path.join(res_dir, "t2d_remission_phase_portrait.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()
    
    print("T2D Remission Simulation Completed.")
    print(f"Results saved to {csv_path} and {plot_path}")

if __name__ == "__main__":
    run_simulation()
