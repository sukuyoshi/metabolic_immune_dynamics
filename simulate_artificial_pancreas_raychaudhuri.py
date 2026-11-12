import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def run_artificial_pancreas_simulation():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    res_dir = os.path.join(base_dir, "results")
    os.makedirs(res_dir, exist_ok=True)
    
    dt = 1.0
    total_minutes = 360
    n_steps = int(total_minutes / dt)
    time_min = np.linspace(0.0, total_minutes, n_steps)
    
    g_target = 110.0
    
    r0 = 0.60
    eg0 = 0.001
    si = 0.0005
    ke = 0.05
    
    meal_rate = np.zeros(n_steps)
    for i, t in enumerate(time_min):
        if 60.0 <= t <= 120.0:
            meal_rate[i] = 4.8 * np.sin(np.pi * (t - 60.0) / 60.0)
            
    g_pid = np.zeros(n_steps)
    i_pid = np.zeros(n_steps)
    u_pid = np.zeros(n_steps)
    
    g_ray = np.zeros(n_steps)
    i_ray = np.zeros(n_steps)
    u_ray = np.zeros(n_steps)
    theta_ray = np.zeros(n_steps)
    
    g_pid[0] = 110.0
    i_pid[0] = 10.0
    g_ray[0] = 110.0
    i_ray[0] = 10.0
    
    kp = 0.04
    ki = 0.0002
    kd = 0.4
    integral_err = 0.0
    
    d_dim = 4
    tau_step = 3
    window_m = 25
    g_history_ray = [110.0] * (d_dim * tau_step + window_m)
    
    for k in range(n_steps - 1):
        err = g_pid[k] - g_target
        integral_err += err * dt
        deriv_err = (g_pid[k] - g_pid[k - 1]) / dt if k > 0 else 0.0
        
        u_base = 0.50
        u_p = u_base + kp * err + ki * integral_err + kd * deriv_err
        u_pid[k] = max(0.05, min(5.0, u_p))
        
        dg_pid = r0 - (eg0 + si * i_pid[k]) * g_pid[k] + meal_rate[k]
        di_pid = u_pid[k] - ke * i_pid[k]
        
        g_pid[k + 1] = max(30.0, g_pid[k] + dg_pid * dt)
        i_pid[k + 1] = max(0.1, i_pid[k] + di_pid * dt)
        
        g_history_ray.append(g_ray[k])
        
        if len(g_history_ray) >= (d_dim - 1) * tau_step + window_m:
            vectors = []
            for j in range(window_m):
                idx = len(g_history_ray) - window_m + j
                v = [g_history_ray[idx - m * tau_step] for m in range(d_dim)]
                vectors.append(v)
            mat_x = np.array(vectors).T
            
            y0 = mat_x[:, :-1]
            y1 = mat_x[:, 1:]
            cov0 = y0 @ y0.T + 1e-4 * np.eye(d_dim)
            cross = y1 @ y0.T
            
            try:
                a_mat = cross @ np.linalg.inv(cov0)
                eigs = np.linalg.eigvals(a_mat)
                theta_val = float(np.sum(np.log(np.maximum(1e-6, np.abs(eigs))) / dt))
            except Exception:
                theta_val = theta_ray[k - 1] if k > 0 else 0.0
        else:
            theta_val = 0.0
        theta_ray[k] = theta_val
        
        err_ray = g_ray[k] - g_target
        u_geom = u_base + kp * err_ray + kd * (g_ray[k] - g_ray[k - 1]) if k > 0 else u_base
        if theta_val > 0.04 and g_ray[k] >= 110.0 and (k == 0 or g_ray[k] >= g_ray[k - 1]):
            anticipation_boost = 1.2 * min(1.8, theta_val * 10.0)
            u_geom += anticipation_boost
        u_ray[k] = max(0.05, min(4.5, u_geom))
        
        dg_ray = r0 - (eg0 + si * i_ray[k]) * g_ray[k] + meal_rate[k]
        di_ray = u_ray[k] - ke * i_ray[k]
        
        g_ray[k + 1] = max(30.0, g_ray[k] + dg_ray * dt)
        i_ray[k + 1] = max(0.1, i_ray[k] + di_ray * dt)

    u_pid[-1] = u_pid[-2]
    u_ray[-1] = u_ray[-2]
    theta_ray[-1] = theta_ray[-2]
    
    df_aid = pd.DataFrame({
        "time_min": time_min,
        "meal_flux": meal_rate,
        "glucose_pid": g_pid,
        "insulin_pid": i_pid,
        "infusion_pid": u_pid,
        "glucose_raychaudhuri": g_ray,
        "insulin_raychaudhuri": i_ray,
        "infusion_raychaudhuri": u_ray,
        "theta_expansion": theta_ray
    })
    
    csv_path = os.path.join(res_dir, "artificial_pancreas_comparison.csv")
    df_aid.to_csv(csv_path, index=False)
    
    plt.figure(figsize=(14, 10))
    
    plt.subplot(2, 2, 1)
    plt.plot(time_min, g_pid, "r--", linewidth=2.0, label="Standard PID Control")
    plt.plot(time_min, g_ray, "b-", linewidth=2.5, label="Raychaudhuri-Predictive AID")
    plt.axvspan(60.0, 120.0, color="orange", alpha=0.15, label="Meal Ingestion (75g CHO)")
    plt.axhline(180.0, color="red", linestyle=":", label="Hyperglycemic Limit (180 mg/dL)")
    plt.axhline(70.0, color="purple", linestyle=":", label="Hypoglycemic Floor (70 mg/dL)")
    plt.axhline(110.0, color="green", linestyle="--", label="Target (110 mg/dL)")
    plt.xlabel("Time (Minutes)", fontsize=12)
    plt.ylabel("Blood Glucose (mg/dL)", fontsize=12)
    plt.title("Postprandial Glucose Control: PID vs Raychaudhuri-AID", fontsize=13)
    plt.grid(True, alpha=0.3)
    plt.legend(loc="upper right", fontsize=9)
    
    plt.subplot(2, 2, 2)
    plt.plot(time_min, u_pid, "r--", linewidth=2.0, label="PID Infusion Rate")
    plt.plot(time_min, u_ray, "b-", linewidth=2.5, label="Raychaudhuri Anticipatory Infusion")
    plt.axvspan(60.0, 120.0, color="orange", alpha=0.15)
    plt.xlabel("Time (Minutes)", fontsize=12)
    plt.ylabel("Insulin Infusion (U/min)", fontsize=12)
    plt.title("Autonomous Insulin Infusion Rate Dynamics", fontsize=13)
    plt.grid(True, alpha=0.3)
    plt.legend(loc="upper right", fontsize=9)
    
    plt.subplot(2, 2, 3)
    plt.plot(time_min, theta_ray, "m-", linewidth=2.0, label="Raychaudhuri Scalar Theta (Expansion)")
    plt.axhline(0.0, color="black", linestyle="-", linewidth=1.0)
    plt.axhline(0.05, color="orange", linestyle=":", label="Divergence Threshold (UNSAT)")
    plt.axvspan(60.0, 120.0, color="orange", alpha=0.15)
    plt.xlabel("Time (Minutes)", fontsize=12)
    plt.ylabel("Theta (Bundle Expansion Rate)", fontsize=12)
    plt.title("Phase-Space Raychaudhuri Divergence Diagnostic", fontsize=13)
    plt.grid(True, alpha=0.3)
    plt.legend(loc="upper right", fontsize=9)
    
    plt.subplot(2, 2, 4)
    tir_pid = float(np.mean((g_pid >= 70.0) & (g_pid <= 180.0))) * 100.0
    tir_ray = float(np.mean((g_ray >= 70.0) & (g_ray <= 180.0))) * 100.0
    bars = plt.bar(["Standard PID", "Raychaudhuri-AID"], [tir_pid, tir_ray], color=["salmon", "royalblue"], width=0.5)
    plt.ylim(0, 115)
    plt.ylabel("Time in Range (70-180 mg/dL) %", fontsize=12)
    plt.title("Clinical Time-in-Range (TIR) Performance", fontsize=13)
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, yval + 2.0, f"{yval:.1f}%", ha="center", va="bottom", fontsize=11, fontweight="bold")
    plt.grid(True, alpha=0.3, axis="y")
    
    plt.tight_layout()
    plot_path = os.path.join(res_dir, "artificial_pancreas_raychaudhuri_plot.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()
    
    print("Artificial Pancreas Simulation Completed.")
    print(f"Results saved to {csv_path} and {plot_path}")
    print(f"Standard PID TIR: {tir_pid:.2f}% | Raychaudhuri-AID TIR: {tir_ray:.2f}%")

if __name__ == "__main__":
    run_artificial_pancreas_simulation()
