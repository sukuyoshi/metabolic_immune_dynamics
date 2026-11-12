import numpy as np

class MetabolicImmuneSystem:
    def __init__(
        self,
        r0=864.0,
        eg0=1.44,
        si0=0.72,
        kappa_res=1.2,
        sigma_ins=2.0,
        alpha_g=20000.0,
        ke=20.0,
        kd0=0.03,
        theta_g=140.0,
        kr0=0.05,
        delta_loss=0.0005,
        kappa_kill=0.15,
        gamma_eff=0.20,
        alpha_ag=30.0,
        rho_reg=0.25,
        d_eff=0.05,
        gamma_reg=0.05,
        d_reg=0.05,
        eta_t=0.02,
        d_cyt=0.10
    ):
        self.r0 = r0
        self.eg0 = eg0
        self.si0 = si0
        self.kappa_res = kappa_res
        self.sigma_ins = sigma_ins
        self.alpha_g = alpha_g
        self.ke = ke
        self.kd0 = kd0
        self.theta_g = theta_g
        self.kr0 = 0.25
        self.delta_loss = delta_loss
        self.kappa_kill = kappa_kill
        self.gamma_eff = gamma_eff
        self.alpha_ag = alpha_ag
        self.rho_reg = rho_reg
        self.d_eff = d_eff
        self.gamma_reg = gamma_reg
        self.d_reg = d_reg
        self.eta_t = eta_t
        self.d_cyt = d_cyt

    def vector_field(self, t, state, mu_fat=1.0, u_meta=0.0, u_imm=0.0, d_meal=0.0):
        g, ins, beta_f, beta_d, beta_dead, t_eff, t_reg, c_inf = state
        
        g = max(10.0, min(800.0, g))
        ins = max(0.1, ins)
        beta_f = max(0.0, beta_f)
        beta_d = max(0.0, beta_d)
        t_eff = max(0.0, t_eff)
        t_reg = max(0.01, t_reg)
        c_inf = max(0.0, c_inf)
        
        si = self.si0 / (1.0 + self.kappa_res * mu_fat)
        dg_dt = self.r0 - (self.eg0 + si * ins) * g + d_meal
        
        secretion = (self.sigma_ins * beta_f * (g**2)) / (self.alpha_g + g**2)
        dins_dt = secretion - self.ke * ins
        
        growth_rate = -0.06 + 0.001 * g - 0.000004 * (g**2)
        growth_rate = max(-0.1, min(0.05, growth_rate))
        
        k_dediff = self.kd0 * (g**2 / (self.theta_g**2 + g**2)) * (1.0 + mu_fat)
        k_rediff = self.kr0 / (1.0 + 2.0 * mu_fat)
        
        dbeta_f_dt = (
            growth_rate * beta_f
            - k_dediff * beta_f
            + k_rediff * beta_d
            - self.kappa_kill * t_eff * beta_f
            + u_meta
        )
        dbeta_d_dt = k_dediff * beta_f - k_rediff * beta_d - self.delta_loss * beta_d
        
        apop_rate = max(0.0, - growth_rate) * beta_f
        dbeta_dead_dt = apop_rate + self.kappa_kill * t_eff * beta_f + self.delta_loss * beta_d
        
        auto_ag = 0.05 * beta_f + 0.1 * max(0.0, g - 120.0)
        dt_eff_dt = (
            (self.gamma_eff * auto_ag / (self.alpha_ag + auto_ag)) * t_eff
            - self.rho_reg * t_reg * t_eff
            - self.d_eff * t_eff
        )
        dt_reg_dt = self.gamma_reg * t_reg - self.d_reg * t_reg + u_imm
        dc_inf_dt = self.eta_t * t_eff + 0.001 * max(0.0, g - 100.0) - self.d_cyt * c_inf
        
        return np.array([
            dg_dt, dins_dt, dbeta_f_dt, dbeta_d_dt, dbeta_dead_dt,
            dt_eff_dt, dt_reg_dt, dc_inf_dt
        ])
