import numpy as np
from scipy.optimize import minimize

from EconModel import EconModelClass

from consav.grids import nonlinspace
from consav.linear_interp import interp_1d, interp_1d_vec
from consav.quadrature import log_normal_gauss_hermite

from optimizer import optimizer

class ModelClass(EconModelClass):

    def settings(self):
        """ fundamental settings """

        pass


    def setup(self):
        """ set baseline parameters """

        # unpack
        par = self.par

        par.T = 4

        par.X = 0.8
        par.psi = 0.9
        par.beta = 0.96

        par.h_max = 1
        par.h_min = 0.1

        # simulation
        par.N_sim = 200

        par.mu = -2.0       # mean of log(X)
        par.sigma = 0.5    # std of log(X)

        # method
        par.method = 'NVFI' # 'VFI' or 'NVFI'
        par.labour_market_flexibility = 'flexible' # 'flexible' or 'rigid'


    def allocate(self):

        """ allocate model """

        # unpack
        par = self.par
        sol = self.sol
        sim = self.sim

        # grids
        par.N_a = 200
        par.N_a_max = 4
        par.a_grid = np.arange(0, par.N_a + 1) / (par.N_a / par.N_a_max)
        par.m_grid = np.arange(0, par.N_a + 1) / (par.N_a / (par.N_a_max + par.h_max + par.X))

        # solutions arrays
        shape = (par.T, par.N_a + 1)

        sol.V_work = np.full(shape, np.nan)
        sol.h_work = np.full(shape, np.nan)
        sol.c_work = np.full(shape, np.nan)

        sol.V_notwork = np.full(shape, np.nan)
        sol.h_notwork = np.full(shape, np.nan)
        sol.c_notwork = np.full(shape, np.nan)

        sol.c_given_m = np.full(shape, np.nan)

        # simulations arrays
        sim.a_init = np.clip(np.random.lognormal(mean=par.mu, sigma=par.sigma, size=par.N_sim), 0, par.N_a_max)

        sim.shape = (par.T, par.N_sim)
        sim.shape_a = (par.T + 1, par.N_sim)

        sim.a = np.full(sim.shape_a, np.nan)
        sim.h = np.full(sim.shape, np.nan)
        sim.c = np.full(sim.shape, np.nan)


    def solve(self):

        # a. unpack
        par = self.par
        sol = self.sol

        if par.method == 'VFI':
            solve_VFI(par, sol)
        elif par.method == 'NVFI': 
            solve_NVFI(par, sol)
        
    def simulate(self):

        # a. unpack
        par = self.par
        sol = self.sol
        sim = self.sim

        # b. simulate
        sim.a[0, :] = sim.a_init

        for t in range(par.T):

            for i in range(par.N_sim):
                a = sim.a[t, i]

                sim_V_work = interp_1d(par.a_grid, sol.V_work[t, :], a)
                sim_V_notwork = interp_1d(par.a_grid, sol.V_notwork[t, :], a)

                if sim_V_work > sim_V_notwork:
                    h = interp_1d(par.a_grid, sol.h_work[t, :], a)
                    c = interp_1d(par.a_grid, sol.c_work[t, :], a)
                else:
                    h = 0.0
                    c = interp_1d(par.a_grid, sol.c_notwork[t, :], a)
                
                sim.h[t, i] = h
                sim.c[t, i] = c

                # update assets
                a_next = a + h + B(h, t, par) - c

                if t < par.T - 1:
                    sim.a[t+1, i] = a_next


def scalar(x):
    return x[0] if np.ndim(x) > 0 else x

def value_function_last_period(h, a, t, par):
    h = scalar(h)
    c = h + a + B(h, t, par)

    if c <= 0:
        c = 0.0
    
    return - (np.log(c+1e-5) - (h**2)/2)


def value_function_inner(c, h, sol_V_work, sol_V_notwork, a, t, par):
    c = scalar(c)
    h = scalar(h)

    max_c = h + a + B(h, t, par)
    a_next = max_c - c

    sol_V_next = max(interp_1d(par.a_grid, sol_V_work[t+1, :], a_next), 
                     interp_1d(par.a_grid, sol_V_notwork[t+1, :], a_next))

    return - (np.log(c+1e-5) - (h**2)/2 + par.beta*sol_V_next)

def B(h, t, par):
    h = scalar(h)
    if t < 2:
        return 0.5 * par.X if h == 0.0 else 0.0
        
    else:
        return (1-par.psi*h)*par.X
    

## VFI
def value_function_VFI(h, sol_V_work, sol_V_notwork, a, t, par):

    max_c = h + a + B(h, t, par)

    c_star = optimizer(value_function_inner, 
                        a=1e-8,
                        b=max_c,
                        args=(h, sol_V_work, sol_V_notwork, a, t, par)
                        )

    v_star = value_function_inner(c_star, h, sol_V_work, sol_V_notwork, a, t, par)

    return v_star

def solve_VFI(par, sol):

    for t in range(par.T - 1, -1, -1):
    
        for a_idx, a in enumerate(par.a_grid):
            
            if t == par.T-1:

                if par.labour_market_flexibility == 'flexible':
                    sol.h_work[t, a_idx] = optimizer(
                                value_function_last_period,
                                a=par.h_min,
                                b=par.h_max,
                                args=(a,t,par)
                    )
                elif par.labour_market_flexibility == 'rigid':
                    sol.h_work[t, a_idx] = par.h_max
                else:
                    raise ValueError("Invalid labour market flexibility setting")
                
                sol.V_work[t, a_idx] = - value_function_last_period(sol.h_work[t, a_idx], a, t, par)
                sol.c_work[t, a_idx] = sol.h_work[t, a_idx] + a + B(sol.h_work[t, a_idx], t, par)

                sol.h_notwork[t, a_idx] = 0.0
                sol.V_notwork[t, a_idx] = - value_function_last_period(0.0, a, t, par)
                sol.c_notwork[t, a_idx] = a + B(sol.h_notwork[t, a_idx], t, par) 


            else:

                if par.labour_market_flexibility == 'flexible':
                    sol.h_work[t, a_idx] = optimizer(value_function_VFI,
                                            a=par.h_min,
                                            b=par.h_max,
                                            args=(sol.V_work, sol.V_notwork, a, t, par)
                    )
                elif par.labour_market_flexibility == 'rigid':
                    sol.h_work[t, a_idx] = par.h_max
                else:
                    raise ValueError("Invalid labour market flexibility setting")
                
                sol.V_work[t, a_idx] = - value_function_VFI(sol.h_work[t, a_idx], sol.V_work, sol.V_notwork, a, t, par)
                max_c = sol.h_work[t, a_idx] + a + B(sol.h_work[t, a_idx], t, par)
                sol.c_work[t, a_idx] = optimizer(value_function_inner,
                                            a=1e-8,
                                            b=max_c,
                                        args=(sol.h_work[t, a_idx], sol.V_work, sol.V_notwork, a, t, par)
                                        )

                sol.h_notwork[t, a_idx] = 0.0
                sol.V_notwork[t, a_idx] = - value_function_VFI(0.0, sol.V_work, sol.V_notwork, a, t, par)
                max_c = sol.h_notwork[t, a_idx] + a + B(sol.h_notwork[t, a_idx], t, par)
                sol.c_notwork[t, a_idx] = optimizer(value_function_inner,
                                                        a=1e-8,
                                                        b=max_c,
                                                        args=(sol.h_notwork[t, a_idx], sol.V_work, sol.V_notwork, a, t, par)
                                                        )


## NVFI
def value_function_NVFI(h, c_given_m, sol_V_work, sol_V_notwork, a, t, par):
    a = scalar(a)
    h = scalar(h)

    m = h + a + B(h, t, par)

    c_star = interp_1d(par.m_grid, c_given_m[t, :], m)

    v_star = value_function_inner(c_star, h, sol_V_work, sol_V_notwork, a, t, par)

    return v_star

def value_function_given_m(c, sol_V_work, sol_V_notwork, m, t, par):
    c = scalar(c)

    a_next = m - c

    sol_V_next = max(interp_1d(par.a_grid, sol_V_work[t+1, :], a_next), 
                     interp_1d(par.a_grid, sol_V_notwork[t+1, :], a_next))

    return - (np.log(c+1e-5) + par.beta*sol_V_next)


def solve_NVFI(par, sol):
    for t in range(par.T - 1, -1, -1):
        
        for m_idx, m in enumerate(par.m_grid):
            if t != par.T-1:
                sol.c_given_m[t, m_idx] = optimizer(
                            value_function_given_m,
                            a=1e-8,
                            b=(m + 1e-6),
                            args=(sol.V_work, sol.V_notwork, m, t, par)
                )

        for a_idx, a in enumerate(par.a_grid):
            
            if t == par.T-1:

                if par.labour_market_flexibility == 'flexible':
                    sol.h_work[t, a_idx] = optimizer(
                                value_function_last_period,
                                a=par.h_min,
                                b=par.h_max,
                                args=(a,t,par)
                    )
                elif par.labour_market_flexibility == 'rigid':
                    sol.h_work[t, a_idx] = par.h_max
                else:
                    raise ValueError("Invalid labour market flexibility setting")
                
                sol.V_work[t, a_idx] = - value_function_last_period(sol.h_work[t, a_idx], a, t, par)
                sol.c_work[t, a_idx] = sol.h_work[t, a_idx] + a + B(sol.h_work[t, a_idx], t, par)


                sol.h_notwork[t, a_idx] = 0.0
                sol.V_notwork[t, a_idx] = - value_function_last_period(sol.h_notwork[t, a_idx], a, t, par)
                sol.c_notwork[t, a_idx] = a + B(sol.h_notwork[t, a_idx], t, par) 


            else:

                if par.labour_market_flexibility == 'flexible':
                    sol.h_work[t, a_idx] = optimizer(value_function_NVFI,
                                            a=par.h_min,
                                            b=par.h_max,
                                            args=(sol.c_given_m, sol.V_work, sol.V_notwork, a, t, par)
                    )
                elif par.labour_market_flexibility == 'rigid':
                    sol.h_work[t, a_idx] = par.h_max
                else:
                    raise ValueError("Invalid labour market flexibility setting")
                
                sol.V_work[t, a_idx] = - value_function_NVFI(sol.h_work[t, a_idx], sol.c_given_m, sol.V_work, sol.V_notwork, a, t, par)
                m = sol.h_work[t, a_idx] + a + B(sol.h_work[t, a_idx], t, par)
                sol.c_work[t, a_idx] = interp_1d(par.m_grid, sol.c_given_m[t, :], m)


                sol.h_notwork[t, a_idx] = 0.0
                sol.V_notwork[t, a_idx] = - value_function_NVFI(sol.h_notwork[t, a_idx], sol.c_given_m, sol.V_work, sol.V_notwork, a, t, par)
                m = a + B(sol.h_notwork[t, a_idx], t, par)
                sol.c_notwork[t, a_idx] = interp_1d(par.m_grid, sol.c_given_m[t, :], m)
