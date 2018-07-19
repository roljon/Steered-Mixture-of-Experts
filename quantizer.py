import numpy as np
from utils import reduce_params

def quantize_params(smoe, params):

    params = reduce_params(params)

    # TODO add oppptortunity of quantizing the entries of symmetric matrix A*A.T
    step_A = 2 ** smoe.bit_depths[0] - 1
    lb_A = np.amin(params['A'], axis=0, keepdims=True)
    ub_A = np.amax(params['A'], axis=0, keepdims=True)
    normalized = (params['A'] - lb_A) / (ub_A - lb_A + 10e-12)
    qA = np.round(normalized * step_A)

    step_musX = 2 ** smoe.bit_depths[1] - 1
    lb_musX = np.amin(params['musX'], axis=0, keepdims=True)
    ub_musX = np.amax(params['musX'], axis=0, keepdims=True)
    normalized = (params['musX'] - lb_musX) / (ub_musX - lb_musX + 10e-12)
    qmusX = np.round(normalized * step_musX)

    step_nu_e = 2 ** smoe.bit_depths[2] - 1
    lb_nu_e = np.amin(params['nu_e'], axis=0, keepdims=True)
    ub_nu_e = np.amax(params['nu_e'], axis=0, keepdims=True)
    normalized = (params['nu_e'] - lb_nu_e) / (ub_nu_e - lb_nu_e + 10e-12)
    qnu_e = np.round(normalized * step_nu_e)

    step_pis = 2 ** smoe.bit_depths[3] - 1
    lb_pis = np.amin(params['pis'], axis=0, keepdims=True)
    ub_pis = np.amax(params['pis'], axis=0, keepdims=True)
    normalized = (params['pis'] - lb_pis) / (ub_pis - lb_pis + 10e-12)
    qpis = np.round(normalized * step_pis)

    step_gamma_e = 2 ** smoe.bit_depths[4] - 1
    lb_gamma_e = np.amin(params['gamma_e'], axis=0, keepdims=True)
    ub_gamma_e = np.amax(params['gamma_e'], axis=0, keepdims=True)
    normalized = (params['gamma_e'] - lb_gamma_e) / (ub_gamma_e - lb_gamma_e + 10e-12)
    qgamma_e = np.round(normalized * step_gamma_e)

    lower_bounds = {'A': lb_A, 'musX': lb_musX, 'nu_e': lb_nu_e, 'pis': lb_pis, 'gamma_e': lb_gamma_e}
    upper_bounds = {'A': ub_A, 'musX': ub_musX, 'nu_e': ub_nu_e, 'pis': ub_pis, 'gamma_e': ub_gamma_e}
    steps = {'A': step_A, 'musX': step_musX, 'nu_e': step_nu_e, 'pis': step_pis, 'gamma_e': step_gamma_e}
    qparams = {'lower_bounds': lower_bounds, 'upper_bounds': upper_bounds, 'steps': steps,
               'A': qA, 'musX': qmusX, 'nu_e': qnu_e, 'pis': qpis, 'gamma_e': qgamma_e}

    return qparams

def rescaler(smoe, qparams):
    # steps
    steps = qparams["steps"]
    step_A = steps["A"]
    step_musX = steps["musX"]
    step_nu_e = steps["nu_e"]
    step_pis = steps["pis"]
    step_gamma_e = steps["gamma_e"]

    # lower bounds
    lower_bounds = qparams["lower_bounds"]
    lb_A = lower_bounds["A"]
    lb_musX = lower_bounds["musX"]
    lb_nu_e = lower_bounds["nu_e"]
    lb_pis = lower_bounds["pis"]
    lb_gamma_e = lower_bounds["gamma_e"]

    # upper bounds
    upper_bounds = qparams["upper_bounds"]
    ub_A = upper_bounds["A"]
    ub_musX = upper_bounds["musX"]
    ub_nu_e = upper_bounds["nu_e"]
    ub_pis = upper_bounds["pis"]
    ub_gamma_e = upper_bounds["gamma_e"]

    # qparams
    qA = qparams["A"]
    qmusX = qparams["musX"]
    qnu_e = qparams["nu_e"]
    qpis = qparams["pis"]
    qgamma_e = qparams["gamma_e"]

    # rescaling
    rA = qA / step_A * (ub_A - lb_A) + lb_A
    rmusX = qmusX / step_musX * (ub_musX - lb_musX) + lb_musX
    rnu_e = qnu_e / step_nu_e * (ub_nu_e - lb_nu_e) + lb_nu_e
    rpis = qpis / step_pis * (ub_pis - lb_pis) + lb_pis
    rgamma_e = qgamma_e / step_gamma_e * (ub_gamma_e - lb_gamma_e) + lb_gamma_e

    if smoe.radial_as:
        A_mask = np.zeros((len(rA), smoe.dim_domain, smoe.dim_domain))
        for ii in range(A_mask.shape[0]):
            np.fill_diagonal(A_mask[ii, :, :], rA[ii])
        rA = A_mask


    rparams = {'A': rA, 'musX': rmusX, 'nu_e': rnu_e, 'pis': rpis, 'gamma_e': rgamma_e}

    return rparams