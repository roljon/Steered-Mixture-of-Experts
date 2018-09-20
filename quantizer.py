import numpy as np
from utils import reduce_params

def quantize_params(smoe, params):

    params = reduce_params(params)

    if smoe.quantization_mode <= 1 or smoe.quantization_mode == 3:
        lb_A_diagonal = np.amin(params['A_diagonal'], axis=0, keepdims=True)
        ub_A_diagonal = np.amax(params['A_diagonal'], axis=0, keepdims=True)
        if not smoe.radial_as:
            lb_A_corr = np.amin(params['A_corr'], axis=0, keepdims=True)
            ub_A_corr = np.amax(params['A_corr'], axis=0, keepdims=True)
        lb_musX = np.amin(params['musX'], axis=0, keepdims=True)
        ub_musX = np.amax(params['musX'], axis=0, keepdims=True)
        lb_nu_e = np.amin(params['nu_e'], axis=0, keepdims=True)
        ub_nu_e = np.amax(params['nu_e'], axis=0, keepdims=True)
        lb_gamma_e = np.amin(params['gamma_e'], axis=0, keepdims=True)
        ub_gamma_e = np.amax(params['gamma_e'], axis=0, keepdims=True)
    elif smoe.quantization_mode == 2:
        if smoe.radial_as:
            lb_A_diagonal = np.ones((1,)) * smoe.lower_bounds[0]
            ub_A_diagonal = np.ones((1,)) * smoe.upper_bounds[0]
        else:
            lb_A_diagonal = np.ones((1, smoe.dim_domain, smoe.dim_domain)) * smoe.lower_bounds[0]
            ub_A_diagonal = np.ones((1, smoe.dim_domain, smoe.dim_domain)) * smoe.upper_bounds[0]
            lb_A_corr = np.ones((1, smoe.dim_domain, smoe.dim_domain)) * smoe.lower_bounds[0]
            ub_A_corr = np.ones((1, smoe.dim_domain, smoe.dim_domain)) * smoe.upper_bounds[0]
        lb_musX = np.ones((1, smoe.dim_domain)) * smoe.lower_bounds[1]
        ub_musX = np.ones((1, smoe.dim_domain)) * smoe.upper_bounds[1]
        lb_nu_e = np.ones((1, smoe.image.shape[-1])) * smoe.lower_bounds[2]
        ub_nu_e = np.ones((1, smoe.image.shape[-1])) * smoe.upper_bounds[2]
        lb_gamma_e = np.ones((1, smoe.dim_domain, smoe.image.shape[-1])) * smoe.lower_bounds[4]
        ub_gamma_e = np.ones((1, smoe.dim_domain, smoe.image.shape[-1])) * smoe.upper_bounds[4]

    if smoe.quantization_mode <= 1 and not smoe.quantize_pis:
        lb_pis = np.amin(params['pis'], axis=0, keepdims=True)
        ub_pis = np.amax(params['pis'], axis=0, keepdims=True)
    elif smoe.quantization_mode == 2 or smoe.quantize_pis:
        lb_pis = np.ones((1,)) * smoe.lower_bounds[3]
        ub_pis = np.ones((1,)) * smoe.upper_bounds[3]

    lower_bounds = {'A_diagonal': lb_A_diagonal, 'musX': lb_musX, 'nu_e': lb_nu_e, 'pis': lb_pis, 'gamma_e': lb_gamma_e}
    upper_bounds = {'A_diagonal': ub_A_diagonal, 'musX': ub_musX, 'nu_e': ub_nu_e, 'pis': ub_pis, 'gamma_e': ub_gamma_e}
    if not smoe.radial_as:
        lower_bounds.update({'A_corr': lb_A_corr})
        upper_bounds.update({'A_corr': ub_A_corr})

    step_A = 2 ** smoe.bit_depths[0] - 1
    step_musX = 2 ** smoe.bit_depths[1] - 1
    step_nu_e = 2 ** smoe.bit_depths[2] - 1
    step_pis = 2 ** smoe.bit_depths[3] - 1
    step_gamma_e = 2 ** smoe.bit_depths[4] - 1
    steps = {'A': step_A, 'musX': step_musX, 'nu_e': step_nu_e, 'pis': step_pis, 'gamma_e': step_gamma_e}


    # TODO add oppptortunity of quantizing the entries of symmetric matrix A*A.T
    normalized = (params['A_diagonal'] - lb_A_diagonal) / (ub_A_diagonal - lb_A_diagonal + 10e-12)
    qA_diagonal = np.round(normalized * step_A)

    if not smoe.radial_as:
        normalized = (params['A_corr'] - lb_A_corr) / (ub_A_corr - lb_A_corr + 10e-12)
        qA_corr = np.round(normalized * step_A)

    normalized = (params['musX'] - lb_musX) / (ub_musX - lb_musX + 10e-12)
    qmusX = np.round(normalized * step_musX)

    normalized = (params['nu_e'] - lb_nu_e) / (ub_nu_e - lb_nu_e + 10e-12)
    qnu_e = np.round(normalized * step_nu_e)

    normalized = (params['pis'] - lb_pis) / (ub_pis - lb_pis + 10e-12)
    qpis = np.round(normalized * step_pis)

    normalized = (params['gamma_e'] - lb_gamma_e) / (ub_gamma_e - lb_gamma_e + 10e-12)
    qgamma_e = np.round(normalized * step_gamma_e)

    qparams = {'lower_bounds': lower_bounds, 'upper_bounds': upper_bounds, 'steps': steps,
               'A_diagonal': qA_diagonal, 'musX': qmusX, 'nu_e': qnu_e, 'pis': qpis, 'gamma_e': qgamma_e}

    if not smoe.radial_as:
        qparams.update({'A_corr': qA_corr})

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
    lb_A_diagonal = lower_bounds["A_diagonal"]
    if not smoe.radial_as:
        lb_A_corr = lower_bounds["A_corr"]
    lb_musX = lower_bounds["musX"]
    lb_nu_e = lower_bounds["nu_e"]
    lb_pis = lower_bounds["pis"]
    lb_gamma_e = lower_bounds["gamma_e"]

    # upper bounds
    upper_bounds = qparams["upper_bounds"]
    ub_A_diagonal = upper_bounds["A_diagonal"]
    if not smoe.radial_as:
        ub_A_corr = upper_bounds["A_corr"]
    ub_musX = upper_bounds["musX"]
    ub_nu_e = upper_bounds["nu_e"]
    ub_pis = upper_bounds["pis"]
    ub_gamma_e = upper_bounds["gamma_e"]

    # qparams
    qA_diagonal = qparams["A_diagonal"]
    if not smoe.radial_as:
        qA_corr = qparams["A_corr"]
    qmusX = qparams["musX"]
    qnu_e = qparams["nu_e"]
    qpis = qparams["pis"]
    qgamma_e = qparams["gamma_e"]

    # rescaling
    rA_diagonal = qA_diagonal / step_A * (ub_A_diagonal - lb_A_diagonal) + lb_A_diagonal
    if not smoe.radial_as:
        rA_corr = qA_corr / step_A * (ub_A_corr - lb_A_corr) + lb_A_corr
    rmusX = qmusX / step_musX * (ub_musX - lb_musX) + lb_musX
    rnu_e = qnu_e / step_nu_e * (ub_nu_e - lb_nu_e) + lb_nu_e
    rpis = qpis / step_pis * (ub_pis - lb_pis) + lb_pis
    rgamma_e = qgamma_e / step_gamma_e * (ub_gamma_e - lb_gamma_e) + lb_gamma_e

    if smoe.radial_as:
        A_mask = np.zeros((len(rA_diagonal), smoe.dim_domain, smoe.dim_domain))
        for ii in range(A_mask.shape[0]):
            np.fill_diagonal(A_mask[ii, :, :], rA_diagonal[ii])
        rA = A_mask
    else:
        rA = rA_diagonal + rA_corr


    rparams = {'A': rA, 'musX': rmusX, 'nu_e': rnu_e, 'pis': rpis, 'gamma_e': rgamma_e}

    return rparams