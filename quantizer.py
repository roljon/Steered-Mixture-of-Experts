import numpy as np


def quantize_params(params):

    '''
    # define parameter as one parameter matrix
    params_matrix = model_transformer(params)

    # TODO compander opportunity
    ## compander step
    # decompose signum and magnitude
    #sign_data = np.sign(params_matrix)
    #abs_data = np.abs(params_matrix)

    companded_data = params_matrix

    # determine lower and upper bound for each kind of parameter
    lower_bound = np.amin(companded_data, axis=0, keepdims=True)
    upper_bound = np.amax(companded_data, axis=0, keepdims=True)

    # data normalization
    normalized_data = (companded_data - lower_bound) / (upper_bound - lower_bound)

    # find bucket indices
    # TODO steps are hard coded so far !!
    steps = (2**18 - 1, 2*18 - 1, 2**6 - 1, 2**6 - 1, 2**6 - 1, 2**20 - 1, 2**20 - 1, 2**20 - 1, 2**6 - 1, 2**6 - 1,
             2**6 - 1, 2**6 - 1, 2**6 - 1, 2**6 - 1, 2**10 - 1)
    tmp = np.round(normalized_data * steps)

    coeffmat = np.concatenate([lower_bound, upper_bound, tmp])
    '''

    # TODO steps are hard coded so far

    # TODO add oppptortunity of quantizing the entries of symmetric matrix A*A.T
    step_A = 2**20-1
    lb_A = np.amin(params['A'], axis=0, keepdims=True)
    ub_A = np.amax(params['A'], axis=0, keepdims=True)
    normalized = (params['A'] - lb_A) / (ub_A - lb_A + 10e-12)
    qA = np.round(normalized * step_A)

    step_musX = 2**18-1
    lb_musX = np.amin(params['musX'], axis=0, keepdims=True)
    ub_musX = np.amax(params['musX'], axis=0, keepdims=True)
    normalized = (params['musX'] - lb_musX) / (ub_musX - lb_musX + 10e-12)
    qmusX = np.round(normalized * step_musX)

    step_nu_e = 2 ** 6 - 1
    lb_nu_e = np.amin(params['nu_e'], axis=0, keepdims=True)
    ub_nu_e = np.amax(params['nu_e'], axis=0, keepdims=True)
    normalized = (params['nu_e'] - lb_nu_e) / (ub_nu_e - lb_nu_e + 10e-12)
    qnu_e = np.round(normalized * step_nu_e)

    step_pis = 2 ** 10 - 1
    lb_pis = np.amin(params['pis'], axis=0, keepdims=True)
    ub_pis = np.amax(params['pis'], axis=0, keepdims=True)
    normalized = (params['pis'] - lb_pis) / (ub_pis - lb_pis + 10e-12)
    qpis = np.round(normalized * step_pis)

    step_gamma_e = 2 ** 10 - 1
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



def model_transformer(params):
    musX = params['musX']
    nu_e = params['nu_e']
    A = params['A']
    gamma_e = params['gamma_e']
    pis = params['pis']

    A_parameter_mat = []
    for ii in range(A.shape[0]):
        # TODO introduce flag for quantize triangular parameter or symmetric matrix (in that case: A_matrix = np.matmul(A[ii, :, :], A[ii, :, :].T)
        A_matrix = A[ii, :, :]
        A_vector = A_matrix[np.tril_indices(A.shape[-1])]
        A_parameter_mat.append(A_vector)
    A_parameter_mat = np.stack(A_parameter_mat)

    gamma_vector = np.reshape(gamma_e, (-1, np.prod(gamma_e.shape[1:])))

    params_matrix = np.concatenate([musX, nu_e, A_parameter_mat, gamma_vector, np.expand_dims(pis, axis=1)], axis=1)

    return params_matrix