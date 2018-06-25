import pickle
import numpy as np
from quantizer import quantize_params

def reduce_params(params):
    idx = params['pis'] > 0
    params['pis'] = params['pis'][idx]
    params['A'] = params['A'][idx]
    params['nu_e'] = params['nu_e'][idx]
    params['gamma_e'] = params['gamma_e'][idx]
    params['musX'] = params['musX'][idx]
    return params


def save_model(smoe, path, best=False, reduce=True, quantize=False):
    if best:
        params = smoe.get_best_params()
    else:
        params = smoe.get_params()
    if reduce:
        params = reduce_params(params)

    mses = smoe.get_mses()
    losses = smoe.get_losses()
    num_pis = smoe.get_num_pis()

    cp = {'params': params, 'mses': mses, 'losses': losses, 'num_pis': num_pis}

    if quantize:
        qparams = quantize_params(params)
        qparams.update({'dim_of_domain': smoe.dim_domain})
        qparams.update({'dim_of_output': smoe.image.shape[-1]})
        qparams.update({'shape_of_img': smoe.image.shape[:-1]})
        qparams.update({'used_ranges': False})
        qparams.update({'quantized_tria_params': True})
        # trained_gamma_flag
        # radial_as_flag
        # trained_pis_flag
        cp.update({'qparams': qparams})

    with open(path, 'wb') as fd:
        pickle.dump(cp, fd)

def load_params(path):
    with open(path, 'rb') as fd:
        params = pickle.load(fd)['params']
        #params['musX'] = np.transpose(params['musX'])
    return params
