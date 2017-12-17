import pickle
import numpy as np

def reduce_params(params):
    idx = params['pis'] > 0
    params['pis'] = params['pis'][idx]
    params['U'] = params['U'][idx]
    params['nu_e'] = params['nu_e'][idx]
    params['gamma_e'] = params['gamma_e'][idx]
    params['musX'] = params['musX'][idx]
    return params


def save_model(smoe, path, best=False, reduce=True):
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
    with open(path, 'wb') as fd:
        pickle.dump(cp, fd)

def load_params(path):
    with open(path, 'rb') as fd:
        params = pickle.load(fd)['params']
        params['musX'] = np.transpose(params['musX'])
    return params
