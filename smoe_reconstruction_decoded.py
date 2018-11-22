import argparse
# for execution without a display
import matplotlib as mpl
mpl.use('Agg')

import pickle
import os
import shutil
import re
import numpy as np

from smoe import Smoe
from utils import read_image, write_image


def main(image_path, results_path, params_file, batches):

    with open(params_file, 'rb') as fd:
        cp = pickle.load(fd)

    # TODO need to be changed, initial grid will be embedded into bitstream itself
    k = list(np.int32(cp["shape_of_img"][0][:] / 4))

    if image_path is not None:
        orig = read_image(image_path)
    else:
        orig = np.zeros((*cp["shape_of_img"][0][:], *cp["dim_of_output"][0][:]), dtype=np.float32)

    smoe = Smoe(orig, kernels_per_dim=k, start_batches=batches, use_determinant=bool(cp["used_determinants"]),
                use_yuv=True)  # Assuming YUV so far!

    rpis = cp['pis'][0]
    rgamma_e = cp['gamma_e']
    rmusX = cp['musX'] + smoe.musX_init[cp['used_kernels'][0].astype(bool), :]
    rnu_e = cp['nu_e']
    rA_diagonal = cp['A_diagonal']
    rA_corr = cp['A_corr']
    rA = np.concatenate((rA_diagonal, rA_corr, np.zeros_like(rA_corr)), axis=1)
    rA = rA[:, [0, 3, 2, 1]].reshape((rA_corr.shape[0], 2, 2))

    try_rec = True
    while try_rec:
        try:
            smoe.rparams = {'A': rA, 'musX': rmusX, 'nu_e': rnu_e, 'pis': rpis, 'gamma_e': rgamma_e}
            loss, mse, _ = smoe.run_batched(train=False, update_reconstruction=True, with_quantized_params=True)
            try_rec = False
        except:
            smoe.session.close()
            batches = 2 * batches
            smoe = Smoe(orig, kernels_per_dim=k, start_batches=batches, use_determinant=bool(cp["used_determinants"]))
    reconstruction = smoe.get_qreconstruction()

    if results_path is not None:
        if not os.path.exists(results_path):
        #   shutil.rmtree(results_path)
            os.mkdir(results_path)
    else:
        results_path = '/tmp'

    rec_path = results_path + '/output'

    write_image(reconstruction, rec_path, smoe.dim_domain, smoe.use_yuv)

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--image_path', type=str, required=False, help="input image")
    parser.add_argument('-r', '--results_path', type=str, required=False, help="results path")
    parser.add_argument('-p', '--params_file', type=str, required=True, help="parameter file for model initialization.")
    parser.add_argument('-b', '--batches', type=int, default=1, help="number of batches to split the input domain for reconstruction")

    args = parser.parse_args()

    main(**vars(args))