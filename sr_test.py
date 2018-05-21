import argparse
# for execution without a display
import matplotlib as mpl
mpl.use('Agg')

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import os
import shutil
from glob import glob

from smoe import Smoe
from utils import load_params


def main(image_path, out_dir, params_dir, upsampling_factor, batches):
    orig = plt.imread(image_path)
    if orig.dtype == np.uint8:
        orig = orig.astype(np.float32)/255.

    orig = np.vstack(tuple([orig]*upsampling_factor))
    orig = np.hstack(tuple([orig]*upsampling_factor))

    out_dir = "{}/{}".format(out_dir, upsampling_factor)

    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(out_dir)

    for params_file in sorted(glob("{}/*_params.pkl".format(params_dir))):
        tf.reset_default_graph()
        params = load_params(params_file)

        smoe = Smoe(image=orig, init_params=params, start_batches=batches)
        rec = smoe.get_reconstruction()
        smoe.session.close()

        iter_ = os.path.basename(params_file).split('_')[0]
        plt.imsave("{}/{}.png".format(out_dir, iter_), rec, cmap='gray', vmin=0, vmax=1)
        np.save("{}/{}.npy".format(out_dir, iter_), rec)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--image_path', type=str, required=True, help="input image")
    parser.add_argument('-r', '--out_dir', type=str, required=True, help="results path")
    parser.add_argument('-p', '--params_dir', type=str, required=True, help="parameter file dir for model initialization")
    parser.add_argument('-s', '--upsampling_factor', type=int, default=2)  # TODO needs to work without image
    parser.add_argument('-b', '--batches', type=int, default=1, help="number of batches to split the training into (will be automaticly reduced when number of pis drops")


    args = parser.parse_args()

    main(**vars(args))
