import argparse
# for execution without a display
import matplotlib as mpl
mpl.use('Agg')

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import os
import shutil

from smoe import Smoe
from plotter import ImagePlotter, LossPlotter
from logger import ModelLogger
from utils import save_model, load_params


def main(image_path, results_path, iterations, validation_iterations, kernels_per_dim, params_file, l1reg, base_lr, batches):
    orig = plt.imread(image_path)
    if orig.dtype == np.uint8:
        orig = orig.astype(np.float32)/255.

    if params_file is not None:
        init_params = load_params(params_file)
    else:
        init_params = None

    if results_path is not None:
        if os.path.exists(results_path):
            shutil.rmtree(results_path)
        os.mkdir(results_path)

    loss_plotter = LossPlotter(path=results_path + "/loss.png", quiet=True)
    image_plotter = ImagePlotter(path=results_path, options=['orig', 'reconstruction', 'gating', 'pis_hist'], quiet=True)
    logger = ModelLogger(path=results_path)

    smoe = Smoe(orig, kernels_per_dim, init_params=init_params, pis_relu=True, train_pis=True, start_batches=batches)

    optimizer1 = tf.train.AdamOptimizer(base_lr)
    optimizer2 = tf.train.AdamOptimizer(base_lr / 100)

    smoe.train(iterations, val_iter=validation_iterations, optimizer1=optimizer1, optimizer2=optimizer2, pis_l1=l1reg,
               callbacks=[loss_plotter.plot, image_plotter.plot, logger.log]) #grad_clip_value_abs=0.01

    save_model(smoe, results_path + "/params_best.pkl", best=True)
    save_model(smoe, results_path + "/params_last.pkl", best=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--image', type=str, required=True, help="input image")
    parser.add_argument('-r', '--results', type=str, required=True, help="results path")
    parser.add_argument('-n', '--iterations', type=int, default=10000, help="number of iterations")
    parser.add_argument('-v', '--validation_iterations', type=int, default=100, help="number of iterations between validations")
    parser.add_argument('-k', '--kernels', type=int, default=12, help="number of kernels per dimension")
    parser.add_argument('-p', '--params', type=str, default=None, help="parameter file for model initialization")
    parser.add_argument('-reg', '--l1reg', type=float, default=0, help="l1 regularization for pis")
    parser.add_argument('-lr', '--learningrate', type=float, default=0.001, help="base learning rate")
    parser.add_argument('-b', '--batches', type=int, default=1, help="number of batches to split the training into (will be automaticly reduced when number of pis drops")

    args = parser.parse_args()

    main(args.image, args.results, args.iterations, args.validation_iterations, args.kernels, args.params, args.l1reg, args.learningrate,
         args.batches)
