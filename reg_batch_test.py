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
from logger import ModelLogger
from plotter import ImagePlotter, LossPlotter
from utils import save_model, load_params


def main(image_path, results_path, iterations, validation_iterations, kernels_per_dim, params_file, l1reg, base_lr,
         batches, checkpoint_path, restart, lr_div, lr_mult, start_reg, end_reg, num_reg):
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

    smoe = Smoe(orig, kernels_per_dim, init_params=init_params, train_pis=True, start_batches=batches)

    lr1 = base_lr
    lr2 = base_lr / lr_div # 10
    lr3 = base_lr * lr_mult

    optimizer1 = tf.train.AdamOptimizer(lr1)
    optimizer2 = tf.train.AdamOptimizer(lr2)
    optimizer3 = tf.train.AdamOptimizer(lr3)

    # optimizers have to be set before the restore
    smoe.set_optimizer(optimizer1, optimizer2, optimizer3)

    if checkpoint_path is not None:
        smoe.restore(checkpoint_path)
    else:
        print("no checkpoint supplied, train from scratch...")
        smoe.train(iterations, val_iter=validation_iterations,
               callbacks=[loss_plotter.plot, image_plotter.plot, logger.log])


    regs = np.linspace(start_reg, end_reg, num_reg)
    regs **= 2
    print("reg schedule: ", regs)

    for reg in list(regs):
        out_path = results_path + '/{0:.8f}'.format(float(reg))
        loss_plotter = LossPlotter(path=out_path + "/loss.png", quiet=True)
        image_plotter = ImagePlotter(path=out_path, options=['orig', 'reconstruction', 'gating', 'pis_hist'],
                                     quiet=True)
        logger = ModelLogger(path=out_path)

        if restart:
            del smoe
            tf.reset_default_graph()

            smoe = Smoe(orig, kernels_per_dim, init_params=init_params, train_pis=True, start_batches=batches)

            optimizer1 = tf.train.AdamOptimizer(lr1)
            optimizer2 = tf.train.AdamOptimizer(lr2)
            optimizer3 = tf.train.AdamOptimizer(lr3)

            if checkpoint_path is not None:
                smoe.restore(checkpoint_path)

            # optimizers have to be set before the restore
            smoe.set_optimizer(optimizer1, optimizer2, optimizer3)
        else:
            print("continue with reg {0:.8f}".format(reg))

        smoe.train(iterations, val_iter=validation_iterations, pis_l1=reg, callbacks=[loss_plotter.plot, image_plotter.plot, logger.log])

        save_model(smoe, out_path + "/params_best.pkl", best=True)
        save_model(smoe, out_path + "/params_last.pkl", best=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--image_path', type=str, required=True, help="input image")
    parser.add_argument('-r', '--results_path', type=str, required=True, help="results path")
    parser.add_argument('-n', '--iterations', type=int, default=10000, help="number of iterations")
    parser.add_argument('-v', '--validation_iterations', type=int, default=100,
                        help="number of iterations between validations")
    parser.add_argument('-k', '--kernels_per_dim', type=int, default=12, help="number of kernels per dimension")
    parser.add_argument('-p', '--params_file', type=str, default=None, help="parameter file for model initialization")
    parser.add_argument('-reg', '--l1reg', type=float, default=0, help="l1 regularization for pis")
    parser.add_argument('-lr', '--base_lr', type=float, default=0.001, help="base learning rate")
    parser.add_argument('-b', '--batches', type=int, default=1,
                        help="number of batches to split the training into (will be automaticly reduced when number of pis drops")
    parser.add_argument('-c', '--checkpoint_path', type=str, default=None,
                        help="path to a checkpoint file to continue the training. EXPERIMENTAL.")
    parser.add_argument('--restart', type=bool, default=False)
    parser.add_argument('-d', '--lr_div', type=float, default=10, help="div for pis lr")
    parser.add_argument('-m', '--lr_mult', type=float, default=1000, help="mult for a lr")
    parser.add_argument('-sr', '--start_reg', type=float, default=1, help="")
    parser.add_argument('-er', '--end_reg', type=float, default=15, help="")
    parser.add_argument('-rn', '--num_reg', type=float, default=50, help="")

    args = parser.parse_args()

    main(**vars(args))