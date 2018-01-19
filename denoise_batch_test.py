import argparse
# for execution without a display
import matplotlib as mpl

mpl.use('Agg')

import tensorflow as tf
import numpy as np
import pickle
import os
import shutil

from scipy.io import loadmat, savemat

from itertools import product

from smoe import Smoe
from logger import ModelLogger
from plotter import ImagePlotter, LossPlotter, DenoisePlotter
from utils import save_model, load_params

#mat_paths = ["tok1_bm3d_sigma10.mat", "tok1_bm3d_sigma25.mat", "peppers128_bm3d_sigma25.mat",
#             "peppers128_bm3d_sigma10.mat"]
mat_paths = ["peppers128_bm3d_sigma10.mat"]

base_lrs = [0.01, 0.001, 0.0001, 0.00001]
base_lr_divs = [1., 10., 50., 100]
u_l1regs = [-0.001, -0.0005, -0.0001, -0.00005, -0.00001, -0.000005, -0.000001]

training_iters = 5000
validation_iterations = 100
l1reg = 0.

kernels_per_dim = 64
batches = 1

for mat_path, base_lr, base_lr_div, u_l1reg in product(mat_paths, base_lrs, base_lr_divs, u_l1regs):
    tf.reset_default_graph()

    mat = loadmat(mat_path)
    y = mat['y'].astype(np.float32)
    y_est = mat['y_est'].astype(np.float32)
    z = mat['z'].astype(np.float32)

    smoe = Smoe(z, kernels_per_dim, pis_relu=True, train_pis=True, start_batches=batches)
    results_path = "denoise_batch/{mat_path}/base_lr_{base_lr}/" \
                   "base_lr_div_{base_lr_div}/u_l1reg_{u_l1reg}".format(mat_path=mat_path,
                                                                        base_lr=base_lr,
                                                                        base_lr_div=base_lr_div,
                                                                        u_l1reg=u_l1reg)
    if results_path is not None:
        if os.path.exists(results_path):
            shutil.rmtree(results_path)
        os.makedirs(results_path)

    loss_plotter = LossPlotter(path=results_path + "/loss.png", quiet=False)
    image_plotter = ImagePlotter(path=results_path, options=['orig', 'reconstruction', 'pis_hist'], quiet=False)
    denoise_plotter = DenoisePlotter(y, z, y_est, path=results_path + "/denoise/")

    optimizer1 = tf.train.AdamOptimizer(base_lr, beta1=0.1, beta2=0.9, epsilon=0.001)
    optimizer2 = tf.train.AdamOptimizer(base_lr / base_lr_div, beta1=0.1, beta2=0.9, epsilon=0.001)

    try:
        smoe.train(training_iters, val_iter=validation_iterations, optimizer1=optimizer1, optimizer2=optimizer2,
                   pis_l1=l1reg, u_l1=u_l1reg, callbacks=[loss_plotter.plot, image_plotter.plot, denoise_plotter.plot])
    except:
        print("error computing {}".format(results_path))
        pass

    try:
        psnrs = denoise_plotter.psnrs
        mses = smoe.get_mses()
        losses = smoe.get_losses()
        num_pis = smoe.get_num_pis()

        cp = {'psnrs': psnrs, 'mses': mses, 'losses': losses, 'num_pis': num_pis,
              'u_l1reg': u_l1reg, 'base_lr_div': base_lr_div, 'base_lr': base_lr, 'mat_path': mat_path}

        path = results_path + "/results.pkl"

        with open(path, 'wb') as fd:
            pickle.dump(cp, fd)
    except:
        print("error saving {}".format(results_path))
        pass
