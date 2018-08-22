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
from utils import save_model, load_params, read_image


def main(image_path, results_path, iterations, validation_iterations, kernels_per_dim, params_file, l1reg, base_lr,
         batches, checkpoint_path, lr_div, lr_mult, disable_train_pis, disable_train_gammas, radial_as, use_determinant,
         normalize_pis, quantization_mode, bit_depths, quantize_pis, lower_bounds, upper_bounds, use_yuv):

    if len(bit_depths) != 5:
        raise ValueError("Number of bit depths must be five!")

    if quantization_mode == 2:
        quantize_pis = True
    elif quantization_mode == 0:
        quantize_pis = False

    orig = read_image(image_path, use_yuv)

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
    logger = ModelLogger(path=results_path, as_media=True)

    smoe = Smoe(orig, kernels_per_dim, init_params=init_params, train_pis=not disable_train_pis,
                train_gammas=not disable_train_gammas, radial_as=radial_as, start_batches=batches,
                use_determinant=use_determinant, normalize_pis=normalize_pis, quantization_mode=quantization_mode,
                bit_depths=bit_depths, quantize_pis=quantize_pis, lower_bounds=lower_bounds, upper_bounds=upper_bounds)

    if orig.shape[-1] == 3:
        smoe.use_yuv = use_yuv
    else:
        smoe.use_yuv = False

    optimizer1 = tf.train.AdamOptimizer(base_lr)
    optimizer2 = tf.train.AdamOptimizer(base_lr/lr_div)
    optimizer3 = tf.train.AdamOptimizer(base_lr*lr_mult)

    # optimizers have to be set before the restore
    smoe.set_optimizer(optimizer1, optimizer2, optimizer3)

    if checkpoint_path is not None:
        smoe.restore(checkpoint_path)

    smoe.train(iterations, val_iter=validation_iterations, pis_l1=l1reg,
               callbacks=[loss_plotter.plot, image_plotter.plot, logger.log])

    save_model(smoe, results_path + "/params_best.pkl", best=True)
    save_model(smoe, results_path + "/params_last.pkl", best=False)

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--image_path', type=str, required=True, help="input image")
    parser.add_argument('-r', '--results_path', type=str, required=True, help="results path")
    parser.add_argument('-n', '--iterations', type=int, default=10000, help="number of iterations")
    parser.add_argument('-v', '--validation_iterations', type=int, default=100, help="number of iterations between validations")
    parser.add_argument('-k', '--kernels_per_dim', type=int, default=[12], nargs='+', help="number of kernels per dimension")
    parser.add_argument('-p', '--params_file', type=str, default=None, help="parameter file for model initialization")
    parser.add_argument('-reg', '--l1reg', type=float, default=0, help="l1 regularization for pis")
    parser.add_argument('-lr', '--base_lr', type=float, default=0.001, help="base learning rate")
    parser.add_argument('-b', '--batches', type=int, default=1, help="number of batches to split the training into (will be automaticly reduced when number of pis drops")
    parser.add_argument('-c', '--checkpoint_path', type=str, default=None, help="path to a checkpoint file to continue the training. EXPERIMENTAL.")
    parser.add_argument('-d', '--lr_div', type=float, default=100, help="div for pis lr")
    parser.add_argument('-m', '--lr_mult', type=float, default=1000, help="mult for a lr")

    parser.add_argument('-dp', '--disable_train_pis', type=str2bool, nargs='?',
                        const=False, default=False, help="disable_train_pis")
    parser.add_argument('-dg', '--disable_train_gammas', type=str2bool, nargs='?',
                        const=False, default=False, help="disable_train_gammas")
    parser.add_argument('-ra', '--radial_as', type=str2bool, nargs='?',
                        const=False, default=False, help="radial_as")
    parser.add_argument('-ud', '--use_determinant', type=str2bool, nargs='?',
                        const=True, default=True, help="use determinants for gaussian normalization")
    parser.add_argument('-np', '--normalize_pis', type=str2bool, nargs='?',
                        const=True, default=True, help="set all pis to 1/K for initialization")

    parser.add_argument('-qm', '--quantization_mode', type=int, default=0,
                        help="Quantization mode: 0 - no quantization, 1 - quantization each validation step,"
                             " 2 - fake quantization optimization")
    parser.add_argument('-bd', '--bit_depths', type=int, default=[20, 18, 6, 10, 10], nargs='+',
                        help="bit depths of each kind of parameter. number of numbers must be 5 in the order: A, musX, nu_e, pis, gamma_e")
    parser.add_argument('-qp', '--quantize_pis', type=str2bool, nargs='?',
                        const=True, default=True, help="Quantize Pis while optimization (only valid for quantization mode 1, for quantization mode 2 always True)")
    parser.add_argument('-lb', '--lower_bounds', type=float, default=[-2500, -.3, -5, 0, -32], nargs='+',
                        help="lower bounds of parameters for quantization while optimization")
    parser.add_argument('-ub', '--upper_bounds', type=float, default=[2500, 1.3, 5, 2, 32], nargs='+',
                        help="upper bounds of parameters for quantization while optimization")

    parser.add_argument('-yuv', '--use_yuv', type=str2bool, nargs='?',
                        const=True, default=True, help="uses YUV color space for modeling if three channels are provided.")


    args = parser.parse_args()

    main(**vars(args))
