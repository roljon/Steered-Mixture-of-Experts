import argparse
# for execution without a display
import matplotlib as mpl
mpl.use('Agg')

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import os
import shutil
import cv2

from smoe import Smoe
from plotter import ImagePlotter, LossPlotter
from logger import ModelLogger
from utils import save_model, load_params


def main(image_path, results_path, iterations, validation_iterations, kernels_per_dim, params_file, l1reg, base_lr,
         batches, checkpoint_path, lr_div, lr_mult, disable_train_pis, disable_train_gammas, radial_as, use_determinant, quantization_mode, bit_depths):

    if len(bit_depths) != 5:
        raise ValueError("Number of bit depths must be five!")

    if image_path.lower().endswith(('.png', '.tif', '.tiff', '.pgm', '.ppm', '.jpg', '.jpeg')):
        orig = plt.imread(image_path)
        if orig.ndim == 2:
            orig = np.expand_dims(orig, axis=-1)
        if orig.dtype == np.uint8:
            orig = orig.astype(np.float32) / 255.
    elif image_path.lower().endswith(('.mp4', '.avi', '.mov', '.mkv', '.flv')):
        # TODO expand dimension in case of grayscale video
        cap = cv2.VideoCapture(image_path)
        num_of_frames = np.array(cap.get(7), dtype=np.int32)
        height = np.array(cap.get(3), dtype=np.int32)
        width = np.array(cap.get(4), dtype=np.int32)
        orig = np.empty((width, height, num_of_frames, 3))
        idx_frame = np.array(0, dtype=np.int32)
        while(idx_frame < num_of_frames):
            ret, curr_frame = cap.read()
            #curr_frame = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
            orig[:, :, idx_frame, :] = curr_frame
            idx_frame += 1
        orig = orig.astype(np.float32) / 255.

    elif image_path.lower().endswith('.yuv'):
        # TODO read raw video by OpenCV
        raise ValueError("Raw Video Data is not supported yet!")
    else:
        raise ValueError("Unknown data format")

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
                use_determinant=use_determinant, quantization_mode=quantization_mode, bit_depths=bit_depths)

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

    parser.add_argument('-dp', '--disable_train_pis', type=bool, default=False, help="disable_train_pis")
    parser.add_argument('-dg', '--disable_train_gammas', type=bool, default=False, help="disable_train_gammas")
    parser.add_argument('-ra', '--radial_as', type=bool, default=False, help="radial_as")
    parser.add_argument('-ud', '--use_determinant', type=bool, default=True, help="use determinants for gaussian normalization")

    parser.add_argument('-qm', '--quantization_mode', type=int, default=0,
                        help="Quantization mode: 0 - no quantization, 1 - quantization each validation step,"
                             " 2 - fake quantization optimization (not yet supported)")
    parser.add_argument('-bd', '--bit_depths', type=int, default=[20, 18, 6, 10, 10], nargs='+',
                        help="bit depths of each kind of parameter. number of numbers must be 5 in the order: A, musX, nu_e, pis, gamma_e")

    args = parser.parse_args()

    main(**vars(args))
