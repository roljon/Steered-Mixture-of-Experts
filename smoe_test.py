import argparse
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from smoe import Smoe
from plotter import ImagePlotter, LossPlotter
from utils import save_model


def main(image_path, results_path, iterations, kernels_per_dim):
    orig = plt.imread(image_path)
    if orig.dtype == np.uint8:
        orig = orig.astype(np.float32)/255.

    loss_plotter = LossPlotter(path=results_path + "/loss.png")
    image_plotter = ImagePlotter(path=results_path, options=['orig', 'reconstruction', 'gating', 'pis_hist'], quiet=False)

    smoe = Smoe(orig, kernels_per_dim, pis_relu=True, train_pis=True) #, pis_l1=0.0001)

    #optimizer1 = tf.train.AdamOptimizer(0.005, beta1=0.05, beta2=0.1, epsilon=0.1)
    #optimizer2 = tf.train.GradientDescentOptimizer(0.0001)
    optimizer1 = tf.train.AdamOptimizer(0.001)
    optimizer2 = tf.train.AdamOptimizer(0.00001)

    smoe.train(iterations, optimizer1=optimizer1, optimizer2=optimizer2,
               callbacks=[loss_plotter.plot, image_plotter.plot])

    save_model(smoe, results_path+"/params.pkl", best=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--image', type=str, required=True, help="input image")
    parser.add_argument('-r', '--results', type=str, required=True, help="results path")
    parser.add_argument('-n', '--iterations', type=int, default=10000, help="number of iterations")
    parser.add_argument('-k', '--kernels', type=int, default=12, help="number of kernels per dimension")

    args = parser.parse_args()

    main(args.image, args.results, args.iterations, args.kernels)