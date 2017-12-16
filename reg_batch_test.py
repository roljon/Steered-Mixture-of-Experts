import tensorflow as tf
from tensorflow.python.framework import ops
import os
import shutil
from time import time

from smoe import Smoe
from plotter import *
from utils import save_model, load_params

image = plt.imread("/media/nvme/rolf/Tok1.png")
out_base = "/media/nvme/tf_smoe/batch_test/"
params_file = "/media/nvme/smoe/1024_120000.pkl"

num = 100
start_reg = 0.0001
end_reg = 0.1
regs = np.linspace(start_reg, end_reg, num)
regs = np.flipud(regs)

train_iters = 300000


if not os.path.exists(out_base):
    os.mkdir(out_base)
else:
    shutil.rmtree(out_base + "/")
    os.mkdir(out_base)


for idx, reg in enumerate(list(regs)):
    ops.reset_default_graph()

    #lr_scale = start_reg/reg

    out_dir = out_base + '/{0:.8f}'.format(reg)
    os.mkdir(out_dir)

    loss_plotter = LossPlotter(path=out_dir+"/loss.png", quiet=True)
    image_plotter = ImagePlotter(path=out_dir, options=['orig', 'reconstruction', 'gating', 'pis_hist'], quiet=True)

    optimizer1 = tf.train.AdamOptimizer(0.001)
    optimizer2 = tf.train.AdamOptimizer(0.000005)

    #optimizer1 = tf.train.AdamOptimizer(0.005, beta1=0.05, beta2=0.1, epsilon=0.1)
    #optimizer1 = tf.train.GradientDescentOptimizer(0.001 * lr_scale)
    #optimizer2 = tf.train.GradientDescentOptimizer(0.0001 * lr_scale)
    #optimizer1 = tf.train.GradientDescentOptimizer(0.001)
    #optimizer2 = tf.train.GradientDescentOptimizer(0.0001)

    #params = load_params(params_file)

    smoe = Smoe(image=image, kernels_per_dim=32, train_pis=True, pis_l1=reg, pis_relu=True)
    start = time()
    smoe.train(train_iters, optimizer1=optimizer1, optimizer2=optimizer2, callbacks=[loss_plotter.plot, image_plotter.plot])
    end = time()
    print('[{0:.0f}/{1:.0f}] reg={2:.8f}, time={3:.2f}s'.format(idx, num, reg, end - start))

    save_model(smoe, out_dir + "/params_best.pkl", best=True)
    save_model(smoe, out_dir + "/params_last.pkl", best=False)


