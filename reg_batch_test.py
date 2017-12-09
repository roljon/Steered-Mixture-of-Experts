import tensorflow as tf
from tensorflow.python.framework import ops
import os
import shutil
import pickle
from time import time

from smoe import Smoe
from plotter import *

image = plt.imread("/media/nvme/rolf/Tok1.png")
out_base = "/media/nvme/smoe/batch_test/"
params_file = "/media/nvme/smoe/1024_120000.pkl"

step_size = 0.001
start_reg = 0.001

num = 20
train_iters = 30000


def reduce_params(params):
    idx = params['pis'] > 0
    idx = idx[0]
    params['pis'] = params['pis'][:,idx]
    params['U'] = params['U'][idx]
    params['nu_e'] = params['nu_e'][idx]
    params['gamma_e'] = params['gamma_e'][idx]
    params['musX'] = params['musX'][:,idx]
    return params


def save_model(smoe, path):
    params = smoe.get_params()
    mses = smoe.get_mses()
    losses = smoe.get_losses()
    num_pis = smoe.get_num_pis()

    cp = {'params': params, 'mses': mses, 'losses': losses, 'num_pis': num_pis}
    with open(path, 'wb') as fd:
        pickle.dump(cp, fd)


if not os.path.exists(out_base):
    os.mkdir(out_base)
else:
    shutil.rmtree(out_base + "/")
    os.mkdir(out_base)


for i in range(num):
    ops.reset_default_graph()

    reg = start_reg + step_size*i

    lr_scale = start_reg/reg

    out_dir = out_base + '/{0:.3f}'.format(reg)
    os.mkdir(out_dir)

    loss_plotter = LossPlotter(path=out_dir+"/loss.png", quiet=True)
    image_plotter = ImagePlotter(path=out_dir, options=['orig', 'reconstruction', 'gating', 'pis_hist'], quiet=True)

    optimizer1 = tf.train.AdamOptimizer(0.005, beta1=0.05, beta2=0.1, epsilon=0.1)
    #optimizer1 = tf.train.GradientDescentOptimizer(0.001 * lr_scale)
    #optimizer2 = tf.train.GradientDescentOptimizer(0.0001 * lr_scale)
    #optimizer1 = tf.train.GradientDescentOptimizer(0.001)
    optimizer2 = tf.train.GradientDescentOptimizer(0.0001)

    with open(params_file, 'rb') as fd:
        params = pickle.load(fd)['params']
        params = reduce_params(params)

    smoe = Smoe(image=image, init_params=params, train_pis=True, pis_l1=reg, pis_relu=True)
    start = time()
    smoe.train(train_iters, optimizer1=optimizer1, optimizer2=optimizer2, callbacks=[loss_plotter.plot, image_plotter.plot])
    end = time()
    print('[{0:.0f}/{1:.0f}] reg={2:.0f}, time={3:.2f}s'.format(i, num, reg, end - start))

    params_file = out_dir + "/params.pkl"
    save_model(smoe, params_file)


