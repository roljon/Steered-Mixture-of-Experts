from skimage.util.shape import view_as_blocks
from scipy.io import savemat
import numpy as np
import os
import shutil
import tensorflow as tf
from tensorflow.python.ops.special_math_ops import _exponential_space_einsum as einsum
import argparse

# for execution without a display
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

from sklearn.metrics import mean_squared_error
from skimage.measure import compare_ssim


def gen_domain(in_):
    if type(in_) is np.ndarray:
        assert len(in_.shape) == 2, "only 2d images supported!"
        assert in_.shape[0] == in_.shape[1], "only quadratic images supported!"
        num_per_dim = in_.shape[0]
    else:
        num_per_dim = in_
    domain = np.zeros((num_per_dim ** 2, 2))

    for row in range(num_per_dim):
        for col in range(num_per_dim):
            # equal spacing between domain positions and boarder
            domain[row * num_per_dim + col, 0] = (1 / num_per_dim) / 2 + row * (1 / num_per_dim)
            domain[row * num_per_dim + col, 1] = (1 / num_per_dim) / 2 + col * (1 / num_per_dim)

    return domain


def gen_nus(musX, image):
    stride = musX[0, 0]
    height, width = image.shape
    mean = np.empty((musX.shape[0],), dtype=np.float32)
    for k, (y, x) in enumerate(zip(*musX.T)):
        x0 = int(round((x - stride) * width))
        x1 = int(round((x + stride) * width))
        y0 = int(round((y - stride) * height))
        y1 = int(round((y + stride) * height))

        mean[k] = np.mean(image[y0:y1, x0:x1])
    return mean


def psnr(mse):
    return 10 * np.log10(255 ** 2 / mse)


def save_to_mat(path):
    cp = {'mu': musX.eval().squeeze(),
          'm': nu_e.eval(),
          'a': a.eval(),
          'mse': mse,
          'psnr': psnr(mse),
          'ssim': compare_ssim(orig, get_reconstruction(), data_range=1.),
          'iter': i,
          'mu_min': mu_min,
          'mu_max': mu_max,
          'm_min': m_min,
          'a_min': a_min,
          'a_max': a_max,
          'mu_bits': mu_bits,
          'm_bits': m_bits,
          'a_bits': a_bits}

    savemat(path, cp)

def get_reconstruction():
    reconstruction = res.eval().reshape((blocks_per_dim, blocks_per_dim, block_size, block_size))
    reconstruction = reconstruction.transpose(0, 2, 1, 3).reshape(-1, reconstruction.shape[1] * reconstruction.shape[3])
    return reconstruction

def save_reconstruction(path):
    reconstruction = get_reconstruction()
    plt.imsave(path, reconstruction, cmap='gray', vmin=0, vmax=1, dpi=600)


results_path = "block_results"
restore_path = None #"block_results/checkpoint/model.ckpt"
save_path = "block_results/checkpoint/model.ckpt"

iterations = 1000
block_size = 16
kernel_per_dim = 2

A_init = 7.5

mu_min = 0
mu_max = 1
m_min = 0
m_max = 1
a_min = 0
a_max = 45

mu_bits = 4
m_bits = 4
a_bits = 4

with_limits = False
with_quant = True

parser = argparse.ArgumentParser()
parser.add_argument('--results_path', type=str, required=True)
parser.add_argument('--save_path', type=str, default=None)
parser.add_argument('--restore_path', type=str, default=None)
parser.add_argument('--mu_bits', type=int, default=0)
parser.add_argument('--m_bits', type=int, default=0)
parser.add_argument('--a_bits', type=int, default=0)
parser.add_argument('--iterations', type=int, default=10000)
parser.add_argument('--skip_best', type=bool, default=False)

args = parser.parse_args()
results_path = args.results_path
save_path = args.save_path
restore_path = args.restore_path
mu_bits = args.mu_bits
m_bits = args.m_bits
a_bits = args.a_bits
iterations = args.iterations
skip_best = args.skip_best

if mu_bits == 0 and m_bits == 0 and a_bits == 0:
    with_limits = True
    with_quant = False
else:
    with_limits = False
    with_quant = True


if with_quant:
    a_min = 5
    a_max = 20

orig = plt.imread('images/lena.png')

if os.path.exists(results_path):
    shutil.rmtree(results_path)
os.mkdir(results_path)
os.mkdir(results_path+"/params")
os.mkdir(results_path+"/reconstructions")

if orig.dtype == np.uint8:
    orig = orig.astype(np.float32) / 255.

orig_blocked = view_as_blocks(orig, (block_size, block_size))
orig_blocked = orig_blocked.reshape((-1, block_size, block_size))
num_blocks = orig_blocked.shape[0]
blocks_per_dim = int(np.sqrt(num_blocks))

domain_init = gen_domain(orig_blocked[0])
musX_init = gen_domain(kernel_per_dim)
musX_init = np.tile(musX_init, (num_blocks, 1, 1))
As_init = np.ones((num_blocks,)) * A_init
nus_list = []
for block in orig_blocked:
    nus_list.append(np.flip(gen_nus(musX_init[0], block), axis=0))  # nobody knows why the flip is neccessary
nus_init = np.stack(nus_list)

gpu_options = tf.GPUOptions(allow_growth=True)
session = tf.InteractiveSession(config=tf.ConfigProto(gpu_options=gpu_options))

nu_e_var = tf.Variable(nus_init, dtype=tf.float32)
musX_var = tf.Variable(musX_init, dtype=tf.float32)
a_var = tf.Variable(As_init, dtype=tf.float32)

target_op = tf.constant(orig_blocked, dtype=tf.float32)
domain_op = tf.constant(domain_init, dtype=tf.float32)

if with_limits:
    musX_cap = tf.minimum(tf.maximum(musX_var, mu_min), mu_max)
    musX_limit_op = musX_var.assign(musX_cap)

    nu_e_cap = tf.minimum(tf.maximum(nu_e_var, m_min), m_max)
    nu_e_limit_op = nu_e_var.assign(nu_e_cap)

    a_cap = tf.minimum(tf.maximum(a_var, a_min), a_max)
    a_limit_op = a_var.assign(a_cap)

    musX = musX_cap
    nu_e = nu_e_cap
    a = a_cap

    limit_op = tf.group(musX_limit_op, nu_e_limit_op, a_limit_op)
else:
    musX = musX_var
    nu_e = nu_e_var
    a = a_var
    limit_op = tf.no_op()

# fake quant
if with_quant:
    musX = tf.fake_quant_with_min_max_args(musX, min=mu_min, max=mu_max, num_bits=mu_bits)
    nu_e = tf.fake_quant_with_min_max_args(nu_e, min=m_min, max=m_max, num_bits=m_bits)
    a = tf.fake_quant_with_min_max_args(a, min=a_min, max=a_max, num_bits=a_bits)

musX = tf.expand_dims(musX, axis=2)

# prepare domain
domain_exp = domain_op
domain_exp = tf.tile(tf.expand_dims(tf.expand_dims(domain_exp, axis=0), axis=0),
                     (tf.shape(musX)[0], tf.shape(musX)[1], 1, 1))

x_sub_mu = tf.expand_dims(domain_exp - musX, axis=-1)
a = tf.expand_dims(tf.expand_dims(a, axis=1), axis=1)
n_exp = tf.exp(a * einsum('abcij,abcjk->abc', tf.transpose(x_sub_mu, perm=[0, 1, 2, 4, 3]), x_sub_mu), name="exploder")

n_w_norm = tf.reduce_sum(n_exp, axis=1, keep_dims=True)
n_w_norm = tf.maximum(10e-8, n_w_norm)

w_e_op = tf.divide(n_exp, n_w_norm, name="skdjfbk")

res = tf.reduce_sum(w_e_op * tf.expand_dims(nu_e, axis=-1), axis=1)
res = tf.minimum(tf.maximum(res, 0), 1)
res = tf.reshape(res, tf.shape(target_op))

mse = tf.reduce_sum(tf.square(res - target_op)) / tf.cast(tf.size(target_op), dtype=tf.float32)

loss_op = mse
mse_op = mse * (255 ** 2)

optimizer1 = tf.train.AdamOptimizer(0.001)
train_op = optimizer1.minimize(loss_op)

check_op = tf.add_check_numerics_ops()

init_new_vars_op = tf.global_variables_initializer()
session.run(init_new_vars_op)

saver = tf.train.Saver()

if restore_path is not None:
    saver.restore(session, restore_path)

fig, axes = plt.subplots(2, 1)
mses = []
psnrs = []
iters = []
best_mse = 1000000000
last_mse = 1000000000

for i in range(iterations):
    _, _, mse = session.run([train_op, check_op, mse_op])
    session.run([limit_op])

    if not skip_best and mse < best_mse:
        save_to_mat(results_path + "/best.mat".format(iter_=i, mse=mse))
        save_reconstruction(results_path + "/best.png".format(iter_=i, mse=mse))
        best_mse = mse

    if len(mses) == 0 or last_mse-mse > 1.0:
        last_mse = mse
        save_to_mat(results_path + "/params/{iter_:08d}_{mse:.2f}.mat".format(iter_=i, mse=mse))
        save_reconstruction(results_path + "/reconstructions/{iter_:08d}_{mse:.2f}.png".format(iter_=i, mse=mse))

    if i % 100 == 0:
        mses.append(mse)
        psnrs.append(psnr(mse))
        iters.append(i)
        axes[0].clear()
        axes[1].clear()
        axes[0].set_title("MSE: " + str(round(mses[-1], 2)))
        axes[0].plot(iters, mses)
        axes[1].set_title("PSNR: " + str(round(psnrs[-1], 2)))
        axes[1].plot(iters, psnrs)
        fig.savefig(results_path + "/learning_curve.png")

if save_path is not None:
    saver.save(session, save_path)