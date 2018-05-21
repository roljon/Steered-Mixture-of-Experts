import tensorflow as tf
import numpy as np
import os
import shutil

# for execution without a display
import matplotlib as mpl
mpl.use('Agg')

import matplotlib.pyplot as plt
import glob


from smoe import Smoe
from plotter import ImagePlotter, LossPlotter
from logger import ModelLogger
from utils import save_model, load_params


image_path = "images/baboon.png"
in_base = "/home/bochinski/nue200/smoe/final_div100/batch_baboon_128_lr0-001-reg0.1-15.2-50_10000-1000"
in_pkl = "params_last.pkl"

results_base = "final/finetuned/batch_baboon_128_lr0-001-reg0.1-15.2-50_10000-1000"


# in_model_paths = \
#     ['/home/bochinski/nue200/smoe/final_div100/batch_peppers_128_lr0-001-reg0.1-15.2-50_10000-1000/6.41433153/params/00000250_params.pkl',
#  '/home/bochinski/nue200/smoe/final_div100/batch_peppers_128_lr0-001-reg0.1-15.2-50_10000-1000/18.98469388/params/00000490_params.pkl',
#  '/home/bochinski/nue200/smoe/final_div100/batch_peppers_128_lr0-001-reg0.1-15.2-50_10000-1000/64.09799667/params/00000670_params.pkl',
#  '/home/bochinski/nue200/smoe/final_div100/batch_lena_128_lr0-001-reg0.1-15.2-50_10000-1000/9.86472720/params/00000280_params.pkl',
#  '/home/bochinski/nue200/smoe/final_div100/batch_lena_128_lr0-001-reg0.1-15.2-50_10000-1000/31.06356102/params/00000620_params.pkl',
#  '/home/bochinski/nue200/smoe/final_div100/batch_lena_128_lr0-001-reg0.1-15.2-50_10000-1000/79.53727613/params/00000270_params.pkl',
#  '/home/bochinski/nue200/smoe/final_div100/batch_cameraman_128_lr0-001-reg0.1-15.2-50_10000-1000/11.86732195/params/00000590_params.pkl',
#  '/home/bochinski/nue200/smoe/final_div100/batch_cameraman_128_lr0-001-reg0.1-15.2-50_10000-1000/38.21258226/params/00000410_params.pkl',
#  '/home/bochinski/nue200/smoe/final_div100/batch_cameraman_128_lr0-001-reg0.1-15.2-50_10000-1000/79.53727613/params/00000870_params.pkl']

in_model_paths = ['final_div100/batch_baboon_128_lr0-001-reg0.1-15.2-50_10000-1000/90.75478551/params/00000770_params.pkl',
 'final_div100/batch_baboon_128_lr0-001-reg0.1-15.2-50_10000-1000/143.02207414/params/00000970_params.pkl',
 'final_div100/batch_baboon_128_lr0-001-reg0.1-15.2-50_10000-1000/215.97001666/params/00000610_params.pkl']

for model_path in in_model_paths:
    image_path = "images/{name}.png".format(name=model_path.split('/')[-4].split('_')[1])
    orig = plt.imread(image_path)
    if orig.dtype == np.uint8:
        orig = orig.astype(np.float32) / 255

    results_base = "finetune_table/" + model_path.split('/')[-4]
    if not os.path.exists(results_base):
        os.mkdir(results_base)

    dir_ = model_path.split('/')[-3]
    results_path = results_base + "/" + dir_
    print(results_path)

    init_params = load_params(model_path)
    if init_params['pis'].shape[0] > 6000:
        print("skipping.. ", dir_)
        continue

    smoe = Smoe(orig, init_params=init_params, train_pis=True, start_batches=80)

    loss_plotter = LossPlotter(path=results_path + "/loss.png", quiet=True)
    image_plotter = ImagePlotter(path=results_path, options=['orig', 'reconstruction', 'gating', 'pis_hist'],
                                 quiet=True)

    logger = ModelLogger(path=results_path)

    base_lr = 0.000001
    lr_div = 100
    lr_mult = 1000
    optimizer1 = tf.train.GradientDescentOptimizer(base_lr)
    optimizer2 = tf.train.GradientDescentOptimizer(base_lr / lr_div)
    optimizer3 = tf.train.GradientDescentOptimizer(base_lr * lr_mult)

    iterations = 1000
    validation_iterations = 10
    l1reg = 0.0

    smoe.set_optimizer(optimizer1, optimizer2, optimizer3)

    smoe.train(iterations, val_iter=validation_iterations, pis_l1=l1reg,
               callbacks=[loss_plotter.plot, image_plotter.plot, logger.log])

    save_model(smoe, results_path + "/params_best.pkl", best=True)
    save_model(smoe, results_path + "/params_last.pkl", best=False)

