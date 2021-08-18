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


def main(image_path, results_path, iterations, iterations_inc, iterations_all, inc_steps, threshold_rel, validation_iterations, kernels_per_dim, params_file, l1reg, base_lr,
         batches, batch_size, checkpoint_path, lr_div, lr_mult, lr_mult_sv, disable_train_pis, disable_train_gammas, disable_train_musx,
         use_diff_center, radial_as, use_determinant, normalize_pis, quantization_mode, bit_depths, quantize_pis, lower_bounds,
         upper_bounds, use_yuv, only_y_gamma, ssim_opt, sampling_percentage, update_kernel_list_iterations, overlap_of_batches, svreg, hpc_mode, current_inc_step, kernel_count_norm_l1, train_svs):

    if len(bit_depths) != 5:
        raise ValueError("Number of bit depths must be five!")

    if ssim_opt:
        sampling_percentage = 100

    if sampling_percentage <= 0 or sampling_percentage > 100:
        raise ValueError("Value of Sampling Percentage must be in range (0,100]")

    if quantization_mode >= 2:
        quantize_pis = True

    orig, precision = read_image(image_path, use_yuv)

    if not orig.shape[-1] == 3:
        use_yuv = False
    if not use_yuv:
        only_y_gamma = False

    if params_file is not None:
        init_params = load_params(params_file)
    else:
        init_params = None

    if results_path is not None:
        if os.path.exists(results_path):
            shutil.rmtree(results_path)
        os.mkdir(results_path)

    if len(kernels_per_dim) == 1:
        kernels_per_dim = [kernels_per_dim[0]] * len(orig.shape[:-1])

    loss_plotter = LossPlotter(path=results_path + "/loss.png", quiet=True)
    if train_svs:
        image_plotter = ImagePlotter(path=results_path, options=['orig', 'reconstruction', 'gating', 'supportvectors', 'pis_hist'], quiet=True)
    else:
        image_plotter = ImagePlotter(path=results_path,
                                     options=['orig', 'reconstruction', 'gating', 'pis_hist'],
                                     quiet=True)
    logger = ModelLogger(path=results_path, as_media=True)

    smoe = Smoe(orig, kernels_per_dim, init_params=init_params, train_pis=not disable_train_pis,
                train_gammas=not disable_train_gammas, train_musx=not disable_train_musx, use_diff_center=use_diff_center, radial_as=radial_as, start_batches=batches,
                batch_size=batch_size, use_determinant=use_determinant, normalize_pis=normalize_pis, quantization_mode=quantization_mode,
                bit_depths=bit_depths, quantize_pis=quantize_pis, lower_bounds=lower_bounds, upper_bounds=upper_bounds,
                use_yuv=use_yuv, only_y_gamma=only_y_gamma, ssim_opt=ssim_opt, precision=precision, add_kernel_slots=inc_steps*np.prod(kernels_per_dim), overlap_of_batches=overlap_of_batches, kernel_count_as_norm_l1=kernel_count_norm_l1, train_svs=train_svs)

    if not train_svs:
        lr_mult_sv = 0

    optimizer1 = tf.train.AdamOptimizer(base_lr)
    optimizer2 = tf.train.AdamOptimizer(base_lr/lr_div)
    optimizer3 = tf.train.AdamOptimizer(base_lr*lr_mult)
    optimizer4 = tf.train.AdamOptimizer(base_lr * lr_mult_sv)

    # optimizers have to be set before the restore
    smoe.set_optimizer(optimizer1, optimizer2, optimizer3, optimizer4)

    base_lr_inc = base_lr  # *10
    optimizer_inc1 = tf.train.AdamOptimizer(base_lr_inc)
    optimizer_inc2 = tf.train.AdamOptimizer(base_lr_inc / 100)
    optimizer_inc3 = tf.train.AdamOptimizer(base_lr_inc * 1000)
    smoe.set_inc_optimizer(optimizer_inc1, optimizer_inc2, optimizer_inc3)

    if checkpoint_path is not None:
        smoe.restore(checkpoint_path)

    if overlap_of_batches > 0:
        sampling_percentage = 100 # otherwise it is not working!

    if hpc_mode and current_inc_step > 0:
        #smoe.kernel_count = smoe.session.run(smoe.num_pi_op)
        smoe.kernel_count += (current_inc_step - 1) * smoe.num_inc_kernels
        smoe.kernel_list_per_batch = [np.ones((smoe.add_kernel_slots + 2 * smoe.start_pis,),
                                              dtype=bool)] * smoe.start_batches

    if iterations != 0:
        smoe.train(iterations, val_iter=validation_iterations, pis_l1=l1reg, sv_l1_sub_l2=svreg,
                   sampling_percentage=sampling_percentage,
                   callbacks=[loss_plotter.plot, image_plotter.plot, logger.log])

    '''
    optimizer1 = tf.train.AdamOptimizer(base_lr*0)
    optimizer2 = tf.train.AdamOptimizer(base_lr / lr_div*0)
    optimizer3 = tf.train.AdamOptimizer(base_lr * lr_mult*0)
    optimizer4 = tf.train.AdamOptimizer(base_lr * 10 ** -1)

    # optimizers have to be set before the restore
    smoe.set_optimizer(optimizer1, optimizer2, optimizer3, optimizer4)

    smoe.train(iterations, val_iter=validation_iterations, pis_l1=l1reg, sv_l1_sub_l2=svreg, sampling_percentage=sampling_percentage,
               callbacks=[loss_plotter.plot, image_plotter.plot, logger.log])
    '''

    if hpc_mode and iterations == 0 or not hpc_mode:
        for i in range(inc_steps):
            print("[{}/{}]".format(i, inc_steps))
            smoe.reinit_inc(threshold_rel=threshold_rel, plot_dir=results_path)
            #smoe.session.run(smoe.assign_sv_zero)
            #smoe.train(iterations_inc, val_iter=validation_iterations, pis_l1=l1reg, sv_l1_sub_l2=svreg,
            #           with_inc=True, train_inc=True, train_orig=True,
            #           callbacks=[loss_plotter.plot, image_plotter.plot, logger.log])

            '''
            optimizer1 = tf.train.AdamOptimizer(base_lr)
            optimizer2 = tf.train.AdamOptimizer(base_lr / lr_div)
            optimizer3 = tf.train.AdamOptimizer(base_lr * lr_mult)
            optimizer4 = tf.train.AdamOptimizer(base_lr * 0)
            smoe.set_optimizer(optimizer1, optimizer2, optimizer3, optimizer4)
            '''

            smoe.apply_inc()
            smoe.train(iterations_inc, val_iter=validation_iterations, pis_l1=0, sv_l1_sub_l2=svreg,
                       callbacks=[loss_plotter.plot, image_plotter.plot, logger.log])
            smoe.train(iterations_all, val_iter=validation_iterations, pis_l1=l1reg, sv_l1_sub_l2=svreg,
                       callbacks=[loss_plotter.plot, image_plotter.plot, logger.log])

            if hpc_mode:
                break


    save_model(smoe, results_path + "/params_best.pkl", best=True, quantize=False if quantization_mode == 0 else True)
    save_model(smoe, results_path + "/params_last.pkl", best=False, quantize=False if quantization_mode == 0 else True)

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
    parser.add_argument('-ni', '--iterations_inc', type=int, default=1000, help="number of inc training iterations")
    parser.add_argument('-na', '--iterations_all', type=int, default=1000, help="number of training iterations after applying")
    parser.add_argument('-is', '--inc_steps', type=int, default=100, help="number of inc training iterations")
    parser.add_argument('-tr', '--threshold_rel', type=float, default=0.2, help="relative threshold for peak calculation")

    parser.add_argument('-v', '--validation_iterations', type=int, default=100, help="number of iterations between validations")
    parser.add_argument('-k', '--kernels_per_dim', type=int, default=[12], nargs='+', help="number of kernels per dimension")
    parser.add_argument('-p', '--params_file', type=str, default=None, help="parameter file for model initialization")
    parser.add_argument('-reg', '--l1reg', type=float, default=0, help="l1 regularization for pis")
    parser.add_argument('-lr', '--base_lr', type=float, default=0.001, help="base learning rate")
    parser.add_argument('-b', '--batches', type=int, default=1, help="number of batches to split the training into")
    parser.add_argument('-bz', '--batch_size', type=int, default=[None], nargs='+',
                        help="number of kernels per dimension")
    parser.add_argument('-c', '--checkpoint_path', type=str, default=None, help="path to a checkpoint file to continue the training. EXPERIMENTAL.")
    parser.add_argument('-d', '--lr_div', type=float, default=100, help="div for pis lr")
    parser.add_argument('-m', '--lr_mult', type=float, default=1000, help="mult for a lr")
    parser.add_argument('-msv', '--lr_mult_sv', type=float, default=1, help="mult for a lr for support vectors")

    parser.add_argument('-dp', '--disable_train_pis', type=str2bool, nargs='?',
                        const=False, default=False, help="disable_train_pis")
    parser.add_argument('-dg', '--disable_train_gammas', type=str2bool, nargs='?',
                        const=False, default=False, help="disable_train_gammas")
    parser.add_argument('-dm', '--disable_train_musx', type=str2bool, nargs='?',
                        const=False, default=False, help="disable_train_musX")
    parser.add_argument('-udc', '--use_diff_center', type=str2bool, nargs='?',
                        const=False, default=False, help="train deviations of mesh grid centers")
    parser.add_argument('-ra', '--radial_as', type=str2bool, nargs='?',
                        const=False, default=False, help="radial_as")
    parser.add_argument('-ud', '--use_determinant', type=str2bool, nargs='?',
                        const=True, default=True, help="use determinants for gaussian normalization")
    parser.add_argument('-np', '--normalize_pis', type=str2bool, nargs='?',
                        const=True, default=True, help="set all pis to 1/K for initialization")

    parser.add_argument('-qm', '--quantization_mode', type=int, default=0,
                        help="Quantization mode: 0 - no quantization, 1 - quantization each validation step,"
                             " 2 - fake quantization optimization (fix min/max),"
                             " 3 - fake quantization optimization (variable min/max, except for pis)")
    parser.add_argument('-bd', '--bit_depths', type=int, default=[20, 18, 6, 10, 10], nargs='+',
                        help="bit depths of each kind of parameter. number of numbers must be 5 in the order: A, musX, nu_e, pis, gamma_e")
    parser.add_argument('-qp', '--quantize_pis', type=str2bool, nargs='?',
                        const=True, default=True, help="Quantize Pis while optimization (only valid for quantization mode 1, for quantization mode 2,3 always True)")
    parser.add_argument('-lb', '--lower_bounds', type=float, default=[-2500, -.3, -5, 0, -32], nargs='+',
                        help="lower bounds of parameters for quantization while optimization")
    parser.add_argument('-ub', '--upper_bounds', type=float, default=[2500, 1.3, 5, 2, 32], nargs='+',
                        help="upper bounds of parameters for quantization while optimization")

    parser.add_argument('-yuv', '--use_yuv', type=str2bool, nargs='?',
                        const=True, default=True, help="uses YUV color space for modeling if three channels are provided.")
    parser.add_argument('-oyg', '--only_y_gamma', type=str2bool, nargs='?',
                        const=False, default=False,
                        help="train only slopes for y channel if yuv channels are used for modeling.")

    parser.add_argument('-ssim', '--ssim_opt', type=str2bool, nargs='?',
                        const=False, default=False,
                        help="SSIM optimization instead of MSE.")
    parser.add_argument('-sp', '--sampling_percentage', type=int, default=100, help="How many samples were used for each"
                                                                                    " update step in percentage (only"
                                                                                    " working if mse optimzed and batch"
                                                                                    " overlap equal 0")
    parser.add_argument('-ukl', '--update_kernel_list_iterations', type=int, default=None,
                        help="number of iterations between kernel list updates")
    parser.add_argument('-ovl', '--overlap_of_batches', type=int, default=0,
                        help="number of pixels they overlap between batches in each dimension")
    parser.add_argument('-svreg', '--svreg', type=float, default=0, help="l1-l2 regularization for support vectors")
    parser.add_argument('-hpc', '--hpc_mode', type=str2bool, nargs='?',
                        const=False, default=False,
                        help="if hpc mode only one inc step will be done.")
    parser.add_argument('-cis', '--current_inc_step', type=int, default=0,
                        help="current inc step to continue progress for HPC mode")
    parser.add_argument('-kcn', '--kernel_count_norm_l1', type=str2bool, nargs='?',
                        const=False, default=False, help="Normalization of l1-Regulariation on pis for Sparsifiacation (relevant for Kernel Adding Approach")
    parser.add_argument('-tvs', '--train_svs', type=str2bool, nargs='?',
                        const=False, default=False, help="Train additional Support Vectors")

    args = parser.parse_args()

    main(**vars(args))
