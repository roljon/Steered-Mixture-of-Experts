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
         upper_bounds, use_yuv, only_y_gamma, ssim_opt, sampling_percentage, update_kernel_list_iterations, overlap_of_batches, svreg, hpc_mode, current_inc_step, kernel_count_norm_l1, train_svs, train_trafo, num_params_model, train_inverse_cov, init_flag, only_rec_from_checkpoint, loss_mask_path):

    if len(bit_depths) != 5:
        raise ValueError("Number of bit depths must be five!")

    if not (num_params_model == 2 or num_params_model == 4 or num_params_model == 6 or num_params_model == 8):
        raise ValueError("num_params_model == {0:d} is not a valid motion parameter model".format(num_params_model))

    if ssim_opt:
        sampling_percentage = 100

    if sampling_percentage <= 0 or sampling_percentage > 100:
        raise ValueError("Value of Sampling Percentage must be in range (0,100]")

    if quantization_mode >= 2:
        quantize_pis = True

    orig, precision, affines = read_image(image_path, use_yuv)

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

    use_loss_mask = False
    loss_mask = None
    if loss_mask_path is not None:
        loss_mask = np.load(loss_mask_path)["loss_mask"]
        use_loss_mask = True

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
                use_yuv=use_yuv, only_y_gamma=only_y_gamma, ssim_opt=ssim_opt, precision=precision, add_kernel_slots=inc_steps*np.prod(kernels_per_dim), overlap_of_batches=overlap_of_batches, kernel_count_as_norm_l1=kernel_count_norm_l1, train_svs=train_svs, affines=affines, train_trafo=train_trafo, num_params_model=num_params_model, train_inverse_cov=train_inverse_cov, init_flag=init_flag, only_rec_from_checkpoint=only_rec_from_checkpoint, loss_mask=loss_mask)


    if not train_svs:
        lr_mult_sv = 0

    optimizer1 = tf.train.AdamOptimizer(base_lr)
    optimizer2 = tf.train.AdamOptimizer(base_lr/lr_div)
    optimizer3 = tf.train.AdamOptimizer(base_lr*lr_mult*1)
    optimizer4 = tf.train.AdamOptimizer(base_lr * lr_mult_sv)
    optimizer5 = tf.train.AdamOptimizer(base_lr)

    # optimizers have to be set before the restore
    smoe.set_optimizer(optimizer1, optimizer2, optimizer3, optimizer4, optimizer5)

    base_lr_inc = base_lr  # *10
    optimizer_inc1 = tf.train.AdamOptimizer(base_lr_inc)
    optimizer_inc2 = tf.train.AdamOptimizer(base_lr_inc / 100)
    optimizer_inc3 = tf.train.AdamOptimizer(base_lr_inc * 1000)
    smoe.set_inc_optimizer(optimizer_inc1, optimizer_inc2, optimizer_inc3)

    if checkpoint_path is not None:
        smoe.restore(checkpoint_path)
        if normalize_pis:
            smoe.get_reconstruction()
            kernel_list = np.zeros((smoe.start_pis,))
            for ii in range(len(smoe.kernel_list_per_batch)):
                kernel_list = np.logical_or(kernel_list, smoe.kernel_list_per_batch[ii])
            smoe.session.run([smoe.re_normalize_pis_op], feed_dict={smoe.kernel_list: kernel_list})
        smoe.update_kernel_list()

    if overlap_of_batches > 0:
        sampling_percentage = 100 # otherwise it is not working!

    if hpc_mode and current_inc_step > 0:
        #smoe.kernel_count = smoe.session.run(smoe.num_pi_op)
        smoe.kernel_count += (current_inc_step - 1) * smoe.num_inc_kernels
        smoe.kernel_list_per_batch = [np.ones((smoe.add_kernel_slots + 2 * smoe.start_pis,),
                                              dtype=bool)] * smoe.start_batches

    if iterations != 0:
        smoe.train(iterations, val_iter=validation_iterations, ukl_iter=update_kernel_list_iterations, pis_l1=l1reg, sv_l1_sub_l2=svreg,
                   sampling_percentage=sampling_percentage, use_loss_mask=use_loss_mask,
                   callbacks=[loss_plotter.plot, image_plotter.plot, logger.log])

        if not only_rec_from_checkpoint: # and any other conditions actually
            optimizer2 = tf.train.AdamOptimizer(base_lr / lr_div * 10)
            smoe.set_optimizer(optimizer1, optimizer2, optimizer3, optimizer4, optimizer5)

            further_iterations = 1000
            for kk in range(kernels_per_dim[2]):
                rec = smoe.get_reconstruction()
                diff = np.average(np.power(255 * (smoe.image - rec), 2), axis=-1, weights=[6/8, 1/8, 1/8])

                ## Code to sample direct randomly prop to diff-image
                diff = diff**2
                idx = np.random.choice(np.arange(0, np.prod(smoe.image.shape[0:-1])), p=diff.flatten() / np.sum(diff),
                                       size=np.prod(kernels_per_dim[0:2]), replace=False)
                idx_3d = np.unravel_index(idx, (smoe.image.shape[0:-1]), order='C')
                musX_3d = np.stack([idx_3d[0] / (smoe.image.shape[0] - 1), idx_3d[1] / (smoe.image.shape[1] - 1),
                                    idx_3d[2] / (smoe.image.shape[2] - 1)], axis=1)
                '''
                x_coord = np.linspace(0, smoe.image.shape[1] - 1, smoe.image.shape[1])
                y_coord = np.linspace(0, smoe.image.shape[0] - 1, smoe.image.shape[0])
                z_coord = np.linspace(0, smoe.image.shape[2] - 1, smoe.image.shape[2])
                X, Y, Z = np.meshgrid(x_coord, y_coord, z_coord)
                density_grid = np.ones((smoe.image.shape[0:-1]))
                adaptation_map = np.ones((smoe.image.shape[0:-1])) * diff**2
                bw = 2
        
                xj = []
                yj = []
                zj = []
                for jj in range(np.prod(kernels_per_dim)):
                    print(jj)
                    density_grid_tmp = density_grid * adaptation_map
                    idx = np.random.choice(np.arange(0, np.prod(smoe.image.shape[0:-1])),
                                           p=density_grid_tmp.flatten() / (np.sum(density_grid_tmp)))
                    idx = np.unravel_index(idx, (smoe.image.shape[0:-1]), order='C')
                    density_grid = density_grid * (
                            1 - np.exp(
                        - ((Y - y_coord[idx[0]]) ** 2 + (X - x_coord[idx[1]]) ** 2 + (Z - z_coord[idx[2]]) ** 2) / (
                                    bw ** 2))) ** 7
                    xj.append(idx[1])
                    yj.append(idx[0])
                    zj.append(idx[2])
                x_means = x_coord[np.stack(xj)] / (smoe.image.shape[1] - 1)
                y_means = y_coord[np.stack(yj)] / (smoe.image.shape[0] - 1)
                z_means = z_coord[np.stack(zj)] / (smoe.image.shape[2] - 1)
                musX_3d = np.stack([y_means, x_means, z_means], axis=1)
                '''
                '''
                A_3d = []
                for ii in range(len(xj)):
                    print(ii)
                    dist = np.sqrt(((x_coord[xj[ii]] - x_coord[np.stack(xj)]) / smoe.image.shape[1]) ** 2 + (
                            (y_coord[yj[ii]] - y_coord[np.stack(yj)]) / smoe.image.shape[0]) ** 2 + (
                                           (z_coord[zj[ii]] - z_coord[np.stack(zj)]) / smoe.image.shape[2]) ** 2)
                    dist = dist[dist > 0]
                    a = np.ones((smoe.dim_domain,)) * 1.3 / (np.min(dist))
                    A = np.diag(a)
                    A_3d.append(A)
                '''

                old_pis, old_nues, old_musX = smoe.session.run([smoe.pis_var, smoe.nu_e_var, smoe.musX_var])

                if kk == 0:
                    num_2d_kernels = np.sum(old_pis != 0)
                idx = np.zeros((smoe.start_pis,), dtype=np.bool)
                idx[num_2d_kernels + kk * np.prod(kernels_per_dim[0:2]):num_2d_kernels + (
                            kk + 1) * np.prod(kernels_per_dim[0:2])] = True
                old_pis[idx] = 1
                old_musX[idx] = musX_3d
                smoe.session.run([smoe.re_assign_pis_op, smoe.re_assign_musX_op], feed_dict={smoe.pis: old_pis, smoe.musX: old_musX})
                smoe.update_kernel_list()
                smoe.valid = False
                w = smoe.get_weight_matrix_argmax()
                for ii in np.where(idx)[0]:
                    old_nues[ii] = np.mean(smoe.image[w == ii], axis=0)
                if np.any(np.isnan(old_nues)):
                    old_nues[np.isnan(old_nues)] = 0.5
                smoe.session.run([smoe.re_assign_nue_op], feed_dict={smoe.nu_e: old_nues})

                if kk == kernels_per_dim[2] - 1:
                    further_iterations = 5000

                smoe.train(further_iterations, val_iter=validation_iterations, ukl_iter=update_kernel_list_iterations, pis_l1=l1reg,
                           sv_l1_sub_l2=svreg,
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
    parser.add_argument('-tt', '--train_trafo', type=str2bool, nargs='?',
                        const=False, default=False, help="train affine/homography transformation in the pixel domain for each frame for video")
    parser.add_argument('-npm', '--num_params_model', type=int, default=6,
                        help="kind of parameter model which can be chosen for global motion compensation (2-,4-,6-,8-Params)")
    parser.add_argument('-tiv', '--train_inverse_cov', type=str2bool, nargs='?',
                        const=False, default=False, help="train directly the inverse covariance matrix (which won't be necessarily a valid CovMat)")
    parser.add_argument('-if', '--init_flag', type=float, default=1, help="Init Flag for Kernel Init in case of Video with Trafo: 1 - kinda affine trafo on regular kernel grid"
                                                                                                                             ", 2 - along t-axis num of kernel depends on lum-var"
                                                                                                                             ", 3 - along t-axis num of kernel depends on num of frames in this spatial location"
                                                                                                                             ", {2,3}.0 - lonely kernels along t-axis are init'ed by var and mean of time coord"
                                                                                                                             ", {2,3}.5 - lonely kernels along t-axis are init'ed by mean=.5 and regular bandwidth")
    parser.add_argument('-orfc', '--only_rec_from_checkpoint', type=str2bool, nargs='?',
                        const=False, default=False,
                        help="flag to signalize that we are only interested in a reconstruction from a checkpoint. Makes things easier.")
    parser.add_argument('-mask', '--loss_mask_path', type=str, default=None, help="input image")

    args = parser.parse_args()

    main(**vars(args))
