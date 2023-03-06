import numpy as np
import os
from skimage.feature import peak_local_max
from skimage.measure import compare_ssim
import matplotlib.pyplot as plt
import tensorflow as tf
from ops.special_math_ops import exponential_space_einsum as einsum
import progressbar
from itertools import product, combinations
from quantizer import quantize_params, rescaler
from scipy.ndimage import gaussian_filter
from ops.image_ops_impl import custom_ssim
import cv2
from scipy.cluster.vq import vq, kmeans, whiten
from scipy.cluster.vq import kmeans2


def sliding_window(image, Overlap, BatchSize):
    # slide a window across the image
    if len(image.shape) == 3: # image case
        image = np.pad(image, ((Overlap, Overlap), (Overlap, Overlap), (0, 0)), 'constant', constant_values=0)
        for y in range(0, image.shape[0] - 2 * Overlap, BatchSize[0]):
            for x in range(0, image.shape[1] - 2 * Overlap, BatchSize[1]):
                # yield the current window
                yield (np.array([y - Overlap, x - Overlap]), image[y:y + BatchSize[0] + 2 * Overlap,
                                                             x:x + BatchSize[1] + 2 * Overlap, :])
    elif len(image.shape) == 4: # video case
        image = np.pad(image, ((Overlap, Overlap), (Overlap, Overlap), (Overlap, Overlap), (0, 0)), 'constant', constant_values=0)
        for y in range(0, image.shape[0] - 2 * Overlap, BatchSize[0]):
            for x in range(0, image.shape[1] - 2 * Overlap, BatchSize[1]):
                for z in range(0, image.shape[2] - 2 * Overlap, BatchSize[2]):
                    # yield the current window
                    yield (np.array([y - Overlap, x - Overlap, z - Overlap]),
                           image[y:y + BatchSize[0] + 2 * Overlap, x:x + BatchSize[1] + 2 * Overlap,
                           z:z + BatchSize[2] + 2 * Overlap, :])

class Smoe:
    def __init__(self, image, kernels_per_dim=None, train_pis=True, init_params=None, start_batches=1,
                 batch_size=None, train_gammas=True, train_musx=True, use_diff_center=False, radial_as=False, use_determinant=False,
                 normalize_pis=True, quantization_mode=0, bit_depths=None, quantize_pis=False, lower_bounds=None,
                 upper_bounds=None, use_yuv=True, only_y_gamma=False, ssim_opt=False, precision=8, add_kernel_slots=0, iter_offset=0, margin=0.5, overlap_of_batches=0, kernel_count_as_norm_l1=False, train_svs=False, affines=None, train_trafo=False, num_params_model=6, train_inverse_cov=True, init_flag=1, only_rec_from_checkpoint=False, loss_mask=None):
        self.batch_shape = None
        self.use_yuv = use_yuv
        self.only_y_gamma = only_y_gamma
        self.ssim_opt = ssim_opt
        self.use_diff_center = use_diff_center
        self.precision = precision
        self.train_mask = None
        self.add_kernel_slots = add_kernel_slots
        self.kernel_count_as_norm_l1 = kernel_count_as_norm_l1

        # init params
        self.pis_init = None
        self.musX_init = None
        self.A_init = None
        self.gamma_e_init = None
        self.nu_e_init = None

        self.qparams = None
        self.rparams = None

        # tf vars
        self.pis_var = None
        self.musX_var = None
        self.A_diagonal_var = None
        self.A_corr_var = None
        self.gamma_e_var = None
        self.nu_e_var = None

        self.pis_best_var = None
        self.musX_best_var = None
        self.A_diagonal_best_var = None
        self.A_corr_best_var = None
        self.gamma_e_best_var = None
        self.nu_e_best_var = None

        # tf inc vars
        self.pis_inc_var = None
        self.musX_inc_var = None
        self.A_diagonal_inc_var = None
        self.A_corr_inc_var = None
        self.gamma_e_inc_var = None
        self.nu_e_inc_var = None

        # tf affine trafo vars
        self.h11_var = None
        self.h12_var = None
        self.h13_var = None
        self.h21_var = None
        self.h22_var = None
        self.h23_var = None
        self.h31_var = None
        self.h32_var = None
        self.stop_first_gradient_op = None
        self.num_params_model = num_params_model
        self.loss_mask = loss_mask

        self.h11_best_var = None
        self.h12_best_var = None
        self.h13_best_var = None
        self.h21_best_var = None
        self.h22_best_var = None
        self.h23_best_var = None
        self.h31_best_var = None
        self.h32_best_var = None

        self.qh11 = None
        self.qh12 = None
        self.qh13 = None
        self.qh21 = None
        self.qh22 = None
        self.qh23 = None
        self.qh31 = None
        self.qh32 = None

        # tf inc ops
        self.stack_inc = None
        self.train_inc_op = None
        self.zero_inc_op = None
        self.reset_optimizers_op = None
        self.accum_inc_ops = None
        self.optimizer_inc1 = None
        self.optimizer_inc2 = None
        self.optimizer_inc3 = None
        self.insert_pos = None
        self.assign_inc_vars_op = None
        self.reinit_inc_vars_op = None
        self.assign_inc_opt_vars_op = None
        self.var_inc_opt1 = None
        self.var_inc_opt2 = None
        self.var_inc_opt3 = None


        # tf ops
        self.restoration_op = None
        self.w_e_op = None
        self.w_e_max_op = None
        self.loss_op = None
        self.train_op = None
        self.checkpoint_best_op = None
        self.mse_op = None
        self.target_op = None
        self.domain_op = None
        self.domain_final = None
        self.joint_domain_batched_op = None
        self.pis_l1 = None
        self.u_l1 = None
        self.zero_op = None
        self.accum_ops = None
        self.current_batch_number = None
        self.num_pi_op = None
        self.save_op = None
        self.kernel_list = None
        self.indices = None
        self.maha_dist = None
        self.var_opt1 = None
        self.var_opt2 = None
        self.var_opt3 = None
        self.var_opt4 = None
        self.var_opt5 = None
        # tf ops - feeding points
        self.musX = None
        self.nu_e = None
        self.gamma_e = None
        self.A = None
        self.pis = None
        # tf ops - quant variable
        self.qA = None
        self.qmusX = None
        self.qnu_e = None
        self.qgamma_e = None
        self.qpis = None

        # optimizers
        self.optimizer1 = None
        self.optimizer2 = None
        self.optimizer3 = None
        self.optimizer4 = None
        self.optimizer5 = None

        # others
        # TODO refactor to logger class
        self.losses = []
        self.qlosses = []
        self.losses_history = []
        self.best_loss = None
        self.best_qloss = None
        self.mses = []
        self.qmses = []
        self.mses_history = []
        self.best_mse = []
        self.best_qmse = []
        self.num_pis = []
        self.num_svs = []

        self.iter = iter_offset
        self.valid = False
        self.qvalid = False
        self.reconstruction_image = None
        self.weight_matrix_argmax = None
        self.qreconstruction_image = None
        self.qweight_matrix_argmax = None
        self.weight_matrix = None
        self.qweight_matrix = None
        self.train_pis = train_pis
        self.train_gammas = train_gammas
        self.train_musx = train_musx
        self.radial_as = radial_as
        self.use_determinant = use_determinant
        self.quantization_mode = quantization_mode
        self.bit_depths = bit_depths
        self.quantize_pis = quantize_pis
        self.lower_bounds = lower_bounds
        self.upper_bounds = upper_bounds
        self.with_SV = train_svs
        self.train_trafo = train_trafo
        self.train_inverse_cov = train_inverse_cov

        self.affines = affines
        self.transformed_domain = None
        self.only_rec_from_checkpoint = only_rec_from_checkpoint

        # generate initializations
        self.start_batches = start_batches  # start_batches corresponds to desired batch numbers
        self.image = image
        self.image_flat = None
        self.dim_domain = image.ndim - 1
        self.num_pixel = np.prod(image.shape[0:self.dim_domain])
        self.init_domain_and_target()

        if batch_size[0] is not None:
            if len(batch_size) == self.dim_domain:
                self.batch_size_valued = batch_size
            elif len(batch_size) == 1:
                self.batch_size_valued = batch_size * np.ones((self.dim_domain,), dtype=np.int)
            else:
                raise ValueError("Required BatchSize doesn't fit to input dimension")
            # sanity check otherwise error
            for ii in range(self.dim_domain):
                if self.joint_domain.shape[ii] % self.batch_size_valued[ii] > 0:
                    raise ValueError("Required BatchSize is not compatible to input dimensions")
        else:
            self.batch_size_valued = self.batch_shape[0:-1]
        self.overlap = overlap_of_batches
        self.batch_size = tuple(np.array(self.batch_size_valued) + 2*self.overlap)

        self.start_batches = np.int(np.prod(np.ceil(np.array(self.image.shape[:-1]) / np.array(self.batch_size_valued))))

        assert kernels_per_dim is not None or init_params is not None, \
            "You need to specify the kernel grid size or give initial parameters."

        if init_params:
            self.pis_init = init_params['pis']
            self.musX_init = init_params['musX']
            #self.A_init = init_params['A']
            self.A_init = init_params['A_diagonal'] + init_params['A_corr']
            self.gamma_e_init = init_params['gamma_e']
            self.nu_e_init = init_params['nu_e']
        else:
            self.generate_kernel_grid(kernels_per_dim)
            self.generate_experts()
            self.generate_pis(normalize_pis)

        self.start_pis = self.pis_init.size
        # for add_kernel
        self.kernel_count = self.pis_init.size
        self.num_inc_kernels = None

        self.margin = margin

        self.random_sampling_per_batch = [np.ones((np.prod(self.batch_size),),
                                                  dtype=np.float32) / np.prod(
            self.batch_size)] * self.start_batches
        self.get_train_mask()

        gpu_options = tf.GPUOptions(allow_growth=True)
        self.session = tf.InteractiveSession(config=tf.ConfigProto(gpu_options=gpu_options))
        # self.session = tf.Session()

        if self.affines is not None:
            #kernels_per_dim[1] = np.round(kernels_per_dim[1] * (np.max(self.affines[:, 0, 2] / (self.image.shape[1] - 1)) - np.min(self.affines[:, 0, 2] / (self.image.shape[1] - 1)) + 1 ))
            #kernels_per_dim[0] = np.round(kernels_per_dim[0] * (np.max(
            #    self.affines[:, 1, 2] / (self.image.shape[0] - 1)) - np.min(
            #    self.affines[:, 1, 2] / (self.image.shape[0] - 1)) + 1 ))

            musx_init = self.musX_init.copy()
            A_init = self.A_init.copy()
            nu_e_init = self.nu_e_init.copy()
            gamma_e_init = self.gamma_e_init.copy()
            pis_init = self.pis_init.copy()
            self.do_perspectiveTransform(affines, kernels_per_dim, init_flag)
        if self.affines is not None or self.dim_domain == 3 and self.train_trafo:
            self.initialize_frames_list()
            if self.affines is None:
                self.transformed_domain = self.joint_domain.copy()

        self.kernel_assignment_to_model = [np.concatenate([np.ones((self.A_init.shape[0],), dtype=bool), np.zeros((A_init.shape[0],), dtype=bool)], axis=0),
                                           np.concatenate([np.zeros((self.A_init.shape[0],), dtype=bool), np.ones((A_init.shape[0],), dtype=bool)], axis=0)]

        self.A_init = np.concatenate([self.A_init, A_init], axis=0)
        self.nu_e_init = np.concatenate([self.nu_e_init, nu_e_init], axis=0)
        self.gamma_e_init = np.concatenate([self.gamma_e_init, gamma_e_init], axis=0)
        self.pis_init = np.concatenate([self.pis_init, np.zeros_like(pis_init)], axis=0) # do zeros_like to remove bg kernel first
        self.musX_init[:, 2] = -5
        self.musX_init = np.concatenate([self.musX_init, musx_init], axis=0)

        self.start_pis = self.pis_init.size
        # for add_kernel
        self.kernel_count = self.pis_init.size

        #self.dim_domain = 2

        self.init_model(self.nu_e_init, self.gamma_e_init, self.pis_init, self.musX_init, self.A_init, self.affines, add_kernel_slots)
        if True: #self.only_rec_from_checkpoint:
            self.kernel_list_per_batch = [np.ones((self.start_pis,), dtype=bool)] * self.start_batches
        else:
            self.initialize_kernel_list(add_kernel_slots)

        if self.affines is not None and not self.only_rec_from_checkpoint:
            w = self.get_weight_matrix_argmax()
            w_e = self.get_weight_matrix()
            nue_init_transformed = []
            for ii in range(len(self.pis_init)):
                nue_init_transformed.append(np.mean(self.image[w == ii], axis=0))
            nue_init_transformed_ = np.stack(nue_init_transformed)
            if np.any(np.isnan(nue_init_transformed_)):
                print('Some Kernels are not belonging to the argmax weighting matrix!')
                nue_init_transformed_[np.isnan(nue_init_transformed_)] = 0.5
            self.session.run([self.re_assign_nue_op], feed_dict={self.nu_e: nue_init_transformed_})

    def init_model(self, nu_e_init, gamma_e_init, pis_init, musX_init, A_init, affines, add_kernel_slots=0):

        # TODO make radial work again, uncomment for pcs scripts
        if A_init.ndim == 1:
            self.radial_as = True

        num_of_all_kernels = self.start_pis

        if add_kernel_slots > 0:
            num_of_all_kernels = add_kernel_slots + 2 * self.start_pis

            #assert A_init.ndim != 1 and self.radial_as is False, "sorry, no radial kernels with add_kernel feature at the moment"
            # TODO allow different num_inc_kernels
            self.num_inc_kernels = pis_init.size
            self.nu_e_inc_var = tf.Variable(nu_e_init, dtype=tf.float32)
            self.gamma_e_inc_var = tf.Variable(gamma_e_init, trainable=self.train_gammas, dtype=tf.float32)
            self.musX_inc_var = tf.Variable(musX_init, dtype=tf.float32)

            if self.radial_as:
                if A_init.ndim == 1:
                    self.A_diagonal_inc_var = tf.Variable(A_init, dtype=tf.float32)
                else:
                    self.A_diagonal_inc_var = tf.Variable(A_init[:, 0, 0], dtype=tf.float32)
                self.A_corr_inc_var = tf.Variable(np.zeros_like(A_init), trainable=False, dtype=tf.float32)
            else:
                self.A_diagonal_inc_var = tf.Variable(A_init, dtype=tf.float32)
                self.A_corr_inc_var = tf.Variable(np.zeros_like(A_init), dtype=tf.float32)
            self.pis_inc_var = tf.Variable(np.zeros_like(pis_init), trainable=self.train_pis, dtype=tf.float32)

            self.nu_e_inc_new = tf.placeholder(tf.float32, nu_e_init.shape)
            self.gamma_e_inc_new = tf.placeholder(tf.float32, gamma_e_init.shape)
            self.musX_inc_new = tf.placeholder(tf.float32, musX_init.shape)
            if self.radial_as and A_init.ndim != 1:
                self.A_diagonal_inc_new = tf.placeholder(tf.float32, A_init[:, 0, 0].shape)
            else:
                self.A_diagonal_inc_new = tf.placeholder(tf.float32, A_init.shape)
            self.A_corr_inc_new = tf.placeholder(tf.float32, A_init.shape)
            self.pis_inc_new = tf.placeholder(tf.float32, pis_init.shape)

            nu_e_inc_assign = self.nu_e_inc_var.assign(self.nu_e_inc_new)
            gamma_e_inc_assign = self.gamma_e_inc_var.assign(self.gamma_e_inc_new)
            musX_inc_assign = self.musX_inc_var.assign(self.musX_inc_new)
            A_diagonal_inc_assign = self.A_diagonal_inc_var.assign(self.A_diagonal_inc_new)
            A_corr_inc_assign = self.A_corr_inc_var.assign(self.A_corr_inc_new)
            pis_inc_assign = self.pis_inc_var.assign(self.pis_inc_new)

            self.reinit_inc_vars_op = tf.group(nu_e_inc_assign, gamma_e_inc_assign, musX_inc_assign,
                                               A_diagonal_inc_assign, A_corr_inc_assign, pis_inc_assign)

            nu_e_init = np.vstack((nu_e_init, np.zeros((add_kernel_slots,) + nu_e_init.shape[1:])))
            gamma_e_init = np.vstack((gamma_e_init, np.zeros((add_kernel_slots,) + gamma_e_init.shape[1:])))
            musX_init = np.vstack((musX_init, np.zeros((add_kernel_slots,) + musX_init.shape[1:])))
            pis_init = np.hstack((pis_init, np.zeros((add_kernel_slots,) + pis_init.shape[1:])))
            A_init = np.vstack((A_init, np.zeros((add_kernel_slots,) + A_init.shape[1:])))



        self.nu_e_var = tf.Variable(nu_e_init, dtype=tf.float32)
        self.gamma_e_var = tf.Variable(gamma_e_init, trainable=self.train_gammas, dtype=tf.float32)
        if self.use_diff_center:
            self.musX_var = tf.Variable(np.zeros_like(musX_init), trainable=self.train_musx, dtype=tf.float32)
            musX_grid = tf.constant(musX_init, dtype=tf.float32)
        else:
            self.musX_var = tf.Variable(musX_init, trainable=self.train_musx, dtype=tf.float32)
        #self.A_var = tf.Variable(A_init, dtype=tf.float32)
        self.pis_var = tf.Variable(pis_init, trainable=self.train_pis, dtype=tf.float32)

        self.stack_inc = tf.constant([1.], dtype=tf.float32)
        self.stack_orig = tf.constant([1.], dtype=tf.float32)
        self.insert_pos = tf.placeholder(tf.int32)

        if self.with_SV:
            '''
            #self.SV_var = tf.Variable(np.zeros((np.prod(self.joint_domain.shape[0:-1]), 1)), dtype=tf.float32)
            #self.SV_var = tf.Variable(np.zeros((np.prod(np.array(self.image.shape[0:-1]) + self.overlap), 1)), dtype=tf.float32)
            self.SV_var = tf.Variable(np.zeros((np.prod(np.array(self.image.shape[0:-1]) + self.overlap), num_of_all_kernels)),
                                      dtype=tf.float32)
            self.bw_SV = tf.Variable(-34 / 2 * 50 / 32 * np.sqrt(np.prod(self.joint_domain.shape[0:-1])), trainable=True, dtype=tf.float32)
    
            '''
            gamma_init = np.sqrt(34 / 2 * 50 / 32 * np.sqrt(np.prod(self.joint_domain.shape[0:-1])))

            #A_prototype = np.zeros((2, 2))
            #np.fill_diagonal(A_prototype, gamma_init)
            #self.bw_SV = tf.Variable(A_prototype, dtype=tf.float32)
            #A_SV = tf.tile(tf.expand_dims(self.bw_SV, axis=0), (np.prod(self.joint_domain.shape[0:-1]), 1, 1))

            #self.SV_var = tf.Variable(np.zeros((np.prod(self.joint_domain.shape[0:-1]), 1)), dtype=tf.float32)
            self.SV_var = tf.Variable(np.zeros((np.prod(np.array(self.image.shape[0:-1]) + 2*self.overlap), 1)),
                                      dtype=tf.float32)

            A_prototype = np.zeros((2, 2))
            np.fill_diagonal(A_prototype, gamma_init)
            A_init_SV = np.tile(A_prototype, (np.prod(self.joint_domain.shape[0:-1]), 1, 1))
            self.bw_diag_SV = tf.Variable(A_init_SV, dtype=tf.float32)
            self.bw_corr_SV = tf.Variable(np.zeros_like(A_init_SV), dtype=tf.float32)


        if self.radial_as:
            if A_init.ndim == 1:
                self.A_diagonal_var = tf.Variable(A_init, dtype=tf.float32)
            else:
                self.A_diagonal_var = tf.Variable(A_init[:, 0, 0], dtype=tf.float32)
            self.A_corr_var = tf.Variable(np.zeros_like(A_init), trainable=False, dtype=tf.float32)
        else:
            self.A_diagonal_var = tf.Variable(A_init, dtype=tf.float32)
            self.A_corr_var = tf.Variable(np.zeros_like(A_init), dtype=tf.float32)


        #assert add_kernel_slots > 0, "this is just for testing"

        if add_kernel_slots > 0:
            nue_e_assign = self.nu_e_var[self.insert_pos:self.insert_pos + self.num_inc_kernels].assign(self.nu_e_inc_var)
            gamma_e_assign = self.gamma_e_var[self.insert_pos:self.insert_pos + self.num_inc_kernels].assign(
                self.gamma_e_inc_var)
            musX_assign = self.musX_var[self.insert_pos:self.insert_pos + self.num_inc_kernels].assign(self.musX_inc_var)
            A_diagonal_assign = self.A_diagonal_var[self.insert_pos:self.insert_pos + self.num_inc_kernels].assign(self.A_diagonal_inc_var)
            A_corr_assign = self.A_corr_var[self.insert_pos:self.insert_pos + self.num_inc_kernels].assign(
                self.A_corr_inc_var)
            pis_assign = self.pis_var[self.insert_pos:self.insert_pos + self.num_inc_kernels].assign(self.pis_inc_var)

            self.assign_inc_vars_op = tf.group(nue_e_assign, gamma_e_assign, musX_assign, A_diagonal_assign, A_corr_assign, pis_assign)

            if self.with_SV:
                self.assign_sv_zero = self.SV_var.assign(np.zeros((np.prod(np.array(self.image.shape[0:-1]) + 2*self.overlap), 1)))

            # concat and give them name like VAR_concat and then quantize!
            concat_musX, concat_nu_e, concat_gamma_e, concat_A_diagonal, concat_A_corr, concat_pis = \
                [tf.concat([self.musX_var, self.musX_inc_var], 0),
                 tf.concat([self.nu_e_var, self.nu_e_inc_var], 0),
                 tf.concat([self.gamma_e_var, self.gamma_e_inc_var], 0),
                 tf.concat([self.A_diagonal_var, self.A_diagonal_inc_var], 0),
                 tf.concat([self.A_corr_var, self.A_corr_inc_var], 0),
                 tf.concat([self.pis_var*self.stack_orig, self.pis_inc_var*self.stack_inc], 0)]
        else:
            concat_musX = self.musX_var
            concat_nu_e = self.nu_e_var
            concat_gamma_e = self.gamma_e_var
            concat_A_diagonal = self.A_diagonal_var
            concat_A_corr = self.A_corr_var
            concat_pis = self.pis_var

        # Quantization of parameters (if requested)
        if self.quantization_mode >= 2 or self.quantize_pis:
            self.qpis = tf.quantization.fake_quant_with_min_max_args(concat_pis, min=self.lower_bounds[3],
                                                        max=self.upper_bounds[3], num_bits=self.bit_depths[3])
        else:
            self.qpis = concat_pis

        pis_mask = self.qpis > 0

        if self.quantization_mode == 2:
            self.qA_diagonal = tf.quantization.fake_quant_with_min_max_args(concat_A_diagonal, min=self.lower_bounds[0],
                                                               max=self.upper_bounds[0], num_bits=self.bit_depths[0])
            self.qA_corr = tf.quantization.fake_quant_with_min_max_args(concat_A_corr, min=self.lower_bounds[0],
                                                           max=self.upper_bounds[0], num_bits=self.bit_depths[0])

            self.qmusX = tf.quantization.fake_quant_with_min_max_args(concat_musX,
                                                         min=self.lower_bounds[1], max=self.upper_bounds[1],
                                                         num_bits=self.bit_depths[1])
            self.qnu_e = tf.quantization.fake_quant_with_min_max_args(concat_nu_e,
                                                         min=self.lower_bounds[2], max=self.upper_bounds[2],
                                                         num_bits=self.bit_depths[2])
            self.qgamma_e = tf.quantization.fake_quant_with_min_max_args(concat_gamma_e,
                                                            min=self.lower_bounds[4], max=self.upper_bounds[4],
                                                            num_bits=self.bit_depths[4])
        elif self.quantization_mode == 3:
            if self.radial_as:
                min_A_diagonal = tf.reduce_min(tf.boolean_mask(concat_A_diagonal, pis_mask))
                max_A_diagonal = tf.reduce_max(tf.boolean_mask(concat_A_diagonal, pis_mask))
                qA_diagonal = tf.quantization.fake_quant_with_min_max_vars(concat_A_diagonal, min=0,
                                                              max=max_A_diagonal - min_A_diagonal,
                                                              num_bits=self.bit_depths[0])
                self.qA_diagonal = qA_diagonal + min_A_diagonal
            else:
                min_A_diagonal = tf.reduce_min(tf.linalg.diag_part(tf.boolean_mask(concat_A_diagonal, pis_mask)))
                max_A_diagonal = tf.reduce_max(tf.linalg.diag_part(tf.boolean_mask(concat_A_diagonal, pis_mask)))
                qA_diagonal = tf.quantization.fake_quant_with_min_max_vars(concat_A_diagonal - min_A_diagonal, min=0,
                                                              max=max_A_diagonal - min_A_diagonal,
                                                              num_bits=self.bit_depths[0])
                self.qA_diagonal = qA_diagonal + min_A_diagonal
            self.qA_corr = tf.quantization.fake_quant_with_min_max_vars(concat_A_corr,
                                                           min=tf.reduce_min(tf.boolean_mask(concat_A_corr, pis_mask)),
                                                           max=tf.reduce_max(tf.boolean_mask(concat_A_corr, pis_mask)),
                                                           num_bits=self.bit_depths[0])
            if self.train_musx:
                self.qmusX = tf.quantization.fake_quant_with_min_max_vars(concat_musX,
                                                             min=tf.reduce_min(tf.boolean_mask(concat_musX, pis_mask)),
                                                             max=tf.reduce_max(tf.boolean_mask(concat_musX, pis_mask)),
                                                             num_bits=self.bit_depths[1])
            else:
                self.qmusX = concat_musX

            min_nu_e = tf.reduce_min(tf.boolean_mask(concat_nu_e, pis_mask))
            max_nu_e = tf.reduce_max(tf.boolean_mask(concat_nu_e, pis_mask))
            qnu_e = tf.quantization.fake_quant_with_min_max_vars(concat_nu_e - min_nu_e, min=0, max=max_nu_e-min_nu_e, num_bits=self.bit_depths[2])
            self.qnu_e = qnu_e + min_nu_e

            self.qgamma_e = tf.quantization.fake_quant_with_min_max_vars(concat_gamma_e,
                                                            min=tf.reduce_min(tf.boolean_mask(concat_gamma_e, pis_mask)),
                                                            max=tf.reduce_max(tf.boolean_mask(concat_gamma_e, pis_mask)),
                                                            num_bits=self.bit_depths[4])
        else:
            self.qA_diagonal = concat_A_diagonal
            self.qA_corr = concat_A_corr
            self.qmusX = concat_musX
            self.qnu_e = concat_nu_e
            self.qgamma_e = concat_gamma_e



        num_channels = gamma_e_init.shape[-1]


        self.joint_domain_batched_op = tf.placeholder(shape=(None, self.dim_domain + num_channels), dtype=tf.float32)

        self.target_op = self.joint_domain_batched_op[:, self.dim_domain:]
        self.domain_op = self.joint_domain_batched_op[:, :self.dim_domain]

        self.loss_weights = tf.placeholder_with_default(tf.ones((tf.shape(self.target_op)[0], 1), dtype=tf.float32), shape=(None, 1))

        self.kernel_list = tf.placeholder(shape=(None,), dtype=tf.bool)

        if self.dim_domain == 3 and self.train_trafo or affines is not None:
            self.frames_list = tf.placeholder(shape=(None,), dtype=tf.bool)

            if affines is not None:
                self.h11_var = tf.Variable(affines[:, 0, 0], trainable=self.train_trafo and self.num_params_model >= 4, dtype=tf.float32)
                self.h12_var = tf.Variable(affines[:, 0, 1], trainable=self.train_trafo and self.num_params_model >= 4, dtype=tf.float32)
                self.h13_var = tf.Variable(affines[:, 0, 2] / (self.image.shape[1] - 1), trainable=self.train_trafo,
                                           dtype=tf.float32)

                self.h21_var = tf.Variable(affines[:, 1, 0], trainable=self.train_trafo and self.num_params_model >= 6, dtype=tf.float32)
                self.h22_var = tf.Variable(affines[:, 1, 1], trainable=self.train_trafo and self.num_params_model >= 6, dtype=tf.float32)
                self.h23_var = tf.Variable(affines[:, 1, 2] / (self.image.shape[0] - 1), trainable=self.train_trafo,
                                           dtype=tf.float32)

                if affines.shape[1] == 3:
                    self.h31_var = tf.Variable(affines[:, 2, 0], trainable=self.train_trafo and self.num_params_model == 8, dtype=tf.float32)
                    self.h32_var = tf.Variable(affines[:, 2, 1], trainable=self.train_trafo and self.num_params_model == 8, dtype=tf.float32)
                else:
                    self.h31_var = tf.Variable(np.zeros_like(affines[:, 0, 0]), trainable=self.train_trafo and self.num_params_model == 8,
                                               dtype=tf.float32)
                    self.h32_var = tf.Variable(np.zeros_like(affines[:, 0, 0]), trainable=self.train_trafo and self.num_params_model == 8,
                                               dtype=tf.float32)
            else:
                self.h11_var = tf.Variable(np.ones((self.image.shape[2],)), trainable=self.train_trafo and self.num_params_model >= 4, dtype=tf.float32)
                self.h12_var = tf.Variable(np.zeros((self.image.shape[2],)), trainable=self.train_trafo and self.num_params_model >= 4, dtype=tf.float32)
                self.h13_var = tf.Variable(np.zeros((self.image.shape[2],)), trainable=self.train_trafo, dtype=tf.float32)

                self.h21_var = tf.Variable(np.zeros((self.image.shape[2],)), trainable=self.train_trafo and self.num_params_model >= 6, dtype=tf.float32)
                self.h22_var = tf.Variable(np.ones((self.image.shape[2],)), trainable=self.train_trafo and self.num_params_model >= 6, dtype=tf.float32)
                self.h23_var = tf.Variable(np.zeros((self.image.shape[2],)), trainable=self.train_trafo, dtype=tf.float32)

                self.h31_var = tf.Variable(np.zeros((self.image.shape[2],)), trainable=self.train_trafo and self.num_params_model == 8, dtype=tf.float32)
                self.h32_var = tf.Variable(np.zeros((self.image.shape[2],)), trainable=self.train_trafo and self.num_params_model == 8, dtype=tf.float32)

            if self.quantization_mode > 1: # dont distinguish between qm2 and qm3
                min_h11 = tf.reduce_min(self.h11_var)
                max_h11 = tf.reduce_max(self.h11_var)
                self.qh11 = tf.fake_quant_with_min_max_vars(self.h11_var - min_h11,
                                                            min=0,
                                                            max=max_h11 - min_h11,
                                                            num_bits=8) + min_h11

                min_h12 = tf.reduce_min(self.h12_var)
                max_h12 = tf.reduce_max(self.h12_var)
                self.qh12 = tf.fake_quant_with_min_max_vars(self.h12_var - min_h12,
                                                            min=0,
                                                            max=max_h12 - min_h12,
                                                            num_bits=8) + min_h12
                min_h13 = tf.reduce_min(self.h13_var)
                max_h13 = tf.reduce_max(self.h13_var)
                self.qh13 = tf.fake_quant_with_min_max_vars(self.h13_var - min_h13,
                                                            min=0,
                                                            max=max_h13 - min_h13,
                                                            num_bits=8) + min_h13

                min_h21 = tf.reduce_min(self.h21_var)
                max_h21 = tf.reduce_max(self.h21_var)
                self.qh21 = tf.fake_quant_with_min_max_vars(self.h21_var - min_h21,
                                                            min=0,
                                                            max=max_h21 - min_h21,
                                                            num_bits=8) + min_h21

                min_h22 = tf.reduce_min(self.h22_var)
                max_h22 = tf.reduce_max(self.h22_var)
                self.qh22 = tf.fake_quant_with_min_max_vars(self.h22_var - min_h22,
                                                            min=0,
                                                            max=max_h22 - min_h22,
                                                            num_bits=8) + min_h22
                min_h23 = tf.reduce_min(self.h23_var)
                max_h23 = tf.reduce_max(self.h23_var)
                self.qh23 = tf.fake_quant_with_min_max_vars(self.h23_var - min_h23,
                                                            min=0,
                                                            max=max_h23 - min_h23,
                                                            num_bits=8) + min_h23

                min_h31 = tf.reduce_min(self.h31_var)
                max_h31 = tf.reduce_max(self.h31_var)
                self.qh31 = tf.fake_quant_with_min_max_vars(self.h31_var - min_h31,
                                                            min=0,
                                                            max=max_h31 - min_h31,
                                                            num_bits=8) + min_h31

                min_h32 = tf.reduce_min(self.h32_var)
                max_h32 = tf.reduce_max(self.h32_var)
                self.qh32 = tf.fake_quant_with_min_max_vars(self.h32_var - min_h32,
                                                            min=0,
                                                            max=max_h32 - min_h32,
                                                            num_bits=8) + min_h32
            else:
                self.qh11 = self.h11_var
                self.qh12 = self.h12_var
                self.qh13 = self.h13_var

                self.qh21 = self.h21_var
                self.qh22 = self.h22_var
                self.qh23 = self.h23_var

                self.qh31 = self.h31_var
                self.qh32 = self.h32_var

            h11_f = tf.tile(tf.boolean_mask(self.qh11, self.frames_list), [np.prod(self.batch_shape[:-2]), ])
            h12_f = tf.tile(tf.boolean_mask(self.qh12, self.frames_list), [np.prod(self.batch_shape[:-2]), ])
            h13_f = tf.tile(tf.boolean_mask(self.qh13, self.frames_list), [np.prod(self.batch_shape[:-2]), ])

            h21_f = tf.tile(tf.boolean_mask(self.qh21, self.frames_list), [np.prod(self.batch_shape[:-2]), ])
            h22_f = tf.tile(tf.boolean_mask(self.qh22, self.frames_list), [np.prod(self.batch_shape[:-2]), ])
            h23_f = tf.tile(tf.boolean_mask(self.qh23, self.frames_list), [np.prod(self.batch_shape[:-2]), ])

            h31_f = tf.tile(tf.boolean_mask(self.qh31, self.frames_list), [np.prod(self.batch_shape[:-2]), ])
            h32_f = tf.tile(tf.boolean_mask(self.qh32, self.frames_list), [np.prod(self.batch_shape[:-2]), ])

            w_dash = 1
            if self.num_params_model == 2:
                x_dash = self.domain_op[:, 1] + h13_f
                y_dash = self.domain_op[:, 0] + h23_f

            elif self.num_params_model == 4:
                x_dash = h11_f * self.domain_op[:, 1] + h12_f * self.domain_op[:, 0] + h13_f
                y_dash = -h12_f * self.domain_op[:, 1] + h11_f * self.domain_op[:, 0] + h23_f

            elif self.num_params_model == 6 or self.num_params_model == 8:
                x_dash = h11_f * self.domain_op[:, 1] + h12_f * self.domain_op[:, 0] + h13_f
                y_dash = h21_f * self.domain_op[:, 1] + h22_f * self.domain_op[:, 0] + h23_f

                if self.num_params_model == 8:
                    w_dash = h31_f * self.domain_op[:, 1] + h32_f * self.domain_op[:, 0] + 1
            else:
                raise ValueError("Invalid motion parameter model!")

            #self.domain_final = tf.stack([y_dash / w_dash, x_dash / w_dash, self.domain_op[:, -1]], axis=1)
            self.domain_final = tf.stack([y_dash / w_dash, x_dash / w_dash, tf.ones_like(self.domain_op[:, -1]) * -5], axis=1)
        else:
            self.domain_final = self.domain_op

        if self.with_SV:
            self.mask_of_sv_in_batch = tf.placeholder(shape=(None,), dtype=tf.bool)
            '''
            dist = tf.reduce_sum(tf.square(self.domain_op), 1)
            dist = tf.reshape(dist, [-1, 1])
            sq_dists = tf.add(tf.subtract(dist, tf.multiply(2., tf.matmul(self.domain_op, tf.transpose(self.domain_op)))),
                              tf.transpose(dist))
            my_kernel = tf.exp(tf.multiply(self.bw_SV, tf.abs(sq_dists)))
            '''
            domain_exp_sv = self.domain_final
            domain_exp_sv = tf.tile(tf.expand_dims(domain_exp_sv, axis=0), (tf.shape(domain)[0], 1, 1))
            x_sub_mu_sv = tf.expand_dims(domain_exp_sv - tf.expand_dims(self.domain_final, axis=1), axis=-1)
            A_SV = tf.linalg.band_part(self.bw_diag_SV, 0, 0) \
                + tf.linalg.band_part(tf.linalg.set_diag(self.bw_corr_SV, np.zeros((np.prod(self.joint_domain.shape[0:-1]), self.dim_domain))), -1, 0)
            A_SV = tf.boolean_mask(A_SV, self.mask_of_sv_in_batch)
            my_kernel = tf.exp(-1.0 * einsum('abli,alm,anm,abnj->ab', x_sub_mu_sv, A_SV, A_SV, x_sub_mu_sv))


            SV = tf.boolean_mask(self.SV_var, self.mask_of_sv_in_batch)
            self.threshold_sv = tf.placeholder_with_default(tf.constant(0.0), shape=None)
            SV_bool = tf.greater_equal(tf.abs(SV), self.threshold_sv)
            SV = SV * tf.cast(SV_bool, dtype=tf.float32)
        #res_sv = tf.matmul(tf.transpose(SV), my_kernel)
        #res_sv_tiled = tf.tile(res_sv, (3, 1)) * [[1], [0], [0]]


        if self.radial_as:
           #A_mask = np.zeros((self.dim_domain, self.dim_domain))
           #np.fill_diagonal(A_mask, 1)
           #A_mask = np.tile(A_mask, (A_init.shape[0]*self.add_kernel_slots, 1, 1))
           A = tf.tile(tf.expand_dims(tf.expand_dims(self.qA_diagonal, axis=-1), axis=-1), (1, self.dim_domain, self.dim_domain))
           A_diagonal = A  # * A_mask
        else:
           A_diagonal = self.qA_diagonal
        A_corr = self.qA_corr


        if self.use_yuv and self.train_gammas and self.only_y_gamma:
            gamma_mask = np.zeros((self.dim_domain, num_channels))
            gamma_mask[:, 0] = 1
            gamma_mask = np.tile(gamma_mask, (gamma_e_init.shape[0], 1, 1))
            self.qgamma_e = self.qgamma_e * gamma_mask

        # combine A_diagonal and A_corr and use only triangular part
        A = tf.linalg.band_part(A_diagonal, 0, 0) \
            + tf.linalg.band_part(tf.linalg.set_diag(A_corr, np.zeros((num_of_all_kernels, self.dim_domain))), -1, 0)
        if self.train_inverse_cov:
            A += tf.transpose(tf.linalg.band_part(tf.linalg.set_diag(A_corr, np.zeros((num_of_all_kernels, self.dim_domain))), -1, 0), perm=(0, 2, 1))
            #A = tf.square(A)

        bool_mask = tf.logical_and(self.kernel_list, pis_mask)

        # track indices of used and necessary kernels
        self.indices = tf.constant(np.arange(num_of_all_kernels), dtype=tf.int32)
        self.indices = tf.boolean_mask(self.indices, bool_mask)


        # using self-Variables to define feed point
        if self.use_diff_center:
            self.musX = tf.boolean_mask(self.qmusX + musX_grid, bool_mask)
        else:
            self.musX = tf.boolean_mask(self.qmusX, bool_mask)
        self.nu_e = tf.boolean_mask(self.qnu_e, bool_mask)
        self.gamma_e = tf.boolean_mask(self.qgamma_e, bool_mask)
        self.A = tf.boolean_mask(A, bool_mask)
        self.pis = tf.boolean_mask(self.qpis, bool_mask)

        self.mask_model_0 = tf.boolean_mask(self.kernel_assignment_to_model[0], bool_mask)
        self.mask_model_1 = tf.boolean_mask(self.kernel_assignment_to_model[1], bool_mask)

        musX = self.musX
        nu_e = self.nu_e
        gamma_e = self.gamma_e
        A = self.A
        pis = self.pis

        musX_0 = tf.boolean_mask(musX, self.mask_model_0)
        A_0 = tf.boolean_mask(A, self.mask_model_0)

        musX_1 = tf.boolean_mask(musX, self.mask_model_1)
        A_1 = tf.boolean_mask(A, self.mask_model_1)

        self.re_assign_nue_op = tf.assign(self.nu_e_var, nu_e)
        self.re_assign_pis_op = tf.assign(self.pis_var, pis)
        self.re_assign_musX_op = tf.assign(self.musX_var, musX)

        normalized_pis = self.pis_var / tf.reduce_sum(pis)
        self.re_normalize_pis_op = tf.assign(self.pis_var, normalized_pis)

        musX_0 = tf.expand_dims(musX_0, axis=1)
        # prepare domain
        domain_exp = self.domain_final
        domain_exp = tf.tile(tf.expand_dims(domain_exp, axis=0), (tf.shape(musX_0)[0], 1, 1))

        x_sub_mu = tf.expand_dims(domain_exp - musX_0, axis=-1)

        musX_1 = tf.expand_dims(musX_1, axis=1)
        # prepare domain
        domain_exp_1 = self.domain_op
        domain_exp_1 = tf.tile(tf.expand_dims(domain_exp_1, axis=0), (tf.shape(musX_1)[0], 1, 1))

        x_sub_mu_1 = tf.expand_dims(domain_exp_1 - musX_1, axis=-1)

        if self.train_inverse_cov:
            #self.maha_dist = einsum('abli,alm,ablj->ab', x_sub_mu, A, x_sub_mu)
            self.maha_dist = einsum('abli,alm,abmj->ab', x_sub_mu, A, x_sub_mu)
            #self.maha_dist = einsum('abli,alm,abnj->ab', x_sub_mu, A, x_sub_mu) # Der hier ist falsch!!
        else:
            maha_dist_0 = einsum('abli,alm,anm,abnj->ab', x_sub_mu, A_0, A_0, x_sub_mu)
            maha_dist_1 = einsum('abli,alm,anm,abnj->ab', x_sub_mu_1, A_1, A_1, x_sub_mu_1)
            if False: # Background
                maha_dist_1 = tf.ones_like(maha_dist_1) * 10**6
                self.maha_dist = tf.concat([maha_dist_0, maha_dist_1], axis=0)
            elif False: # Foreground
                maha_dist_0 = tf.ones_like(maha_dist_0) * 10 ** 6
                self.maha_dist = tf.concat([maha_dist_0, maha_dist_1], axis=0)
            else: # Both
                self.maha_dist = tf.concat([maha_dist_0, maha_dist_1], axis=0)
        self.maha_dist_ind = tf.boolean_mask(self.indices, tf.reduce_any(self.maha_dist < 800, axis=1))
        n_exp = tf.exp(-0.5 * self.maha_dist)

        if self.use_determinant:
            n_div = tf.reduce_prod(tf.matrix_diag_part(A), axis=-1)
            p = self.image.ndim - 1
            n_dis = np.sqrt(np.power(2 * np.pi, p))
            n_quo = n_div / n_dis

            N = tf.tile(tf.expand_dims(n_quo, axis=1), (1, tf.shape(n_exp)[1])) * n_exp
        else:
            N = n_exp

        n_w = N * tf.expand_dims(pis, axis=-1)
        n_w_norm = tf.reduce_sum(n_w, axis=0)
        n_w_norm = tf.maximum(10e-12, n_w_norm)

        self.w_e_op = n_w / n_w_norm

        minimum_influence = 0.5 * 1/(2**self.precision)
        bool_mask_infl = tf.cast(tf.greater(self.w_e_op, minimum_influence), tf.float32)
        self.w_e_op = self.w_e_op * bool_mask_infl

        kernel_list_batch_op = tf.reduce_sum(bool_mask_infl, axis=1) > 0 # 10 ** -9
        if self.only_rec_from_checkpoint:
            self.w_e_max_op = tf.zeros((self.batch_size), dtype=tf.int64)
        else:
            self.w_e_max_op = tf.reshape(tf.argmax(tf.boolean_mask(self.w_e_op, kernel_list_batch_op), axis=0),
                                         self.batch_size)
        #
        self.indices = tf.boolean_mask(self.indices, kernel_list_batch_op)
        self.w_e_out_op = tf.boolean_mask(self.w_e_op, kernel_list_batch_op)
        self.w_e_out_op = tf.reshape(self.w_e_out_op, (tf.shape(self.w_e_out_op)[0],) + self.batch_size)

        nu_e = tf.expand_dims(tf.transpose(nu_e), axis=-1)
        if self.train_gammas:
            # TODO reorder nu_e and gamma_e to avoid unnecessary transpositions
            domain_tiled = tf.expand_dims(tf.transpose(self.domain_final), axis=0)
            domain_tiled = tf.tile(domain_tiled, (num_channels, 1, 1))
            sloped_out = tf.matmul(tf.transpose(gamma_e, perm=[2, 0, 1]), domain_tiled)
            self.res = tf.reduce_sum(self.w_e_op * (sloped_out + nu_e), axis=1)
        else:
            self.res = tf.reduce_sum(self.w_e_op * nu_e, axis=1)

        if self.with_SV:
            SV = tf.transpose(SV)
            #SV = tf.boolean_mask(SV, bool_mask)
            res_sv = tf.matmul(SV, my_kernel)
            #res_sv = tf.reduce_sum(self.w_e_op * res_sv, axis=0, keepdims=True)
            res_sv_tiled = tf.tile(res_sv, (3, 1)) * [[1], [0], [0]]
            self.res = self.res + res_sv_tiled
        self.res = tf.clip_by_value(self.res, 0, 1)
        self.res = tf.transpose(self.res)

        # checkpoint op
        self.pis_best_var = tf.Variable(self.qpis)
        self.musX_best_var = tf.Variable(self.qmusX)
        self.A_diagonal_best_var = tf.Variable(self.qA_diagonal)
        self.A_corr_best_var = tf.Variable(self.qA_corr)
        self.gamma_e_best_var = tf.Variable(self.qgamma_e)
        self.nu_e_best_var = tf.Variable(self.qnu_e)
        if self.dim_domain == 3 and self.train_trafo or affines is not None:
            self.h11_best_var = tf.Variable(self.qh11)
            self.h12_best_var = tf.Variable(self.qh12)
            self.h13_best_var = tf.Variable(self.qh13)
            self.h21_best_var = tf.Variable(self.qh21)
            self.h22_best_var = tf.Variable(self.qh22)
            self.h23_best_var = tf.Variable(self.qh23)
            self.h31_best_var = tf.Variable(self.qh31)
            self.h32_best_var = tf.Variable(self.qh32)
            self.checkpoint_best_op = tf.group(tf.assign(self.pis_best_var, self.qpis),
                                               tf.assign(self.musX_best_var, self.qmusX),
                                               tf.assign(self.A_diagonal_best_var, self.qA_diagonal),
                                               tf.assign(self.A_corr_best_var, self.qA_corr),
                                               tf.assign(self.gamma_e_best_var, self.qgamma_e),
                                               tf.assign(self.nu_e_best_var, self.qnu_e),
                                               tf.assign(self.h11_best_var, self.qh11),
                                               tf.assign(self.h12_best_var, self.qh12),
                                               tf.assign(self.h13_best_var, self.qh13),
                                               tf.assign(self.h21_best_var, self.qh21),
                                               tf.assign(self.h22_best_var, self.qh22),
                                               tf.assign(self.h23_best_var, self.qh23),
                                               tf.assign(self.h31_best_var, self.qh31),
                                               tf.assign(self.h32_best_var, self.qh32))
        else:
            self.checkpoint_best_op = tf.group(tf.assign(self.pis_best_var, self.qpis),
                                               tf.assign(self.musX_best_var, self.qmusX),
                                               tf.assign(self.A_diagonal_best_var, self.qA_diagonal),
                                               tf.assign(self.A_corr_best_var, self.qA_corr),
                                               tf.assign(self.gamma_e_best_var, self.qgamma_e),
                                               tf.assign(self.nu_e_best_var, self.qnu_e))


        self.res = tf.quantization.fake_quant_with_min_max_args(self.res, min=0, max=1, num_bits=self.precision)
        #mse = tf.reduce_mean(tf.square(tf.round(self.res * 255) / 255 - self.target_op))

        if self.dim_domain >= 4:
            diff = tf.boolean_mask(self.res - self.target_op, self.train_mask)
        else:
            diff = self.res - self.target_op
        err_map = tf.reduce_mean(tf.square(diff), axis=1)
        self.sampl_prob = err_map / tf.reduce_sum(err_map)

        if self.overlap > 0:
            '''
            valued_array = np.ones(self.batch_size, dtype=np.bool)
            valued_array[self.overlap:-self.overlap, self.overlap:-self.overlap] = False
            valued_bool = tf.constant(valued_array, dtype=tf.bool )
            valued_bool = tf.reshape(valued_bool, [-1,])
            diff = diff * tf.expand_dims(tf.cast(valued_bool, dtype=tf.float32), axis=-1)

            '''
            diff = tf.reshape(diff, self.batch_size + (num_channels,))
            if self.dim_domain == 2:
                diff = diff[self.overlap:-self.overlap, self.overlap:-self.overlap, :]
            elif self.dim_domain == 3:
                diff = diff[self.overlap:-self.overlap, self.overlap:-self.overlap, self.overlap:-self.overlap, :]
            diff = tf.reshape(diff, [-1, num_channels])


        squared_diff = tf.square(diff)
        mse = tf.reduce_mean(squared_diff)

        if not self.ssim_opt:
            # margin in pixel to determine epsilon
            epsilon = self.margin * 1 / (2 ** self.precision)
            loss_pixel = tf.maximum(0., tf.square(tf.subtract(tf.abs(diff), epsilon))) * self.loss_weights
            if self.use_yuv:
                loss_pixel = 6/8 * tf.reduce_mean(loss_pixel[:, 0]) + 1/8 * tf.reduce_sum(tf.reduce_mean(loss_pixel[:, 1::],
                                                                                                         axis=0))
            else:
                loss_pixel = tf.reduce_mean(loss_pixel)
        else:
            if False: #self.use_yuv: # keep that code snippet for future ssim per frame option
                res_y = tf.reshape(self.res[:, 0], self.batch_shape[:-1] + (1,))
                res_u = tf.reshape(self.res[:, 1], self.batch_shape[:-1] + (1,))
                res_v = tf.reshape(self.res[:, 2], self.batch_shape[:-1] + (1,))
                target_y = tf.reshape(self.target_op[:, 0], self.batch_shape[:-1] + (1,))
                target_u = tf.reshape(self.target_op[:, 1], self.batch_shape[:-1] + (1,))
                target_v = tf.reshape(self.target_op[:, 2], self.batch_shape[:-1] + (1,))

                if self.dim_domain == 2:
                    paddings = tf.constant([[5, 5], [5, 5], [0, 0]])
                    res_y = tf.pad(res_y, paddings, "SYMMETRIC")
                    res_u = tf.pad(res_u, paddings, "SYMMETRIC")
                    res_v = tf.pad(res_v, paddings, "SYMMETRIC")
                    target_y = tf.pad(target_y, paddings, "SYMMETRIC")
                    target_u = tf.pad(target_u, paddings, "SYMMETRIC")
                    target_v = tf.pad(target_v, paddings, "SYMMETRIC")

                    loss_pixel = 1 - (6 / 8 * tf.image.ssim(res_y, target_y, max_val=1)
                                      + 1 / 8 * tf.image.ssim(res_u, target_u, max_val=1)
                                      + 1 / 8 * tf.image.ssim(res_v, target_v, max_val=1))
                elif self.dim_domain == 3:
                    ssim_y = []
                    ssim_u = []
                    ssim_v = []
                    paddings = tf.constant([[5, 5], [5, 5], [0, 0], [0, 0]])
                    res_y = tf.pad(res_y, paddings, "SYMMETRIC")
                    res_u = tf.pad(res_u, paddings, "SYMMETRIC")
                    res_v = tf.pad(res_v, paddings, "SYMMETRIC")
                    target_y = tf.pad(target_y, paddings, "SYMMETRIC")
                    target_u = tf.pad(target_u, paddings, "SYMMETRIC")
                    target_v = tf.pad(target_v, paddings, "SYMMETRIC")

                    for ii in range(self.batch_shape[2]):
                        ssim_y.append(tf.image.ssim(res_y[:, :, ii, :], target_y[:, :, ii, :], max_val=1))
                        ssim_u.append(tf.image.ssim(res_u[:, :, ii, :], target_u[:, :, ii, :], max_val=1))
                        ssim_v.append(tf.image.ssim(res_v[:, :, ii, :], target_v[:, :, ii, :], max_val=1))

                    loss_pixel = 1 - (6/8 * tf.reduce_mean(ssim_y)
                                      + 1/8 * tf.reduce_mean(ssim_u)
                                      + 1/8 * tf.reduce_mean(ssim_v))


            else: # Use Custom SSIM
                res = tf.reshape(self.res, self.batch_size + (num_channels,))
                self.target_op = tf.reshape(self.target_op, self.batch_size + (num_channels,))

                if self.overlap > 0:
                    if self.dim_domain == 2:
                        res = res[self.overlap:-self.overlap, self.overlap:-self.overlap, :]
                        self.target_op = self.target_op[self.overlap:-self.overlap, self.overlap:-self.overlap, :]
                    elif self.dim_domain == 3:
                        res = res[self.overlap:-self.overlap, self.overlap:-self.overlap, self.overlap:-self.overlap, :]
                        self.target_op = self.target_op[self.overlap:-self.overlap, self.overlap:-self.overlap, self.overlap:-self.overlap, :]

                if self.dim_domain == 2:
                    paddings = tf.constant([[5, 5], [5, 5], [0, 0]])
                    res = tf.pad(res, paddings, "SYMMETRIC")
                    self.target_op = tf.pad(self.target_op, paddings, "SYMMETRIC")

                    ssim_per_channel = custom_ssim(res, self.target_op, max_val=1)
                elif self.dim_domain == 3:
                    paddings = tf.constant([[5, 5], [5, 5], [5, 5], [0, 0]])
                    res = tf.pad(res, paddings, "SYMMETRIC")
                    self.target_op = tf.pad(self.target_op, paddings, "SYMMETRIC")

                    ssim_per_channel = custom_ssim(res, self.target_op, max_val=1, ndim=3)

                if self.use_yuv:
                    ssim = tf.reduce_sum(ssim_per_channel * [6, 1, 1], [-1]) / 8
                else:
                    ssim = tf.reduce_mean(ssim_per_channel, [-1])
                loss_pixel = 1 - ssim

        self.num_pi_op = tf.count_nonzero(pis_mask)
        if self.with_SV:
            self.num_sv_op = tf.count_nonzero(tf.abs(self.SV_var) > 5*10**-3)
        else:
            self.num_sv_op = tf.count_nonzero(0.0)

        self.pis_l1 = tf.placeholder(tf.float32)
        self.u_l1 = tf.placeholder(tf.float32)
        self.sv_l1_sub_l2 = tf.placeholder(tf.float32)

        if self.kernel_count_as_norm_l1:
            pis_l1_norm = tf.cast(self.num_pi_op, dtype=tf.float32)
        else:
            pis_l1_norm = self.start_pis

        pis_l1 = self.pis_l1 * tf.reduce_sum(pis) / pis_l1_norm

        if self.with_SV:
            P1 = tf.reduce_sum(tf.abs(SV))
            P2 = tf.sqrt(tf.nn.l2_loss(SV) * 2.0 + 10 ** -9)
            P = P1 - P2
            SV_l1_sub_l2 = self.sv_l1_sub_l2 * 10 ** -1 * P / np.prod(self.batch_size_valued)  #tf.cast(tf.shape(SV)[0], dtype=tf.float32) # mse
            #SV_l1 = 0.08 * 10 ** -2 * P  # ssim
        else:
            SV_l1_sub_l2 = 0.0

        # TODO work in progess
        #rxx_det = 1/tf.reduce_prod(tf.matrix_diag_part(A), axis=-1)**2
        #u_l1 = self.u_l1 * tf.reduce_sum(tf.reduce_prod(tf.matrix_diag_part(A), axis=-1)**2)
        #u_l1 = self.u_l1 * tf.reduce_sum(tf.matrix_diag_part(A) ** 2)
        #u_l1 = self.u_l1 * tf.reduce_sum(rxx_det) #/ self.start_pis # * (tf.cast(self.num_pi_op, tf.float32) / self.start_pis)
        #u_l1 = self.u_l1 * tf.reduce_sum(tf.pow(tf.matrix_diag_part(A)-40, 2))
        u_l1 = self.u_l1 * tf.reduce_sum(tf.linalg.diag_part(A)) # * (tf.cast(self.num_pi_op, tf.float32) / self.start_pis)
        #'''

        #tf.log(   tf.linalg.det(einsum('alm,anm->aln', A_SV, A_SV)))

        #kl_div = 10**-3 * tf.reduce_sum(tf.log(tf.linalg.det(einsum('alm,anm->aln', A_SV, A_SV))) - tf.linalg.trace( tf.linalg.inv( einsum('alm,anm->aln', A_SV, A_SV) +  10**-2*tf.eye(2, batch_shape=[tf.shape(A_SV)[0]])  ) -  tf.eye(2, batch_shape=[tf.shape(A_SV)[0]])))

        self.loss_op = loss_pixel + pis_l1 + u_l1 + SV_l1_sub_l2

        self.mse_op = mse * ((2**self.precision) ** 2)

        #self.res = tf.reshape(self.res, self.batch_shape[:-1] + (num_channels,))
        self.res = tf.reshape(self.res, self.batch_size + (num_channels,))
        if self.with_SV:
            self.res_sv = tf.reshape(res_sv, self.batch_size)
        else:
            self.res_sv = tf.constant(0.0)


        init_new_vars_op = tf.global_variables_initializer()
        self.session.run(init_new_vars_op)

    def checkpoint(self, path):
        if self.save_op is None:
            self.save_op = tf.train.Saver(max_to_keep=None)
        save_path = self.save_op.save(self.session, path)
        print("Model saved in file: %s" % save_path)

    def restore(self, path):
        if self.save_op is None:
            self.save_op = tf.train.Saver(max_to_keep=None)

        self.save_op.restore(self.session, path)
        print("Model restored from file: %s" % path)

    def set_optimizer(self, optimizer1, optimizer2=None, optimizer3=None, optimizer4=None, optimizer5=None, grad_clip_value_abs=None):
        self.optimizer1 = optimizer1

        if optimizer2 is None:
            self.optimizer2 = optimizer1
        else:
            self.optimizer2 = optimizer2

        if optimizer3 is None:
            self.optimizer3 = optimizer1
        else:
            self.optimizer3 = optimizer3

        if optimizer4 is None:
            self.optimizer4 = optimizer1
        else:
            self.optimizer4 = optimizer4

        if optimizer5 is None:
            self.optimizer5 = optimizer1
        else:
            self.optimizer5 = optimizer5

        var_opt1 = [self.nu_e_var, self.gamma_e_var, self.musX_var]
        var_opt2 = [self.pis_var]
        var_opt3 = [self.A_diagonal_var, self.A_corr_var]
        if self.with_SV:
            var_opt4 = [self.SV_var, self.bw_diag_SV, self.bw_corr_SV]
        else:
            var_opt4 = []
        var_opt5 = [self.h11_var, self.h12_var, self.h13_var, self.h21_var, self.h22_var, self.h23_var, self.h31_var, self.h32_var]

        # sort out not trainable vars
        self.var_opt1 = [var for var in var_opt1 if var in tf.trainable_variables()]
        self.var_opt2 = [var for var in var_opt2 if var in tf.trainable_variables()]
        self.var_opt3 = [var for var in var_opt3 if var in tf.trainable_variables()]
        if self.with_SV:
            self.var_opt4 = [var for var in var_opt4 if var in tf.trainable_variables()]
        self.var_opt5 = [var for var in var_opt5 if var in tf.trainable_variables()]

        variables = []
        if not optimizer1._lr == 0:
            variables += self.var_opt1
            len1 = len(self.var_opt1)
        else:
            len1 = 0
        if not optimizer2._lr == 0:
            variables += self.var_opt2
            len2 = len(self.var_opt2)
        else:
            len2 = 0
        if not optimizer3._lr == 0:
            variables += self.var_opt3
            len3 = len(self.var_opt3)
        else:
            len3 = 0
        if not optimizer4._lr == 0:
            variables += self.var_opt4
            len4 = len(self.var_opt4)
        else:
            len4 = 0
        if not optimizer5._lr == 0:
            variables += self.var_opt5
            len5 = len(self.var_opt5)
        else:
            len5 = 0
        accum_gradients = [tf.Variable(tf.zeros_like(var.initialized_value()), trainable=False)
                           for var in variables]
        self.zero_op = [grad.assign(tf.zeros_like(grad)) for grad in accum_gradients]
        gradients = tf.gradients(self.loss_op, variables)
        if not self.only_rec_from_checkpoint:
            self.accum_ops = [accum_gradients[i].assign_add(gv) for i, gv in enumerate(gradients)]

        if grad_clip_value_abs is not None:
            accum_gradients = [tf.clip_by_value(g, -grad_clip_value_abs, grad_clip_value_abs) for g in accum_gradients]

        if self.affines is not None and self.train_trafo or self.dim_domain == 3 and self.train_trafo:
            manip = np.ones((self.image.shape[2],))
            manip[0] = 0 # first frame doesn't need to be transformed
            self.stop_first_gradient_op = [accum_gradients[ii].assign(accum_gradients[ii] * manip) for ii in range(len(accum_gradients) - len5, len(accum_gradients))]

        '''
        gradients1 = accum_gradients[:len(self.var_opt1)]
        gradients2 = accum_gradients[len(self.var_opt1):len(self.var_opt1) + len(self.var_opt2)]
        gradients3 = accum_gradients[len(self.var_opt1) + len(self.var_opt2):len(self.var_opt1) + len(self.var_opt2) + len(self.var_opt3)]
        gradients4 = accum_gradients[len(self.var_opt1) + len(self.var_opt2) + len(self.var_opt3):]
        '''

        gradients1 = [accum_gradients.pop(0) for g in range(len1)]
        gradients2 = [accum_gradients.pop(0) for g in range(len2)]
        gradients3 = [accum_gradients.pop(0) for g in range(len3)]
        gradients4 = [accum_gradients.pop(0) for g in range(len4)]
        gradients5 = [accum_gradients.pop(0) for g in range(len5)]

        if len1 > 0:
            train_op1 = self.optimizer1.apply_gradients(zip(gradients1, self.var_opt1))
        else:
            train_op1 = tf.no_op()
        if len2 > 0:
            train_op2 = self.optimizer2.apply_gradients(zip(gradients2, self.var_opt2))
        else:
            train_op2 = tf.no_op()
        if len3 > 0:
            train_op3 = self.optimizer3.apply_gradients(zip(gradients3, self.var_opt3))
        else:
            train_op3 = tf.no_op()
        if len4 > 0:
            train_op4 = self.optimizer4.apply_gradients(zip(gradients4, self.var_opt4))
        else:
            train_op4 = tf.no_op()
        if len5 > 0:
            train_op5 = self.optimizer5.apply_gradients(zip(gradients5, self.var_opt5))
        else:
            train_op5 = tf.no_op()
        self.train_op = tf.group(train_op1, train_op2, train_op3, train_op4, train_op5)

        uninitialized_vars = []
        for var in tf.global_variables():
            try:
                self.session.run(var)
            except tf.errors.FailedPreconditionError:
                if var is not None:
                    uninitialized_vars.append(var)

        init_new_vars_op = tf.variables_initializer(uninitialized_vars)
        self.session.run(init_new_vars_op)

    def set_inc_optimizer(self, optimizer1, optimizer2=None, optimizer3=None, grad_clip_value_abs=None):
        assert optimizer2 is not None and optimizer3 is not None, "this may break if one optimizer is reused"
        self.optimizer_inc1 = optimizer1

        if optimizer2 is None:
            self.optimizer_inc2 = optimizer1
        else:
            self.optimizer_inc2 = optimizer2

        if optimizer3 is None:
            self.optimizer_inc3 = optimizer1
        else:
            self.optimizer_inc3 = optimizer3

        var_inc_opt1 = [self.nu_e_inc_var, self.gamma_e_inc_var, self.musX_inc_var]
        var_inc_opt2 = [self.pis_inc_var]
        var_inc_opt3 = [self.A_diagonal_inc_var, self.A_corr_inc_var]

        # sort out not trainable vars
        self.var_inc_opt1 = [var for var in var_inc_opt1 if var in tf.trainable_variables()]
        self.var_inc_opt2 = [var for var in var_inc_opt2 if var in tf.trainable_variables()]
        self.var_inc_opt3 = [var for var in var_inc_opt3 if var in tf.trainable_variables()]

        accum_gradients = [tf.Variable(tf.zeros_like(var.initialized_value()), trainable=False)
                           for var in self.var_inc_opt1 + self.var_inc_opt2 + self.var_inc_opt3]
        self.accum_inc_gradients = accum_gradients
        self.zero_inc_op = [grad.assign(tf.zeros_like(grad)) for grad in accum_gradients]
        gradients = tf.gradients(self.loss_op, self.var_inc_opt1 + self.var_inc_opt2 + self.var_inc_opt3)
        self.accum_inc_ops = [accum_gradients[i].assign_add(gv) for i, gv in enumerate(gradients)]
        if grad_clip_value_abs is not None:
            accum_gradients = [tf.clip_by_value(g, -grad_clip_value_abs, grad_clip_value_abs) for g in accum_gradients]

        gradients1 = accum_gradients[:len(self.var_inc_opt1)]
        gradients2 = accum_gradients[len(self.var_inc_opt1):len(self.var_inc_opt1) + len(self.var_inc_opt2)]
        gradients3 = accum_gradients[len(self.var_inc_opt1) + len(self.var_inc_opt2):]

        if len(self.var_inc_opt1) > 0:
            train_op1 = self.optimizer_inc1.apply_gradients(zip(gradients1, self.var_inc_opt1))
        else:
            train_op1 = tf.no_op()
        if len(self.var_inc_opt2) > 0:
            train_op2 = self.optimizer_inc2.apply_gradients(zip(gradients2, self.var_inc_opt2))
        else:
            train_op2 = tf.no_op()
        if len(self.var_inc_opt3) > 0:
            train_op3 = self.optimizer_inc3.apply_gradients(zip(gradients3, self.var_inc_opt3))
        else:
            train_op3 = tf.no_op()

        self.train_inc_op = tf.group(train_op1, train_op2, train_op3)

        uninitialized_vars = []
        for var in tf.global_variables():
            try:
                self.session.run(var)
            except tf.errors.FailedPreconditionError:
                if var is not None:
                    uninitialized_vars.append(var)

        init_new_vars_op = tf.variables_initializer(uninitialized_vars)
        self.session.run(init_new_vars_op)

        # self.reset_optimizers_op = tf.group([tf.variables_initializer(self.optimizer_inc1.variables()),
        #                                      tf.variables_initializer(self.optimizer_inc2.variables()),
        #                                      tf.variables_initializer(self.optimizer_inc3.variables())])
        #TODO use the above (i.e. upgrade tf)
        reset_vars = []
        for name in self.optimizer_inc1.get_slot_names():
            for var_inc in self.var_inc_opt1:
                reset_vars.append(self.optimizer_inc1.get_slot(var_inc, name))
        for name in self.optimizer_inc2.get_slot_names():
            for var_inc in self.var_inc_opt2:
                reset_vars.append(self.optimizer_inc2.get_slot(var_inc, name))
        for name in self.optimizer_inc3.get_slot_names():
            for var_inc in self.var_inc_opt3:
                reset_vars.append(self.optimizer_inc3.get_slot(var_inc, name))

        self.reset_optimizers_op = tf.variables_initializer(reset_vars)

        # apply optimizer vars
        assert self.optimizer1 is not None, "call set_optimizer before set_inc_optimizer"

        opt_inc_assign_ops = []
        for name in self.optimizer1.get_slot_names():
            for var, var_inc in zip(self.var_opt1, self.var_inc_opt1):
                slot = self.optimizer1.get_slot(var, name)
                slot_inc = self.optimizer_inc1.get_slot(var_inc, name)
                assign_op = slot[self.insert_pos:self.insert_pos + self.num_inc_kernels].assign(slot_inc)
                opt_inc_assign_ops.append(assign_op)

        for name in self.optimizer2.get_slot_names():
            for var, var_inc in zip(self.var_opt2, self.var_inc_opt2):
                slot = self.optimizer2.get_slot(var, name)
                slot_inc = self.optimizer_inc2.get_slot(var_inc, name)
                assign_op = slot[self.insert_pos:self.insert_pos + self.num_inc_kernels].assign(slot_inc)
                opt_inc_assign_ops.append(assign_op)

        for name in self.optimizer3.get_slot_names():
            for var, var_inc in zip(self.var_opt3, self.var_inc_opt3):
                slot = self.optimizer3.get_slot(var, name)
                slot_inc = self.optimizer_inc3.get_slot(var_inc, name)
                assign_op = slot[self.insert_pos:self.insert_pos + self.num_inc_kernels].assign(slot_inc)
                opt_inc_assign_ops.append(assign_op)

        self.assign_inc_opt_vars_op = tf.group(*opt_inc_assign_ops)

    def calc_peaks_inc(self, threshold_rel, max_iters=100, plot_dir=None):
        rec = self.get_reconstruction()
        sigma = 1
        # diff = np.abs(self.image - rec)
        if self.use_yuv:
            weights = [6/8, 1/8, 1/8]
        else:
            weights = [1, 1, 1]

        if True: #self.ssim_opt:
            diff = np.average(1 - compare_ssim(self.image, rec, data_range=1, multichannel=True, full=True)[1], axis=-1, weights=weights)
        else:
            diff = np.average(np.power(255 * (self.image - rec), 2), axis=-1, weights=weights)

        '''
        reverse = None
        for i in range(1, max_iters):
            if reverse:
                i = 1 / (self.num_inc_kernels - i + 1)

            blurred = gaussian_filter(diff, i)
            blurred_mean = np.mean(blurred)
            peaks = peak_local_max(blurred, threshold_abs=blurred_mean, threshold_rel=threshold_rel)

            if reverse is None:
                if peaks.shape[0] < self.num_inc_kernels:
                    reverse = True
                    continue
                else:
                    reverse = False

            # print("[{0}] blurred_mean: {1:.2f} num_peaks: {2} sigma: {3:.3f}".format(self.iter, blurred_mean, peaks.shape[0], i))
            if peaks.shape[0] < self.num_inc_kernels:
                break
            sigma = i
        

        blurred = gaussian_filter(diff, sigma)
        blurred_mean = np.mean(blurred)
        blurred_median = np.median(blurred)
        peaks = peak_local_max(blurred, num_peaks=self.num_inc_kernels)
        


        a = 1 / (sigma / self.image.shape[0]) * 2/1
        '''

        #diff[128, 128] = 20

        blurred = diff
        blurred_mean = np.mean(blurred)
        blurred_median = np.median(blurred)
        blurred_max = np.max(blurred)
        min_distance_peaks = 8

        '''
        blurred[0,0] = 20
        blurred_max = np.max(blurred)
        '''


        #peaks = peak_local_max(blurred, num_peaks=self.num_inc_kernels, min_distance=min_distance_peaks, threshold_abs=0.5 * blurred_max)
        _, used = zip(*self.get_num_pis())
        used = used[-1]
        num_new_kernels = self.start_pis - used
        peaks = peak_local_max(blurred, num_peaks=num_new_kernels, min_distance=min_distance_peaks)
        # peaks = peak_local_max(blurred, num_peaks=self.num_inc_kernels)
        a = 16 * self.image.shape[0] / min_distance_peaks

        if plot_dir:
            if not os.path.exists(plot_dir):
                os.mkdir(plot_dir)

            fig = plt.figure()
            if self.dim_domain == 2:
                plt.imshow(blurred, cmap='gray')
            elif self.dim_domain == 3:
                plt.imshow(blurred[:, :, 0], cmap='gray')
            elif self.dim_domain == 4:
                plt.imshow(blurred[int(blurred.shape[0] / 2), int(blurred.shape[1] / 2), :, :], cmap='gray')

            plt.colorbar()
            plt.title(
                "num peaks: {0:.2f}, sigma: {1:.2f} (a^2: {2:.2f}) bmen: {3:.2f} bmed: {4:.2f}".format(peaks.shape[0],
                                                                                                       sigma, a,
                                                                                                       blurred_mean,
                                                                                                       blurred_median))
            plt.scatter(peaks[:, 1], peaks[:, 0])
            plt.savefig("{}/inc_{}.png".format(plot_dir, self.iter))
            plt.close(fig)

        return peaks, a

    def reinit_inc(self, plot_dir=None, threshold_rel=0.2):
        # rec = self.get_reconstruction()
        # blurred = cv2.GaussianBlur(np.abs(self.image - rec), (0, 0), 5, 5)
        # peaks = peak_local_max(blurred, num_peaks=self.num_inc_kernels)
        peaks, a = self.calc_peaks_inc(threshold_rel=threshold_rel, plot_dir=plot_dir)
        musX_inc = np.zeros_like(self.musX_init)
        #center_peaks = peaks / self.image.shape[0:-1]
        center_peaks = self.joint_domain[peaks[:, 0], peaks[:, 1], :self.dim_domain]
        musX_inc[:len(center_peaks)] = center_peaks
        curr_pis = self.get_params()['pis']
        pi_mean = np.mean(curr_pis[curr_pis > 0])
        pi_median = np.median(curr_pis[curr_pis > 0])
        #pis_inc = np.ones_like(self.pis_init) * pi_mean
        pis_inc = np.zeros_like(self.pis_init)
        pis_inc[:len(center_peaks)] = pi_median
        # pis_inc = self.pis_init
        gamma_e_inc = np.zeros_like(self.gamma_e_init)
        nu_e_inc = np.zeros_like(self.nu_e_init)
        nu_e_inc[:len(center_peaks)] = self.joint_domain[peaks[:, 0], peaks[:, 1], self.dim_domain:]

        A_diagonal_inc = np.zeros_like(self.A_init)
        A_corr_inc = np.zeros_like(self.A_init)
        # a = 2 * (self.kernel_count + 1)
        A_diagonal_inc[:len(center_peaks), 0, 0] = a
        A_diagonal_inc[:len(center_peaks), 1, 1] = a
        if self.radial_as:
            A_diagonal_inc = A_diagonal_inc[:len(center_peaks), 0, 0]

        # fig = plt.figure()
        # plt.imshow(blurred, vmin=-1, vmax=1, cmap='gray')
        # plt.scatter(peaks[:, 1], peaks[:, 0])
        # plt.savefig("inc_{}.png".format(self.iter))
        # plt.close(fig)
        '''
        SVs, BW_diag, BW_corr = self.session.run([self.SV_var, self.bw_diag_SV, self.bw_corr_SV])
        if self.overlap > 0 :
            shape = tuple(np.array(self.image.shape[0:-1]) + 2*self.overlap)
            SVs = np.reshape(SVs, shape)[self.overlap:-self.overlap, self.overlap:-self.overlap]
            SVs = np.reshape(SVs, [-1, ])
        idx = np.squeeze(np.argsort(np.abs(SVs), axis=0))
        domain = np.reshape(self.joint_domain[:, :, :self.dim_domain], [-1, self.dim_domain])[idx]
        SVs = SVs[idx]
        np.sort(SVs)
        BW_diag = BW_diag[idx]
        BW_corr = BW_corr[idx]

        nu_es = np.reshape(self.reconstruction_image, [-1, 3])[idx, 0]


        musX_inc = np.zeros_like(self.musX_init)
        musX_inc[:self.musX_init.shape[0], :] = domain[:self.musX_init.shape[0], :]
        A_corr_inc = np.zeros_like(self.A_init)
        A_corr_inc[:self.musX_init.shape[0]] = BW_corr[:self.musX_init.shape[0]]
        A_diagonal_inc = np.zeros_like(self.A_init)
        A_diagonal_inc[:self.musX_init.shape[0]] = BW_diag[:self.musX_init.shape[0]]
        nu_e_inc = np.ones_like(self.nu_e_init) * 0.5
        nu_e_inc[:self.musX_init.shape[0], 0] = nu_es[:self.musX_init.shape[0]]
        '''


        self.session.run(self.reinit_inc_vars_op, feed_dict={self.nu_e_inc_new: nu_e_inc, self.musX_inc_new: musX_inc,
                                                             self.pis_inc_new: pis_inc,
                                                             self.gamma_e_inc_new: gamma_e_inc,
                                                             self.A_diagonal_inc_new: A_diagonal_inc,
                                                             self.A_corr_inc_new: A_corr_inc})

        #self.update_kernel_list(self.add_kernel_slots)

        proto = np.zeros((self.add_kernel_slots + 2 * self.start_pis,), dtype=bool)
        proto[self.add_kernel_slots + self.start_pis::] = True
        for k in range(self.start_batches):
            self.kernel_list_per_batch[k] = np.logical_or(self.kernel_list_per_batch[k], proto)
        self.kernel_list_per_batch = [np.ones((self.add_kernel_slots + 2*self.start_pis,), dtype=bool)] * self.start_batches

    def apply_inc(self):
        #self.session.run([self.assign_inc_opt_vars_op], feed_dict={self.insert_pos: self.kernel_count})
        self.session.run([self.assign_inc_vars_op], feed_dict={self.insert_pos: self.kernel_count})
        self.session.run(self.reset_optimizers_op)
        self.kernel_count += self.num_inc_kernels

    def train(self, num_iter, val_iter=100, ukl_iter=None, optimizer1=None, optimizer2=None, optimizer3=None, grad_clip_value_abs=None, pis_l1=0,
              u_l1=0, sv_l1_sub_l2=0, sampling_percentage=100, callbacks=(), with_inc=False, train_inc=False, train_orig=True, use_loss_mask=False):
        if ukl_iter == None:
            ukl_iter = val_iter

        if optimizer1:
            self.set_optimizer(optimizer1, optimizer2, optimizer3, grad_clip_value_abs=grad_clip_value_abs)
        assert self.optimizer1 is not None, "no optimizer found, you have to specify one!"

        # TODO history nutzen oder weg
        # self.losses = []
        # self.mses = []
        # self.num_pis = []
        if self.quantization_mode >= 1:
            self.qparams = quantize_params(self, self.get_params())
        if self.quantization_mode == 1:
            self.rparams = rescaler(self, self.qparams)
            self.best_qloss, self.best_qmse, _ = self.run_batched(pis_l1=pis_l1, u_l1=u_l1, sv_l1_sub_l2=sv_l1_sub_l2, train=False, update_reconstruction=True,
                                       with_quantized_params=True, with_inc=with_inc, train_inc=False)
            self.qlosses.append((0, self.best_qloss))
            self.qmses.append((0, self.best_qmse))

        self.best_loss, self.best_mse, num_pi, num_sv = self.run_batched(pis_l1=pis_l1, u_l1=u_l1, sv_l1_sub_l2=sv_l1_sub_l2, train=False,
                                                                         update_reconstruction=True, with_inc=with_inc,
                                                                         train_inc=False, use_loss_mask=use_loss_mask)


        self.losses.append((self.iter, self.best_loss))
        self.mses.append((self.iter, self.best_mse))
        self.num_pis.append((self.iter, num_pi))
        self.num_svs.append((self.iter, num_sv))

        # run callbacks
        for callback in callbacks:
            callback(self)

        for i in range(1, num_iter + 1):
            self.iter += 1
            try:
                validate = i % val_iter == 0
                update_kernel_list = i % ukl_iter == 0

                loss_val, mse_val, num_pi, num_sv = self.run_batched(pis_l1=pis_l1, u_l1=u_l1, sv_l1_sub_l2=sv_l1_sub_l2, train=train_orig,
                                                             update_reconstruction=False, sampling_percentage=sampling_percentage,
                                                             with_inc=with_inc, train_inc=train_inc, use_loss_mask=use_loss_mask)

                if update_kernel_list:
                    self.update_kernel_list(self.add_kernel_slots)
                    if not validate:
                        loss_val, mse_val, num_pi, num_sv = self.run_batched(pis_l1=pis_l1, u_l1=u_l1, train=False,
                                                                             update_reconstruction=False, with_inc=with_inc,
                                                                             train_inc=False, thr_sv=5 * 10 ** -3)

                if validate:
                    if self.quantization_mode >= 1:
                        self.qparams = quantize_params(self, self.get_params())
                    if self.quantization_mode == 1:
                        self.rparams = rescaler(self, self.qparams)
                        qloss_val, qmse_val, _, _ = self.run_batched(pis_l1=pis_l1,
                                                                  u_l1=u_l1, sv_l1_sub_l2=sv_l1_sub_l2, train=False, update_reconstruction=True,
                                                                  with_quantized_params=True, with_inc=with_inc, train_inc=False, use_loss_mask=use_loss_mask)


                    '''
                    ### HACK um Threshold Zeugs schnell zu kontrollieren! ###
                    _, mse_0, _, _ = self.run_batched(pis_l1=pis_l1, u_l1=u_l1, sv_l1_sub_l2=sv_l1_sub_l2, train=False,
                                                                         update_reconstruction=True, with_inc=with_inc,
                                                                         train_inc=False, thr_sv=0)
                    _, mse_inf, _, _ = self.run_batched(pis_l1=pis_l1, u_l1=u_l1, sv_l1_sub_l2=sv_l1_sub_l2, train=False,
                                                      update_reconstruction=True, with_inc=with_inc,
                                                      train_inc=False, thr_sv=1.0*10**6)
                    '''
                    loss_val, mse_val, num_pi, num_sv = self.run_batched(pis_l1=pis_l1, u_l1=u_l1, train=False,
                                                                 update_reconstruction=True, with_inc=with_inc, train_inc=False, thr_sv=5*10**-3, use_loss_mask=use_loss_mask)

                    # run batched with quant params
                    #print('PSNR_0 = {0:.2f},  PSNR_5 = {1:.2f},  PSNR_INF = {2:.2f}'.format(10*np.log10(255**2/mse_0), 10*np.log10(255**2/mse_val), 10*np.log10(255**2/mse_inf)))
                    ######

                # TODO take loss_history into account
                if np.isnan(loss_val) or (len(self.losses) > 0 and loss_val + 1 > (
                            self.losses[0][1] + 100) * 10):  # TODO +100 is a hotfix to handle negative losses properly
                    # self.session.run(
                    #     [self.loss_op, self.train_op, self.global_norm_op, self.global_norm1_op, self.global_norm2_op])
                    print("stop")
                    break

                if validate:
                    # TODO take loss_history into account
                    if not self.best_loss or loss_val < self.best_loss:
                        self.best_loss = loss_val
                        self.session.run(self.checkpoint_best_op)
                        # break
                    self.losses.append((self.iter, loss_val))

                    # TODO take mses_history into account
                    if not self.best_mse or mse_val < self.best_mse:
                        self.best_mse = mse_val
                    self.mses.append((self.iter, mse_val))

                    if self.quantization_mode == 1:
                        self.qmses.append((i, qmse_val))
                        self.qlosses.append((i, qloss_val))

                    self.num_pis.append((self.iter, num_pi))
                    self.num_svs.append((self.iter, num_sv))

                    # run callbacks
                    for callback in callbacks:
                        callback(self)

                        # print("loss at iter %d: %f" % (i, session.run(loss)))
            except KeyboardInterrupt:
                break

        self.losses_history.append(self.losses)
        self.mses_history.append(self.mses)
        print("end loss/mse: ", loss_val, "/", mse_val, "@iter: ", i)
        print("best loss/mse: ", self.best_loss, "/", self.best_mse)


    def run_batched(self, pis_l1=0, u_l1=0, sv_l1_sub_l2=0, train=True, update_reconstruction=False, with_quantized_params=False, sampling_percentage=100, with_inc=False, train_inc=False, thr_sv=None, use_loss_mask=False):

        self.valid = False
        if with_quantized_params:
            self.qvalid = False

        if train:
            self.session.run(self.zero_op)

        if train_inc:
            assert with_inc, "need inc to train inc"
            self.session.run(self.zero_inc_op)


        loss_val = 0
        mse_val = 0
        num_pi = -1

        # only for update update_reconstruction=True
        reconstruction_ = np.zeros_like(self.image)
        reconstruction_sv = np.zeros(self.image.shape[0:-1])
        w_e_ = np.zeros(self.image.shape[0:-1])
        if self.add_kernel_slots > 0:
            num_all_kernels = 2 * self.start_pis + self.add_kernel_slots
        else:
            num_all_kernels = self.start_pis
        w_matrix = np.zeros((num_all_kernels,) + self.image.shape[:-1])

        widgets = [
            progressbar.AnimatedMarker(markers='◐◓◑◒'),
            ' Iteration: {0}  '.format(self.iter),
            ' ', progressbar.Counter('%(value)d/{0}'.format(self.start_batches)),
            ' ', progressbar.Timer(),
        ]
        bar = progressbar.ProgressBar(widgets=widgets)

        ii = -1
        for (coord, batch) in bar(sliding_window(self.joint_domain, self.overlap, self.batch_size_valued)):
        #for (coord, batch) in sliding_window(self.joint_domain, self.overlap, self.batch_size_valued):
            ii = ii + 1
            retrieve = [self.loss_op, self.mse_op, self.num_pi_op, self.num_sv_op]
            if not with_quantized_params:
                retrieve.append(self.indices)

            img_patch = batch.reshape(-1, batch.shape[-1])
            if self.with_SV:
                if self.dim_domain > 2:
                    assert "only works for imgs so far"
                bool_mask_sv = np.zeros(self.image.shape[0:-1], dtype=np.bool)
                bool_mask_sv = np.pad(bool_mask_sv, ((self.overlap, self.overlap), (self.overlap, self.overlap)), 'constant', constant_values=False)
                bool_mask_sv[coord[0] + self.overlap:batch.shape[0] + coord[0] + self.overlap, coord[1] + self.overlap:batch.shape[1] + coord[1] + self.overlap] = True
                bool_mask_sv = bool_mask_sv.reshape(-1,)



            if train and self.dim_domain >= 4:
                img_patch = img_patch[self.train_mask]

            if train and not self.ssim_opt and sampling_percentage < 100:
                num_samples = np.uint32(np.round(img_patch.shape[0] * sampling_percentage/100))
                samples = np.random.choice(img_patch.shape[0], (num_samples,), replace=False, p=self.random_sampling_per_batch[ii])
                img_patch = img_patch[samples, :]



            feed_dict = {self.joint_domain_batched_op: img_patch, self.pis_l1: pis_l1, self.u_l1: u_l1,
                         self.kernel_list: self.kernel_list_per_batch[ii], self.stack_inc: [1.] if with_inc else [0.], self.stack_orig: [1.]}

            if use_loss_mask:
                batch_loss_mask = self.loss_mask[coord[0]:self.batch_size_valued[0] + coord[0], coord[1]:self.batch_size_valued[1] + coord[1],
                coord[2]:self.batch_size_valued[2] + coord[2]].reshape(-1, 1)
                feed_dict.update({self.loss_weights: batch_loss_mask})

            if self.with_SV:
                feed_dict.update({self.mask_of_sv_in_batch: bool_mask_sv, self.sv_l1_sub_l2: sv_l1_sub_l2})
                if thr_sv is not None:
                    feed_dict.update({self.threshold_sv: thr_sv})

            if train and not self.only_rec_from_checkpoint:
                retrieve.append(self.accum_ops)
            if update_reconstruction:
                retrieve += [self.res, self.w_e_max_op, self.res_sv]
                if with_quantized_params:
                    feed_dict.update({self.A: self.rparams["A"], self.musX: self.rparams["musX"], self.nu_e: self.rparams["nu_e"], self.gamma_e: self.rparams["gamma_e"], self.pis: self.rparams["pis"]})
                #das muss wieder weg!!!
                #feed_dict.update({self.A: self.A_init, self.musX: self.musX_init, self.nu_e: self.nu_e_init, self.gamma_e: self.gamma_e_init, self.pis: self.pis_init})
            if train_inc:
                retrieve.append(self.accum_inc_ops)

            if update_reconstruction:
                retrieve.append(self.w_e_out_op)
                retrieve.append(self.sampl_prob)

            if self.affines is not None or self.dim_domain == 3 and self.train_trafo:
                feed_dict.update({self.frames_list: self.frames_list_per_batch[ii]})

            results = self.session.run(retrieve, feed_dict=feed_dict)
            if update_reconstruction:
                if train:
                    rec_batch = results[6]
                    w_e_batch = results[4][results[7]]
                else:
                    if with_quantized_params:
                        rec_batch = results[4]
                        w_e_batch = results[4][results[5]]
                    else:
                        rec_batch = results[5]
                        if results[4].shape[0] == 0:
                            w_e_batch = np.zeros(self.batch_size)
                        else:
                            w_e_batch = results[4][results[6]]
                        rec_sv_batch = results[7]

                start_y = coord[0] + self.overlap
                start_x = coord[1] + self.overlap
                end_y = start_y + self.batch_size_valued[0]
                end_x = start_x + self.batch_size_valued[1]
                if self.dim_domain == 2:
                    reconstruction_[start_y:end_y, start_x:end_x, :] \
                        = rec_batch[self.overlap:self.overlap + self.batch_size_valued[0],
                          self.overlap:self.overlap + self.batch_size_valued[1], :]
                    if self.with_SV:
                        reconstruction_sv[start_y:end_y, start_x:end_x] \
                            = rec_sv_batch[self.overlap:self.overlap + self.batch_size_valued[0],
                              self.overlap:self.overlap + self.batch_size_valued[1]]
                    w_e_[start_y:end_y, start_x:end_x] \
                        = w_e_batch[self.overlap:self.overlap + self.batch_size_valued[0],
                          self.overlap:self.overlap + self.batch_size_valued[1]]
                elif self.dim_domain == 3:
                    start_z = coord[2] + self.overlap
                    end_z = start_z + self.batch_size_valued[2]
                    reconstruction_[start_y:end_y, start_x:end_x, start_z:end_z, :] \
                        = rec_batch[self.overlap:self.overlap + self.batch_size_valued[0],
                          self.overlap:self.overlap + self.batch_size_valued[1],
                          self.overlap:self.overlap + self.batch_size_valued[2], :]
                    w_e_[start_y:end_y, start_x:end_x, start_z:end_z] \
                        = w_e_batch[self.overlap:self.overlap + self.batch_size_valued[0],
                          self.overlap:self.overlap + self.batch_size_valued[1],
                          self.overlap:self.overlap + self.batch_size_valued[2]]


            if update_reconstruction:
                if self.dim_domain == 2:
                    w_matrix[results[4], start_y:end_y, start_x:end_x] = results[-2][:,
                                self.overlap:self.overlap + self.batch_size_valued[0],
                                self.overlap:self.overlap + self.batch_size_valued[1]]
                elif self.dim_domain == 3:
                    w_matrix[results[4], start_y:end_y, start_x:end_x, start_z:end_z] = results[-2][:,
                                                self.overlap:self.overlap + self.batch_size_valued[0],
                                                self.overlap:self.overlap + self.batch_size_valued[1],
                                                self.overlap:self.overlap + self.batch_size_valued[2]]

            loss_val += results[0] * np.prod(self.batch_size_valued) / self.num_pixel
            mse_val += results[1] * np.prod(self.batch_size_valued) / self.num_pixel

            num_pi = results[2]
            num_sv = results[3]
            if not with_quantized_params:
                bool_mask = np.zeros_like(self.kernel_list_per_batch[ii])
                bool_mask[results[4]] = True
                self.kernel_list_per_batch[ii] = bool_mask

            if update_reconstruction:
                self.random_sampling_per_batch[ii] = results[-1]

        if update_reconstruction:
            if not with_quantized_params:
                self.reconstruction_image = reconstruction_
                self.reconstruction_sv = reconstruction_sv
                self.weight_matrix_argmax = w_e_

                self.weight_matrix = w_matrix
                self.valid = True
            else:
                self.qreconstruction_image = reconstruction_
                self.qweight_matrix_argmax = w_e_
                self.qweight_matrix = w_matrix
                self.qvalid = True

        if train:
            if self.stop_first_gradient_op is not None:
                self.session.run(self.stop_first_gradient_op)
            self.session.run(self.train_op)
        if train_inc:
            self.session.run(self.train_inc_op)


        return loss_val, mse_val, num_pi, num_sv

    def get_params(self):
        pis, musX, A_diagonal, A_corr, gamma_e, nu_e = self.session.run([self.qpis, self.qmusX,
                                                                         self.qA_diagonal, self.qA_corr,
                                                                         self.qgamma_e, self.qnu_e])

        out_dict = {'pis': pis, 'musX': musX, 'A_diagonal': A_diagonal, 'A_corr': A_corr, 'gamma_e': gamma_e, 'nu_e': nu_e}

        if self.dim_domain == 3 and (self.affines is not None or self.train_trafo):
            h11, h12, h13, h21, h22, h23, h31, h32 = self.session.run([self.qh11, self.qh12,
                                                                       self.qh13, self.qh21,
                                                                       self.qh22, self.qh23,
                                                                       self.qh31, self.qh32])
            out_dict.update({'h11': h11, 'h12': h12, 'h13': h13, 'h21': h21, 'h22': h22, 'h23': h23, 'h31': h31, 'h32': h32})


        return out_dict

    def get_gradients(self):
        raise NotImplementedError

    def get_reconstruction(self):
        if not self.valid:
            self.run_batched(train=False, update_reconstruction=True)
        return self.reconstruction_image

    def get_qreconstruction(self):
        if not self.qvalid:
            self.run_batched(train=False, update_reconstruction=True, with_quantized_params=True)
        return self.qreconstruction_image

    def get_weight_matrix_argmax(self):
        if not self.valid:
            self.run_batched(train=False, update_reconstruction=True)
        return self.weight_matrix_argmax

    def get_weight_matrix(self):
        if not self.valid:
            self.run_batched(train=False, update_reconstruction=True)
        return self.weight_matrix

    def get_best_params(self):
        pis, musX, A_diagonal, A_corr, gamma_e, nu_e = self.session.run([self.pis_best_var, self.musX_best_var,
                                                                         self.A_diagonal_best_var, self.A_corr_best_var,
                                                                         self.gamma_e_best_var, self.nu_e_best_var])
        out_dict = {'pis': pis, 'musX': musX, 'A_diagonal': A_diagonal, 'A_corr': A_corr, 'gamma_e': gamma_e,
                    'nu_e': nu_e}

        if self.dim_domain == 3 and (self.affines is not None or self.train_trafo):
            h11, h12, h13, h21, h22, h23, h31, h32 = self.session.run([self.h11_best_var, self.h12_best_var,
                                                                       self.h13_best_var, self.h21_best_var,
                                                                       self.h22_best_var, self.h23_best_var,
                                                                       self.h31_best_var, self.h32_best_var])
            out_dict.update({'h11': h11, 'h12': h12, 'h13': h13, 'h21': h21, 'h22': h22, 'h23': h23, 'h31': h31, 'h32': h32})

        return out_dict

    def get_best_reconstruction(self):
        raise NotImplemented

    def get_best_weight_matrix(self):
        raise NotImplemented

    def get_losses(self):
        return self.losses

    def get_qlosses(self):
        return self.qlosses

    def get_best_loss(self):
        return self.best_loss

    def get_losses_history(self):
        return self.losses_history

    def get_mses(self):
        return self.mses

    def get_qmses(self):
        return self.qmses

    def get_best_mse(self):
        return self.best_mse

    def get_mses_history(self):
        return self.mses_history

    def get_num_pis(self):
        return self.num_pis

    def get_num_svs(self):
        return self.num_svs

    def get_original_image(self):
        return np.squeeze(self.image)

    def init_domain_and_target(self):
        joint_domain = self.gen_domain(self.image, self.dim_domain)
        self.batch_shape = self.get_batch_shape(self.start_batches, joint_domain.shape)
        self.joint_domain = joint_domain

    def do_perspectiveTransform(self, affines, kernels_per_dim, init_flag=1):
        transformed_domain = self.joint_domain.copy()
        transformed_domain[..., 2] = -5

        for ii, affine in enumerate(affines):
            if self.num_params_model == 2:
                transformed_domain[:, :, ii, 0] = self.joint_domain[:, :, ii, 0] + affine[1, 2] / (self.image.shape[1] - 1)
                transformed_domain[:, :, ii, 1] = self.joint_domain[:, :, ii, 1] + affine[0, 2] / (self.image.shape[0] - 1)
            elif self.num_params_model == 4:
                transformed_domain[:, :, ii, 1] = affine[0, 0] * self.joint_domain[:, :, ii, 1] + affine[0, 1] * self.joint_domain[:, :, ii, 0] + affine[0, 2] / (self.image.shape[0] - 1)
                transformed_domain[:, :, ii, 0] = -affine[0, 1] * self.joint_domain[:, :, ii, 1] + affine[0, 0] * self.joint_domain[:, :, ii, 0] + affine[1, 2] / (self.image.shape[1] - 1)
            else:
                transformed_domain[:, :, ii, 0] = affine[1, 0] * self.joint_domain[:, :, ii, 1] + affine[1, 1] * self.joint_domain[:, :, ii, 0] + affine[1, 2] / (self.image.shape[1] - 1)
                transformed_domain[:, :, ii, 1] = affine[0, 0] * self.joint_domain[:, :, ii, 1] + affine[0, 1] * self.joint_domain[:, :, ii, 0] + affine[0, 2] / (self.image.shape[0] - 1)

            if self.num_params_model == 8 and affines.shape[1] == 3:
                w_dash = affine[2, 0] * self.joint_domain[:, :, ii, 1] + affine[2, 1] * self.joint_domain[:, :, ii, 0] + 1
                transformed_domain[:, :, ii, 0] /= w_dash
                transformed_domain[:, :, ii, 1] /= w_dash

        if init_flag == 1: # do kinda affine trafo onto the kernel grid
            cnt = 0
            musX_new = np.zeros_like(self.musX_init)
            for xx in range(kernels_per_dim[1]):
                for yy in range(kernels_per_dim[0]):
                    for zz in range(kernels_per_dim[2]):
                        zz_start = np.int(np.floor(self.image.shape[2]/kernels_per_dim[2])*zz)
                        zz_end = np.int(np.minimum(np.ceil(self.image.shape[2]/kernels_per_dim[2])*(zz+1), self.image.shape[2]))
                        #print(zz_start, zz_end)

                        xx_start = np.int(np.floor(self.image.shape[1]/kernels_per_dim[1])*xx)
                        xx_end = np.int(np.minimum(np.ceil(self.image.shape[1]/kernels_per_dim[1])*(xx+1), self.image.shape[1]))
                        #print(xx_start, xx_end)

                        yy_start = np.int(np.floor(self.image.shape[0] / kernels_per_dim[0]) * yy)
                        yy_end = np.int(np.minimum(np.ceil(self.image.shape[0] / kernels_per_dim[0]) * (yy + 1),
                                            self.image.shape[0]))
                        #print(yy_start, yy_end)

                        musX_new[cnt, :] = np.mean(transformed_domain[yy_start:yy_end, xx_start:xx_end, zz_start:zz_end, 0:3], axis=(0, 1, 2))
                        cnt = cnt + 1


            self.nu_e_init = np.ones_like(self.nu_e_init) * .5

            ''' # here: after affine trafo of kernel grid, try to find new time_means and corresponding bandwidth by using kmeans on xy-Plane
            _, labels = kmeans2(transformed_domain[:, :, :, 0:2].reshape((-1, 2)), musX_new[:, 0:2], 1)
            for ii in range(musX_new.shape[0]):
                new_time_bandwidth = np.var(transformed_domain[:, :, :, 2].reshape((-1, 1))[labels == ii])
                if not self.train_inverse_cov:
                    new_time_bandwidth = np.sqrt(new_time_bandwidth)
                if new_time_bandwidth > 0:
                    self.A_init[ii, 2, 2] = 1/new_time_bandwidth
                    musX_new[ii, 2] = np.mean(transformed_domain[:, :, :, 2].reshape((-1, 1))[labels == ii])
            '''
            self.musX_init = musX_new
        elif init_flag > 1 and init_flag < 4: # define 2D regular grid on xy-plane and define depending on max kernels in time and variances number of kernels for each spatial coord.

            flat_center = self.gen_domain(kernels_per_dim, 2)

            '''
            flat_center[:, 0] = (np.max(transformed_domain[..., 0]) - np.min(transformed_domain[..., 0])) * flat_center[
                                                                                                            :,
                                                                                                            0] + np.min(
                transformed_domain[..., 0])
            flat_center[:, 1] = (np.max(transformed_domain[..., 1]) - np.min(transformed_domain[..., 1])) * flat_center[
                                                                                                            :,
                                                                                                            1] + np.min(
                transformed_domain[..., 1])
            '''

                            #### alternative:#####
            min_y = np.sign(np.min(transformed_domain[..., 0])) * np.ceil(np.abs(np.min(transformed_domain[..., 0])))
            min_x = np.sign(np.min(transformed_domain[..., 1])) * np.ceil(np.abs(np.min(transformed_domain[..., 1])))
            max_y = np.ceil(np.max(transformed_domain[..., 0]))
            max_x = np.ceil(np.max(transformed_domain[..., 1]))
            # find recombinations
            flat_center_s = []
            for yy in range(int(min_y), int(max_y), 1):
                for xx in range(int(min_x), int(max_x), 1):
                    flat_center_s.append(flat_center +  np.array([yy, xx]))
            flat_center = np.vstack(flat_center_s)

            _, labels = kmeans2(transformed_domain[:, :, :, 0:2].reshape((-1, 2)), flat_center, 1)


            variances_over_lum = []
            for ii in np.unique(labels):
                variances_over_lum.append(np.var(transformed_domain[..., 3].reshape((-1, 1))[labels == ii]))
            varspace = np.linspace(np.min(variances_over_lum), np.max(variances_over_lum), kernels_per_dim[2])
            num_kernel_per_flat_center = np.argmin(np.abs((np.expand_dims(variances_over_lum, 0) - np.expand_dims(varspace, -1) )), axis=0) + 1

            musX_new = []
            A_new = []
            cnt = 0
            for ii in np.unique(labels):
                current_time_coords = transformed_domain[..., 2].reshape((-1, 1))[labels == ii]

                # if true the upcoming kernel will be assigned to one frame anyway
                if np.any(np.mean(current_time_coords, axis=0) == transformed_domain[0, 0, :, 2]) and len(np.unique(current_time_coords)) == 1:
                    num_kernel_per_flat_center[cnt] = 1

                if num_kernel_per_flat_center[cnt] == 1:
                    if init_flag % 1 == 0:
                        musX_new.append(np.hstack([flat_center[ii], np.mean(current_time_coords, axis=0)]))
                        # the bandwidth depends on the variance (or span) of the labeled time-coords but is never more precise than kernels init'ed in regular as many as frames
                        time_bandwidth = np.minimum(1 / np.sqrt(np.var(current_time_coords)), 2 * (self.image.shape[2] + 1))
                    elif init_flag % 1 == .5:
                        musX_new.append(np.hstack([flat_center[ii], .5]))
                        time_bandwidth = 2 * (1 + 1) # bandwidth as precise as it is supposed at regular grid
                    #A_new.append(np.diag([2 * (kernels_per_dim[0] + 1) / (np.max(transformed_domain[..., 0]) - np.min(transformed_domain[..., 0])), 2 * (kernels_per_dim[1] + 1) / (np.max(transformed_domain[..., 1]) - np.min(transformed_domain[..., 1])), time_bandwidth]) * [1, 1, 1])
                    A_new.append(np.diag([2 * (kernels_per_dim[0] + 1),
                                          2 * (kernels_per_dim[1] + 1), time_bandwidth]) * [1, 1, 1])
                else:


                    if np.floor(init_flag) == 2: #num_kernel_depends_on_variance:
                        # time center regular depending on variance in luminance in corresponding samples
                        time_means = np.linspace(np.min(current_time_coords),
                                                 np.max(current_time_coords),
                                                 num_kernel_per_flat_center[cnt])

                        labels_for_time_bandwidth = np.argmin(
                            np.abs(current_time_coords - time_means), axis=1)

                        num_kernel = len(np.unique(labels_for_time_bandwidth))

                        for jj in np.unique(labels_for_time_bandwidth):
                            time_bandwidth = np.minimum(
                                1 / (np.sqrt(np.var(current_time_coords[labels_for_time_bandwidth == jj])) + 10 ** -5),
                                2 * (self.image.shape[2] + 1) * num_kernel)
                            if np.isnan(time_bandwidth):
                                continue
                            musX_new.append(np.hstack([flat_center[ii], time_means[jj]]))

                            #A_new.append(np.diag([2 * (kernels_per_dim[0] + 1) / (
                            #        np.max(transformed_domain[..., 0]) - np.min(transformed_domain[..., 0])),
                            #                      2 * (kernels_per_dim[1] + 1) / (
                            #                              np.max(transformed_domain[..., 1]) - np.min(
                            #                          transformed_domain[..., 1])), time_bandwidth]) * [1, 1, 1])
                            A_new.append(np.diag([2 * (kernels_per_dim[0] + 1),
                                                  2 * (kernels_per_dim[1] + 1), time_bandwidth]) * [1, 1, 1])
                    elif np.floor(init_flag) == 3:
                        # time center regular depending of existing time coords depending on maximal kernel in time domain:
                        time_means = self.gen_domain([np.ceil(len(np.unique(current_time_coords)) * kernels_per_dim[2] / self.image.shape[2])], 1) * (np.max(current_time_coords) - np.min(current_time_coords)) + np.min(current_time_coords)
                        for jj in range(len(time_means)):
                            musX_new.append(np.hstack([flat_center[ii], time_means[jj]]))
                            #A_new.append(np.diag([2 * (kernels_per_dim[0] + 1) / (
                            #        np.max(transformed_domain[..., 0]) - np.min(transformed_domain[..., 0])),
                            #                      2 * (kernels_per_dim[1] + 1) / (
                            #                              np.max(transformed_domain[..., 1]) - np.min(
                            #                          transformed_domain[..., 1])), 2 * (len(time_means) + 1)]) * [1, 1, 1])
                            A_new.append(np.diag([2 * (kernels_per_dim[0] + 1),
                                                  2 * (kernels_per_dim[1] + 1), 2 * (len(time_means) + 1)]) * [1, 1, 1])




                cnt += 1
            musX_new = np.stack(musX_new)
            A_new = np.stack(A_new)
            K = musX_new.shape[0]
            print('Number of Kernels are ' + str(K))


            ''' good working state
            K = len(np.unique(labels))
            musX_new = np.zeros((K, 3))
            A_new = np.zeros((K, 3, 3))
            cnt = 0
            for ii in np.unique(labels):
                musX_new[cnt] = np.hstack([flat_center[ii], np.mean(transformed_domain[..., 2].reshape((-1, 1))[labels == ii], axis=0)])
                time_bandwidth = np.minimum(1/np.sqrt(np.var(transformed_domain[..., 2].reshape((-1, 1))[labels == ii])),2 * (self.image.shape[2] + 1) )
                A_new[cnt] = np.diag([2 * (kernels_per_dim[0] + 1) / (np.max(transformed_domain[..., 0]) - np.min(transformed_domain[..., 0])), 2 * (kernels_per_dim[1] + 1) / (np.max(transformed_domain[..., 1]) - np.min(transformed_domain[..., 1])), time_bandwidth ])
                cnt += 1
            '''

            self.musX_init = musX_new
            self.A_init = A_new
            self.start_pis = K
            self.kernel_count = K
            self.nu_e_init = np.ones((K, 3)) * .5
            self.gamma_e_init = np.zeros((K, 3, 3))
            self.pis_init = np.ones((K,))
            #self.pis_init[0:K] = 1
            #self.pis_init[K::] = 0
            '''
            K = flat_center.shape[0]
            self.musX_init = np.hstack([flat_center, np.ones((K, 1)) * .5])
            self.A_init = np.tile(np.diag([2 * (kernels_per_dim[0] + 1) / (np.max(transformed_domain[..., 0]) - np.min(transformed_domain[..., 0])), 2 * (kernels_per_dim[1] + 1) / (np.max(transformed_domain[..., 1]) - np.min(transformed_domain[..., 1])), 2 * (1+1)]), (K, 1, 1))
            self.start_pis = K
            self.kernel_count = K
            self.nu_e_init = self.nu_e_init[0:K]
            self.gamma_e_init = self.gamma_e_init[0:K]
            self.pis_init = self.pis_init[0:K]
            #self.pis_init[0:K] = 1
            #self.pis_init[K::] = 0
            '''
        elif init_flag == 4 or init_flag == 5:
            # find recombinations
            flat_center_s = []
            kernels_per_dim_2d = kernels_per_dim.copy()
            kernels_per_dim_2d[2] = 1
            if init_flag == 5:
                for ii in range(2):
                    kernels_per_dim_2d[ii] = int(np.ceil(kernels_per_dim_2d[ii] * 1.1*np.sqrt(kernels_per_dim[2])))
            flat_center = self.gen_domain(kernels_per_dim_2d, 3)
            if init_flag == 4:
                min_y = np.sign(np.min(transformed_domain[..., 0])) * np.ceil(
                    np.abs(np.min(transformed_domain[..., 0])))
                min_x = np.sign(np.min(transformed_domain[..., 1])) * np.ceil(
                    np.abs(np.min(transformed_domain[..., 1])))
                max_y = np.ceil(np.max(transformed_domain[..., 0]))
                max_x = np.ceil(np.max(transformed_domain[..., 1]))
                for yy in range(int(min_y), int(max_y), 1):
                    for xx in range(int(min_x), int(max_x), 1):
                        flat_center_s.append(flat_center + np.array([yy, xx, 0]))
                flat_center = np.vstack(flat_center_s)
                _, labels = kmeans2(transformed_domain[:, :, :, 0:3].reshape((-1, 3)), flat_center, 1)
                musX_new = flat_center[np.unique(labels)]
            else:
                min_y = np.sign(np.min(transformed_domain[..., 0])) * np.abs(np.min(transformed_domain[..., 0]))
                min_x = np.sign(np.min(transformed_domain[..., 1])) * np.abs(np.min(transformed_domain[..., 1]))
                max_y = np.max(transformed_domain[..., 0])
                max_x = np.max(transformed_domain[..., 1])
                flat_center[:, 0] = flat_center[:, 0] * (max_y - min_y) + min_y
                flat_center[:, 1] = flat_center[:, 1] * (max_x - min_x) + min_x
                _, labels = kmeans2(transformed_domain[:, :, :, 0:2].reshape((-1, 2)), flat_center[:, 0:2], 1)
                musX_new = flat_center[np.unique(labels)]

            K = musX_new.shape[0]
            A_values = np.ones((3,))
            for ii in range(2):
                A_values[ii] = 2 * (kernels_per_dim_2d[ii] + 1)
            A_prototype = np.diag(A_values)
            A_new = np.tile(A_prototype, (K, 1, 1))
            self.musX_init = musX_new
            self.A_init = A_new
            self.start_pis = K
            self.kernel_count = K
            self.nu_e_init = np.ones((K, 3)) * .5
            self.gamma_e_init = np.zeros((K, 3, 3))
            self.pis_init = np.ones((K,))
        self.transformed_domain = transformed_domain



    def get_iter(self):
        return self.iter

    # quadratic for 2d in [0,1]
    def generate_kernel_grid(self, kernels_per_dim):
        dim_of_domain = self.image.ndim-1
        self.musX_init = self.gen_domain(kernels_per_dim, dim_of_domain)


        if len(kernels_per_dim) > 1:
            A_values = np.zeros((dim_of_domain,))
            for ii in range(dim_of_domain):
                A_values[ii] = 2 * (kernels_per_dim[ii] + 1)
            A_prototype = np.diag(A_values)
            number_of_kernel = np.prod(kernels_per_dim)
        else:
            A_prototype = np.zeros((dim_of_domain, dim_of_domain))
            np.fill_diagonal(A_prototype, 2 * (kernels_per_dim[0] + 1))
            number_of_kernel = kernels_per_dim[0] ** dim_of_domain
        self.A_init = np.tile(A_prototype, (number_of_kernel, 1, 1))
        if self.train_inverse_cov:
            self.A_init = self.A_init**2

    def generate_experts(self, with_means=True):
        assert self.musX_init is not None, "need musX to generate experts"

        num_channels = self.image.shape[-1]

        self.gamma_e_init = np.zeros((self.musX_init.shape[0], self.dim_domain, num_channels))

        # TODO generalize initialization of nu_e's for arbitrary dim of input space
        if with_means:
            if self.dim_domain == 2:
                # assumes that muX_init are in a square grid
                stride = self.musX_init[0]
                height, width = self.image.shape[0], self.image.shape[1]
                mean = np.empty((self.musX_init.shape[0], num_channels), dtype=np.float32)
                for k, (y, x) in enumerate(zip(*self.musX_init.T)):
                    x0 = int(round((x - stride[1]) * width))
                    x1 = int(round((x + stride[1]) * width))
                    y0 = int(round((y - stride[0]) * height))
                    y1 = int(round((y + stride[0]) * height))

                    mean[k] = np.mean(self.image[y0:y1, x0:x1], axis=(0, 1))
            elif self.dim_domain == 3:
                stride = self.musX_init[0]
                height, width, frames = self.image.shape[0], self.image.shape[1], self.image.shape[2]
                mean = np.empty((self.musX_init.shape[0], num_channels), dtype=np.float32)
                for k, (y, x, z) in enumerate(zip(*self.musX_init.T)):
                    x0 = int(round((x - stride[1]) * width))
                    x1 = int(round((x + stride[1]) * width))
                    y0 = int(round((y - stride[0]) * height))
                    y1 = int(round((y + stride[0]) * height))
                    z0 = int(round((z - stride[2]) * frames))
                    z1 = int(round((z + stride[2]) * frames))

                    mean[k] = np.mean(self.image[y0:y1, x0:x1, z0:z1], axis=(0, 1, 2))
            elif self.dim_domain == 4:
                stride = self.musX_init[0]
                num_disp_1, num_disp_2, height, width = self.image.shape[0], self.image.shape[1], self.image.shape[2], self.image.shape[3]
                mean = np.empty((self.musX_init.shape[0], num_channels), dtype=np.float32)
                for k, (a1, a2, y, x) in enumerate(zip(*self.musX_init.T)):
                    x0 = int(round((x - stride[3]) * width))
                    x1 = int(round((x + stride[3]) * width))
                    y0 = int(round((y - stride[2]) * height))
                    y1 = int(round((y + stride[2]) * height))
                    a10 = int(round((a1 - stride[0]) * num_disp_1))
                    a11 = int(round((a1 + stride[0]) * num_disp_1))
                    a20 = int(round((a2 - stride[1]) * num_disp_2))
                    a21 = int(round((a2 + stride[1]) * num_disp_2))
                    a10 = np.maximum(a10, 4)
                    a20 = np.maximum(a20, 4)
                    a11 = np.minimum(a11, 11)
                    a21 = np.minimum(a21, 11)

                    mean[k] = np.mean(self.image[a10:a11, a20:a21, y0:y1, x0:x1], axis=(0, 1, 2, 3))

            '''    
            stride = self.musX_init[0, :]
            size_of_img = self.image.shape
            mean = np.empty((self.musX_init.shape[0], num_channels), dtype=np.float32)
            for k in range(self.musX_init.shape[0]):
                range_dim = []
                for dim in range(dim_of_domain):
                    dim0 = int(round((self.musX_init[k, dim] - stride[dim])))
                    dim1 = int(round((self.musX_init[k, dim] + stride[dim])))
                    range_dim.append(dim0:dim1)
                mean[k] = np.mean(self.image[*range_dim], axis= )    
            '''
        else:
            # choose nu_e to be 0.5 at center of kernel
            mean = np.ones((self.musX_init.shape[0], num_channels)) * 0.5

        self.nu_e_init = mean

    def generate_pis(self, normalize_pis):
        number = self.musX_init.shape[0]
        if normalize_pis:
            self.pis_init = np.ones((number,), dtype=np.float32) / (number)
        else:
            self.pis_init = np.ones((number,), dtype=np.float32)

    def initialize_kernel_list(self, add_kernel_slots=0):
        num_of_all_kernels = self.start_pis
        if add_kernel_slots > 0:
            num_of_all_kernels = 2*num_of_all_kernels + add_kernel_slots

        if self.affines is not None or self.dim_domain == 3 and self.train_trafo:
            joint_domain = self.transformed_domain
        else:
            joint_domain = self.joint_domain

        batch_center = []
        for (coord, batch) in sliding_window(joint_domain, self.overlap, self.batch_size_valued):
            batch_center.append(np.mean(batch, axis=tuple(np.arange(self.dim_domain))))
        batch_center = np.stack(batch_center)

        batch_center_1 = []
        for (coord, batch) in sliding_window(self.joint_domain, self.overlap, self.batch_size_valued):
            batch_center_1.append(np.mean(batch, axis=tuple(np.arange(self.dim_domain))))
        batch_center_1 = np.stack(batch_center_1)

        maha_dists = []
        for k in range(batch_center.shape[0]):
            results = self.session.run([self.maha_dist], feed_dict={self.domain_final:  np.expand_dims(batch_center[k][:self.dim_domain], axis=0),
                                                                    self.domain_op: np.expand_dims(
                                                                        batch_center_1[k][:self.dim_domain], axis=0),
                                                                 self.kernel_list: np.ones((num_of_all_kernels,),
                                                                                           dtype=bool),
                                                                    self.stack_inc: [0.],
                                                                    self.stack_orig: [1.]})
            maha_dists.append(results[0])
        maha_dists = np.concatenate(maha_dists, axis=1)
        maha_dist_min_ind = np.argmin(maha_dists, axis=1)
        self.kernel_list_per_batch = []
        for k in range(batch_center.shape[0]):
            if add_kernel_slots > 0:
                self.kernel_list_per_batch.append(np.concatenate(
                    [maha_dist_min_ind == k, np.zeros((add_kernel_slots + self.start_pis,), dtype=bool)]))
            else:
                self.kernel_list_per_batch.append(maha_dist_min_ind == k)


        self.update_kernel_list(add_kernel_slots)

    def update_kernel_list(self, add_kernel_slots=0):
        num_of_all_kernels = self.start_pis
        if add_kernel_slots > 0:
            num_of_all_kernels = 2*num_of_all_kernels + add_kernel_slots

        if self.affines is not None or self.dim_domain == 3 and self.train_trafo:
            if self.num_params_model == 2:
                h13, h23 = self.session.run([self.qh13, self.qh23])
                h11 = np.ones_like(h13)
                h12 = np.zeros_like(h13)
                h21 = np.zeros_like(h13)
                h22 = np.ones_like(h13)
            elif self.num_params_model == 4:
                h11, h12, h13, h23 = self.session.run([self.qh11, self.qh12, self.qh13, self.qh23])
                h22 = h11
                h21 = -h12
            elif self.num_params_model == 6:
                h11, h12, h13, h21, h22, h23 = self.session.run([self.qh11, self.qh12, self.qh13, self.qh21, self.qh22, self.qh23])
            elif self.num_params_model == 8:
                h11, h12, h13, h21, h22, h23, h31, h32 = self.session.run([self.qh11, self.qh12, self.qh13, self.qh21, self.qh22, self.qh23, self.qh31, self.qh32])
            transformed_domain = self.joint_domain.copy()
            transformed_domain[..., 2] = -5
            for ii in range(self.image.shape[2]):
                transformed_domain[:, :, ii, 0] = h21[ii] * self.joint_domain[:, :, ii, 1] + h22[ii] * self.joint_domain[:, :, ii, 0] + h23[ii]
                transformed_domain[:, :, ii, 1] = h11[ii] * self.joint_domain[:, :, ii, 1] + h12[ii] * self.joint_domain[:, :, ii, 0] + h13[ii]
                if self.num_params_model == 8:
                    w_dash = h31[ii] * self.joint_domain[:, :, ii, 1] + h32[ii] * self.joint_domain[:, :, ii, 0] + 1
                    transformed_domain[:, :, ii, 0] /= w_dash
                    transformed_domain[:, :, ii, 1] /= w_dash

            joint_domain = transformed_domain
        else:
            joint_domain = self.joint_domain

        # DAMIT DAS HIER WIEDER GEHT, ZUNÄCHST BATCHES ERSTELLEN MIT GENERATOR UND DANN MINS; MAXS BESTIMMEN!!
        mins = []
        maxs = []
        for (coord, batch) in sliding_window(joint_domain, self.overlap, self.batch_size_valued):
            mins.append(np.min(batch, axis=tuple(np.arange(self.dim_domain))))
            maxs.append(np.max(batch, axis=tuple(np.arange(self.dim_domain))))
        mins = np.stack(mins)
        maxs = np.stack(maxs)

        mins = mins[:, 0:self.dim_domain]
        maxs = maxs[:, 0:self.dim_domain]
        # get all corner points of (hyper-)cube and middle points of edges
        tt = np.concatenate((np.expand_dims(mins, axis=-1), np.expand_dims(maxs, axis=-1), np.expand_dims((mins + maxs)/2, axis=-1)), axis=-1)

        ######## For non tranformed bla ##########
        mins = []
        maxs = []
        for (coord, batch) in sliding_window(self.joint_domain, self.overlap, self.batch_size_valued):
            mins.append(np.min(batch, axis=tuple(np.arange(self.dim_domain))))
            maxs.append(np.max(batch, axis=tuple(np.arange(self.dim_domain))))
        mins = np.stack(mins)
        maxs = np.stack(maxs)

        mins = mins[:, 0:self.dim_domain]
        maxs = maxs[:, 0:self.dim_domain]
        # get all corner points of (hyper-)cube and middle points of edges
        tt_1 = np.concatenate((np.expand_dims(mins, axis=-1), np.expand_dims(maxs, axis=-1), np.expand_dims((mins + maxs)/2, axis=-1)), axis=-1)

        for k in range(mins.shape[0]):
            edges_batch = np.array(list(product(*tt[k, :, :])))
            edges_batch = np.concatenate((edges_batch, np.zeros((edges_batch.shape[0], self.image.shape[-1]))), axis=1)

            edges_batch_1 = np.array(list(product(*tt_1[k, :, :])))
            edges_batch_1 = np.concatenate((edges_batch_1, np.zeros((edges_batch_1.shape[0], self.image.shape[-1]))), axis=1)
            results = self.session.run([self.maha_dist_ind], feed_dict={self.domain_final: edges_batch[:,:self.dim_domain],
                                                                        self.domain_op: edges_batch_1[:,
                                                                                           :self.dim_domain],
                                                                  self.kernel_list: np.ones((num_of_all_kernels,),
                                                                                            dtype=bool),
                                                                    self.stack_inc: [1.],
                                                                    self.stack_orig: [1.]})
            #bool_mask = np.zeros((self.start_pis,), dtype=bool)
            bool_mask = np.zeros((num_of_all_kernels,), dtype=bool)
            bool_mask[results[0]] = True
            self.kernel_list_per_batch[k] = np.logical_or(self.kernel_list_per_batch[k], bool_mask)

    def initialize_frames_list(self):
        self.frames_list_per_batch = []
        for (coord, batch) in sliding_window(self.joint_domain, self.overlap, self.batch_size_valued):
            frames_list = np.zeros((self.image.shape[2],), dtype=bool)
            frames_list[ coord[2] : coord[2] + self.batch_size_valued[2] ] = True
            self.frames_list_per_batch.append(frames_list)

    def get_train_mask(self):
        if self.dim_domain >= 4:
            train_mask = np.ones(self.batch_shape[:-1], dtype=bool)
            train_mask[0, 0:4, :, :] = False
            train_mask[0, 11:, :, :] = False
            train_mask[1, 0:2, :, :] = False
            train_mask[1, 13:, :, :] = False
            train_mask[2:4, 0, :, :] = False
            train_mask[2:4, 14, :, :] = False
            train_mask[11:13, 0, :, :] = False
            train_mask[11:13, 14, :, :] = False
            train_mask[13, 0:2, :, :] = False
            train_mask[13, 13:, :, :] = False
            train_mask[14, 0:4, :, :] = False
            train_mask[14, 11:, :, :] = False
            self.train_mask = train_mask.reshape(-1)





    @staticmethod
    def gen_domain(in_, dim_of_input_space=2):
        num_per_dim = np.zeros((dim_of_input_space,), dtype=np.int32)
        if type(in_) is np.ndarray:
            for ii in range(dim_of_input_space):
                num_per_dim[ii] = in_.shape[ii]
        else:
            for ii in range(dim_of_input_space):
                if len(in_) > 1:
                    num_per_dim[ii] = in_[ii]
                else:
                    num_per_dim[ii] = in_[0]

        # create coordinates for each dimension
        coord = []
        for ii in range(dim_of_input_space):
            if type(in_) is np.ndarray:
                coord.append(np.linspace(0, 1, num_per_dim[ii]))
            else:
                # equal spacing between domain positions and boarder
                coord.append(np.linspace((1 / num_per_dim[ii]) / 2, 1 - (1 / num_per_dim[ii]) / 2, num_per_dim[ii]))

        # create grids
        grids = np.meshgrid(*coord, indexing='ij')

        if type(in_) is np.ndarray:
            domain = np.stack(grids, axis=-1)
            domain = np.append(domain, in_, axis=-1)
        else:
            domain = np.reshape(np.stack(grids, axis=-1), (np.prod(num_per_dim), dim_of_input_space))

        return domain

    @staticmethod
    def calc_intervals(in_size, batches):
        start = 0
        steps = np.linspace(0, in_size, num=max(2, batches+1), endpoint=True)
        intervals = []
        for end in steps[1:]:
            end = int(end)
            intervals.append((start, end))
            start = end

        return intervals

    @staticmethod
    def cubify(arr, newshape):
        oldshape = np.array(arr.shape)
        repeats = (oldshape / newshape).astype(int)
        tmpshape = np.column_stack([repeats, newshape]).ravel()
        order = np.arange(len(tmpshape))
        order = np.concatenate([order[::2], order[1::2]])
        # newshape must divide oldshape evenly or else ValueError will be raised
        return arr.reshape(tmpshape).transpose(order).reshape(-1, *newshape)

    @staticmethod
    def uncubify(arr, oldshape):
        N, newshape = arr.shape[0], arr.shape[1:]
        oldshape = np.array(oldshape)
        repeats = (oldshape / newshape).astype(int)
        tmpshape = np.concatenate([repeats, newshape])
        order = np.arange(len(tmpshape)).reshape(2, -1).ravel(order='F')
        return arr.reshape(tmpshape).transpose(order).reshape(oldshape)

    @staticmethod
    def get_batch_shape(desired_batches, joint_domain_shape):

        def divisors(n):
            # get factors and their counts
            factors = {}
            nn = n
            i = 2
            while i * i <= nn:
                while nn % i == 0:
                    if not i in factors:
                        factors[i] = 0
                    factors[i] += 1
                    nn //= i
                i += 1
            if nn > 1:
                factors[nn] = 1

            primes = list(factors.keys())

            # generates factors from primes[k:] subset
            def generate(k):
                if k == len(primes):
                    yield 1
                else:
                    rest = generate(k + 1)
                    prime = primes[k]
                    for factor in rest:
                        prime_to_i = 1
                        # prime_to_i iterates prime**i values, i being all possible exponents
                        for _ in range(factors[prime] + 1):
                            yield factor * prime_to_i
                            prime_to_i *= prime

            # in python3, `yield from generate(0)` would also work
            for factor in generate(0):
                yield factor


        # determine all divisors for each dimension (except last dimension)
        factors = []
        for ii in range(joint_domain_shape.__len__()-1):
            divisor = []
            for fac in divisors(joint_domain_shape[ii]):
                divisor.append(fac)
            factors.append(divisor)
        factors.append([1])
        # TODO LF HACK
        if joint_domain_shape.__len__() > 4:
            factors[0] = [1]
            factors[1] = [1]

        # get all variations of dimension divisors as tuples
        shapes = list(product(*factors))

        # determine all possible batch sizes
        possible_batches = np.zeros((shapes.__len__(), 1))
        for ii, shape in enumerate(shapes):
            # possible_batches.append(np.prod(shape[:-1]))
            possible_batches[ii] = np.prod(shape[:-1])

        # determine that batch size which is the closest to the desired one (and bigger)
        diff = possible_batches - desired_batches
        # diff = diff * -1
        diff[diff < 0] = np.inf
        aimed_batch = possible_batches[np.argmin(diff)]
        indices = np.where(possible_batches == aimed_batch)[0]
        batch_shape_divisor = []
        for idx in indices:
            batch_shape_divisor.append(shapes[idx])

        # determine that divisor tuple so that cubes come out as close as possible
        sum_of_dim_divisors = np.zeros((batch_shape_divisor.__len__(), 1))
        for ii, divs in enumerate(batch_shape_divisor):
            if len(divs) > 4:
                sum_of_dim_divisors[ii] = np.sum(divs[2:3])
            else:
                sum_of_dim_divisors[ii] = np.sum(divs)

        batch_shape_divisor = batch_shape_divisor[np.argmin(sum_of_dim_divisors)]
        new_shape = []
        for ii in range(joint_domain_shape.__len__()):
            new_shape.append(int(joint_domain_shape[ii] / batch_shape_divisor[ii]))

        return tuple(new_shape)

    @staticmethod
    def remap_kernel_indices(w_es_mat, kernel_list):
        w_es_flat = w_es_mat.flatten()
        w_es_tmp = w_es_flat
        idx_shift = int(0)
        for ii in range(kernel_list.size):
            #if ii + idx_shift != kernel_list[ii]:
            #    idx_shift += kernel_list[ii] - (ii + idx_shift)
            #while ii + idx_shift != kernel_list[ii]:
             #   idx_shift += 1

             #   w_es_tmp[np.where(w_es_flat == ii)] += idx_shift
            #if ii != kernel_list[ii]:
            #    w_es_flat[np.where(w_es_flat >= ii)] += 1
            w_es_tmp[np.where(w_es_flat == ii)] = kernel_list[ii]

        remapped_wes = np.reshape(w_es_tmp, w_es_mat.shape)

        return remapped_wes

    # Not used method, Not sure if it's correct anyway
    @staticmethod
    def find_neighboring_indices(img_shape, batch_shape):
        segments_per_dim = [1]
        for k in range(len(img_shape) - 1, 0, -1):
            segments_per_dim.append(int(img_shape[k - 1] / batch_shape[k]))
        ind_jump_per_dim = np.cumprod(segments_per_dim)[0:-1]
        ind_jump_per_dim = np.concatenate((ind_jump_per_dim, -ind_jump_per_dim))
        indices = [ind_jump_per_dim]
        for k in range(2, len(img_shape), 1):
            indices.append(np.sum(np.array(list(combinations(ind_jump_per_dim, k))), axis=1, keepdims=False))
        indices = np.unique(np.concatenate(indices))

        return indices


