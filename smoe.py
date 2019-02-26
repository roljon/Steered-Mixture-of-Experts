import numpy as np
import math
import cv2
import os
from skimage.feature import peak_local_max
from skimage.measure import compare_ssim
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.python.ops.special_math_ops import _exponential_space_einsum as einsum
import progressbar
from itertools import product, combinations
from quantizer import quantize_params, rescaler
from scipy.ndimage import gaussian_filter

class Smoe:
    def __init__(self, image, kernels_per_dim=None, train_pis=True, init_params=None, start_batches=1,
                 train_gammas=True, train_musx=True, use_diff_center=False, radial_as=False, use_determinant=False,
                 normalize_pis=True, quantization_mode=0, bit_depths=None, quantize_pis=False, lower_bounds=None,
                 upper_bounds=None, use_yuv=True, only_y_gamma=False, ssim_opt=False, precision=8, add_kernel_slots=0, iter_offset=0, margin=0.5):
        self.batch_shape = None
        self.use_yuv = use_yuv
        self.only_y_gamma = only_y_gamma
        self.ssim_opt = ssim_opt
        self.use_diff_center = use_diff_center
        self.precision = precision
        self.train_mask = None
        self.add_kernel_slots = add_kernel_slots

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
        self.A_var = None
        self.gamma_e_var = None
        self.nu_e_var = None

        self.pis_best_var = None
        self.musX_best_var = None
        self.A_best_var = None
        self.gamma_e_best_var = None
        self.nu_e_best_var = None

        # tf inc vars
        self.pis_inc_var = None
        self.musX_inc_var = None
        self.A_diagonal_inc_var = None
        self.A_corr_inc_var = None
        self.gamma_e_inc_var = None
        self.nu_e_inc_var = None

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

        self.iter = iter_offset
        self.valid = False
        self.qvalid = False
        self.reconstruction_image = None
        self.weight_matrix_argmax = None
        self.qreconstruction_image = None
        self.qweight_matrix_argmax = None
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

        # generate initializations
        self.start_batches = start_batches  # start_batches corresponds to desired batch numbers
        self.image = image
        self.image_flat = None
        self.joint_domain_batched = None
        self.dim_domain = image.ndim - 1
        self.num_pixel = np.prod(image.shape[0:self.dim_domain])
        self.init_domain_and_target()
        self.neighboring_batches_ind = self.find_neighboring_indices(self.image.shape, self.joint_domain_batched.shape)
        self.batches = start_batches
        self.start_batches = self.joint_domain_batched.shape[0]

        assert kernels_per_dim is not None or init_params is not None, \
            "You need to specify the kernel grid size or give initial parameters."

        if init_params:
            self.pis_init = init_params['pis']
            self.musX_init = init_params['musX']
            self.A_init = init_params['A']
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

        self.random_sampling_per_batch = [np.ones((np.prod(self.joint_domain_batched.shape[1:self.dim_domain+1]),), dtype=np.float32) / np.prod(self.joint_domain_batched.shape[1:self.dim_domain+1])] * self.start_batches
        self.get_train_mask()

        gpu_options = tf.GPUOptions(allow_growth=True)
        self.session = tf.InteractiveSession(config=tf.ConfigProto(gpu_options=gpu_options))
        # self.session = tf.Session()

        self.init_model(self.nu_e_init, self.gamma_e_init, self.pis_init, self.musX_init, self.A_init, add_kernel_slots)
        self.initialize_kernel_list(add_kernel_slots)

    def init_model(self, nu_e_init, gamma_e_init, pis_init, musX_init, A_init, add_kernel_slots=0):

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
            self.qpis = tf.fake_quant_with_min_max_args(concat_pis, min=self.lower_bounds[3],
                                                        max=self.upper_bounds[3], num_bits=self.bit_depths[3])
        else:
            self.qpis = concat_pis

        pis_mask = self.qpis > 0

        if self.quantization_mode == 2:
            self.qA_diagonal = tf.fake_quant_with_min_max_args(concat_A_diagonal, min=self.lower_bounds[0],
                                                               max=self.upper_bounds[0], num_bits=self.bit_depths[0])
            self.qA_corr = tf.fake_quant_with_min_max_args(concat_A_corr, min=self.lower_bounds[0],
                                                           max=self.upper_bounds[0], num_bits=self.bit_depths[0])

            self.qmusX = tf.fake_quant_with_min_max_args(concat_musX,
                                                         min=self.lower_bounds[1], max=self.upper_bounds[1],
                                                         num_bits=self.bit_depths[1])
            self.qnu_e = tf.fake_quant_with_min_max_args(concat_nu_e,
                                                         min=self.lower_bounds[2], max=self.upper_bounds[2],
                                                         num_bits=self.bit_depths[2])
            self.qgamma_e = tf.fake_quant_with_min_max_args(concat_gamma_e,
                                                            min=self.lower_bounds[4], max=self.upper_bounds[4],
                                                            num_bits=self.bit_depths[4])
        elif self.quantization_mode == 3:
            if self.radial_as:
                min_A_diagonal = tf.reduce_min(tf.boolean_mask(concat_A_diagonal, pis_mask))
                max_A_diagonal = tf.reduce_max(tf.boolean_mask(concat_A_diagonal, pis_mask))
                qA_diagonal = tf.fake_quant_with_min_max_vars(concat_A_diagonal, min=0,
                                                              max=max_A_diagonal - min_A_diagonal,
                                                              num_bits=self.bit_depths[0])
                self.qA_diagonal = qA_diagonal + min_A_diagonal
            else:
                min_A_diagonal = tf.reduce_min(tf.matrix_diag_part(tf.boolean_mask(concat_A_diagonal, pis_mask)))
                max_A_diagonal = tf.reduce_max(tf.matrix_diag_part(tf.boolean_mask(concat_A_diagonal, pis_mask)))
                qA_diagonal = tf.fake_quant_with_min_max_vars(concat_A_diagonal - min_A_diagonal, min=0,
                                                              max=max_A_diagonal - min_A_diagonal,
                                                              num_bits=self.bit_depths[0])
                self.qA_diagonal = qA_diagonal + min_A_diagonal
            self.qA_corr = tf.fake_quant_with_min_max_vars(concat_A_corr,
                                                           min=tf.reduce_min(tf.boolean_mask(concat_A_corr, pis_mask)),
                                                           max=tf.reduce_max(tf.boolean_mask(concat_A_corr, pis_mask)),
                                                           num_bits=self.bit_depths[0])
            if self.train_musx:
                self.qmusX = tf.fake_quant_with_min_max_vars(concat_musX,
                                                             min=tf.reduce_min(tf.boolean_mask(concat_musX, pis_mask)),
                                                             max=tf.reduce_max(tf.boolean_mask(concat_musX, pis_mask)),
                                                             num_bits=self.bit_depths[1])
            else:
                self.qmusX = concat_musX

            min_nu_e = tf.reduce_min(tf.boolean_mask(concat_nu_e, pis_mask))
            max_nu_e = tf.reduce_max(tf.boolean_mask(concat_nu_e, pis_mask))
            qnu_e = tf.fake_quant_with_min_max_vars(concat_nu_e - min_nu_e, min=0, max=max_nu_e-min_nu_e, num_bits=self.bit_depths[2])
            self.qnu_e = qnu_e + min_nu_e

            self.qgamma_e = tf.fake_quant_with_min_max_vars(concat_gamma_e,
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
        #self.target_op = tf.reshape(self.target_op, [-1])
        self.domain_op = self.joint_domain_batched_op[:, :self.dim_domain]

        self.kernel_list = tf.placeholder(shape=(None,), dtype=tf.bool)


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
        A = tf.matrix_band_part(A_diagonal, 0, 0) \
            + tf.matrix_band_part(tf.matrix_set_diag(A_corr, np.zeros((num_of_all_kernels, self.dim_domain))), -1, 0)

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


        musX = self.musX
        nu_e = self.nu_e
        gamma_e = self.gamma_e
        A = self.A
        pis = self.pis

        musX = tf.expand_dims(musX, axis=1)
        # prepare domain
        domain_exp = self.domain_op
        domain_exp = tf.tile(tf.expand_dims(domain_exp, axis=0), (tf.shape(musX)[0], 1, 1))

        x_sub_mu = tf.expand_dims(domain_exp - musX, axis=-1)

        self.maha_dist = einsum('abli,alm,anm,abnj->ab', x_sub_mu, A, A, x_sub_mu)
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

        #self.w_e_max_op = tf.reshape(tf.argmax(self.w_e_op, axis=0), self.batch_shape[:-1])

        kernel_list_batch_op = tf.reduce_sum(bool_mask_infl, axis=1) > 0 # 10 ** -9
        self.w_e_max_op = tf.reshape(tf.argmax(tf.boolean_mask(self.w_e_op, kernel_list_batch_op), axis=0), self.batch_shape[:-1])
        self.indices = tf.boolean_mask(self.indices, kernel_list_batch_op)

        nu_e = tf.expand_dims(tf.transpose(nu_e), axis=-1)
        if self.train_gammas:
            # TODO reorder nu_e and gamma_e to avoid unnecessary transpositions
            domain_tiled = tf.expand_dims(tf.transpose(self.domain_op), axis=0)
            domain_tiled = tf.tile(domain_tiled, (num_channels, 1, 1))
            sloped_out = tf.matmul(tf.transpose(gamma_e, perm=[2, 0, 1]), domain_tiled)
            self.res = tf.reduce_sum(self.w_e_op * (sloped_out + nu_e), axis=1)
        else:
            self.res = tf.reduce_sum(self.w_e_op * nu_e, axis=1)

        self.res = tf.clip_by_value(self.res, 0, 1)
        self.res = tf.transpose(self.res)

        # checkpoint op
        self.pis_best_var = tf.Variable(self.qpis)
        self.musX_best_var = tf.Variable(self.qmusX)
        self.A_diagonal_best_var = tf.Variable(self.qA_diagonal)
        self.A_corr_best_var = tf.Variable(self.qA_corr)
        self.gamma_e_best_var = tf.Variable(self.qgamma_e)
        self.nu_e_best_var = tf.Variable(self.qnu_e)
        self.checkpoint_best_op = tf.group(tf.assign(self.pis_best_var, self.qpis),
                                           tf.assign(self.musX_best_var, self.qmusX),
                                           tf.assign(self.A_diagonal_best_var, self.qA_diagonal),
                                           tf.assign(self.A_corr_best_var, self.qA_corr),
                                           tf.assign(self.gamma_e_best_var, self.qgamma_e),
                                           tf.assign(self.nu_e_best_var, self.qnu_e))

        self.res = tf.fake_quant_with_min_max_args(self.res, min=0, max=1, num_bits=self.precision)
        #mse = tf.reduce_mean(tf.square(tf.round(self.res * 255) / 255 - self.target_op))

        if self.dim_domain >= 4:
            diff = tf.boolean_mask(self.res - self.target_op, self.train_mask)
        else:
            diff = self.res - self.target_op
        squared_diff = tf.square(diff)
        mse = tf.reduce_mean(squared_diff)
        err_map = tf.reduce_mean(squared_diff, axis=1)
        self.sampl_prob = err_map / tf.reduce_sum(err_map)

        if not self.ssim_opt:
            # margin in pixel to determine epsilon
            epsilon = self.margin * 1 / (2 ** self.precision)
            loss_pixel = tf.maximum(0., tf.square(tf.subtract(tf.abs(diff), epsilon)))
            if self.use_yuv:
                loss_pixel = 6/8 * tf.reduce_mean(loss_pixel[:, 0]) + 1/8 * tf.reduce_sum(tf.reduce_mean(loss_pixel[:, 1::],
                                                                                                         axis=0))
            else:
                loss_pixel = tf.reduce_mean(loss_pixel)
        else:
            if self.use_yuv:
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

            else:
                res = tf.reshape(self.res, self.batch_shape[:-1] + (num_channels,))
                self.target_op = tf.reshape(self.target_op, self.batch_shape[:-1] + (num_channels,))

                if self.dim_domain == 2:
                    paddings = tf.constant([[5, 5], [5, 5], [0, 0]])
                    res = tf.pad(res, paddings, "SYMMETRIC")
                    self.target_op = tf.pad(self.target_op, paddings, "SYMMETRIC")

                    loss_pixel = 1 - tf.image.ssim(res, self.target_op, max_val=1)
                elif self.dim_domain == 3:
                    ssim = []
                    paddings = tf.constant([[5, 5], [5, 5], [0, 0], [0, 0]])
                    res = tf.pad(res, paddings, "SYMMETRIC")
                    self.target_op = tf.pad(self.target_op, paddings, "SYMMETRIC")
                    for ii in range(self.batch_shape[2]):
                        ssim.append(tf.image.ssim(res[:, :, ii, :], self.target_op[:, :, ii, :], max_val=1))
                    ssim = tf.reduce_mean(ssim)
                    loss_pixel = 1 - ssim

        self.num_pi_op = tf.count_nonzero(pis_mask)

        self.pis_l1 = tf.placeholder(tf.float32)
        self.u_l1 = tf.placeholder(tf.float32)
        pis_l1 = self.pis_l1 * tf.reduce_sum(pis) / self.start_pis

        # TODO work in progess
        #rxx_det = 1/tf.reduce_prod(tf.matrix_diag_part(A), axis=-1)**2
        #u_l1 = self.u_l1 * tf.reduce_sum(tf.reduce_prod(tf.matrix_diag_part(A), axis=-1)**2)
        #u_l1 = self.u_l1 * tf.reduce_sum(tf.matrix_diag_part(A) ** 2)
        #u_l1 = self.u_l1 * tf.reduce_sum(rxx_det) #/ self.start_pis # * (tf.cast(self.num_pi_op, tf.float32) / self.start_pis)
        #u_l1 = self.u_l1 * tf.reduce_sum(tf.pow(tf.matrix_diag_part(A)-40, 2))
        u_l1 = self.u_l1 * tf.reduce_sum(tf.matrix_diag_part(A)) # * (tf.cast(self.num_pi_op, tf.float32) / self.start_pis)
        #'''

        self.loss_op = loss_pixel + pis_l1 + u_l1

        self.mse_op = mse * ((2**self.precision) ** 2)

        self.res = tf.reshape(self.res, self.batch_shape[:-1] + (num_channels,))

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

    def set_optimizer(self, optimizer1, optimizer2=None, optimizer3=None, grad_clip_value_abs=None):
        self.optimizer1 = optimizer1

        if optimizer2 is None:
            self.optimizer2 = optimizer1
        else:
            self.optimizer2 = optimizer2

        if optimizer3 is None:
            self.optimizer3 = optimizer1
        else:
            self.optimizer3 = optimizer3

        var_opt1 = [self.nu_e_var, self.gamma_e_var, self.musX_var]
        var_opt2 = [self.pis_var]
        var_opt3 = [self.A_diagonal_var, self.A_corr_var]

        # sort out not trainable vars
        self.var_opt1 = [var for var in var_opt1 if var in tf.trainable_variables()]
        self.var_opt2 = [var for var in var_opt2 if var in tf.trainable_variables()]
        self.var_opt3 = [var for var in var_opt3 if var in tf.trainable_variables()]

        accum_gradients = [tf.Variable(tf.zeros_like(var.initialized_value()), trainable=False)
                           for var in self.var_opt1 + self.var_opt2 + self.var_opt3]
        self.zero_op = [grad.assign(tf.zeros_like(grad)) for grad in accum_gradients]
        gradients = tf.gradients(self.loss_op, self.var_opt1 + self.var_opt2 + self.var_opt3)
        self.accum_ops = [accum_gradients[i].assign_add(gv) for i, gv in enumerate(gradients)]

        if grad_clip_value_abs is not None:
            accum_gradients = [tf.clip_by_value(g, -grad_clip_value_abs, grad_clip_value_abs) for g in accum_gradients]

        gradients1 = accum_gradients[:len(self.var_opt1)]
        gradients2 = accum_gradients[len(self.var_opt1):len(self.var_opt1) + len(self.var_opt2)]
        gradients3 = accum_gradients[len(self.var_opt1) + len(self.var_opt2):]

        train_op1 = self.optimizer1.apply_gradients(zip(gradients1, self.var_opt1))
        if len(var_opt2) > 0:
            train_op2 = self.optimizer2.apply_gradients(zip(gradients2, self.var_opt2))
        else:
            train_op2 = tf.no_op()
        train_op3 = self.optimizer3.apply_gradients(zip(gradients3, self.var_opt3))
        self.train_op = tf.group(train_op1, train_op2, train_op3)

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

        if self.ssim_opt:
            diff = np.average(1 - compare_ssim(self.image, rec, data_range=1, multichannel=True, full=True)[1], axis=-1, weights=weights)
        else:
            diff = np.average(np.power(255 * (self.image - rec), 2), axis=-1, weights=weights)

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
        peaks = peak_local_max(blurred, num_peaks=self.num_inc_kernels)
        blurred_mean = np.mean(blurred)
        blurred_median = np.median(blurred)

        a = 1 / (sigma / self.image.shape[0]) * 2/1

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
        musX_inc = peaks / self.image.shape[0:-1]
        curr_pis = self.get_params()['pis']
        pi_mean = np.mean(curr_pis[curr_pis > 0])
        pi_median = np.median(curr_pis[curr_pis > 0])
        #pis_inc = np.ones_like(self.pis_init) * pi_mean
        pis_inc = np.ones_like(self.pis_init) * pi_median
        # pis_inc = self.pis_init
        gamma_e_inc = np.zeros_like(self.gamma_e_init)
        nu_e_inc = np.ones_like(self.nu_e_init) * 0.5

        A_diagonal_inc = np.zeros_like(self.A_init)
        A_corr_inc = np.zeros_like(self.A_init)
        # a = 2 * (self.kernel_count + 1)
        A_diagonal_inc[:, 0, 0] = a
        A_diagonal_inc[:, 1, 1] = a
        if self.radial_as:
            A_diagonal_inc = A_diagonal_inc[:, 0, 0]

        # fig = plt.figure()
        # plt.imshow(blurred, vmin=-1, vmax=1, cmap='gray')
        # plt.scatter(peaks[:, 1], peaks[:, 0])
        # plt.savefig("inc_{}.png".format(self.iter))
        # plt.close(fig)

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
        self.session.run([self.assign_inc_opt_vars_op], feed_dict={self.insert_pos: self.kernel_count})
        self.session.run([self.assign_inc_vars_op], feed_dict={self.insert_pos: self.kernel_count})
        self.session.run(self.reset_optimizers_op)
        self.kernel_count += self.num_inc_kernels

    def train(self, num_iter, val_iter=100, ukl_iter=None, optimizer1=None, optimizer2=None, optimizer3=None, grad_clip_value_abs=None, pis_l1=0,
              u_l1=0, sampling_percentage=100, callbacks=(), with_inc=False, train_inc=False, train_orig=True):
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
            self.best_qloss, self.best_qmse, _ = self.run_batched(pis_l1=pis_l1, u_l1=u_l1, train=False, update_reconstruction=True,
                                       with_quantized_params=True, with_inc=with_inc, train_inc=False)
            self.qlosses.append((0, self.best_qloss))
            self.qmses.append((0, self.best_qmse))

        self.best_loss, self.best_mse, num_pi = self.run_batched(pis_l1=pis_l1, u_l1=u_l1, train=False,
                                                                 update_reconstruction=True, with_inc=with_inc, train_inc=False)


        self.losses.append((self.iter, self.best_loss))
        self.mses.append((self.iter, self.best_mse))
        self.num_pis.append((self.iter, num_pi))

        # run callbacks
        for callback in callbacks:
            callback(self)

        for i in range(1, num_iter + 1):
            self.iter += 1
            try:
                validate = i % val_iter == 0
                update_kernel_list = i % ukl_iter == 0

                loss_val, mse_val, num_pi = self.run_batched(pis_l1=pis_l1, u_l1=u_l1, train=train_orig,
                                                             update_reconstruction=False, sampling_percentage=sampling_percentage,
                                                             with_inc=with_inc, train_inc=train_inc)

                if update_kernel_list:
                    self.update_kernel_list(self.add_kernel_slots)

                if validate:
                    if self.quantization_mode >= 1:
                        self.qparams = quantize_params(self, self.get_params())
                    if self.quantization_mode == 1:
                        self.rparams = rescaler(self, self.qparams)
                        qloss_val, qmse_val, _ = self.run_batched(pis_l1=pis_l1,
                                                                  u_l1=u_l1, train=False, update_reconstruction=True,
                                                                  with_quantized_params=True, with_inc=with_inc, train_inc=False)
                    loss_val, mse_val, num_pi = self.run_batched(pis_l1=pis_l1, u_l1=u_l1, train=False,
                                                                 update_reconstruction=True, with_inc=with_inc, train_inc=False)
                    # run batched with quant params

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

    def run_batched(self, pis_l1=0, u_l1=0, train=True, update_reconstruction=False, with_quantized_params=False, sampling_percentage=100, with_inc=False, train_inc=False):
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
        reconstructions = []
        w_es = []

        widgets = [
            progressbar.AnimatedMarker(markers='◐◓◑◒'),
            ' Iteration: {0}  '.format(self.iter),
            progressbar.Percentage(),
            ' ', progressbar.Counter('%(value)d/{0}'.format(self.start_batches)),
            ' ', progressbar.Bar('>'),
            ' ', progressbar.Timer(),
            ' ', progressbar.ETA()
        ]
        bar = progressbar.ProgressBar(widgets=widgets)
        for ii in bar(range(self.start_batches)):
            retrieve = [self.loss_op, self.mse_op, self.num_pi_op]
            if not with_quantized_params:
                retrieve.append(self.indices)

            img_patch = self.joint_domain_batched[ii].reshape(-1, self.joint_domain_batched[ii].shape[-1])

            if train and self.dim_domain >= 4:
                img_patch = img_patch[self.train_mask]

            if train and not self.ssim_opt and sampling_percentage < 100:
                num_samples = np.uint32(np.round(img_patch.shape[0] * sampling_percentage/100))
                samples = np.random.choice(img_patch.shape[0], (num_samples,), replace=False, p=self.random_sampling_per_batch[ii])
            else:
                samples = np.arange(img_patch.shape[0])

            feed_dict = {self.joint_domain_batched_op: img_patch[samples, :], self.pis_l1: pis_l1, self.u_l1: u_l1,
                         self.kernel_list: self.kernel_list_per_batch[ii], self.stack_inc: [1.] if with_inc else [0.],
                         self.stack_orig: [1.]}
            if train:
                retrieve.append(self.accum_ops)
            if update_reconstruction:
                retrieve += [self.res, self.w_e_max_op]
                if with_quantized_params:
                    feed_dict.update({self.A: self.rparams["A"], self.musX: self.rparams["musX"], self.nu_e: self.rparams["nu_e"], self.gamma_e: self.rparams["gamma_e"], self.pis: self.rparams["pis"]})

            if train_inc:
                retrieve.append(self.accum_inc_ops)

            retrieve.append(self.sampl_prob)

            results = self.session.run(retrieve, feed_dict=feed_dict)
            if update_reconstruction:
                if train:
                    reconstructions.append(results[5])
                    w_es.append(results[3][results[6]])
                else:
                    if with_quantized_params:
                        reconstructions.append(results[3])
                        w_es.append(results[4])
                    else:
                        reconstructions.append(results[4])
                        w_es.append(results[3][results[5]])

            loss_val += results[0] * np.prod(self.joint_domain_batched.shape[1:-1]) / self.num_pixel
            mse_val += results[1] * np.prod(self.joint_domain_batched.shape[1:-1]) / self.num_pixel
            num_pi = results[2]
            if not with_quantized_params:
                bool_mask = np.zeros_like(self.kernel_list_per_batch[ii])
                bool_mask[results[3]] = True
                self.kernel_list_per_batch[ii] = bool_mask

            if update_reconstruction:
                self.random_sampling_per_batch[ii] = results[-1]

        if update_reconstruction:
            reconstruction = np.stack(reconstructions)
            w_e = np.stack(w_es)
            if not with_quantized_params:
                self.reconstruction_image = self.uncubify(reconstruction, self.image.shape)
                self.weight_matrix_argmax = self.uncubify(w_e, self.image.shape[0:self.image.ndim-1])
                self.valid = True
            else:
                self.qreconstruction_image = self.uncubify(reconstruction, self.image.shape)
                self.qweight_matrix_argmax = self.uncubify(w_e, self.image.shape[0:self.image.ndim - 1])
                self.qvalid = True

        if train:
            self.session.run(self.train_op)
        if train_inc:
            self.session.run(self.train_inc_op)


        return loss_val, mse_val, num_pi

    def get_params(self):
        pis, musX, A_diagonal, A_corr, gamma_e, nu_e = self.session.run([self.qpis, self.qmusX,
                                                                         self.qA_diagonal, self.qA_corr,
                                                                         self.qgamma_e, self.qnu_e])

        out_dict = {'pis': pis, 'musX': musX, 'A_diagonal': A_diagonal, 'A_corr': A_corr, 'gamma_e': gamma_e, 'nu_e': nu_e}
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

    def get_best_params(self):
        pis, musX, A_diagonal, A_corr, gamma_e, nu_e = self.session.run([self.pis_best_var, self.musX_best_var,
                                                                         self.A_diagonal_best_var, self.A_corr_best_var,
                                                                         self.gamma_e_best_var, self.nu_e_best_var])

        out_dict = {'pis': pis, 'musX': musX, 'A_diagonal': A_diagonal, 'A_corr': A_corr, 'gamma_e': gamma_e, 'nu_e': nu_e}
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

    def get_original_image(self):
        return np.squeeze(self.image)

    def init_domain_and_target(self):
        joint_domain = self.gen_domain(self.image, self.dim_domain)
        self.joint_domain_batched = joint_domain
        self.batch_shape = self.get_batch_shape(self.start_batches, joint_domain.shape)
        self.joint_domain_batched = self.cubify(self.joint_domain_batched, self.batch_shape)

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
            self.pis_init = np.ones((number,), dtype=np.float32) / number
        else:
            self.pis_init = np.ones((number,), dtype=np.float32)

    def initialize_kernel_list(self, add_kernel_slots=0):
        num_of_all_kernels = self.start_pis
        if add_kernel_slots > 0:
            num_of_all_kernels = 2*num_of_all_kernels + add_kernel_slots

        batch_center = np.mean(self.joint_domain_batched, axis=tuple(np.arange(self.dim_domain) + 1))

        maha_dists = []
        for k in range(self.joint_domain_batched.shape[0]):
            results = self.session.run([self.maha_dist], feed_dict={self.joint_domain_batched_op:  np.expand_dims(batch_center[k], axis=0),
                                                                 self.kernel_list: np.ones((num_of_all_kernels,),
                                                                                           dtype=bool),
                                                                    self.stack_inc: [0.],
                                                                    self.stack_orig: [1.]})
            maha_dists.append(results[0])
        maha_dists = np.concatenate(maha_dists, axis=1)
        maha_dist_min_ind = np.argmin(maha_dists, axis=1)
        self.kernel_list_per_batch = []
        for k in range(self.joint_domain_batched.shape[0]):
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

        mins = np.min(self.joint_domain_batched, axis=tuple(np.arange(self.dim_domain) + 1))
        maxs = np.max(self.joint_domain_batched, axis=tuple(np.arange(self.dim_domain) + 1))
        mins = mins[:, 0:self.dim_domain]
        maxs = maxs[:, 0:self.dim_domain]
        # get all corner points of (hyper-)cube and middle points of edges
        tt = np.concatenate((np.expand_dims(mins, axis=-1), np.expand_dims(maxs, axis=-1), np.expand_dims((mins + maxs)/2, axis=-1)), axis=-1)
        for k in range(self.joint_domain_batched.shape[0]):
            edges_batch = np.array(list(product(*tt[k, :, :])))
            edges_batch = np.concatenate((edges_batch, np.zeros((edges_batch.shape[0], self.image.shape[-1]))), axis=1)
            results = self.session.run([self.maha_dist_ind], feed_dict={self.joint_domain_batched_op: edges_batch,
                                                                  self.kernel_list: np.ones((num_of_all_kernels,),
                                                                                            dtype=bool),
                                                                    self.stack_inc: [1.],
                                                                    self.stack_orig: [1.]})
            #bool_mask = np.zeros((self.start_pis,), dtype=bool)
            bool_mask = np.zeros((num_of_all_kernels,), dtype=bool)
            bool_mask[results[0]] = True
            self.kernel_list_per_batch[k] = np.logical_or(self.kernel_list_per_batch[k], bool_mask)

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
