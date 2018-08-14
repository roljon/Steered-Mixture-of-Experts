import numpy as np
import math
import tensorflow as tf
from tensorflow.python.ops.special_math_ops import _exponential_space_einsum as einsum
import progressbar
from itertools import product
from quantizer import quantize_params, rescaler

class Smoe:
    def __init__(self, image, kernels_per_dim=None, train_pis=True, init_params=None, start_batches=1,
                 train_gammas=True, radial_as=False, use_determinant=True, normalize_pis=True, quantization_mode=0,
                 bit_depths=None, quantize_pis=True, lower_bounds=None, upper_bounds=None, iter_offset=0, margin=0.5):
        self.batch_shape = None

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
        self.radial_as = radial_as
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
        self.margin = margin
        self.kernel_list_per_batch = [np.ones((self.start_pis,), dtype=bool)] * self.start_batches

        gpu_options = tf.GPUOptions(allow_growth=True)
        self.session = tf.InteractiveSession(config=tf.ConfigProto(gpu_options=gpu_options))
        # self.session = tf.Session()

        self.init_model(self.joint_domain_batched, self.nu_e_init, self.gamma_e_init, self.pis_init, self.musX_init, self.A_init,
                        train_pis, train_gammas, radial_as, use_determinant)

    def init_model(self, joint_domain_batched, nu_e_init, gamma_e_init, pis_init, musX_init, A_init, train_pis=True, train_gammas=True, radial_as=False, use_determinant=True):

        self.nu_e_var = tf.Variable(nu_e_init, dtype=tf.float32)
        self.gamma_e_var = tf.Variable(gamma_e_init, trainable=train_gammas, dtype=tf.float32)
        self.musX_var = tf.Variable(musX_init, dtype=tf.float32)
        #self.A_var = tf.Variable(A_init, dtype=tf.float32)
        self.pis_var = tf.Variable(pis_init, trainable=train_pis, dtype=tf.float32)


        # TODO make radial work again, uncomment for pcs scripts
        if A_init.ndim == 1:
            radial_as = True

        if radial_as:
            if A_init.ndim == 1:
                self.A_var = tf.Variable(A_init, dtype=tf.float32)
            else:
                self.A_var = tf.Variable(A_init[:, 0, 0], dtype=tf.float32)
        else:
            self.A_var = tf.Variable(A_init, dtype=tf.float32)

        # Quantization of parameters (if requested)
        if self.quantization_mode == 2:
            self.qA = tf.fake_quant_with_min_max_args(self.A_var, min=self.lower_bounds[0],
                                                      max=self.upper_bounds[0], num_bits=self.bit_depths[0])
            self.qmusX = tf.fake_quant_with_min_max_args(self.musX_var,
                                                         min=self.lower_bounds[1], max=self.upper_bounds[1],
                                                         num_bits=self.bit_depths[1])
            self.qnu_e = tf.fake_quant_with_min_max_args(self.nu_e_var,
                                                         min=self.lower_bounds[2], max=self.upper_bounds[2],
                                                         num_bits=self.bit_depths[2])
            self.qgamma_e = tf.fake_quant_with_min_max_args(self.gamma_e_var,
                                                            min=self.lower_bounds[4], max=self.upper_bounds[4],
                                                            num_bits=self.bit_depths[4])
        else:
            self.qA = self.A_var
            self.qmusX = self.musX_var
            self.qnu_e = self.nu_e_var
            self.qgamma_e = self.gamma_e_var

        if self.quantization_mode == 2 or (self.quantization_mode == 1 and self.quantize_pis):
            self.qpis = tf.fake_quant_with_min_max_args(self.pis_var, min=self.lower_bounds[3],
                                                        max=self.upper_bounds[3], num_bits=self.bit_depths[3])
        else:
            self.qpis = self.pis_var

        num_channels = gamma_e_init.shape[-1]


        self.joint_domain_batched_op = tf.constant(joint_domain_batched, dtype=tf.float32)

        self.current_batch_number = tf.placeholder(dtype=tf.int32)

        self.joint_domain_batched_op = self.joint_domain_batched_op[self.current_batch_number]
        self.joint_domain_batched_op = tf.reshape(self.joint_domain_batched_op,
                                                  (tf.reduce_prod(self.batch_shape[:-1]),
                                                   tf.reduce_prod(self.batch_shape[-1])))
        self.target_op = self.joint_domain_batched_op[:, self.dim_domain:]
        self.target_op = tf.reshape(self.target_op, [-1])
        self.domain_op = self.joint_domain_batched_op[:, :self.dim_domain]

        self.kernel_list = tf.placeholder(shape=(None,), dtype=tf.bool)


        if radial_as:
           A_mask = np.zeros((self.dim_domain, self.dim_domain))
           np.fill_diagonal(A_mask, 1)
           A_mask = np.tile(A_mask, (A_init.shape[0], 1, 1))
           A = tf.tile(tf.expand_dims(tf.expand_dims(self.qA, axis=-1), axis=-1), (1, self.dim_domain, self.dim_domain))
           A = A * A_mask
        else:
           A = self.qA


        pis_mask = self.qpis > 0
        bool_mask = tf.logical_and(self.kernel_list, pis_mask)

        # track indices of used and necessary kernels
        self.indices = tf.constant(np.arange(self.start_pis), dtype=tf.int32)
        self.indices = tf.boolean_mask(self.indices, bool_mask)

        # using self-Variables to define feed point
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

        # use only triangular part
        A_lower = tf.matrix_band_part(A, -1, 0)
        n_exp = tf.exp(-0.5 * einsum('abli,alm,anm,abnj->ab', x_sub_mu, A_lower, A_lower, x_sub_mu))

        if use_determinant:
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
        self.w_e_max_op = tf.reshape(tf.argmax(self.w_e_op, axis=0), self.batch_shape[:-1])

        kernel_list_batch_op = tf.reduce_sum(tf.cast(tf.greater(self.w_e_op, 0), tf.int32), axis=1) > 0 # 10 ** -9
        self.indices = tf.boolean_mask(self.indices, kernel_list_batch_op)
        #self.indices = tf.Print(self.indices, [tf.count_nonzero(kernel_list_batch_op)])

        # TODO reorder nu_e and gamma_e to avoid unnecessary transpositions
        domain_tiled = tf.expand_dims(tf.transpose(self.domain_op), axis=0)
        domain_tiled = tf.tile(domain_tiled, (num_channels, 1, 1))
        sloped_out = tf.matmul(tf.transpose(gamma_e, perm=[2, 0, 1]), domain_tiled)
        nu_e = tf.expand_dims(tf.transpose(nu_e), axis=-1)
        self.res = tf.reduce_sum(self.w_e_op * (sloped_out + nu_e), axis=1)

        self.res = tf.minimum(tf.maximum(self.res, 0), 1)
        self.res = tf.transpose(self.res)

        # checkpoint op
        self.pis_best_var = tf.Variable(self.pis_var)
        self.musX_best_var = tf.Variable(self.musX_var)
        self.A_best_var = tf.Variable(self.A_var)
        self.gamma_e_best_var = tf.Variable(self.gamma_e_var)
        self.nu_e_best_var = tf.Variable(self.nu_e_var)
        self.checkpoint_best_op = tf.group(tf.assign(self.pis_best_var, self.pis_var),
                                           tf.assign(self.musX_best_var, self.musX_var),
                                           tf.assign(self.A_best_var, self.A_var),
                                           tf.assign(self.gamma_e_best_var, self.gamma_e_var),
                                           tf.assign(self.nu_e_best_var, self.nu_e_var))

        # mse = tf.reduce_sum(tf.square(self.restoration_op - target)) / tf.size(target, out_type=tf.float32)

        mse = tf.reduce_sum(tf.square(tf.round(tf.reshape(self.res, [-1]) * 255) / 255 - self.target_op)) / tf.cast(tf.size(self.target_op), dtype=tf.float32)
        # margin in pixel to determine epsilon
        epsilon = self.margin * 1/(2**8)
        loss_pixel = tf.reduce_mean(tf.maximum(0., tf.square(tf.subtract(tf.abs(tf.subtract(tf.reshape(self.res, [-1]), self.target_op)), epsilon))))

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

        self.mse_op = mse * (255 ** 2)

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
        var_opt3 = [self.A_var]

        # sort out not trainable vars
        var_opt1 = [var for var in var_opt1 if var in tf.trainable_variables()]
        var_opt2 = [var for var in var_opt2 if var in tf.trainable_variables()]
        var_opt3 = [var for var in var_opt3 if var in tf.trainable_variables()]

        accum_gradients = [tf.Variable(tf.zeros_like(var.initialized_value()), trainable=False)
                           for var in var_opt1 + var_opt2 + var_opt3]
        self.zero_op = [grad.assign(tf.zeros_like(grad)) for grad in accum_gradients]
        gradients = tf.gradients(self.loss_op, var_opt1 + var_opt2 + var_opt3)
        self.accum_ops = [accum_gradients[i].assign_add(gv) for i, gv in enumerate(gradients)]

        if grad_clip_value_abs is not None:
            accum_gradients = [tf.clip_by_value(g, -grad_clip_value_abs, grad_clip_value_abs) for g in accum_gradients]

        gradients1 = accum_gradients[:len(var_opt1)]
        gradients2 = accum_gradients[len(var_opt1):len(var_opt1) + len(var_opt2)]
        gradients3 = accum_gradients[len(var_opt1) + len(var_opt2):]

        train_op1 = self.optimizer1.apply_gradients(zip(gradients1, var_opt1))
        if len(var_opt2) > 0:
            train_op2 = self.optimizer2.apply_gradients(zip(gradients2, var_opt2))
        else:
            train_op2 = tf.no_op()
        train_op3 = self.optimizer3.apply_gradients(zip(gradients3, var_opt3))
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

    def train(self, num_iter, val_iter=100, optimizer1=None, optimizer2=None, optimizer3=None, grad_clip_value_abs=None, pis_l1=0,
              u_l1=0, callbacks=()):
        if optimizer1:
            self.set_optimizer(optimizer1, optimizer2, optimizer3, grad_clip_value_abs=grad_clip_value_abs)
        assert self.optimizer1 is not None, "no optimizer found, you have to specify one!"

        # TODO history nutzen oder weg
        # self.losses = []
        # self.mses = []
        # self.num_pis = []
        if self.quantization_mode == 1 or self.quantization_mode == 2:
            self.qparams = quantize_params(self, self.get_params())
        if self.quantization_mode == 1:
            self.rparams = rescaler(self, self.qparams)
            self.best_qloss, self.best_qmse, _ = self.run_batched(pis_l1=pis_l1, u_l1=u_l1, train=False, update_reconstruction=True,
                                       with_quantized_params=True)
            self.qlosses.append((0, self.best_qloss))
            self.qmses.append((0, self.best_qmse))

        self.best_loss, self.best_mse, num_pi = self.run_batched(pis_l1=pis_l1, u_l1=u_l1, train=False,
                                                                 update_reconstruction=True)


        self.losses.append((0, self.best_loss))
        self.mses.append((0, self.best_mse))
        self.num_pis.append((0, num_pi))

        # run callbacks
        for callback in callbacks:
            callback(self)

        for i in range(1, num_iter + 1):
            self.iter += 1
            try:
                validate = i % val_iter == 0

                loss_val, mse_val, num_pi = self.run_batched(pis_l1=pis_l1, u_l1=u_l1, train=True,
                                                             update_reconstruction=False)

                if validate:
                    if self.quantization_mode == 1 or self.quantization_mode == 2:
                        self.qparams = quantize_params(self, self.get_params())
                    if self.quantization_mode == 1:
                        self.rparams = rescaler(self, self.qparams)
                        self.kernel_list_per_batch = [np.ones((self.start_pis,), dtype=bool)] * self.start_batches
                        qloss_val, qmse_val, _ = self.run_batched(pis_l1=pis_l1,
                                                                  u_l1=u_l1, train=False, update_reconstruction=True,
                                                                  with_quantized_params=True)
                    loss_val, mse_val, num_pi = self.run_batched(pis_l1=pis_l1, u_l1=u_l1, train=False,
                                                                 update_reconstruction=True)
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
                    self.losses.append((i, loss_val))

                    # TODO take mses_history into account
                    if not self.best_mse or mse_val < self.best_mse:
                        self.best_mse = mse_val
                    self.mses.append((i, mse_val))

                    if self.quantization_mode == 1:
                        self.qmses.append((i, qmse_val))
                        self.qlosses.append((i, qloss_val))

                    self.num_pis.append((i, num_pi))

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

    def run_batched(self, pis_l1=0, u_l1=0, train=True, update_reconstruction=False, with_quantized_params=False):
        self.valid = False
        if with_quantized_params:
            self.qvalid = False

        if train:
            self.session.run(self.zero_op)

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
            retrieve = [self.loss_op, self.mse_op, self.num_pi_op, self.indices]
            feed_dict = {self.current_batch_number: ii, self.pis_l1: pis_l1, self.u_l1: u_l1, self.kernel_list: self.kernel_list_per_batch[ii]}
            if train:
                retrieve.append(self.accum_ops)
            if update_reconstruction:
                retrieve += [self.res, self.w_e_max_op]
                if with_quantized_params:
                    feed_dict.update({self.A: self.rparams["A"], self.musX: self.rparams["musX"], self.nu_e: self.rparams["nu_e"], self.gamma_e: self.rparams["gamma_e"], self.pis: self.rparams["pis"]})

            results = self.session.run(retrieve, feed_dict=feed_dict)
            if update_reconstruction:
                if train:
                    reconstructions.append(results[5])
                    w_es.append(results[6])
                else:
                    reconstructions.append(results[4])
                    #remapped_wes = self.remap_kernel_indices(results[5], results[3])
                    w_es.append(results[5])

            loss_val += results[0] * np.prod(self.joint_domain_batched.shape[1:-1]) / self.num_pixel
            mse_val += results[1] * np.prod(self.joint_domain_batched.shape[1:-1]) / self.num_pixel
            num_pi = results[2]
            if not with_quantized_params:
                bool_mask = np.zeros((self.start_pis,), dtype=bool)
                bool_mask[results[3]] = True
                self.kernel_list_per_batch[ii] = bool_mask

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

        return loss_val, mse_val, num_pi

    def get_params(self):
        pis, musX, A, gamma_e, nu_e = self.session.run([self.qpis, self.qmusX, self.qA,
                                                        self.qgamma_e, self.qnu_e])

        out_dict = {'pis': pis, 'musX': musX, 'A': A, 'gamma_e': gamma_e, 'nu_e': nu_e}
        return out_dict

    def get_gradients(self):
        raise NotImplementedError

    def get_reconstruction(self):
        if not self.valid:
            self.run_batched(train=False, update_reconstruction=True)
        return np.squeeze(self.reconstruction_image)

    def get_qreconstruction(self):
        if not self.qvalid:
            self.run_batched(train=False, update_reconstruction=True, with_quantized_params=True)
        return np.squeeze(self.qreconstruction_image)

    def get_weight_matrix_argmax(self):
        if not self.valid:
            self.run_batched(train=False, update_reconstruction=True)
        return self.weight_matrix_argmax

    def get_best_params(self):
        pis, musX, A, gamma_e, nu_e = self.session.run([self.pis_best_var, self.musX_best_var, self.A_best_var,
                                                        self.gamma_e_best_var, self.nu_e_best_var])

        out_dict = {'pis': pis, 'musX': musX, 'A': A, 'gamma_e': gamma_e, 'nu_e': nu_e}
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
                    x0 = int(round((x - stride[0]) * width))
                    x1 = int(round((x + stride[0]) * width))
                    y0 = int(round((y - stride[0]) * height))
                    y1 = int(round((y + stride[0]) * height))

                    mean[k] = np.mean(self.image[y0:y1, x0:x1], axis=(0, 1))
            elif self.dim_domain == 3:
                stride = self.musX_init[0]
                height, width, frames = self.image.shape[0], self.image.shape[1], self.image.shape[2]
                mean = np.empty((self.musX_init.shape[0], num_channels), dtype=np.float32)
                for k, (y, x, z) in enumerate(zip(*self.musX_init.T)):
                    x0 = int(round((x - stride[0]) * width))
                    x1 = int(round((x + stride[0]) * width))
                    y0 = int(round((y - stride[1]) * height))
                    y1 = int(round((y + stride[1]) * height))
                    z0 = int(round((z - stride[2]) * frames))
                    z1 = int(round((z + stride[2]) * frames))

                    mean[k] = np.mean(self.image[y0:y1, x0:x1, z0:z1], axis=(0, 1, 2))
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
            sum_of_dim_divisors[ii] = np.sum(divs[0:2])  # also possible np.sum(divs[0:2]) to make sure that spatial blocks are more divided

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

