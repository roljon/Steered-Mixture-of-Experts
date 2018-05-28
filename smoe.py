import numpy as np
import math
import tensorflow as tf
from tensorflow.python.ops.special_math_ops import _exponential_space_einsum as einsum

class Smoe:
    def __init__(self, image, kernels_per_dim=None, train_pis=True, init_params=None, start_batches=1, train_gammas=True, radial_as=False, iter_offset=0, margin=0.5):
        self.domain = None

        # init params
        self.pis_init = None
        self.musX_init = None
        self.A_init = None
        self.gamma_e_init = None
        self.nu_e_init = None

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
        self.pis_l1 = None
        self.u_l1 = None
        self.zero_op = None
        self.accum_ops = None
        self.start = None
        self.end = None
        self.num_pi_op = None
        self.save_op = None

        # optimizers
        self.optimizer1 = None
        self.optimizer2 = None
        self.optimizer3 = None

        # others
        # TODO refactor to logger class
        self.losses = []
        self.losses_history = []
        self.best_loss = None
        self.mses = []
        self.mses_history = []
        self.best_mse = []
        self.num_pis = []

        self.iter = iter_offset
        self.valid = False
        self.reconstruction_image = None
        self.weight_matrix_argmax = None

        # generate initializations
        self.image = image
        self.dim_domain = image.ndim - 1
        self.num_pixel = np.prod(image.shape[0:self.dim_domain])
        self.init_domain_and_target()
        self.intervals = self.calc_intervals(self.num_pixel, start_batches)
        self.batches = start_batches
        self.start_batches = start_batches

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
            self.generate_pis(kernels_per_dim)

        self.start_pis = self.pis_init.size
        self.margin = margin

        gpu_options = tf.GPUOptions(allow_growth=True)
        self.session = tf.InteractiveSession(config=tf.ConfigProto(gpu_options=gpu_options))
        # self.session = tf.Session()

        self.init_model(self.domain, self.nu_e_init, self.gamma_e_init, self.pis_init, self.musX_init, self.A_init,
                        train_pis, train_gammas, radial_as)

    def init_model(self, domain_init, nu_e_init, gamma_e_init, pis_init, musX_init, A_init, train_pis=True, train_gammas=True, radial_as=False):

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

        num_channels = gamma_e_init.shape[-1]

        self.target_op = tf.constant(self.image_flat, dtype=tf.float32)
        self.domain_op = tf.constant(domain_init, dtype=tf.float32)

        self.start = tf.placeholder(dtype=tf.int32)
        self.end = tf.placeholder(dtype=tf.int32)
        # assume output channels of target are in the inner dimension
        self.target_op = self.target_op[self.start*num_channels:self.end*num_channels]
        self.domain_op = self.domain_op[self.start:self.end]

        if radial_as:
           A_mask = np.ones(shape=(A_init.shape[0], 2, 2), dtype=np.float32)
           A_mask[:, 0, 1] = 0
           A_mask[:, 1, 0] = 0
           A = tf.tile(tf.expand_dims(tf.expand_dims(self.A_var, axis=-1), axis=-1), (1, 2, 2))
           #print(A_mask.shape)
           #print(A.shape)
           A = A * A_mask
           #self.a_test = A
        else:
           A = self.A_var
        #A = self.A_var

        musX = tf.expand_dims(self.musX_var, axis=1)

        pis_mask = self.pis_var > 0
        
        musX = tf.boolean_mask(musX, pis_mask)
        nu_e = tf.boolean_mask(self.nu_e_var, pis_mask)
        gamma_e = tf.boolean_mask(self.gamma_e_var, pis_mask)
        A = tf.boolean_mask(A, pis_mask)
        pis = tf.boolean_mask(self.pis_var, pis_mask)


        n_div = tf.reduce_prod(tf.matrix_diag_part(A), axis=-1)
        p = self.image.ndim-1
        n_dis = np.sqrt(np.power(2*np.pi, p))
        n_quo = n_div / n_dis

        # prepare domain
        domain_exp = self.domain_op
        domain_exp = tf.tile(tf.expand_dims(domain_exp, axis=0), (tf.shape(musX)[0], 1, 1))

        x_sub_mu = tf.expand_dims(domain_exp - musX, axis=-1)
        n_exp = tf.exp(-0.5 * einsum('abli,alm,anm,abnj->ab', x_sub_mu, A, A, x_sub_mu))

        N = tf.tile(tf.expand_dims(n_quo, axis=1), (1, tf.shape(n_exp)[1])) * n_exp

        n_w = N * tf.expand_dims(pis, axis=-1)
        n_w_norm = tf.reduce_sum(n_w, axis=0)
        n_w_norm = tf.maximum(10e-12, n_w_norm)

        self.w_e_op = n_w / n_w_norm
        self.w_e_max_op = tf.argmax(self.w_e_op, axis=0)

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
        #tf.round(tf.reshape(self.res, [-1]) * 255) / 255
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

                # TODO this should be refactored into samples_per_batch or removed
                self.batches = math.ceil(self.start_batches * (num_pi / self.start_pis))
                # print("{0} -> {1} batches".format(self.start_batches, self.batches))
                self.intervals = self.calc_intervals(self.num_pixel, self.batches)

                loss_val, mse_val, num_pi = self.run_batched(pis_l1=pis_l1, u_l1=u_l1, train=True,
                                                             update_reconstruction=validate)

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

    def run_batched(self, pis_l1=0, u_l1=0, train=True, update_reconstruction=False):
        self.valid = False

        if train:
            self.session.run(self.zero_op)

        loss_val = 0
        mse_val = 0
        num_pi = -1
        # only for update update_reconstruction=True
        reconstructions = []
        w_es = []

        for start, end in self.intervals:
            retrieve = [self.loss_op, self.mse_op, self.num_pi_op]
            if train:
                retrieve.append(self.accum_ops)
            if update_reconstruction:
                retrieve += [self.res, self.w_e_max_op]

            results = self.session.run(retrieve,
                                       feed_dict={self.start: start,
                                                  self.end: end,
                                                  self.pis_l1: pis_l1,
                                                  self.u_l1: u_l1})

            if update_reconstruction:
                if train:
                    reconstructions.append(results[4])
                    w_es.append(results[5])
                else:
                    reconstructions.append(results[3])
                    w_es.append(results[4])

            loss_val += results[0] * (end - start) / self.num_pixel
            mse_val += results[1] * (end - start) / self.num_pixel
            num_pi = results[2]

        if update_reconstruction:
            reconstruction = np.concatenate(reconstructions)
            w_e = np.concatenate(w_es)
            self.reconstruction_image = reconstruction.reshape(self.image.shape)
            self.weight_matrix_argmax = w_e.reshape(self.image.shape[0:self.image.ndim-1])
            self.valid = True

        if train:
            self.session.run(self.train_op)

        return loss_val, mse_val, num_pi

    def get_params(self):
        pis, musX, A, gamma_e, nu_e = self.session.run([self.pis_var, self.musX_var, self.A_var,
                                                        self.gamma_e_var, self.nu_e_var])

        out_dict = {'pis': pis, 'musX': musX, 'A': A, 'gamma_e': gamma_e, 'nu_e': nu_e}
        return out_dict

    def get_gradients(self):
        raise NotImplementedError

    def get_reconstruction(self):
        if not self.valid:
            self.run_batched(train=False, update_reconstruction=True)
        return np.squeeze(self.reconstruction_image)

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

    def get_best_loss(self):
        return self.best_loss

    def get_losses_history(self):
        return self.losses_history

    def get_mses(self):
        return self.mses

    def get_best_mse(self):
        return self.best_mse

    def get_mses_history(self):
        return self.mses_history

    def get_num_pis(self):
        return self.num_pis

    def get_original_image(self):
        return np.squeeze(self.image)

    def init_domain(self):
        self.domain = self.gen_domain(self.image, self.image.ndim-1)

    def init_domain_and_target(self):
        dim_of_domain = self.image.ndim - 1
        joint_domain = self.gen_domain(self.image, self.image.ndim-1)
        joint_domain = np.reshape(joint_domain, (self.num_pixel, joint_domain.shape[-1]))
        self.domain = joint_domain[:, :dim_of_domain]
        self.image_flat = joint_domain[:, dim_of_domain:].flatten()

    def get_iter(self):
        return self.iter

    # quadratic for 2d in [0,1]
    def generate_kernel_grid(self, kernels_per_dim):
        dim_of_domain = self.image.ndim-1
        self.musX_init = self.gen_domain(kernels_per_dim, dim_of_domain)

        A_prototype = np.zeros((dim_of_domain, dim_of_domain))
        np.fill_diagonal(A_prototype, 2 * (kernels_per_dim + 1))
        self.A_init = np.tile(A_prototype, (kernels_per_dim ** dim_of_domain, 1, 1))

    def generate_experts(self, with_means=True):
        assert self.musX_init is not None, "need musX to generate experts"

        num_channels = self.image.shape[-1]

        self.gamma_e_init = np.zeros((self.musX_init.shape[0], self.dim_domain, num_channels))

        # TODO generalize initialization of nu_e's for arbitrary dim of input space
        if with_means:
            if self.dim_domain == 2:
                # assumes that muX_init are in a square grid
                stride = self.musX_init[0, 0]
                height, width = self.image.shape[0], self.image.shape[1]
                mean = np.empty((self.musX_init.shape[0], num_channels), dtype=np.float32)
                for k, (y, x) in enumerate(zip(*self.musX_init.T)):
                    x0 = int(round((x - stride) * width))
                    x1 = int(round((x + stride) * width))
                    y0 = int(round((y - stride) * height))
                    y1 = int(round((y + stride) * height))

                    mean[k] = np.mean(self.image[y0:y1, x0:x1], axis=(0, 1))
            elif self.dim_domain == 3:
                stride = self.musX_init[0, 0]
                height, width, frames = self.image.shape[0], self.image.shape[1], self.image.shape[2]
                mean = np.empty((self.musX_init.shape[0], num_channels), dtype=np.float32)
                for k, (y, x, z) in enumerate(zip(*self.musX_init.T)):
                    x0 = int(round((x - stride) * width))
                    x1 = int(round((x + stride) * width))
                    y0 = int(round((y - stride) * height))
                    y1 = int(round((y + stride) * height))
                    z0 = int(round((z - stride) * frames))
                    z1 = int(round((z + stride) * frames))

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

    def generate_pis(self, grid_size):
        number = grid_size ** self.musX_init.shape[1]
        self.pis_init = np.ones((number,), dtype=np.float32) / number

    @staticmethod
    def gen_domain(in_, dim_of_input_space=2):
        num_per_dim = np.zeros((dim_of_input_space,), dtype=np.int32)
        if type(in_) is np.ndarray:
            for ii in range(dim_of_input_space):
                num_per_dim[ii] = in_.shape[ii]
        else:
            for ii in range(dim_of_input_space):
                num_per_dim[ii] = in_

        # create coordinates for each dimension
        coord = []
        for ii in range(dim_of_input_space):
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
