import numpy as np
import tensorflow as tf


class Smoe:
    def __init__(self, image, kernels_per_dim=None, train_pis=True, sqrt_pis=False, pis_l1=None, pis_relu=False, init_params=None):
        self.domain = None

        # init params
        self.pis_init = None
        self.musX_init = None
        self.U_init = None
        self.gamma_e_init = None
        self.nu_e_init = None

        # tf vars
        self.pis_var = None
        self.musX_var = None
        self.U_var = None
        self.gamma_e_var = None
        self.nu_e_var = None

        self.pis_best_var = None
        self.musX_best_var = None
        self.U_best_var = None
        self.gamma_e_best_var = None
        self.nu_e_best_var = None

        # tf ops
        self.restoration_op = None
        self.w_e_op = None
        self.loss_op = None
        self.train_op = None
        self.checkpoint_best_op = None
        self.gradients = None
        self.mse_op = None

        # optimizers
        self.optimizer1 = None
        self.optimizer2 = None

        # others
        # TODO refactor to logger class
        self.losses = []
        self.losses_history = []
        self.best_loss = None
        self.mses = []
        self.mses_history = []
        self.best_mse = []
        self.num_pis = []

        # generate initializations
        self.image = image
        self.init_domain()

        assert kernels_per_dim is not None or init_params is not None, \
            "You need to specify the kernel grid size or give initial parameters."

        if init_params:
            self.pis_init = init_params['pis']
            self.musX_init = init_params['musX']
            self.U_init = init_params['U']
            self.gamma_e_init = init_params['gamma_e']
            self.nu_e_init = init_params['nu_e'].T
        else:
            self.generate_kernel_grid(kernels_per_dim)
            self.generate_experts()
            self.generate_pis(kernels_per_dim, sqrt_pis)

        self.session = tf.Session()
        self.init_model(self.domain, self.nu_e_init, self.gamma_e_init, self.pis_init, self.musX_init, self.U_init,
                        train_pis, sqrt_pis, pis_l1, pis_relu)

    # TODO use self for init vars or refactor to a ModelParams class
    def init_model(self, domain_init, nu_e_init, gamma_e_init, pis_init, musX_init, U_init, train_pis=True,
                   sqrt_pis=False, pis_l1=None, pis_relu=False):
        domain = tf.constant(domain_init, dtype=tf.float32)
        domain_exp = tf.transpose(domain)
        domain_exp = tf.tile(tf.expand_dims(domain_exp, axis=0), (musX_init.shape[1], 1, 1))

        # self.pis_var = tf.Variable(pis_init, trainable=train_pis,dtype=tf.float32)
        if train_pis:
            self.pis_var = tf.Variable(pis_init, trainable=train_pis, dtype=tf.float32)
        else:
            self.pis_var = tf.constant(pis_init, dtype=tf.float32)

        pis = self.pis_var

        if pis_relu:
            pis = tf.nn.relu(pis)

        if sqrt_pis:
            pis **= 2

        self.musX_var = tf.Variable(musX_init, dtype=tf.float32)
        musX = tf.transpose(self.musX_var)
        musX = tf.expand_dims(musX, axis=1)

        U_mask_init = np.ones_like(U_init)
        U_mask_init[:, 0, 1] = 0

        U_mask = tf.constant(U_mask_init, tf.float32)
        self.U_var = tf.Variable(U_init, tf.float32)
        U = self.U_var * U_mask
        U = tf.maximum(10e-8, U)  # TODO Hotfix to prevent <= 0 values in diag(U) for log calculation for c

        self.nu_e_var = tf.Variable(nu_e_init.T, dtype=tf.float32)
        self.gamma_e_var = tf.Variable(gamma_e_init, dtype=tf.float32)

        target = tf.constant(self.image, dtype=tf.float32)

        X = domain_exp - musX
        Q = tf.linalg.triangular_solve(U, tf.transpose(X, perm=[0, 2, 1]))
        q = tf.reduce_sum(Q * Q, axis=1)
        d = domain_init.shape[0]
        c = d * tf.log(2 * np.pi) + 2 * tf.reduce_sum(tf.log(tf.matrix_diag_part(U)), axis=1)
        y = -(tf.expand_dims(c, axis=1) + q) / 2
        w_e = tf.exp(y)

        w_e *= tf.transpose(pis)
        w_dewnom = tf.reduce_sum(w_e, axis=0)
        self.w_e_op = (w_e * (1. / w_dewnom))

        res = tf.reduce_sum((tf.matmul(self.gamma_e_var, domain) + self.nu_e_var) * self.w_e_op, axis=0)
        res = tf.reshape(res, self.image.shape)
        self.restoration_op = tf.transpose(res)  # transpose only needed for compatibility and should be removed

        # checkpoint op
        self.pis_best_var = tf.Variable(self.pis_var)
        self.musX_best_var = tf.Variable(self.musX_var)
        self.U_best_var = tf.Variable(self.U_var)
        self.gamma_e_best_var = tf.Variable(self.gamma_e_var)
        self.nu_e_best_var = tf.Variable(self.nu_e_var)
        self.checkpoint_best_op = tf.group(tf.assign(self.pis_best_var, self.pis_var),
                                           tf.assign(self.musX_best_var, self.musX_var),
                                           tf.assign(self.U_best_var, self.U_var),
                                           tf.assign(self.gamma_e_best_var, self.gamma_e_var),
                                           tf.assign(self.nu_e_best_var, self.nu_e_var))

        mse = tf.reduce_sum(tf.square(self.restoration_op - target)) / tf.size(target, out_type=tf.float32)
        if pis_l1 is not None:
            self.loss_op = mse + pis_l1 * tf.reduce_sum(pis)
        else:
            self.loss_op = mse

        self.mse_op = mse * (255**2)

        init_new_vars_op = tf.global_variables_initializer()
        self.session.run(init_new_vars_op)

    def set_optimizer(self, optimizer1, optimizer2=None, grad_clip_value_abs=None):
        # optimizer = tf.train.GradientDescentOptimizer(0.00005)
        # optimizer = tf.train.MomentumOptimizer(0.000001, 0.1)
        # optimizer = tf.train.AdamOptimizer(0.00005, epsilon=0.01) #best 0.00005
        # optimizer = tf.train.AdamOptimizer(0.00005, beta1=0.05, beta2=0.1, epsilon=0.1)
        # optimizer = tf.train.AdamOptimizer(0.000001, beta1=0.1, beta2=0.9, epsilon=0.1)
        # optimizer = tf.train.AdamOptimizer(0.00005)

        """
        var_list1 = [pis_var]
        var_list2 = [musX_var,nu_e,gamma_e,U_var] #pis
        #opt1 = tf.train.GradientDescentOptimizer(0.0000001) #tf.train.AdamOptimizer(0.000005, beta1=0.05, beta2=0.1, epsilon=1)
        opt1 = tf.train.AdamOptimizer(0.000005, beta1=0.05, beta2=0.1, epsilon=0.1)
        opt2 = tf.train.AdamOptimizer(0.00005, beta1=0.05, beta2=0.1, epsilon=0.1)
        grads = tf.gradients(loss, var_list1 + var_list2)
        grads1 = grads[:len(var_list1)]
        grads2 = grads[len(var_list1):]
        train_op1 = opt1.apply_gradients(zip(grads1, var_list1))
        train_op2 = opt2.apply_gradients(zip(grads2, var_list2))
        train = tf.group(train_op1, train_op2)
        #"""
        # """

        self.optimizer1 = optimizer1
        if optimizer2 is not None:
            self.optimizer2 = optimizer2

        var_opt1 = [self.musX_var, self.nu_e_var, self.gamma_e_var, self.U_var]
        var_opt2 = [self.pis_var]

        # sort out not trainable vars
        var_opt1 = [var for var in var_opt1 if var in tf.trainable_variables()]
        var_opt2 = [var for var in var_opt2 if var in tf.trainable_variables()]

        self.gradients = tf.gradients(self.loss_op, var_opt1 + var_opt2)

        if grad_clip_value_abs is not None:
            # gradients, _ = tf.clip_by_global_norm(gradients, 1)
            # print(type(gradients), gradients)
            self.gradients = [tf.clip_by_value(g, -grad_clip_value_abs, grad_clip_value_abs) for g in self.gradients]

            # self.gradients = [tf.clip_by_norm(g, grad_clip_value_abs) for g in self.gradients]

        if self.optimizer2 is None or len(var_opt2) == 0:
            self.train_op = self.optimizer1.apply_gradients(zip(self.gradients, var_opt1 + var_opt2))
        else:
            gradients1 = self.gradients[:len(var_opt1)]
            gradients2 = self.gradients[len(var_opt1):]

            train_op1 = self.optimizer1.apply_gradients(zip(gradients1, var_opt1))
            train_op2 = self.optimizer2.apply_gradients(zip(gradients2, var_opt2))
            self.train_op = tf.group(train_op1, train_op2)

        uninitialized_vars = []
        for var in tf.global_variables():
            try:
                self.session.run(var)
            except tf.errors.FailedPreconditionError:
                if var is not None:
                    uninitialized_vars.append(var)

        init_new_vars_op = tf.variables_initializer(uninitialized_vars)
        self.session.run(init_new_vars_op)

        # print(self.session.run(tf.report_uninitialized_variables()))
        # print([v for v in tf.global_variables() if v.name.split(':')[0]])
        # print([v for v in tf.global_variables() if
        #      v.name.split(':')[0] in set(self.session.run(tf.report_uninitialized_variables()))
        #      ])
        # self.session.run(tf.variables_initializer(tf.report_uninitialized_variables()))

        # initialize optimizer vars, super hacky, probably does not work for all optimizers
        """
        if optimizer1 is not None:
            optimizer_vars = [self.optimizer1.get_slot(self.loss_op, name) for name in self.optimizer1.get_slot_names()
                              if self.optimizer1.get_slot(self.loss_op, name) is not None]
            print(optimizer_vars)
            try:
                optimizer_vars += list(self.optimizer1._get_beta_accumulators())
            except:
                pass
            print(optimizer_vars)
            self.session.run(tf.initialize_variables(optimizer_vars))

        if optimizer2 is not None:
            optimizer_vars = [self.optimizer2.get_slot(self.loss_op, name) for name in self.optimizer2.get_slot_names()
                              if self.optimizer2.get_slot(self.loss_op, name)]
            try:
                optimizer_vars += list(self.optimizer2._get_beta_accumulators())
            except:
                pass
            self.session.run(tf.initialize_variables(optimizer_vars))
        """

    def get_gradients(self):
        return self.session.run(self.gradients)

    def train(self, num_iter, val_iter=100, optimizer1=None, optimizer2=None, grad_clip_value_abs=None, callbacks=()):
        if optimizer1:
            self.set_optimizer(optimizer1, optimizer2, grad_clip_value_abs=grad_clip_value_abs)
        assert self.optimizer1 is not None, "no optimizer found, you have to specify one!"

        self.losses = []
        self.mses = []
        for i in range(num_iter):
            try:
                loss_val, mse_val, _ = self.session.run([self.loss_op, self.mse_op, self.train_op])

                # TODO take loss_history into account
                if np.isnan(loss_val) or (len(self.losses) > 0 and loss_val > self.losses[0][1]*10):
                    # self.session.run(
                    #     [self.loss_op, self.train_op, self.global_norm_op, self.global_norm1_op, self.global_norm2_op])
                    break

                if i % val_iter == 0:
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

                    params = self.get_params()
                    used = np.count_nonzero(params['pis'][0] > 0)
                    self.num_pis.append((i, used))

                    # run callbacks
                    for callback in callbacks:
                        callback(self)

                        # print("loss at iter %d: %f" % (i, session.run(loss)))
            except KeyboardInterrupt:
                break

        self.losses_history.append(self.losses)
        self.mses_history.append(self.mses)
        print("end loss/mse: ", loss_val, "/", mse_val,  "@iter: ", i)
        print("best loss/mse: ", self.best_loss, "/", self.best_mse)

    def get_params(self):
        pis, musX, U, gamma_e, nu_e = self.session.run([self.pis_var, self.musX_var, self.U_var,
                                                        self.gamma_e_var, self.nu_e_var])

        out_dict = {'pis': pis, 'musX': musX, 'U': U, 'gamma_e': gamma_e, 'nu_e': nu_e}
        return out_dict

    def get_reconstruction(self):
        return self.session.run(self.restoration_op)

    def get_weight_matrix(self):
        return self.session.run(self.w_e_op)

    def get_best_params(self):
        pis, musX, U, gamma_e, nu_e = self.session.run([self.pis_best_var, self.musX_best_var, self.U_best_var,
                                                        self.gamma_e_best_var, self.nu_e_best_var])

        out_dict = {'pis': pis, 'musX': musX, 'U': U, 'gamma_e': gamma_e, 'nu_e': nu_e}
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
        return self.image

    def init_domain(self):
        self.domain = self._gen_domain(self.image)

    # quadratic for 2d in [0,1]
    def generate_kernel_grid(self, kernels_per_dim):
        self.musX_init = self._gen_domain(kernels_per_dim)
        RsXX = np.zeros((2, 2, kernels_per_dim ** 2))

        for row in range(kernels_per_dim):
            for col in range(kernels_per_dim):
                sig_1 = 1 / (kernels_per_dim + 1) ** 2  # 1/((num_per_dim+1)**2)
                sig_2 = 1 / (kernels_per_dim + 1) ** 2  # 1/((num_per_dim+1)**2)
                roh = 0  # np.random.uniform(-0.1, 0.1)

                RsXX[:, :, row * kernels_per_dim + col] = np.array([[sig_1, roh * sig_1 * sig_2],
                                                                    [roh * sig_1 * sig_2, sig_2]])

        self.U_init = np.zeros(shape=np.roll(RsXX.shape, 1), dtype=np.float32)
        for k in range(self.U_init.shape[0]):
            self.U_init[k] = np.linalg.cholesky(RsXX[:, :, k])

    def generate_experts(self, with_means=True):
        assert self.musX_init is not None, "need musX to generate experts"
        # random expert slopes
        # self.gamma_e_init = np.random.uniform(-0.1, 0.1, size=(self.musX_init.shape[1], 2))

        self.gamma_e_init = np.zeros((self.musX_init.shape[1], 2))

        # choose nu_e to be 0.5 at center of kernel
        if with_means:
            # assumes that muX_init are in a square grid
            stride = self.musX_init[0, 0]
            height, width = self.image.shape
            mean = np.empty((self.musX_init.shape[1],), dtype=np.float32)
            for k, (x, y) in enumerate(zip(*self.musX_init)):
                x0 = int(round((x - stride) * width))
                x1 = int(round((x + stride) * width))
                y0 = int(round((y - stride) * height))
                y1 = int(round((y + stride) * height))
                # print(k, x0,x1, y0, y1, np.mean(self.image[y0:y1, x0:x1]))
                mean[k] = np.mean(self.image[y0:y1, x0:x1])
        else:
            mean = 0.5
        self.nu_e_init = mean - np.sum(self.gamma_e_init * self.musX_init.T, axis=1)
        self.nu_e_init = self.nu_e_init.reshape(1, self.musX_init.shape[1])
        # add a bit jitter
        # jitter = np.random.uniform(-0.05, 0.05, nu_e.shape)
        # print(nu_e)
        # nu_e += jitter
        # print(jitter)
        # print(nu_e)

    def generate_pis(self, grid_size, sqrt_pis=False):
        number = grid_size ** 2
        self.pis_init = np.ones((1, number), dtype=np.float32) / number
        if sqrt_pis:
            self.pis_init = np.sqrt(self.pis_init)
            # jitter_range = 0.01
            # self.pis = np.random.uniform(-jitter_range, jitter_range, size=(1, number)) / number

    @staticmethod
    def _gen_domain(in_):
        if type(in_) is np.ndarray:
            assert len(in_.shape) == 2, "only 2d images supported!"
            assert in_.shape[0] == in_.shape[1], "only quadratic images supported!"
            num_per_dim = in_.shape[0]
        else:
            num_per_dim = in_
        domain = np.zeros((2, num_per_dim ** 2))

        for row in range(num_per_dim):
            for col in range(num_per_dim):
                # equal spacing between domain positions and boarder
                domain[0, row * num_per_dim + col] = (1 / num_per_dim) / 2 + row * (1 / num_per_dim)
                domain[1, row * num_per_dim + col] = (1 / num_per_dim) / 2 + col * (1 / num_per_dim)

        return domain
