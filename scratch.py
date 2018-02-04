from tensorflow.python.client import timeline


# ga = tf.reduce_max(tf.gradients(self.loss_op, self.pis_var))
# gl = tf.reduce_max(tf.gradients(pis_l1, self.pis_var))
# pm = tf.reduce_max(self.pis_var)
# self.loss_op = tf.Print(self.loss_op, [ga, gl, pm, tf.reduce_any(tf.is_nan(self.pis_var))])
tmp = []  # [(self.pis_var, 'pis '), (self.musX_var, 'mu  '), (self.U_var, 'U   '), (self.nu_e_var, 'nu  '), (self.gamma_e_var, 'gamma')]
for op, name in tmp:
    g = tf.gradients(self.loss_op, op)
    self.loss_op = tf.Print(self.loss_op, [tf.reduce_max(g),
                                           tf.reduce_max(op),
                                           tf.count_nonzero(op > 0),
                                           tf.count_nonzero(tf.is_nan(op)),
                                           tf.count_nonzero(tf.is_nan(g))], name)

    # tp = self.U_var
    # self.loss_op = tf.Print(self.loss_op, [tf.gradients(self.loss_op, tp)])



    # builder = tf.profiler.ProfileOptionBuilder
    # opts = builder(builder.time_and_memory()).order_by('micros').build()
    # with tf.contrib.tfprof.ProfileContext('profile',
    #                                      trace_steps=[],
    #                                      dump_steps=[]) as pctx:
    #    # Enable tracing for next session.run.
    #    pctx.trace_next_step()
    #    # Dump the profile to '/tmp/train_dir' after the step.
    #    pctx.dump_next_step()
    # metadata = tf.RunMetadata()

    # '''
    if self.mini_idxs is not None:
        if 'done' in locals():
            break
        results = self.session.run(retrieve,
                                   feed_dict={self.mini_idxs: np.arange(self.image_flat.size),
                                              self.pis_l1: pis_l1,
                                              self.u_l1: u_l1})
        done = True

    else:
        results = self.session.run(retrieve,
                                   feed_dict={self.start: start,
                                              self.end: end,
                                              self.pis_l1: pis_l1,
                                              self.u_l1: u_l1})  # ,options=tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE,
    # output_partition_graphs=True),
    #             run_metadata=metadata)

    # timeline_ = timeline.Timeline(metadata.step_stats)
    # with open("dynamic_stitch_gpu_profile.json", "w") as f:
    #    f.write(timeline_.generate_chrome_trace_format())
    # with open("dynamic_stitch_gpu_profile.pbtxt", "w") as f:
    #    f.write(str(metadata))
    # exit()

    '''
    builder = tf.profiler.ProfileOptionBuilder
    opts = builder(builder.time_and_memory()).order_by('micros').build()
    with tf.contrib.tfprof.ProfileContext('profile',
                                         trace_steps=[],
                                         dump_steps=[]) as pctx:
       # Enable tracing for next session.run.
       pctx.trace_next_step()
       # Dump the profile to '/tmp/train_dir' after the step.
       pctx.dump_next_step()
    metadata = tf.RunMetadata()

    if self.mini_idxs is not None:
        if 'done' in locals():
            break
        results = self.session.run(retrieve,
                                   feed_dict={self.mini_idxs: np.arange(self.image_flat.size),
                                              self.pis_l1: pis_l1,
                                              self.u_l1: u_l1})
        done = True

    else:
        results = self.session.run(retrieve,
                                   feed_dict={self.start: start,
                                              self.end: end,
                                              self.pis_l1: pis_l1,
                                              self.u_l1: u_l1},options=tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE,
                 output_partition_graphs=True),
                 run_metadata=metadata)

    timeline_ = timeline.Timeline(metadata.step_stats)
    with open("dynamic_stitch_gpu_profile.json", "w") as f:
        f.write(timeline_.generate_chrome_trace_format())
    with open("dynamic_stitch_gpu_profile.pbtxt", "w") as f:
        f.write(str(metadata))
    exit()
    #'''

    #    pctx.profiler.profile_operations(options=opts)
    #    exit()






    # normalize pis to 1
    # with tf.control_dependencies([train_op]):
    # #pis_mask = self.pis_var > 0
    # #pis = tf.where(pis_mask, self.pis_var, tf.zeros_like(self.pis_var))
    #     pis = tf.nn.relu(tf.stop_gradient(self.pis_var))
    #     pis = tf.where(tf.is_nan(pis), tf.zeros_like(pis), pis)
    #     pis /= tf.reduce_sum(pis)
    #     pis = tf.where(tf.is_nan(pis), tf.zeros_like(pis), pis)
    #     assign_op = self.pis_var.assign(pis)
    #     # assign_op = tf.Print(assign_op, [tf.reduce_max(pis)])

    # self.train_op = train_op #tf.group(assign_op, self.train_op)

    # pis = tf.nn.relu(tf.stop_gradient(self.pis_var))
    # pis = tf.where(tf.is_nan(pis), tf.zeros_like(pis), pis)
    # pis /= tf.reduce_sum(pis)
    # pis = tf.where(tf.is_nan(pis), tf.zeros_like(pis), pis)
    # self.assign_op = self.pis_var.assign(pis)
    # self.assign_op = tf.Print(assign_op, [tf.reduce_sum(self.pis_var)])





    # TODO work in progess
    """
    # for  grad in gradients1:
    #    print (grad.shape)
    pis_relu = tf.nn.relu(tf.stop_gradient(self.pis_var))
    #pis_norm = pis_relu / tf.reduce_sum(pis_relu)
    # pis_norm = tf.expand_dims(pis_norm, axis=1)
    pis_norm = pis_relu / tf.cast(tf.count_nonzero(pis_relu), tf.float32)
    pis_norm = pis_norm#**2
    #pis_norm = tf.Print(pis_norm, [tf.reduce_min(pis_norm), tf.reduce_max(pis_norm)], "minmax")
    #pis_norm = tf.maximum(10e-8, pis_norm)

    pis_norm1 = tf.expand_dims(pis_norm, axis=-1)
    pis_norm2 = tf.expand_dims(pis_norm1, axis=-1)
    new_grads = []
    for grad in accum_gradients:
        if len(grad.shape) == 1:
            grad /= pis_norm
        elif len(grad.shape) == 2:
            grad /= pis_norm1
        elif len(grad.shape) == 3:
            grad /= pis_norm2
        else:
            raise ValueError
        new_grads.append(grad)
    accum_gradients = new_grads

    # gradients1 = [grad / pis_norm if len(grad.shape) == 2 else grad / tf.expand_dims(pis_norm, axis=-1) for grad
    #              in gradients1]
    # """












    """
    metadata = tf.RunMetadata()
    self.session.run(self.accum_ops, feed_dict={self.start: self.intervals[0][0],
                                                           self.end: self.intervals[0][1],
                                                           self.pis_l1: pis_l1 / self.batches},
                     options=tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE,
                     output_partition_graphs=True),
                     run_metadata=metadata)

    timeline_ = timeline.Timeline(metadata.step_stats)
    with open("dynamic_stitch_gpu_profile.json", "w") as f:
        f.write(timeline_.generate_chrome_trace_format())
    with open("dynamic_stitch_gpu_profile.pbtxt", "w") as f:
        f.write(str(metadata))
    exit()
    # """