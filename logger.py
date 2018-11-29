import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
import skvideo.io as skv
from utils import write_image

from utils import save_model


class ModelLogger:
    def __init__(self, path, as_media=False):
        self.params_path = path + "/params"
        self.reconstruction_path = path + "/reconstructions"
        self.checkpoints_path = path + "/checkpoints"
        self.as_media = as_media

        if not os.path.exists(self.params_path):
            os.mkdir(self.params_path)

        if not os.path.exists(self.reconstruction_path):
            os.mkdir(self.reconstruction_path)

        if not os.path.exists(self.checkpoints_path):
            os.mkdir(self.checkpoints_path)

    def log(self, smoe, checkpoint_iter=100):
        iter_ = smoe.get_iter()
        reconstruction = smoe.get_reconstruction()

        save_model(smoe, self.params_path + "/{0:08d}_params.pkl".format(iter_), best=False, reduce=True,
                   quantize=True if (smoe.quantization_mode >= 1) else False)

        if self.as_media:
            write_image(reconstruction, self.reconstruction_path + "/{0:08d}_reconstruction".format(iter_), smoe.dim_domain, smoe.use_yuv, smoe.precision)
            if smoe.quantization_mode == 1:
                qreconstruction = smoe.get_qreconstruction()
                write_image(qreconstruction, self.reconstruction_path + "/{0:08d}_qreconstruction".format(iter_), smoe.dim_domain, smoe.use_yuv, smoe.precision)
        else:
            np.save(self.reconstruction_path + "/{0:08d}_reconstruction.npy".format(iter_), reconstruction)
            if smoe.quantization_mode == 1:
                qreconstruction = smoe.get_qreconstruction()
                np.save(self.reconstruction_path + "/{0:08d}_qreconstruction.npy".format(iter_), qreconstruction)

        if iter_ % checkpoint_iter == 0:
            smoe.checkpoint(self.checkpoints_path + "/{0:08d}_model.ckpt".format(iter_))
