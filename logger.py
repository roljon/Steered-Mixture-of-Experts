import os
import numpy as np
import matplotlib.pyplot as plt

from utils import save_model


class ModelLogger:
    def __init__(self, path, png=False):
        self.params_path = path + "/params"
        self.reconstruction_path = path + "/reconstructions"
        self.checkpoints_path = path + "/checkpoints"
        self.png = png

        if not os.path.exists(self.params_path):
            os.mkdir(self.params_path)

        if not os.path.exists(self.reconstruction_path):
            os.mkdir(self.reconstruction_path)

        if not os.path.exists(self.checkpoints_path):
            os.mkdir(self.checkpoints_path)

    def log(self, smoe, checkpoint_iter=100):
        iter_ = smoe.get_iter()
        reconstruction = smoe.get_reconstruction()

        save_model(smoe, self.params_path + "/{0:08d}_params.pkl".format(iter_), best=False, reduce=True)

        if self.png:
            plt.imsave(self.reconstruction_path + "/{0:08d}_reconstruction.png".format(iter_), reconstruction, cmap='gray', vmin=0, vmax=1)
        else:
            np.save(self.reconstruction_path + "/{0:08d}_reconstruction.npy".format(iter_), reconstruction)

        if iter_ % checkpoint_iter == 0:
            smoe.checkpoint(self.checkpoints_path + "/{0:08d}_model.ckpt".format(iter_))
