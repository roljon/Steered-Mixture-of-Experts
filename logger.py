import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
import skvideo.io as skv

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
        if smoe.quantization_mode == 1:
            qreconstruction = smoe.get_qreconstruction()

        save_model(smoe, self.params_path + "/{0:08d}_params.pkl".format(iter_), best=False, reduce=True, quantize=True if smoe.quantization_mode == 1 else False)

        if self.as_media:
            if smoe.dim_domain == 2:
                plt.imsave(self.reconstruction_path + "/{0:08d}_reconstruction.png".format(iter_), reconstruction, cmap='gray', vmin=0, vmax=1)
                if smoe.quantization_mode == 1:
                    plt.imsave(self.reconstruction_path + "/{0:08d}_qreconstruction.png".format(iter_), qreconstruction, cmap='gray', vmin=0, vmax=1)
            elif smoe.dim_domain == 3:
                out = cv2.VideoWriter(self.reconstruction_path + "/{0:08d}_reconstruction.yuv".format(iter_), cv2.VideoWriter_fourcc(*'I420'), 25, (reconstruction.shape[0:2]))
                for ii in range(reconstruction.shape[2]):
                    frame = np.squeeze(reconstruction[:, :, ii, :])
                    frame = np.uint8(np.round(frame * 255))
                    out.write(frame)
                out.release()

                if smoe.quantization_mode == 1:
                    out = cv2.VideoWriter(self.reconstruction_path + "/{0:08d}_qreconstruction.yuv".format(iter_), cv2.VideoWriter_fourcc(*'I420'), 25, (qreconstruction.shape[0:2]))
                    for ii in range(qreconstruction.shape[2]):
                        frame = np.squeeze(qreconstruction[:, :, ii, :])
                        frame = np.uint8(np.round(frame * 255))
                        out.write(frame)
                    out.release()
        else:
            np.save(self.reconstruction_path + "/{0:08d}_reconstruction.npy".format(iter_), reconstruction)
            if smoe.quantization_mode == 1:
                np.save(self.reconstruction_path + "/{0:08d}_qreconstruction.npy".format(iter_), qreconstruction)

        if iter_ % checkpoint_iter == 0:
            smoe.checkpoint(self.checkpoints_path + "/{0:08d}_model.ckpt".format(iter_))
