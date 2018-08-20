import argparse
# for execution without a display
import matplotlib as mpl
mpl.use('Agg')

import pickle
import matplotlib.pyplot as plt
import numpy as np
import os
import shutil
import cv2
import re

from smoe import Smoe
from utils import save_model, load_params
from quantizer import quantize_params, rescaler

def main(image_path, results_path, params_file, batches, bit_depths, quant_params):

    if len(bit_depths) != 5:
        raise ValueError("Number of bit depths must be five!")


    if image_path.lower().endswith(('.png', '.tif', '.tiff', '.pgm', '.ppm', '.jpg', '.jpeg')):
        orig = plt.imread(image_path)
        if orig.ndim == 2:
            orig = np.expand_dims(orig, axis=-1)
        if orig.dtype == np.uint8:
            orig = orig.astype(np.float32) / 255.
    elif image_path.lower().endswith(('.mp4', '.avi', '.mov', '.mkv', '.flv')):
        # TODO expand dimension in case of grayscale video
        cap = cv2.VideoCapture(image_path)
        num_of_frames = np.array(cap.get(7), dtype=np.int32)
        height = np.array(cap.get(3), dtype=np.int32)
        width = np.array(cap.get(4), dtype=np.int32)
        orig = np.empty((width, height, num_of_frames, 3))
        idx_frame = np.array(0, dtype=np.int32)
        while(idx_frame < num_of_frames):
            ret, curr_frame = cap.read()
            #curr_frame = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
            orig[:, :, idx_frame, :] = curr_frame
            idx_frame += 1
        orig = orig.astype(np.float32) / 255.

    elif image_path.lower().endswith('.yuv'):
        # TODO read raw video by OpenCV
        raise ValueError("Raw Video Data is not supported yet!")
    else:
        raise ValueError("Unknown data format")

    init_params = load_params(params_file)

    if results_path is not None:
        if not os.path.exists(results_path):
        #   shutil.rmtree(results_path)
            os.mkdir(results_path)

    smoe = Smoe(orig, init_params=init_params, start_batches=batches,
                bit_depths=bit_depths)

    with open(params_file, 'rb') as fd:
        cp = pickle.load(fd)
        smoe.quantization_mode = cp.get('quantization_mode')
        smoe.quantize_pis = cp.get('quantized_pis')
        smoe.lower_bounds = cp.get('lower_bounds')
        smoe.upper_bounds = cp.get('upper_bounds')

    if smoe.quantization_mode is None:
        smoe.quantization_mode = 0
        smoe.quantize_pis = False


    if smoe.quantization_mode <= 0 and quant_params:
        smoe.qparams = quantize_params(smoe, smoe.get_params())
        smoe.rparams = rescaler(smoe, smoe.qparams)
        with_quantized_params = True
    else:
        with_quantized_params = False

    loss, mse, _ = smoe.run_batched(train=False, update_reconstruction=True, with_quantized_params=with_quantized_params)


    iter_str = re.findall(r'\d+', params_file)[-1]
    reconstruction_path = results_path + '/' + iter_str + "_reconstruction"
    if with_quantized_params:
        reconstruction = smoe.get_qreconstruction()
        bit_depths_add = "_{0:1d}_{1:1d}_{2:1d}_{3:1d}_{4:1d}".format(bit_depths[0], bit_depths[1], bit_depths[2],
                                                                      bit_depths[3], bit_depths[4])
        reconstruction_path = reconstruction_path + bit_depths_add

        qparams_path = results_path + '/' + iter_str + "_params" + bit_depths_add + ".pkl"
        qparams = smoe.qparams
        qparams.update({'dim_of_domain': smoe.dim_domain})
        qparams.update({'dim_of_output': smoe.image.shape[-1]})
        qparams.update({'shape_of_img': smoe.image.shape[:-1]})
        qparams.update({'used_ranges': False})
        qparams.update({'quantized_tria_params': True})
        qparams.update({'trained_gamma': smoe.train_gammas})
        qparams.update({'radial_as': smoe.radial_as})
        qparams.update({'trained_pis': smoe.train_pis})
        with open(qparams_path, 'wb') as fd:
            pickle.dump(qparams, fd)
    else:
        reconstruction = smoe.get_reconstruction()

    if smoe.dim_domain == 2:
        plt.imsave(reconstruction_path + ".png", reconstruction, cmap='gray',
                   vmin=0, vmax=1)
    elif smoe.dim_domain == 3:
        out = cv2.VideoWriter(reconstruction_path + ".yuv",
                              cv2.VideoWriter_fourcc(*'I420'), 25, (reconstruction.shape[0:2]))
        for ii in range(reconstruction.shape[2]):
            frame = np.squeeze(reconstruction[:, :, ii, :])
            frame = np.uint8(np.round(frame * 255))
            out.write(frame)
        out.release()

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
    parser.add_argument('-p', '--params_file', type=str, required=True, help="parameter file for model initialization.")
    parser.add_argument('-b', '--batches', type=int, default=1, help="number of batches to split the training into (will be automaticly reduced when number of pis drops")


    parser.add_argument('-bd', '--bit_depths', type=int, default=[20, 18, 6, 10, 10], nargs='+',
                        help="bit depths of each kind of parameter. number of numbers must be 5 in the order: A, musX, nu_e, pis, gamma_e")

    parser.add_argument('-qp', '--quant_params', type=str2bool, nargs='?',
                        const=True, default=True, help="use quantized parameters for reconstruction")

    args = parser.parse_args()

    main(**vars(args))