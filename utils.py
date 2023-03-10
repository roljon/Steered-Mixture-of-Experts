import pickle
import numpy as np
import matplotlib.pyplot as plt
import cv2
import hdf5storage

def reduce_params(params):
    idx = params['pis'] > 0
    params['pis'] = params['pis'][idx]
    params['A_diagonal'] = params['A_diagonal'][idx]
    params['A_corr'] = params['A_corr'][idx]
    params['nu_e'] = params['nu_e'][idx]
    params['gamma_e'] = params['gamma_e'][idx]
    params['musX'] = params['musX'][idx]
    return params, idx


def save_model(smoe, path, best=False, reduce=True, quantize=True):
    if best:
        params = smoe.get_best_params()
    else:
        params = smoe.get_params()
    if reduce:
        params, bool_idx = reduce_params(params)

    mses = smoe.get_mses()
    losses = smoe.get_losses()
    num_pis = smoe.get_num_pis()

    cp = {'params': params, 'mses': mses, 'losses': losses, 'num_pis': num_pis,
          'quantization_mode': smoe.quantization_mode, 'quantized_pis': smoe.quantize_pis,
          'lower_bounds': smoe.lower_bounds, 'upper_bounds': smoe.upper_bounds,
          'use_yuv': smoe.use_yuv, 'only_y_gamma': smoe.only_y_gamma, 'ssim_opt': smoe.ssim_opt,
          'use_determinant': smoe.use_determinant, 'use_diff_center': smoe.use_diff_center}

    if smoe.dim_domain == 3 and (smoe.train_trafo or smoe.affines is not None):
        cp.update({'train_trafo': smoe.train_trafo, 'num_params_model': smoe.num_params_model})

    if quantize:
        qparams = smoe.qparams
        qparams.update({'dim_of_domain': smoe.dim_domain})
        qparams.update({'dim_of_output': smoe.image.shape[-1]})
        qparams.update({'shape_of_img': smoe.image.shape[:-1]})
        qparams.update({'used_ranges': False})
        qparams.update({'quantized_tria_params': True})
        qparams.update({'trained_gamma': smoe.train_gammas})
        qparams.update({'trained_musx': smoe.train_musx})
        qparams.update({'radial_as': smoe.radial_as})
        qparams.update({'trained_pis': smoe.train_pis})
        qparams.update({'use_yuv': smoe.use_yuv})
        qparams.update({'only_y_gamma': smoe.only_y_gamma})
        qparams.update({'use_determinant': smoe.use_determinant})
        qparams.update({'use_diff_center': smoe.use_diff_center})
        if reduce:
            qparams.update({'used_kernels': bool_idx})
        cp.update({'qparams': qparams})

    with open(path, 'wb') as fd:
        pickle.dump(cp, fd)

def load_params(path):
    with open(path, 'rb') as fd:
        params = pickle.load(fd)['params']
        #params['musX'] = np.transpose(params['musX'])
    return params


def read_image(path, use_yuv=True):
    affines = None

    if path.lower().endswith(('.png', '.tif', '.tiff', '.pgm', '.ppm', '.jpg', '.jpeg')):
        orig = cv2.imread(path)
        # check if orig is grayscale image:
        b1 = orig[:, :, 0] == orig[:, :, 1]
        b2 = orig[:, :, 0] == orig[:, :, 2]
        if np.sum(np.logical_and(b1, b2).flatten()) == np.prod(orig.shape[0:2]):
            orig = orig[:, :, 0]
            orig = np.expand_dims(orig, axis=-1)
        if orig.shape[2] == 3 and use_yuv:
            orig = cv2.cvtColor(orig, cv2.COLOR_BGR2YUV)

    elif path.lower().endswith(('.mp4', '.avi', '.mov', '.mkv', '.flv')):
        cap = cv2.VideoCapture(path)
        num_of_frames = np.array(cap.get(7), dtype=np.int32)
        height = np.array(cap.get(3), dtype=np.int32)
        width = np.array(cap.get(4), dtype=np.int32)
        orig = np.empty((width, height, num_of_frames, 3))
        idx_frame = np.array(0, dtype=np.int32)
        while(idx_frame < num_of_frames):
            ret, curr_frame = cap.read()
            if use_yuv:
                curr_frame = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2YUV)
            orig[:, :, idx_frame, :] = curr_frame
            idx_frame += 1

        orig = np.uint8(orig)


        # check if orig is grayscale video:
        b1 = orig[:, :, :, 1] == orig[:, :, :, 2]
        if np.sum(b1.flatten()) > np.prod(orig.shape[0:3]) * 0.9:  # Experimental!!
            orig = orig[:, :, :, 0]
            orig = np.expand_dims(orig, axis=-1)
    elif path.lower().endswith('.mat'):
        orig = hdf5storage.loadmat(path)["LF"]
        orig = orig[:, :, :, :, 0:3]
        if use_yuv:
            for ii in range(orig.shape[0]):
                for jj in range(orig.shape[1]):
                    orig[ii, jj, :, :, :] = cv2.cvtColor(orig[ii, jj, :, :, :], cv2.COLOR_RGB2YUV)
    elif path.lower().endswith('.yuv'):
        # TODO read raw video by OpenCV
        raise ValueError("Raw Video Data is not supported yet!")
    elif path.lower().endswith('.npz'):
        npz = np.load(path)
        orig = np.moveaxis(npz["imgs"], 0, -2)
        if use_yuv:
            for ii in range(orig.shape[2]):
                yuv_frame = cv2.cvtColor(orig[:,:,ii,:], cv2.COLOR_RGB2YUV)
                orig[:, :, ii, :] = yuv_frame

        affines = npz["affines"]
    else:
        raise ValueError("Unknown data format")

    if orig.dtype == np.uint8:
        orig = orig.astype(np.float32) / 255.
        precision = 8
    elif orig.dtype == np.uint16:
        orig = orig.astype(np.float32) / 2**16.
        precision = 16


    return orig, precision, affines

def write_image(img, path, type, yuv, precision):
    if precision == 8:
        img = np.uint8(np.round(img * 255))
    elif precision == 16:
        img = np.uint16(np.round(img * 2**precision))
    if type == 2:
        if yuv:
            img = cv2.cvtColor(img, cv2.COLOR_YUV2BGR)
        cv2.imwrite(path + ".png", img)
    elif type == 3:
        out = cv2.VideoWriter(path + ".yuv",
                              cv2.VideoWriter_fourcc(*'I420'), 25, (img.shape[0:2]))
        for ii in range(img.shape[2]):
            # TODO grayscale videos do not work
            frame = img[:, :, ii, :]
            if yuv:
                frame = cv2.cvtColor(frame, cv2.COLOR_YUV2BGR)
            out.write(frame)
        out.release()
    elif type == 4:
        if yuv:
            for ii in range(img.shape[0]):
                for jj in range(img.shape[1]):
                    img[ii, jj, :, :, :] = cv2.cvtColor(img[ii, jj, :, :, :], cv2.COLOR_YUV2RGB)
        mat = {}
        mat['LF'] = img
        hdf5storage.write(mat, '/', path + ".mat", matlab_compatible=True)
