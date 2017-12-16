# from sklearn.metrics import mean_squared_error
# from skimage.measure import compare_ssim
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np
import os
import glob

mpl.rcParams['image.cmap'] = 'jet'


def psnr(mse):
    return 10 * np.log10(255**2/mse)


class ImagePlotter:
    def __init__(self, path=None, options=(), quiet=False):
        self.path = path
        self.options = options
        self.quiet = quiet

        if self.path is not None:
            if not os.path.exists(path):
                os.mkdir(path)
            else:
                files = glob.glob(path + "/*")
                list(map(lambda x: os.remove(x), files))

        # TODO not nice
        num_options = len(self.options)
        if num_options <= 4:
            self.rows = 1
            self.cols = num_options
        else:
            self.rows = 2
            self.cols = num_options % 4

        #self.fig, self.axes = plt.subplots(1, self.cols)  # plt.subplots(self.rows, self.cols)

        # TODO only hotfix to display hist properly
        if "pis_hist" in self.options:
            self.fig = plt.figure()

            gs = GridSpec(2, 3)  # 2 rows, 3 columns

            ax1 = self.fig.add_subplot(gs[0, 0])  # First row, first column
            ax2 = self.fig.add_subplot(gs[0, 1])  # First row, second column
            ax3 = self.fig.add_subplot(gs[0, 2])  # First row, third column
            ax4 = self.fig.add_subplot(gs[1, :])  # Second row, span all columns

            self.axes = [ax1, ax2, ax3, ax4]

        else:
            self.fig, self.axes = plt.subplots(1, self.cols)  # plt.subplots(self.rows, self.cols)

        if not self.quiet:
            self.fig.show()
            self.fig.canvas.draw()

    def plot(self, smoe):
        for idx, option in enumerate(self.options):
            row = int(idx / self.cols)
            col = int(idx % self.cols)

            # no row, col indexes if there is only one row
            if len(self.axes) == len(self.options):
                ax = self.axes[col]
            else:
                ax = self.axes[row, col]

            ax.clear()

            if option == "orig":
                ax.imshow(smoe.get_original_image(), cmap='gray', interpolation='None', vmin=0, vmax=1)
            elif option == "reconstruction":
                ax.imshow(smoe.get_reconstruction(), cmap='gray', interpolation='None', vmin=0, vmax=1)
            elif option == "gating":
                w_e_opt = smoe.get_weight_matrix()
                dim_size = int(np.sqrt(w_e_opt.shape[1]))
                ax.imshow(w_e_opt.argmax(axis=0).reshape((dim_size, dim_size)).T, interpolation='None')
            elif option == "pis_hist":
                params = smoe.get_params()
                ax.hist(params['pis'], 500)
                used = np.count_nonzero(params['pis'] > 0)
                total = params['pis'].shape[0]
                ax.set_title('{0:d} / {1:d} ({2:.2f})'.format(used, total, 100.*used/total))

        iters_loss, losses = zip(*smoe.get_losses())
        iters_mse, mses = zip(*smoe.get_mses())
        assert iters_loss == iters_mse, "mse/loss logging out of sync"
        self.fig.suptitle(
            'start, best, last: {0:.6f} / {1:.6f} / {2:.6f}\n'
            'MSE: start, best, last: {3:.2f} / {4:.2f} / {5:.2f}\n'
            'PSNR: start, best, last: {6:.2f} / {7:.2f} / {8:.2f}'.format(losses[0],
                                                                          smoe.get_best_loss(),
                                                                          losses[-1],
                                                                          mses[0],
                                                                          smoe.get_best_mse(),
                                                                          mses[-1],
                                                                          psnr(mses[0]),
                                                                          psnr(smoe.get_best_mse()),
                                                                          psnr(mses[-1]))
        )

        if not self.quiet:
            self.fig.canvas.draw()

        if self.path:
            name = str(iters_loss[-1]) + ".png"
            self.fig.savefig(self.path + "/" + name)

    def __del__(self):
        plt.close(self.fig)


class LossPlotter:
    def __init__(self, path=None, quiet=False):
        self.path = path
        self.quiet = quiet

        self.fig = plt.figure()
        self.ax_loss = self.fig.add_subplot(111)
        self.ax_mse = self.ax_loss.twinx()

        self.ax_loss.set_ylabel('loss', color='b')
        self.ax_loss.tick_params('y', colors='b')
        self.ax_mse.set_ylabel('MSE', color='r')
        self.ax_mse.tick_params('y', colors='r')

        #if self.path is not None:
        #    if not os.path.exists(path):
        #        os.mkdir(path)
        #    else:
        #        files = glob.glob(path + "/*")
        #        list(map(lambda x: os.remove(x), files))


        if not self.quiet:
            self.fig.show()
            self.fig.canvas.draw()

    def plot(self, smoe):
        iters_loss, losses = zip(*smoe.get_losses())
        iters_mse, mses = zip(*smoe.get_mses())
        assert iters_loss == iters_mse, "mse/loss logging out of sync" + str((iters_loss, iters_mse))
        self.ax_loss.clear()
        self.ax_loss.set_title(
            'start, best, last: {0:.6f} / {1:.6f} / {2:.6f}\n'
            'MSE: start, best, last: {3:.2f} / {4:.2f} / {5:.2f}\n'
            'PSNR: start, best, last: {6:.2f} / {7:.2f} / {8:.2f}'.format(losses[0],
                                                                          smoe.get_best_loss(),
                                                                          losses[-1],
                                                                          mses[0],
                                                                          smoe.get_best_mse(),
                                                                          mses[-1],
                                                                          psnr(mses[0]),
                                                                          psnr(smoe.get_best_mse()),
                                                                          psnr(mses[-1]))
        )
        self.ax_loss.plot(iters_loss, losses, color='b')
        self.ax_mse.plot(iters_mse, mses, color='r')

        if self.path:
            self.fig.savefig(self.path)

        if not self.quiet:
            self.fig.canvas.draw()

    def __del__(self):
        plt.close(self.fig)
