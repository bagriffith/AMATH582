import loadVid
import svd
import dmd
import numpy as np
from scipy import linalg
import matplotlib.pyplot as plt


def run_analysis():
    vids = ['monte_carlo.mov', 'ski_drop.mov']
    shapes = [(370, 624), (534, 488)]

    for vid, shape in zip(vids, shapes):
        X1, X2 = loadVid.open_video('HW5/data/' + vid)

        u, s, vh = linalg.svd(X1, full_matrices=False)
        svd.plot_n_modes(X1.T, u.T, vh.shape[0], shape,
                         'HW5/figures/' + vid[:-4] + '-rd.png')
        
        svd.plot_mode_fraction(s, 'HW5/figures/' + vid[:-4] + '-modes.pdf')

        w, psi = dmd.dmd(X2, u, s, vh)
        b, _, _, _ = np.linalg.lstsq(psi, X1[:, 0])

        fg, bg = dmd.frame_bg_sep(5, X1, psi, w, b)
        dmd.show_frame(fg, shape, 'HW5/figures/' + vid[:-4] + '-fg.pdf')
        dmd.show_frame(bg, shape, 'HW5/figures/' + vid[:-4] + '-bg.pdf')
        # dmd.save_fgbg_videos(X1, psi, w, b,
        #                      'HW5/figures/' + vids[:-4] + '-vid.avi')


if __name__ == '__main__':
    run_analysis()
