import os
import numpy as np
import matplotlib.pyplot as plt

from DataPatch import compute_all_brain,compute_all_hippocampus

if __name__ == '__main__':
    if not os.path.exists("ave_patches.npy"):
        compute_all_brain('Philips15/Registration_Philip15')
    if not os.path.exists("hipp_label_patches.npy"):
        compute_all_hippocampus('Philips15/Registration_Philip15_Hippocampus')

    ave_data = np.load('ave_patches.npy')
    # matrix_r = np.corrcoef(ave_data)
    # matrix_r_abs = np.abs(matrix_r)
    hipp_data = np.load('hipp_label_patches.npy')
    data_list = np.where(np.sum(ave_data, axis=1) > 0.05)
    hipp_list = np.where(np.sum(hipp_data, axis=1) > 0)

    for seed in hipp_list[0]:
        if seed not in data_list[0]:
            print("Not all hipp in ave_list")
    end_nodes = ave_data[data_list]
    start_nodes = hipp_data[hipp_list]

    matrix_r = np.corrcoef(end_nodes)
    matrix_r_abs = np.abs(matrix_r)

    fig, ax = plt.subplots()
    im = ax.imshow(matrix_r_abs)

    plt.show()