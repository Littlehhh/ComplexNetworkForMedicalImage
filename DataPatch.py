import os
import glob
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import networkx as nx


class SliceBuilder:
    def __init__(self, raw_dataset, label_dataset, patch_shape, stride_shape):
        self._raw_slices = self._build_slices(raw_dataset, patch_shape, stride_shape)
        if label_dataset is None:
            self._label_slices = None
        else:
            self._label_slices = self._build_slices(label_dataset, patch_shape, stride_shape)
            assert len(self._raw_slices) == len(self._label_slices)

    def __len__(self):
        return len(self.raw_slices)

    @property
    def raw_slices(self):
        return self._raw_slices

    @property
    def label_slices(self):
        return self._label_slices

    @staticmethod
    def _build_slices(dataset, patch_shape, stride_shape):
        """Iterates over a given n-dim dataset patch-by-patch with a given stride
        and builds an array of slice positions.

        Returns:
            list of slices, i.e.
            [(slice, slice, slice, slice), ...] if len(shape) == 4
            [(slice, slice, slice), ...] if len(shape) == 3
        """
        slices = []
        i_z, i_y, i_x = dataset.shape
        k_z, k_y, k_x = patch_shape
        s_z, s_y, s_x = stride_shape
        z_steps = SliceBuilder._gen_indices(i_z, k_z, s_z)
        for z in z_steps:
            y_steps = SliceBuilder._gen_indices(i_y, k_y, s_y)
            for y in y_steps:
                x_steps = SliceBuilder._gen_indices(i_x, k_x, s_x)
                for x in x_steps:
                    slice_idx = (
                        slice(z, z + k_z, 1),
                        slice(y, y + k_y, 1),
                        slice(x, x + k_x, 1)
                    )
                    slices.append(slice_idx)
        return slices

    @staticmethod
    def _gen_indices(i, k, s):
        assert i >= k, 'Sample size has to be bigger than the patch size'
        j = int()
        for j in range(0, i - k + 1, s):
            yield j
        if j + k < i:
            yield i - k


def listdir_with_postfix(path, postfix):
    return sorted(glob.glob(os.path.join(path, ('*'+postfix))))


def nib_save(np_array, origin_data, file_name):
    save_img = nib.Nifti1Image(np_array, origin_data.affine, origin_data.header)
    nib.save(save_img, file_name)


def data_patch(file_name, dtype=np.float32):
    data_proxy = nib.load(file_name)
    brain_data: np.ndarray = data_proxy.get_fdata()
    if dtype == np.bool_:
        # data_proxy.set_data_dtype(dtype=dtype)
        brain_data = brain_data.astype(dtype)
        # return
    # Guarantee the shape is the same
    assert brain_data.shape == (182, 218, 182)
    # Crop the brain into (180, 210, 180)
    crop_brain = brain_data[1:-1, 4:-4, 1:-1]
    assert crop_brain.shape == (180, 210, 180)
    # spilt the brain in Sagittal
    mid = 90
    right_brain = crop_brain[:mid, :, :]
    left_brain = crop_brain[mid:, :, :]
    flip_left_brain = np.flip(left_brain, axis=0)
    # nib_save(flip_left_brain, data_proxy, "flip_left.nii.gz")
    # nib_save(right_brain, data_proxy, "right.nii.gz")
    # means no overlap
    patch_shape = stride_shape = (5, 5, 5)
    right_brain_patches = SliceBuilder(right_brain, None, patch_shape, stride_shape)
    flip_left_brain_patches = SliceBuilder(flip_left_brain, None, patch_shape, stride_shape)
    # flip_right_brain_node = filter_effective_node(right_brain_patches, right_brain)
    # left_brain_node = filter_effective_node(flip_left_brain_patches,  flip_left_brain)
    patches_as_row = list()
    for patch in right_brain_patches.raw_slices:
        patches_as_row.append(right_brain[patch].flatten())
    for patch in flip_left_brain_patches.raw_slices:
        patches_as_row.append(flip_left_brain[patch].flatten())
    # print(len(patches_as_row))
    data = np.asarray(patches_as_row)
    assert data.shape == (54432, 125)
    return data


def build_complex_brain_network(file_name="CC0120_siemens_15_58_F_remove_skull_to_MNI152_T1_1mm_brain.nii.gz"):
    # file_name = "CC0120_siemens_15_58_F_remove_skull_to_MNI152_T1_1mm_brain.nii.gz"
    data_proxy = nib.load(file_name)
    brain_data: np.ndarray = data_proxy.get_fdata()
    # Guarantee the shape is the same
    assert brain_data.shape == (182, 218, 182)
    # Crop the brain into (180, 210, 180)
    crop_brain = brain_data[1:-1, 4:-4, 1:-1]
    assert crop_brain.shape == (180, 210, 180)
    # spilt the brain in Sagittal
    mid = 90
    right_brain = crop_brain[:mid, :, :]
    left_brain = crop_brain[mid:, :, :]
    flip_left_brain = np.flip(left_brain, axis=0)
    # nib_save(flip_left_brain, data_proxy, "flip_left.nii.gz")
    # nib_save(right_brain, data_proxy, "right.nii.gz")
    # means no overlap
    patch_shape = stride_shape = (5, 5, 5)
    right_brain_patches = SliceBuilder(right_brain, None, patch_shape, stride_shape)
    flip_left_brain_patches = SliceBuilder(flip_left_brain, None, patch_shape, stride_shape)
    # flip_right_brain_node = filter_effective_node(right_brain_patches, right_brain)
    # left_brain_node = filter_effective_node(flip_left_brain_patches,  flip_left_brain)
    # patches = list()
    # patches.extend(right_brain_patches.raw_slices)
    # patches.extend(flip_left_brain_patches.raw_slices)
    return right_brain_patches, flip_left_brain_patches


def compute_all_hippocampus():
    file_list = listdir_with_postfix(
        '/home/workshop/Data/medical_image_compute/Siemens15/Registration_Siemens15_Hippocampus',
        '.nii.gz')
    hipp_label_patches = np.ones((54432, 125))
    for file_name in file_list:
        print(os.path.basename(file_name) + ' start')
        hipp_label_patches *= data_patch(file_name)

    np.save('hipp_label_patches', hipp_label_patches)


def compute_all_brain():
    file_list = listdir_with_postfix('/home/workshop/Data/medical_image_compute/Siemens15/Registration_Siemens15',
                                     '.nii.gz')
    ave_patches = np.zeros((54432, 125))
    for file_name in file_list:
        print(os.path.basename(file_name) + ' start')
        ave_patches += data_patch(file_name)
    ave_patches /= len(file_list)
    np.save('ave_patches', ave_patches)
    # return ave_patches


def filter_effective_node(patches: SliceBuilder, data: np.ndarray, threshold=1e-5):
    node = [[data, patch]
            for data, patch in
            zip(data, patches.raw_slices)
            if np.sum(data) > threshold]
    return node


def get_patch_center(patch_idx: tuple):
    xslice: slice = patch_idx[0]
    yslice: slice = patch_idx[1]
    zslice: slice = patch_idx[2]
    x: int = (xslice.start + xslice.stop) // 2
    y: int = (yslice.start + yslice.stop) // 2
    z: int = (zslice.start + zslice.stop) // 2
    return tuple([x, y, z])


def get_flip_patch_center(patch_idx: tuple, x_len=180):
    xslice: slice = patch_idx[0]
    yslice: slice = patch_idx[1]
    zslice: slice = patch_idx[2]
    x: int = x_len - 1 - (xslice.start + xslice.stop) // 2
    y: int = (yslice.start + yslice.stop) // 2
    z: int = (zslice.start + zslice.stop) // 2
    return tuple([x, y, z])


if __name__ == '__main__':
    # A = np.zeros((10, 10, 10))
    # A[8, 2, 1] = 1
    # A[3, 5, 2] = 1
    # B = np.flip(A[5:], 0)
    # print(np.where(B))
    # print(compute_all())
    # a = [[[1, 2, 3],
    #       [4, 5, 6]],
    #      [[1, 2, 3],
    #       [7, 8, 9]]]
    # A = np.asarray(a)
    # x = [[0.1, 0.1, 0.1],
    #      [1, 4, 3],
    #      [1, 2, 3],
    #      [1, 2, 3],
    #      [1, 2, 3]]
    # X = np.asarray(x)
    # r_matrix = np.corrcoef(X)
    # print(r_matrix)

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

    graph = nx.Graph()
    for origin in hipp_list[0]:
        for end in data_list[0]:
            if origin != end:
                x = np.where(data_list == origin)
                graph.add_edge(origin,
                               end,
                               weight=matrix_r_abs[np.where(data_list == origin),
                                                   np.where(data_list == end)])
    # elarge = [(u, v) for (u, v, d) in graph.edges(data=True) if d['weight'] > 0.97]
    # esmall = [(u, v) for (u, v, d) in graph.edges(data=True) if d['weight'] <= 0.97]
    #
    # pos = nx.spring_layout(graph)  # positions for all nodes
    # # nodes
    # nx.draw_networkx_nodes(graph, pos, node_size=1.5)
    #
    # # edges
    # nx.draw_networkx_edges(graph, pos, edgelist=elarge, width=0.8)
    # nx.draw_networkx_edges(graph, pos, edgelist=esmall, width=0.8, alpha=0.5, edge_color='b', style='dashed')
    #
    # # labels
    # # plt.figure(figsize=(14, 10) )
    # nx.draw_networkx_labels(graph, pos, font_size=1, font_family='sans-serif')
    # plt.axis('off')
    # right_patches_position, flip_left_patches_position = build_complex_brain_network()
    # THRESHOLD = 0.005
    # assert len(right_patches_position.raw_slices) == len(flip_left_patches_position.raw_slices) == ave_data.shape[0]/2
    # right_effective_node = filter_effective_node(right_patches_position,
    #                                              ave_data[:len(right_patches_position.raw_slices)],
    #                                              THRESHOLD)
    # flip_left_effective_node = filter_effective_node(flip_left_patches_position,
    #                                                  ave_data[-len(flip_left_patches_position.raw_slices):],
    #                                                  THRESHOLD)
    # effective_node = np.vstack((np.asarray([node_data[0] for node_data in right_effective_node]),
    #                            np.asarray([node_data[0] for node_data in flip_left_effective_node])))


    # compute_all_hippocampus()
    # right_patches_position, flip_left_patches_position = build_complex_brain_network()
    # THRESHOLD = 0.001
    # assert len(right_patches_position.raw_slices) == len(flip_left_patches_position.raw_slices) == ave_data.shape[0]/2
    # right_effective_node = filter_effective_node(right_patches_position,
    #                                              ave_data[:len(right_patches_position.raw_slices)],
    #                                              THRESHOLD)
    # flip_left_effective_node = filter_effective_node(flip_left_patches_position,
    #                                                  ave_data[-len(flip_left_patches_position.raw_slices):],
    #                                                  THRESHOLD)
    # effective_node = np.vstack((np.asarray([node_data[0] for node_data in right_effective_node]),
    #                            np.asarray([node_data[0] for node_data in flip_left_effective_node])))
    # print(len(right_effective_node))
    # print(len(flip_left_effective_node))
    # print(effective_node.shape)
    # matrix_r = np.corrcoef(effective_node)
    # matrix_r_abs = np.abs(matrix_r)
    print("Done")

    # NOT WORK ON NETWORKX
    # graph = nx.Graph()
    # for origin in range(len(effective_node)):
    #     for end in range(origin+1, len(effective_node)):
    #         graph.add_edge(origin, end, weight=matrix_r_abs[origin, end])
    # elarge = [(u, v) for (u, v, d) in G.edges(data=True) if d['weight'] > 0.97]
    # esmall = [(u, v) for (u, v, d) in G.edges(data=True) if d['weight'] <= 0.97]
    #
    # pos = nx.spring_layout(graph)  # positions for all nodes
    # # nodes
    # nx.draw_networkx_nodes(graph, pos, node_size=1.5)
    #
    # # edges
    # nx.draw_networkx_edges(graph, pos, edgelist=elarge, width=0.8)
    # nx.draw_networkx_edges(graph, pos, edgelist=esmall, width=0.8, alpha=0.5, edge_color='b', style='dashed')
    #
    # # labels
    # # plt.figure(figsize=(14, 10) )
    # nx.draw_networkx_labels(graph, pos, font_size=1, font_family='sans-serif')
    # plt.axis('off')

