
# from datasets.hdf5 import SliceBuilder
import nibabel as nib
import SimpleITK as sitk
from torch.utils.data import Dataset, DataLoader, ConcatDataset


class SliceBuilder:
    def __init__(self, raw_dataset, label_dataset, patch_shape, stride_shape):
        self._raw_slices = self._build_slices(raw_dataset, patch_shape, stride_shape)
        if label_dataset is None:
            self._label_slices = None
        else:
            self._label_slices = self._build_slices(label_dataset, patch_shape, stride_shape)
            assert len(self._raw_slices) == len(self._label_slices)

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
        if dataset.GetDimension() == 4:
            in_channels, i_z, i_y, i_x = dataset.GetSize()
        else:
            i_z, i_y, i_x = dataset.GetSize()
            print(i_z)
        k_z, k_y, k_x = patch_shape
        s_z, s_y, s_x = stride_shape
        z_steps = SliceBuilder._gen_indices(i_z, k_z, s_z)
        print(z_steps)
        for z in z_steps:
            y_steps = SliceBuilder._gen_indices(i_y, k_y, s_y)
            for y in y_steps:
                x_steps = SliceBuilder._gen_indices(i_x, k_x, s_x)
                for x in x_steps:
                    slice_idx = (
                        slice(z, z + k_z),
                        slice(y, y + k_y),
                        slice(x, x + k_x)
                    )
                    if dataset.GetDimension() == 4:
                        slice_idx = (slice(0, in_channels),) + slice_idx
                    # print(slice_idx)
                    slices.append(slice_idx)
        return slices

    @staticmethod
    def _gen_indices(i, k, s):
        assert i >= k, 'Sample size has to be bigger than the patch size'
        for j in range(0, i - k + 1, s):
            yield j
        if j + k < i:
            yield i - k


def gen_patches():

    image: sitk.Image = sitk.ReadImage('ISO_Time20160804131327_ID90932902.nii')
    image.GetDimension()
    image.GetSize()
    Slicer = SliceBuilder(image, None, [32, 64, 64], [8, 16, 16])
    print(Slicer)


def NIfTIDataset(Dataset):
    pass


if __name__ == '__main__':
    gen_patches()

