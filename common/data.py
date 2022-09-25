import os
import h5py
import torch.utils.data as data
import numpy as np
import common.utils as utils
from torchvision import transforms

class PUDataset(data.Dataset):
    def __init__(self, h5_file_path="",
                 skip_rate=1, npoint=256, use_random=True, use_norm=True, isTrain=True):
        super().__init__()

        self.isTrain=isTrain

        self.npoint = npoint
        self.use_random = use_random
        self.use_norm = use_norm

        h5_file = h5py.File(h5_file_path)
        self.gt = h5_file['poisson_1024'][:]  # [:] h5_obj => nparray
        self.input = h5_file['poisson_1024'][:] if use_random \
            else h5_file['poisson_256'][:]
        assert len(self.input) == len(self.gt), 'invalid data'
        self.data_npoint = self.input.shape[1]

        centroid = np.mean(self.gt[..., :3], axis=1, keepdims=True)
        furthest_distance = np.amax(np.sqrt(np.sum((self.gt[..., :3] - centroid) ** 2, axis=-1)), axis=1, keepdims=True)
        self.radius = furthest_distance[:, 0]  # not very sure?

        if use_norm:
            self.radius = np.ones(shape=(len(self.input)))
            self.gt[..., :3] -= centroid
            self.gt[..., :3] /= np.expand_dims(furthest_distance, axis=-1)
            self.input[..., :3] -= centroid
            self.input[..., :3] /= np.expand_dims(furthest_distance, axis=-1)

    def __len__(self):
        return len(self.input)

    def __getitem__(self, index):
        input_data = self.input[index]
        gt_data = self.gt[index]
        radius_data = np.array([self.radius[index]])

        sample_idx = utils.nonuniform_sampling(self.data_npoint, sample_num=self.npoint)
        input_data = input_data[sample_idx, :]

        if not self.isTrain:
            return input_data, gt_data, radius_data

        if self.use_norm:
            # for data aug
            input_data, gt_data = utils.rotate_point_cloud_and_gt(input_data, gt_data)
            input_data, gt_data, scale = utils.random_scale_point_cloud_and_gt(input_data, gt_data,
                                                                               scale_low=0.9, scale_high=1.1)
            input_data, gt_data = utils.shift_point_cloud_and_gt(input_data, gt_data, shift_range=0.1)
            radius_data = radius_data * scale

            # for input aug
            #if np.random.rand() > 0.5:
            #    input_data = utils.jitter_perturbation_point_cloud(input_data, sigma=0.025, clip=0.05)
            #if np.random.rand() > 0.5:
            #    input_data = utils.rotate_perturbation_point_cloud(input_data, angle_sigma=0.03, angle_clip=0.09)
        else:
            raise NotImplementedError

        return input_data, gt_data, radius_data


if __name__=="__main__":
    
    dataset=PUDataset()
    (input_data,gt_data,radius_data)=dataset.__getitem__(0)
    print(f"Dataset : {dataset}")
    print(f"Input: {input_data}, GT :{gt_data}, rd : {radius_data}")
    print(input_data.shape,gt_data.shape,radius_data.shape)
