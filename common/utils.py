import torch
import os 
import numpy as np
from time import time
import pc_util
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Introduced from DGCNN. 
def knn(x, k):
    inner = -2*torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x**2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)
    
    dist = pairwise_distance.topk(k=k, dim=-1)[0]
    idx = pairwise_distance.topk(k=k, dim=-1)[1]   # (batch_size, num_points, k)
    return dist.to(device), idx.to(device)


def get_graph_feature(x, k=20, idx=None, dim9=False):
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    if idx is None:
        if dim9 == False:
            _, idx = knn(x, k=k)   # (batch_size, num_points, k)
        else:
            _, idx = knn(x[:, 6:], k=k)

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1)*num_points

    idx = idx + idx_base

    idx = idx.view(-1)
 
    _, num_dims, _ = x.size()

    x = x.transpose(2, 1).contiguous()   # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims) #   batch_size * num_points * k + range(0, batch_size*num_points)
    feature = x.view(batch_size*num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims) 
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)
    
    feature = torch.cat((feature-x, x), dim=3).permute(0, 3, 1, 2).contiguous()
    
    return feature      # (batch_size, 2*num_dims, num_points, k)

def save_model(previous_name, save_dir, epoch, data_threshold, model):
    if os.path.exists(previous_name):
        os.remove(previous_name)

    torch.save(model.state_dict(), '%s/model_%d_%d.pth' % (save_dir, epoch, data_threshold * 100))

    previous_name = '%s/model_%d_%d.pth' % (save_dir, epoch, data_threshold * 100)

    return previous_name
    

def save_model_epoch(save_dir, epoch, model):
    torch.save(model.state_dict(), '%s/epoch_%d.pth' % (save_dir, epoch))


def nonuniform_sampling(num, sample_num):
    sample = set()
    loc = np.random.rand() * 0.8 + 0.1
    while len(sample) < sample_num:
        a = int(np.random.normal(loc=loc, scale=0.3) * num)
        if a < 0 or a >= num:
            continue
        sample.add(a)
    return list(sample)


def rotate_point_cloud_and_gt(input_data, gt_data=None):
    """ Randomly rotate the point clouds to augument the dataset
        rotation is per shape based along up direction
        Input:
          Nx3 array, original point cloud
        Return:
          Nx3 array, rotated point cloud
    """
    angles = np.random.uniform(size=(3)) * 2 * np.pi
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(angles[0]), -np.sin(angles[0])],
                   [0, np.sin(angles[0]), np.cos(angles[0])]])
    Ry = np.array([[np.cos(angles[1]), 0, np.sin(angles[1])],
                   [0, 1, 0],
                   [-np.sin(angles[1]), 0, np.cos(angles[1])]])
    Rz = np.array([[np.cos(angles[2]), -np.sin(angles[2]), 0],
                   [np.sin(angles[2]), np.cos(angles[2]), 0],
                   [0, 0, 1]])
    rotation_matrix = np.dot(Rz, np.dot(Ry, Rx))

    input_data[:, :3] = np.dot(input_data[:, :3], rotation_matrix)
    if input_data.shape[1] > 3:
        input_data[:, 3:] = np.dot(input_data[:, 3:], rotation_matrix)

    if gt_data is not None:
        gt_data[:, :3] = np.dot(gt_data[:, :3], rotation_matrix)
        if gt_data.shape[1] > 3:
            gt_data[:, 3:] = np.dot(gt_data[:, 3:], rotation_matrix)

    return input_data, gt_data


def random_scale_point_cloud_and_gt(input_data, gt_data=None, scale_low=0.5, scale_high=2):
    """ Randomly scale the point cloud. Scale is per point cloud.
        Input:
            Nx3 array, original point cloud
        Return:
            Nx3 array, scaled point cloud
    """
    scale = np.random.uniform(scale_low, scale_high)
    input_data[:, :3] *= scale
    if gt_data is not None:
        gt_data[:, :3] *= scale

    return input_data, gt_data, scale


def shift_point_cloud_and_gt(input_data, gt_data=None, shift_range=0.3):
    """ Randomly shift point cloud. Shift is per point cloud.
        Input:
          Nx3 array, original point cloud
        Return:
          Nx3 array, shifted point cloud
    """
    shifts = np.random.uniform(-shift_range, shift_range, 3)
    input_data[:, :3] += shifts
    if gt_data is not None:
        gt_data[:, :3] += shifts
    return input_data, gt_data


def jitter_perturbation_point_cloud(input_data, sigma=0.005, clip=0.02):
    """ Randomly jitter points. jittering is per point.
        Input:
          Nx3 array, original point cloud
        Return:
          Nx3 array, jittered point cloud
    """
    assert (clip > 0)
    jitter = np.clip(sigma * np.random.randn(*input_data.shape), -1 * clip, clip)
    jitter[:, 3:] = 0
    input_data += jitter
    return input_data


def rotate_perturbation_point_cloud(input_data, angle_sigma=0.03, angle_clip=0.09):
    """ Randomly perturb the point clouds by small rotations
        Input:
          Nx3 array, original point cloud
        Return:
          Nx3 array, rotated point cloud
    """
    angles = np.clip(angle_sigma * np.random.randn(3), -angle_clip, angle_clip)
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(angles[0]), -np.sin(angles[0])],
                   [0, np.sin(angles[0]), np.cos(angles[0])]])
    Ry = np.array([[np.cos(angles[1]), 0, np.sin(angles[1])],
                   [0, 1, 0],
                   [-np.sin(angles[1]), 0, np.cos(angles[1])]])
    Rz = np.array([[np.cos(angles[2]), -np.sin(angles[2]), 0],
                   [np.sin(angles[2]), np.cos(angles[2]), 0],
                   [0, 0, 1]])
    R = np.dot(Rz, np.dot(Ry, Rx))
    input_data[:, :3] = np.dot(input_data[:, :3], R)
    if input_data.shape[1] > 3:
        input_data[:, 3:] = np.dot(input_data[:, 3:], R)
    return input_data

def gen_grid(up_ratio):
    import math
    """
    output [num_grid_point, 2]
    """
    sqrted = int(math.sqrt(up_ratio)) + 1
    for i in range(1, sqrted + 1).__reversed__():
        if (up_ratio % i) == 0:
            num_x = i
            num_y = up_ratio // i
            break
    grid_x = torch.linspace(-0.2, 0.2, num_x)
    grid_y = torch.linspace(-0.2, 0.2, num_y)

    x, y = torch.meshgrid(grid_x, grid_y)
    grid = torch.reshape(torch.stack([x, y], axis=-1), [-1, 2])  # [2, 2, 2] -> [4, 2]
    return grid

def save_xyz_file(numpy_array, xyz_dir):
    num_points = numpy_array.shape[0]
    with open(xyz_dir, 'w') as f:
        for i in range(num_points):
            line = "%f %f %f\n" % (numpy_array[i, 0], numpy_array[i, 1], numpy_array[i, 2])
            f.write(line)
    return
def farthest_point_sample(xyz, npoint):
    """
    Input:
        xyz: pointcloud data, [B, N, 3]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]
    """
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    return centroids



if __name__ == '__main__':
    a = torch.randn((1, 1024, 3))
    seed = farthest_point_sample(a, 20)[0]
    li = seed[:20]
    print(len(li))
    patches = pc_util.extract_knn_patch(a[0][np.asarray(li), :], a[0], 256)
    print(patches.shape)
