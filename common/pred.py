import torch
import numpy as np
import pc_util
from time import time
from tqdm import tqdm
from utils import farthest_point_sample


def patch_prediction(opt, patch_point, model):
    # normalize the point clouds
    patch_point, centroid, furthest_distance = pc_util.normalize_point_cloud(patch_point)
    patch_point = np.expand_dims(patch_point, axis=0)
    patch_point = torch.tensor(patch_point, device=torch.device(opt.device)).transpose(2, 1).contiguous()
    pred = model(patch_point)
    pred = pred.transpose(2, 1).contiguous().detach().cpu().numpy()
    pred = np.squeeze(centroid + pred * furthest_distance, axis=0)
    return pred

def pc_prediction(opt, pc, model):

    points = torch.tensor(np.expand_dims(pc, axis=0), dtype=torch.float32)
    start = time()
    print(f'Input number of point: {pc.shape[0]}')
    print(f'Output number of point: {pc.shape[0] * opt.up_ratio}')
    seed1_num = int(pc.shape[0] / opt.patch_num_point * opt.patch_num_ratio)

    seed = farthest_point_sample(points, seed1_num)[0]
    seed_list = seed[:seed1_num]

    print("farthest distance sampling cost", time() - start)
    print("number of patches: %d" % len(seed_list))

    input_list = []
    up_point_list = []
    patches = pc_util.extract_knn_patch(pc[np.asarray(seed_list), :], pc, opt.patch_num_point)
    patch_time = 0.
    for point in patches:

        start = time()
        up_point = patch_prediction(opt, point, model)
        end = time()
        patch_time += end-start
        input_list.append(point)
        up_point_list.append(up_point)

    return input_list, up_point_list, patch_time/len(patches)