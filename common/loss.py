import torch
from common.utils import knn
from chamfer_distance import ChamferDistance as chamfer_dist

def repulsion_loss(pcd,h=0.0005):

    dist, _= knn(pcd, k=20)#B N k
    
    dist=dist[:,:,1:5]**2 #top 4 cloest neighbors

    loss=torch.clamp(-dist+h,min=0)

    loss=torch.mean(loss)*1e2

    return loss

def chamferloss(source, target):

    # cd_loss = ChamferDistance()
    dist1, dist2, _, _ = chamfer_dist(source, target)
    loss = (torch.mean(dist1)) + (torch.mean(dist2))
    loss = loss*1e2

    return loss

class AccumLoss(object):
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val
        self.count += n
        self.avg = self.sum / self.count