import argparse

class opts():
    def __init__(self):
        self.parser = argparse.ArgumentParser()

    def init(self):
        self.parser.add_argument("--device", default='cuda', type=str)
        self.parser.add_argument('--up_ratio', type=int, default=4)
        self.parser.add_argument('--manualSeed', default=2, type=int, help="deterministic mode")
        self.parser.add_argument('--d', type=int, default=2, help="dilation rate")
        self.parser.add_argument('--k', type=int, default=20, help="number of neighbors (kernel size for GCN)")
        self.parser.add_argument('--batch_size', type=int, default=64)
        self.parser.add_argument('--lr', type=float, default=3e-4)
        self.parser.add_argument('--nepoch', type=int, default=200)
        self.parser.add_argument('--train', action='store_true')
        self.parser.add_argument('--root_path', type=str, default='/home/steven/torch_pcu/')
        self.parser.add_argument('--dataset_path', type=str, default='PUGAN_poisson_256_poisson_1024.h5')
        self.parser.add_argument('--checkpoint', type=str, default='./log/')
        self.parser.add_argument('--dim', type=int, default=6)
        self.parser.add_argument('--num_point', type=int, default=256)
        self.parser.add_argument('--patch_num_point', type=int, default=256)
        self.parser.add_argument('--patch_num_ratio', type=int, default=3)
        self.parser.add_argument('--feature_dim', type=int, default=256, help='number of upsampling dense feature dimentions.')
        self.parser.add_argument('--rep', type=float, default=0.01, help='coefficient of repulsion loss')
        self.parser.add_argument('--vis', action='store_true', help='Visualization of point cloud')
        self.parser.add_argument('--vis_step', type=int, default=100, help='number of step to visualize the point cloud')
        self.parser.add_argument('--eval_dir', type=str, default='', help='path to evaluation data directory')
        self.parser.add_argument('--resume', type=str, default='', help='Checkpoint file')
        self.parser.add_argument('--out_dir', type=str, default='')
    def parse(self):
        self.init()
        self.opt = self.parser.parse_args()
        
        return self.opt