import os
import torch
import random
import logging
import numpy as np
from tqdm import tqdm
from glob import glob
import torch.utils.data
import torch.optim as optim
from torchsummary import summary
from common import pc_util
from common.data import PUDataset
from common.config import opts
from common.utils import save_model_epoch, save_xyz_file, farthest_point_sample
from common.loss import AccumLoss, repulsion_loss, chamferloss
from common.vis_pc import point_cloud_three_views, plot_pcd_three_views
from common.pred import pc_prediction
from model.model import Enhanced_PCU, xavier_init
from torch.utils.tensorboard import SummaryWriter   

opt = opts().parse()
torch.backends.cudnn.benchmark = True

def train(opt, train_dataloader, model, optimizer, epoch):
    return step('train', opt, train_dataloader, model, optimizer, epoch)

def val(opt, eval_dataloader, model):
    with torch.no_grad():
        return step('test', opt, eval_dataloader, model)

def step(phase, opt, dataloader, model, optimizer=None, epoch=None):

    if phase == 'train':
        model.train()
    else: 
        model.eval()
        total_time = 0

    loop = tqdm(dataloader, total=len(dataloader), leave=False)
    for i, data in enumerate(loop):
        loss_all = {'loss' : AccumLoss()}

        if phase == 'train':
            input_pc, gt_pc, radius = data # B, N, C
            input_pc = input_pc.to(opt.device)
            gt_pc = gt_pc.to(opt.device)

            input_pc = input_pc.transpose(2, 1).contiguous()
            output = model(input_pc)

            rep_loss = repulsion_loss(output)
            cd_loss = chamferloss(gt_pc, output.transpose(2, 1).contiguous())
            total_loss = cd_loss + opt.rep * rep_loss

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            loss = total_loss.item()/opt.batch_size

            loss_all['loss'].update(total_loss.item(), input_pc.shape[0])

            writer.add_scalar(tag='loss', scalar_value=loss, global_step=epoch*len(dataloader)+i)
            writer.add_scalar(tag='learning_rate', scalar_value=optimizer.state_dict()['param_groups'][0]['lr'], global_step=epoch*len(dataloader)+i)

            loop.set_description(f"[EPOCH: {(epoch+1)}/{opt.nepoch}]")
            loop.set_postfix(CD_loss=cd_loss.item()/opt.batch_size, Rep_loss=rep_loss.item()/opt.batch_size)

            if opt.vis and (i % opt.vis_step) == 0:
                pcds = [output.permute(0, 2, 1).contiguous().detach().cpu().numpy(), gt_pc.detach().cpu().numpy()]
                
                if os.path.exists(os.path.join(opt.checkpoint, 'plots')):
                    plot_path = os.path.join(opt.checkpoint, 'plots',
                                                 'epoch_%d_step_%d.png' % (epoch+1, i+1))
                else: 
                    os.mkdir(os.path.join(opt.checkpoint, 'plots'))
                    plot_path = os.path.join(opt.checkpoint, 'plots',
                                                 'epoch_%d_step_%d.png' % (epoch+1, i+1))
                plot_pcd_three_views(plot_path, pcds, ['output', 'gt'])
        else:
            print(data.split('/')[-1][:-4])
            pc = pc_util.load(data)[:, :3]
            opt.num_point = pc.shape[0]
            out_point_num = int(opt.num_point * opt.up_ratio)
            pc, centroid, furthest_distance = pc_util.normalize_point_cloud(pc)

            input_list, pred_list, avg_patch_time = pc_prediction(opt, pc, model)
            total_time += avg_patch_time # inference time

            pred_pc = np.concatenate(pred_list, axis=0)
            pred_pc = (pred_pc * furthest_distance) + centroid

            pred_pc = np.reshape(pred_pc, [-1, 3])
            path = os.path.join(opt.out_dir, data.split('/')[-1][:-4])
            pred_pc = torch.tensor(pred_pc, device='cpu')
            idx = farthest_point_sample(pred_pc[np.newaxis, ...], out_point_num)[0]
            pred_pc = pred_pc[idx, 0:3]
            np.savetxt(path + '.xyz', pred_pc, fmt='%.6f')

    if phase == 'train':
        return loss_all['loss'].avg
    else:

        logging.info('Average Inference Time: {} ms'.format(total_time / len(dataloader) * 1000.))
        
        
        # return loss


if __name__ == '__main__':

    random.seed(opt.manualSeed)
    np.random.seed(opt.manualSeed)
    torch.manual_seed(opt.manualSeed)

    device = torch.device(opt.device)

    writer = SummaryWriter(opt.checkpoint)

    if opt.train:
        if os.path.exists(os.path.join(os.getcwd(), opt.checkpoint)):
            logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%Y/%m/%d %H:%M:%S', \
                        filename = os.path.join(opt.checkpoint, 'train.log'), level=logging.INFO)
        else:
            os.mkdir('./log/')
            logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%Y/%m/%d %H:%M:%S', \
                        filename = os.path.join(opt.checkpoint, 'train.log'), level=logging.INFO)

    root_path = opt.root_path
    dataset_path = root_path + opt.dataset_path

    if opt.train:
        train_data = PUDataset(h5_file_path=dataset_path)

        train_dataloader = torch.utils.data.DataLoader(dataset=train_data, batch_size=opt.batch_size,
                                                        shuffle=True, num_workers=0, pin_memory=True)
        model = Enhanced_PCU(opt).to(device)
        model.apply(xavier_init)

        print(f"Total samples of dataset: {len(train_data)}")
        print(f"Total parameters: {sum(p.numel() for p in model.parameters())/1000000:.2f}M")
        summary(model, input_size=(3, 256), batch_size=opt.batch_size) # Some bug for batch size

    else:
      
        samples = glob(os.path.join(opt.eval_dir, '*.xyz'))

        print(f'Total number of samples: {len(samples)}')

        model = Enhanced_PCU(opt).to(device)

        checkpoint = torch.load(opt.resume)
        model.load_state_dict(checkpoint)
        print(model)
                                                                

    
    lr = opt.lr
    optimizer = optim.Adam(params=model.parameters(), lr=opt.lr)
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.3117, patience=5, verbose=True)

    if opt.train:
        for epoch in range(opt.nepoch):

            loss = train(opt, train_dataloader, model, optimizer, epoch)
            scheduler.step(loss)

            if (epoch+1) % 10 == 0:
                save_model_epoch(opt.checkpoint, epoch+1, model)

            print('Epoch: %d, lr: %.7f, loss: %.4f' % (epoch+1, optimizer.state_dict()['param_groups'][0]['lr'], loss))
            logging.info('Epoch: %d, lr: %.7f, loss: %.4f' % (epoch+1, optimizer.state_dict()['param_groups'][0]['lr'], loss))


    else:

        val(opt, samples, model)
        print("-----------------------Finish-----------------------")


