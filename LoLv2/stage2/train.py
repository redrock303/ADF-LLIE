import argparse
import os
import os.path as osp
import logging

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from tensorboardX import SummaryWriter
import sys
sys.path.append('/dataset/kunzhou/project/low_light_noisy/ECCV2024_LDRM')
sys.path.append('/dataset/kunzhou/project/package_l')

from config import config
from utils import common, dataloader, solver, model_opr

from dataloader import  NYU_v2_datset

from models.unet_lap import UNet
from validate import validate



torch.backends.cudnn.enabled = False
def init_dist(local_rank):
    if mp.get_start_method(allow_none=True) != 'spawn':
        mp.set_start_method('spawn', force=True)
    print('local_rank',local_rank)
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl", init_method='env://')
    dist.barrier()

def iteration_enhancement(ins_temporal,gsmooth):
    b,n,c,h,w = ins_temporal.size()
    ins = ins_temporal.view(b*n,c,h,w)
    out = ins.clone()
    for i in range(2):
        out_smooth = gsmooth(out)
        dog = ins - out_smooth
        out = out + dog
    out=torch.clamp(out,0,1.0)
    out = out.view(b,n,c,h,w)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()

    # initialization
    rank = 0
    num_gpu = 1
    distributed = False
    if 'WORLD_SIZE' in os.environ:
        num_gpu = int(os.environ['WORLD_SIZE'])
        distributed = num_gpu > 1
    if distributed:
        rank = args.local_rank
        init_dist(rank)
    common.init_random_seed(config.DATASET.SEED+rank)

    model_name = config.model_version
    # set up dirs and log
    exp_dir, cur_dir = osp.split(osp.split(osp.realpath(__file__))[0])
    root_dir = osp.split(exp_dir)[0]

    model_log_root_dir = osp.join(root_dir, 'logs', cur_dir)
    common.mkdir(model_log_root_dir)

    log_dir = osp.join(model_log_root_dir, model_name)
    model_dir = osp.join(log_dir, 'models')
    solver_dir = osp.join(log_dir, 'solvers')
    if rank <= 0:
        common.mkdir(log_dir)
        
        
        common.mkdir(model_dir)
        common.mkdir(solver_dir)
        save_dir = osp.join(log_dir, 'saved_imgs')
        common.mkdir(save_dir)
        tb_dir = osp.join(log_dir, 'tb_log')
        tb_writer = SummaryWriter(tb_dir)
        common.setup_logger('base', log_dir, 'train', level=logging.INFO, screen=True, to_file=True)
        logger = logging.getLogger('base')

    
    train_dataset = NYU_v2_datset(split = 'train',patch_width = 64,path_height = 64,rank=rank)
    train_loader = dataloader.train_loader(train_dataset, config,rank=rank, is_dist=distributed)
    if rank <= 0:
        print('---per gpu batch size',config.DATALOADER.IMG_PER_GPU)
        val_dataset = NYU_v2_datset(split = 'test',rank=0)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=8, pin_memory=True)
        # print('val_loader',val_loader.__len__())


    # model
    
    model =  UNet()
    print("model have {:.3f}M paramerters in total".format(sum(x.numel() for x in model.parameters())/1000000.0))
    
    if config.CONTINUE_ITER:
        model_path = osp.join(model_dir, '%d.pth' % config.CONTINUE_ITER)
        if rank <= 0:
            logger.info('[Loading] Iter: %d' % config.CONTINUE_ITER)
        model_opr.load_model(model, model_path, strict=False, cpu=True)
    elif config.INIT_MODEL:
        model_opr.load_model(model, config.INIT_MODEL, strict=False, cpu=True)

    device = torch.device(config.MODEL.DEVICE)
    model.to(device)
    # sisr_net.to(device)
    
    if distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[torch.cuda.current_device()])


    # solvers
    optimizer = solver.make_optimizer(config, model)  # lr without X num_gpu
    lr_scheduler = solver.make_lr_scheduler(config, optimizer)
    iteration = 0

    if config.CONTINUE_ITER:
        solver_path = osp.join(solver_dir, '%d.solver' % config.CONTINUE_ITER)
        iteration = model_opr.load_solver(optimizer, lr_scheduler, solver_path)
    

    best_psnr = 0
    for epoch in range(1000):
        for batch_data in train_loader:
            model.train()
            iteration = iteration + 1
            lr_img = batch_data[0].to(device)
            s1_img = batch_data[1].to(device)
            hr_img = batch_data[2].to(device)


            # print('lr_img',lr_img.shape,hr_img.shape)

            loss_dict = model(lr_img,s1_img,hr_img) # ,img_noise,img_clean

            
            total_loss = sum(loss for loss in loss_dict.values())

            # if float(total_loss.item()) < 0.3: # for strange loss here ...
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            lr_scheduler.step()

            if rank <= 0:
                if iteration % config.LOG_PERIOD == 0 or iteration == config.SOLVER.MAX_ITER:
                    log_str = 'Iter: %d, LR: %.3e, ' % (iteration, optimizer.param_groups[0]['lr'])
                    for key in loss_dict:
                        tb_writer.add_scalar(key, loss_dict[key].mean(), global_step=iteration)
                        log_str += key + ': %.4f, ' % float(loss_dict[key])
                    logger.info(log_str)

                if iteration % config.SAVE_PERIOD == 0 or iteration == config.SOLVER.MAX_ITER:
                    logger.info('[Saving] Iter: %d' % iteration)


                    
                   
                if iteration % config.VAL.PERIOD == 0 or iteration == config.SOLVER.MAX_ITER:
                    logger.info('[Validating] Iter: %d' % iteration)
                    model.eval()
                    psnr, ssim = validate(model, val_loader, device, iteration, 
                                         save_path=save_dir, save_img=True
                                        )
                    if best_psnr < psnr:
                        best_psnr = psnr 
                        model_path = osp.join(model_dir, 'best_unet_lolv2.pth')
                        model_opr.save_model(model, model_path)

                    logger.info('[Val Result] Iter: %d, PSNR: %.4f, SSIM: %.4f best psnr: %.4f' % (iteration, psnr, ssim,best_psnr))

                    

                if iteration >= config.SOLVER.MAX_ITER:
                    logger.info('Finish training process!')
                    break


if __name__ == '__main__':
    main()
