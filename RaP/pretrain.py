import argparse
import os
import ruamel.yaml as yaml
import numpy as np
import random
import time
import datetime
import json
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from torch.utils.data import DataLoader
from contextlib import nullcontext
from models.rap_pretrain import Rap_pretrain
import utils
from utils import warmup_lr_schedule, step_lr_schedule
from data import create_dataset, create_sampler, create_loader
from meta_dataloader import MetaLoader,InfiniteIterator
from torch.cuda.amp import autocast,GradScaler


def train(model, data_loader, optimizer, epoch, device, config,args,scaler):

    model.train()  
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=50, fmt='{value:.6f}'))
    metric_logger.add_meter('loss_ita', utils.SmoothedValue(window_size=50, fmt='{value:.4f}'))
    metric_logger.add_meter('loss_itm', utils.SmoothedValue(window_size=50, fmt='{value:.4f}'))    
    metric_logger.add_meter('loss_lm', utils.SmoothedValue(window_size=50, fmt='{value:.4f}'))
    header = 'Train Epoch: [{}]'.format(epoch)
    print_freq = 50
    if config['laion_path']:
        data_loader.dataset.reload_laion(epoch)
    accu_step=args.accu_step
    optimizer.zero_grad()
    print(len(data_loader))
    for i, (task, batch) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        if epoch==0:
            warmup_lr_schedule(optimizer, i, config['warmup_steps'], config['warmup_lr'], config['init_lr'])
            
        image=batch[0]
        imgs_aug= batch[1]
        caption=batch[2]
        alpha = config['alpha']*min(1,(epoch*len(data_loader)+i)/(2*len(data_loader))) 
        image = image.to(device,non_blocking=True)
        with autocast():
            loss_ita, loss_itm, loss_lm = model(image,imgs_aug, caption, alpha = alpha)  
            loss = loss_ita + loss_itm + loss_lm
            loss = loss / accu_step
        scaler.scale(loss).backward()
        if (i+1)%accu_step==0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_norm)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        
        metric_logger.update(loss_ita=loss_ita.item())
        metric_logger.update(loss_itm=loss_itm.item())
        metric_logger.update(loss_lm=loss_lm.item())
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])  

        
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger.global_avg())
    return {k: "{:.3f}".format(meter.global_avg) for k, meter in metric_logger.meters.items()}


def main(args, config):
    utils.init_distributed_mode(args)    
    
    device = torch.device(args.device)
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True


    print("Creating dataset")
    webvid_dataset = [create_dataset('webvid', config, min_scale=0.2)]
    cc3m_video_dataset = [create_dataset('cc3m_video', config, min_scale=0.2)]

    num_tasks = utils.get_world_size()
    global_rank = utils.get_rank()
    webvid_samplers = create_sampler(webvid_dataset, [True], num_tasks, global_rank)
    cc3m_video_samplers = create_sampler(cc3m_video_dataset, [True], num_tasks, global_rank)
    if args.isdebug==1:
        nw = 0
    else :
        nw = 4
    webvid_data_loader = create_loader(webvid_dataset,webvid_samplers,batch_size=[config['batch_size']], num_workers=[nw], is_trains=[True], collate_fns=[None])[0]
    cc3m_video_data_loader = create_loader(cc3m_video_dataset,cc3m_video_samplers,batch_size=[config['batch_size']], num_workers=[nw], is_trains=[True], collate_fns=[None])[0]
    
    train_loaders={'webvid':webvid_data_loader,'cc3m':cc3m_video_data_loader}
    
    #### Model #### 
    print("Creating model")
    model = Rap_pretrain(image_size=config['image_size'], vit=config['vit'], vit_grad_ckpt=config['vit_grad_ckpt'], 
                            vit_ckpt_layer=config['vit_ckpt_layer'], queue_size=config['queue_size'])

    model = model.to(device)

    optimizer = torch.optim.AdamW(params=model.parameters(), lr=config['init_lr'], weight_decay=config['weight_decay'])

    start_epoch = 0
    if args.checkpoint:    
        checkpoint = torch.load(args.checkpoint, map_location='cpu') 
        state_dict = checkpoint['model']    
        model.load_state_dict(state_dict)    
        
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch']+1      
        print('resume checkpoint from %s'%args.checkpoint)    
    
    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module    
        
    print("Start training")
    start_time = time.time()

    scaler = GradScaler()
    for epoch in range(start_epoch, config['max_epoch']):
        data_loader = MetaLoader(train_loaders,accum_steps=1,distributed=dist.get_world_size() > 1,epoch=epoch,output_dir=args.output_dir)
        step_lr_schedule(optimizer, epoch, config['init_lr'], config['min_lr'], config['lr_decay_rate'])
        train_stats = train(model, data_loader, optimizer, epoch, device, config,args,scaler) 
        if utils.is_main_process():  
            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                         'epoch': epoch,
                        }
            save_obj = {
                'model': model_without_ddp.state_dict(),
                'optimizer': optimizer.state_dict(),
                'config': config,
                'epoch': epoch,
            }
            torch.save(save_obj, os.path.join(args.output_dir, 'checkpoint_%02d.pth'%epoch))
            
            with open(os.path.join(args.output_dir, "log.txt"),"a") as f:
                f.write(json.dumps(log_stats) + "\n")
        dist.barrier()
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str)) 

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='./configs/pretrain.yaml')
    parser.add_argument('--output_dir', default='output/Pretrain')
    parser.add_argument('--checkpoint', default='')
    parser.add_argument('--evaluate', action='store_true')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')
    parser.add_argument('--isdebug', default=0, type=int)
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--distributed', default=True, type=bool)
    parser.add_argument('--accu_step', default=1, type=int)
    parser.add_argument('--clip_norm', default=1.0, type=float)
    
    args = parser.parse_args()

    config = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        
    yaml.dump(config, open(os.path.join(args.output_dir, 'config.yaml'), 'w'))    
    
    main(args, config)