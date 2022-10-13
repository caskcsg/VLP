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
from transformers import BertTokenizer
from models.rap_retrieval import rap_retrieval_init
import utils
from data.MSRVTT_dataset import msrvtt7k_dataset,msrvtt9k_compact_dataset
from data.MSVD_dataset import msvd_dataset
from data.LSMDC_dataset import lsmdc_dataset
from data.DiDeMo_dataset import didemo_dataset

from utils import cosine_lr_schedule
from data import create_dataset, create_sampler, create_loader
import copy

def train(model, data_loader, optimizer, epoch, device, config):
    
    # train
    model.train()  
    
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('loss_itm', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    metric_logger.add_meter('loss_ita', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    header = 'Train Epoch: [{}]'.format(epoch)
    print_freq = 50

    for i,(image, caption) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        image = image.to(device,non_blocking=True)   
       
        if epoch>0:
            alpha = config['alpha']
        else:
            alpha = config['alpha']*min(1,i/len(data_loader))

        loss_ita, loss_itm = model(image, caption, alpha=alpha)
        loss = loss_ita + loss_itm
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()    
        
        metric_logger.update(loss_itm=loss_itm.item())
        metric_logger.update(loss_ita=loss_ita.item())
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger.global_avg())     
    return {k: "{:.3f}".format(meter.global_avg) for k, meter in metric_logger.meters.items()}  

@torch.no_grad()
def evaluation(model, data_loader, tokenizer, device, config,args):
    # test
    model.eval() 
    
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Evaluation:'    

    print('Computing features for evaluation...')
    start_time = time.time()

    # texts = data_loader.dataset.text
    # num_text = text_embeds.shape[0]

    # print(num_text)
    texts = data_loader.dataset.text
    num_text = len(texts)
    text_bs = 1024
    text_ids = []
    text_embeds = []  
    text_atts = []
    print('infer text')

    for i in range(0, num_text, text_bs):
        text = texts[i: min(num_text, i+text_bs)]
        text_input = tokenizer(text, padding='max_length', truncation=True, max_length=args.max_len, return_tensors="pt").to(device) 
        text_output = model.text_encoder(text_input.input_ids, attention_mask = text_input.attention_mask, mode='text')  
        text_embed = F.normalize(model.text_proj(text_output.last_hidden_state[:,0,:]))
        text_embeds.append(text_embed.cpu())
        text_ids.append(text_input.input_ids)
        text_atts.append(text_input.attention_mask)
    
    text_embeds = torch.cat(text_embeds,dim=0)
    text_ids = torch.cat(text_ids,dim=0)
    text_atts = torch.cat(text_atts,dim=0)
    text_ids[:,0] = tokenizer.additional_special_tokens_ids[0]

    video_feats = []
    video_embeds = []
    
    print('infer')
    now_cnt =0

    for video,caption in data_loader:
        if now_cnt%10==0:
            import datetime
            print(datetime.datetime.now())
            print(now_cnt)
        # import ipdb;ipdb.set_trace()
        now_cnt+=1
        # print(video.size())
        B,N,C,W,H = video.size()
        video = video.view(-1,C,W,H)
        video = video.to(device,non_blocking=True) 
        video_feat = model.visual_encoder(video) 
        video_embed = model.vision_proj(video_feat[:,0,:])
        video_embed = video_embed.view(B,N,-1).mean(dim=1)
        video_embed = F.normalize(video_embed,dim=-1)  
        video_feat = video_feat.view(B,-1,video_feat.shape[-1])

        video_feats.append(video_feat.cpu())
        video_embeds.append(video_embed.cpu())

    video_feats = torch.cat(video_feats,dim=0)
    video_embeds = torch.cat(video_embeds,dim=0)
    
    sims_matrix = video_embeds @ text_embeds.t()
    
    score_matrix_v2t = torch.full((text_embeds.shape[0],text_embeds.shape[0]),-100.0).to(device) 
    
    num_tasks = utils.get_world_size()
    rank = utils.get_rank() 
    step = sims_matrix.size(0)//num_tasks + 1
    start = rank*step
    end = min(sims_matrix.size(0),start+step)

    print('infer itm')


    sims_matrix = sims_matrix.t()

    score_matrix_t2v = torch.full((text_embeds.shape[0],text_embeds.shape[0]),-100.0).to(device)
    
    step = sims_matrix.size(0)//num_tasks + 1
    start = rank*step
    end = min(sims_matrix.size(0),start+step)    
    config['k_test']= min(config['k_test'],video_embeds.shape[0])
    print(config['k_test'])
    for i,sims in enumerate(metric_logger.log_every(sims_matrix[start:end], 50, header)): 
        
        topk_sim, topk_idx = sims.topk(k=config['k_test'], dim=0)
        encoder_output = video_feats[topk_idx].to(device,non_blocking=True) 
        encoder_att = torch.ones(encoder_output.size()[:-1],dtype=torch.long).to(device,non_blocking=True) 
        output = model.text_encoder(text_ids[start+i].repeat(config['k_test'],1), 
                                    attention_mask = text_atts[start+i].repeat(config['k_test'],1),
                                    encoder_hidden_states = encoder_output,
                                    encoder_attention_mask = encoder_att,                             
                                    return_dict = True,
                                   )
        score = model.itm_head(output.last_hidden_state[:,0,:])[:,1]
        score_matrix_t2v[start+i,topk_idx] = score + topk_sim.to(device)

    if args.distributed:
        dist.barrier()   
        torch.distributed.all_reduce(score_matrix_v2t, op=torch.distributed.ReduceOp.SUM) 
        torch.distributed.all_reduce(score_matrix_t2v, op=torch.distributed.ReduceOp.SUM)        
        
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Evaluation time {}'.format(total_time_str)) 

    return score_matrix_v2t.cpu().numpy(), score_matrix_t2v.cpu().numpy()


            
@torch.no_grad()
def itm_eval(scores_v2t, scores_t2v, txt2vmg):
    
    ranks = np.zeros(scores_t2v.shape[0])
    
    for index,score in enumerate(scores_t2v):
        inds = np.argsort(score)[::-1]
        ranks[index] = np.where(inds == txt2vmg[index])[0][0]
    
    mdR = np.median(ranks+1)
        
    # Compute metrics
    vr1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    vr5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    vr10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)

    vr_mean = (vr1 + vr5 + vr10) / 3


    eval_result =  {'txt_r1': 0,
                    'txt_r5': 0,
                    'txt_r10': 0,
                    'txt_r_mean': 0,
                    'vid_r1': vr1,
                    'vid_r5': vr5,
                    'vid_r10': vr10,
                    'vid_r_mean': vr_mean,
                    'vid_mdR': mdR,
                    'r_mean': 0}
    return eval_result


def main(args, config):
    utils.init_distributed_mode(args)    
    
    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True

    #### Dataset #### 
    print("Creating retrieval dataset")
    task = args.config.split('_')[-1].split('.yaml')[0]
    print(task)
    num_w=args.num_wroker
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    tokenizer.add_special_tokens({'bos_token':'[DEC]'})
    tokenizer.add_special_tokens({'additional_special_tokens':['[ENC]']})       
    tokenizer.enc_token_id = tokenizer.additional_special_tokens_ids[0]  

    if task=='msrvtt9kcompact':
        train_dataset = msrvtt9k_compact_dataset(config['video_root'],data_split='train',num_frm=config['num_frm_train'],
                                    max_img_size=config['image_size'], frm_sampling_strategy=args.frm_sampling_strategy)
        val_dataset = msrvtt7k_dataset(config['video_root'],data_split='test',num_frm=config['num_frm_test'],
                                    max_img_size=config['image_size'], frm_sampling_strategy='uniform')
        test_dataset = msrvtt7k_dataset(config['video_root'],data_split='test',num_frm=config['num_frm_test'],
                                    max_img_size=config['image_size'], frm_sampling_strategy='uniform')
    if task=='msrvtt7k':
        train_dataset = msrvtt7k_dataset(config['video_root'],data_split='train',num_frm=config['num_frm_train'],
                                    max_img_size=config['image_size'], frm_sampling_strategy=args.frm_sampling_strategy)
        val_dataset = msrvtt7k_dataset(config['video_root'],data_split='val',num_frm=config['num_frm_test'],
                                    max_img_size=config['image_size'], frm_sampling_strategy='uniform')
        test_dataset = msrvtt7k_dataset(config['video_root'],data_split='test',num_frm=config['num_frm_test'],
                                    max_img_size=config['image_size'], frm_sampling_strategy='uniform')
    elif task=='msvd':
        train_dataset = msvd_dataset(config['video_root'],data_split='train',num_frm=config['num_frm_train'],
                        max_img_size=config['image_size'])
        test_dataset = msvd_dataset(config['video_root'],data_split='test',num_frm=config['num_frm_test'],
                        max_img_size=config['image_size'])
        val_dataset = msvd_dataset(config['video_root'],data_split='val',num_frm=config['num_frm_test'],
                        max_img_size=config['image_size'])

    elif task=='didemo':
        
        train_dataset = didemo_dataset(config['video_root'],data_split='train',num_frm=config['num_frm_train'],
                                    max_img_size=config['image_size'], frm_sampling_strategy=args.frm_sampling_strategy)
        val_dataset = didemo_dataset(config['video_root'],data_split='val',num_frm=config['num_frm_test'],
                                    max_img_size=config['image_size'], frm_sampling_strategy='uniform')
        test_dataset = didemo_dataset(config['video_root'],data_split='test',num_frm=config['num_frm_test'],
                                    max_img_size=config['image_size'], frm_sampling_strategy='uniform')
    elif task=='lsmdc':
        train_dataset = lsmdc_dataset(config['video_root'],data_split='train',num_frm=config['num_frm_train'],
                                    max_img_size=config['image_size'], frm_sampling_strategy=args.frm_sampling_strategy)
        val_dataset = lsmdc_dataset(config['video_root'],data_split='val',num_frm=config['num_frm_test'],
                                    max_img_size=config['image_size'], frm_sampling_strategy='uniform')
        test_dataset = lsmdc_dataset(config['video_root'],data_split='test',num_frm=config['num_frm_test'],
                                    max_img_size=config['image_size'], frm_sampling_strategy='uniform')

    if args.distributed:
        num_tasks = utils.get_world_size()
        global_rank = utils.get_rank()            
        samplers = create_sampler([train_dataset], [True], num_tasks, global_rank) + [None, None]
    else:
        samplers = [None, None, None]
    
    print('dataloader')

    train_loader, val_loader, test_loader = create_loader([train_dataset, val_dataset, test_dataset],samplers,
                                                          batch_size=[config['batch_size_train']]+[config['batch_size_test']]*2,
                                                        #   num_workers=[4,4,4],
                                                          num_workers=[num_w,num_w,num_w],
                                                          is_trains=[True, False, False], 
                                                          collate_fns=[None,None,None])
    print("Creating model")
    model = rap_retrieval_init(pretrained=args.pretrained, image_size=config['image_size'], vit=config['vit'])
    
    model = model.to(device)
    
    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module   
    

    optimizer = torch.optim.AdamW(params=model.parameters(), lr=args.init_lr, weight_decay=config['weight_decay'])

    print("Start training")
    start_time = time.time()
    best = 0
    best_epoch = 0

    for epoch in range(0, config['max_epoch']):
        if args.evaluate==0:
            if args.distributed:
                train_loader.sampler.set_epoch(epoch)

            cosine_lr_schedule(optimizer, epoch, config['max_epoch'], args.init_lr, config['min_lr'])

            train_stats = train(model, train_loader, optimizer, epoch,  device, config) 

        torch.distributed.barrier()
        if (epoch+1) % args.eval_epoch ==0 or args.evaluate==1:

            score_val_i2t, score_val_t2i, = evaluation(model_without_ddp, val_loader, model_without_ddp.tokenizer, device, config,args)
            val_result = itm_eval(score_val_i2t, score_val_t2i,val_loader.dataset.txt2video) 
            print('eval_result') 
            print(val_result)
            if val_result['vid_r_mean']>best:
                score_test_i2t, score_test_t2i = evaluation(model_without_ddp, test_loader, model_without_ddp.tokenizer, device, config,args)
                test_result = itm_eval(score_test_i2t, score_test_t2i,test_loader.dataset.txt2video)
                best = val_result['vid_r_mean']
                print('test_result')
                print(test_result)
                save_obj = {
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'config': config,
                    'epoch': epoch,
                }
                if utils.is_main_process():
                    torch.save(save_obj, os.path.join(args.output_dir, 'checkpoint_%02d.pth'%epoch))  
                
        if args.evaluate: 
            break
        dist.barrier()     
        torch.cuda.empty_cache()
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str)) 


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='./configs/retrieval_msrvtt.yaml')
    parser.add_argument('--output_dir', default='output/Retrieval_msrvtt')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--max_len', default=64, type=int)
    parser.add_argument('--num_wroker', default=4, type=int)
    parser.add_argument('--eval_epoch', default=1, type=int)
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')    
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--distributed', default=True, type=bool)
    parser.add_argument('--pretrained', type=str)
    parser.add_argument('--frm_sampling_strategy', default='rand', type=str)
    parser.add_argument('--init_lr', default=1e-5, type=float)
    parser.add_argument('--evaluate', default=0, type=int)

    args = parser.parse_args()

    config = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        
    yaml.dump(config, open(os.path.join(args.output_dir, 'config.yaml'), 'w'))    
    
    main(args, config)