from models.med import BertConfig, BertModel
from transformers import BertTokenizer

import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from models.blip import create_vit, init_tokenizer, load_checkpoint

class Rap_retrieval(nn.Module):
    def __init__(self,
                 med_config = 'configs/med_config.json',
                 image_size = 384,
                 vit = 'base',
                 vit_grad_ckpt = False,
                 vit_ckpt_layer = 0,                      
                 embed_dim = 256,
                 queue_size = 57600,
                 momentum = 0.995,
                 negative_all_rank = False,
                 ):
    
        super().__init__()
        
        self.visual_encoder, vision_width = create_vit(vit,image_size, vit_grad_ckpt, vit_ckpt_layer)
        self.tokenizer = init_tokenizer()   
        med_config = BertConfig.from_json_file(med_config)
        med_config.encoder_width = vision_width
        self.text_encoder = BertModel(config=med_config, add_pooling_layer=False)          

        text_width = self.text_encoder.config.hidden_size
        
        self.vision_proj = nn.Linear(vision_width, embed_dim)
        self.text_proj = nn.Linear(text_width, embed_dim)

        self.itm_head = nn.Linear(text_width, 2) 
        
        self.visual_encoder_m, vision_width = create_vit(vit,image_size)              
        self.vision_proj_m = nn.Linear(vision_width, embed_dim)
        self.text_encoder_m = BertModel(config=med_config, add_pooling_layer=False)    
        self.text_proj_m = nn.Linear(text_width, embed_dim)
        
        self.model_pairs = [[self.visual_encoder,self.visual_encoder_m],
                            [self.vision_proj,self.vision_proj_m],
                            [self.text_encoder,self.text_encoder_m],
                            [self.text_proj,self.text_proj_m],
                           ]       
        self.copy_params()

        self.register_buffer("image_queue", torch.randn(embed_dim, queue_size))
        self.register_buffer("text_queue", torch.randn(embed_dim, queue_size))
        self.register_buffer("idx_queue", torch.full((1,queue_size),-100))
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))  

        self.image_queue = nn.functional.normalize(self.image_queue, dim=0)
        self.text_queue = nn.functional.normalize(self.text_queue, dim=0)
        
        self.queue_size = queue_size
        self.momentum = momentum
        self.temp = nn.Parameter(0.07*torch.ones([]))   
        
        self.negative_all_rank = negative_all_rank
        
    def forward(self, image, caption, alpha):

        with torch.no_grad():
            self.temp.clamp_(0.001,0.5)
        
        ori_image= image
        bs = image.shape[0]
        frames=image.shape[1]
        image = image.reshape(image.shape[0]*image.shape[1],image.shape[2],image.shape[3],image.shape[4])
        image_embeds = self.visual_encoder(image) 
        image_embeds=image_embeds.reshape(bs,frames,image_embeds.shape[1],image_embeds.shape[2])

        image_embeds=torch.mean(image_embeds,dim=1)

        image_atts = torch.ones(image_embeds.size()[:-1],dtype=torch.long).to(image.device)        
        image_feat = F.normalize(self.vision_proj(image_embeds[:,0,:]),dim=-1)          
        
        text = self.tokenizer(caption, padding='max_length', truncation=True, max_length=30, 
                              return_tensors="pt").to(image.device)  
        text_output = self.text_encoder(text.input_ids, attention_mask = text.attention_mask,                      
                                        return_dict = True, mode = 'text')            
        text_feat = F.normalize(self.text_proj(text_output.last_hidden_state[:,0,:]),dim=-1)                 
             

        with torch.no_grad():
            self._momentum_update()

            image_embeds_m = self.visual_encoder_m(image) 
            image_embeds_m=image_embeds_m.reshape(bs,frames,image_embeds_m.shape[1],image_embeds_m.shape[2])
            image_embeds_m = torch.mean(image_embeds_m,dim=1)

            image_feat_m = F.normalize(self.vision_proj_m(image_embeds_m[:,0,:]),dim=-1)  


            image_feat_m_l = F.normalize(self.vision_proj_m(image_embeds_m[:,1:,:]),dim=-1)  



            image_feat_all = torch.cat([image_feat_m.t(),self.image_queue.clone().detach()],dim=1)                   
            
            text_output_m = self.text_encoder_m(text.input_ids, attention_mask = text.attention_mask,                      
                                                return_dict = True, mode = 'text')    
            text_feat_m = F.normalize(self.text_proj_m(text_output_m.last_hidden_state[:,0,:]),dim=-1) 

            text_feat_m_l = F.normalize(self.text_proj_m(text_output_m.last_hidden_state[:,1:,:]),dim=-1) 

            text_feat_all = torch.cat([text_feat_m.t(),self.text_queue.clone().detach()],dim=1)

            sim_i2t_m = image_feat_m @ text_feat_all / self.temp  
            sim_t2i_m = text_feat_m @ image_feat_all / self.temp 

            sim_targets = torch.zeros(sim_i2t_m.size()).to(image.device)
            sim_targets.fill_diagonal_(1)          

            sim_i2t_targets = alpha * F.softmax(sim_i2t_m, dim=1) + (1 - alpha) * sim_targets
            sim_t2i_targets = alpha * F.softmax(sim_t2i_m, dim=1) + (1 - alpha) * sim_targets        

        sim_i2t = image_feat @ text_feat_all / self.temp
        sim_t2i = text_feat @ image_feat_all / self.temp

        loss_i2t = -torch.sum(F.log_softmax(sim_i2t, dim=1)*sim_i2t_targets,dim=1).mean()
        loss_t2i = -torch.sum(F.log_softmax(sim_t2i, dim=1)*sim_t2i_targets,dim=1).mean() 

        sim_local_t2i = torch.bmm(text_feat_m_l, image_feat_m_l.permute(0,2,1)) /self.temp
        sim_local_i2t = torch.bmm(image_feat_m_l,text_feat_m_l.permute(0,2,1)) / self.temp

        sim_local_t2i_max =torch.nn.Softmax(dim=-1)(torch.max(sim_local_t2i,dim=-1).values)
        sim_local_i2t_max =torch.nn.Softmax(dim=-1)(torch.max(sim_local_i2t,dim=-1).values)

        loss_t2i_crosMod_l = self.in_batch_g2l_loss(text_feat_m_l, image_feat, self.temp, text.attention_mask[:,1:],sim=sim_local_t2i_max)

        loss_i2t_crosMod_l = self.in_batch_g2l_loss(image_feat_m_l, text_feat, self.temp,sim = sim_local_i2t_max)

        sim_i2i = image_feat @ image_feat_all / self.temp
        sim_t2t = text_feat @ text_feat_all / self.temp

        loss_i2i = -torch.sum(F.log_softmax(sim_i2i, dim=1)*sim_targets,dim=1).mean()
        loss_t2t = -torch.sum(F.log_softmax(sim_t2t, dim=1)*sim_targets,dim=1).mean()

        loss_ita = (loss_i2t+loss_t2i+loss_i2i+loss_t2t)/4+loss_t2i_crosMod_l+loss_i2t_crosMod_l

        self._dequeue_and_enqueue(image_feat_m, text_feat_m)        

        encoder_input_ids = text.input_ids.clone()
        encoder_input_ids[:,0] = self.tokenizer.enc_token_id
        
        output_pos = self.text_encoder(encoder_input_ids,
                                       attention_mask = text.attention_mask,
                                       encoder_hidden_states = image_embeds,
                                       encoder_attention_mask = image_atts,      
                                       return_dict = True,
                                      )            
        with torch.no_grad():       
            weights_t2i = F.softmax(sim_t2i[:,:bs],dim=1)+1e-4 
            weights_t2i.fill_diagonal_(0)            
            weights_i2t = F.softmax(sim_i2t[:,:bs],dim=1)+1e-4  
            weights_i2t.fill_diagonal_(0)   
            

        image_embeds_neg = []    
        for b in range(bs):
            neg_idx = torch.multinomial(weights_t2i[b], 1).item()
            image_embeds_neg.append(image_embeds[neg_idx])
        image_embeds_neg = torch.stack(image_embeds_neg,dim=0)   

        text_ids_neg = []
        text_atts_neg = []
        for b in range(bs):
            neg_idx = torch.multinomial(weights_i2t[b], 1).item()
            text_ids_neg.append(encoder_input_ids[neg_idx])
            text_atts_neg.append(text.attention_mask[neg_idx])

        text_ids_neg = torch.stack(text_ids_neg,dim=0)   
        text_atts_neg = torch.stack(text_atts_neg,dim=0)      

        text_ids_all = torch.cat([encoder_input_ids, text_ids_neg],dim=0)     
        text_atts_all = torch.cat([text.attention_mask, text_atts_neg],dim=0)     

        image_embeds_all = torch.cat([image_embeds_neg,image_embeds],dim=0)
        image_atts_all = torch.cat([image_atts,image_atts],dim=0)

        output_neg = self.text_encoder(text_ids_all,
                                       attention_mask = text_atts_all,
                                       encoder_hidden_states = image_embeds_all,
                                       encoder_attention_mask = image_atts_all,      
                                       return_dict = True,
                                      )                            

        vl_embeddings = torch.cat([output_pos.last_hidden_state[:,0,:], output_neg.last_hidden_state[:,0,:]],dim=0)
        vl_output = self.itm_head(vl_embeddings)            

        itm_labels = torch.cat([torch.ones(bs,dtype=torch.long),torch.zeros(2*bs,dtype=torch.long)],
                               dim=0).to(image.device)
        loss_itm = F.cross_entropy(vl_output, itm_labels)  
                  
        return loss_ita, loss_itm
 
    def patch_pooling(self, x):
        pooled_patch_length = 16
        batch_size, seq_length, dim = x.size()
        b1 = int(np.sqrt(seq_length))
        x = x.reshape(batch_size, b1, b1, dim)
        x = x.permute(0,3,1,2)
        c1 = b1 // int(np.sqrt(pooled_patch_length))
        x = F.avg_pool2d(x, c1, stride=c1)
        x = x.permute(0,2,3,1).reshape(batch_size, pooled_patch_length, dim)
        return x


    def in_batch_g2l_loss(self, l, m, temp, attention_mask=None,sim=None):
        

        m = m.unsqueeze(1)
        N, n_locals, dim = l.size()

        u_p = torch.matmul(l, m.permute(0,2,1)).unsqueeze(2) / temp 

        if attention_mask is not None:
            temp_mask = attention_mask.unsqueeze(2).unsqueeze(3)
            u_p = (temp_mask * u_p) + (10000. * (1-temp_mask)) 

        l_n = l.reshape(-1, dim) 
        m_n = m.reshape(-1, dim) 

        u_n = torch.mm(m_n,l_n.t()) / temp 
        u_n = u_n.reshape(N, 1, N, n_locals).permute(0, 2, 3, 1) 

        
        mask = torch.eye(N)[:, :, None, None].to(l.device)
        n_mask = 1 - mask 


        u_n = (n_mask * u_n) - (10000. * (1 - n_mask)) 

    
        if attention_mask is not None:
            temp_mask = attention_mask.unsqueeze(0).unsqueeze(3).expand(N, -1, -1, -1)
            u_n = (temp_mask * u_n) - (10000. * (1-temp_mask)) 
            
        u_n = u_n.reshape(N, N * n_locals, 1).unsqueeze(dim=1).expand(-1, n_locals, -1, -1)

        pred_lgt = torch.cat([u_p, u_n], dim=2)
        pred_log = F.log_softmax(pred_lgt, dim=2)

        if attention_mask is not None: 
            loss = (torch.sum(-(pred_log[:, :, 0].squeeze()*sim), dim=1) / torch.sum(attention_mask, dim=1)).mean()
        else:
            loss = -(pred_log[:, :, 0].squeeze()*sim).mean()

        return loss

    @torch.no_grad()    
    def copy_params(self):
        for model_pair in self.model_pairs:           
            for param, param_m in zip(model_pair[0].parameters(), model_pair[1].parameters()):
                param_m.data.copy_(param.data)  
                param_m.requires_grad = False    

            
    @torch.no_grad()        
    def _momentum_update(self):
        for model_pair in self.model_pairs:           
            for param, param_m in zip(model_pair[0].parameters(), model_pair[1].parameters()):
                param_m.data = param_m.data * self.momentum + param.data * (1. - self.momentum)
                
    @torch.no_grad()
    def _dequeue_and_enqueue(self, image_feat, text_feat):

        image_feats = concat_all_gather(image_feat)
        text_feats = concat_all_gather(text_feat)

        batch_size = image_feats.shape[0]

        ptr = int(self.queue_ptr)
        assert self.queue_size % batch_size == 0  
        self.image_queue[:, ptr:ptr + batch_size] = image_feats.T
        self.text_queue[:, ptr:ptr + batch_size] = text_feats.T
        ptr = (ptr + batch_size) % self.queue_size 

        self.queue_ptr[0] = ptr 


def rap_retrieval_init(pretrained='',**kwargs):
    model = Rap_retrieval(**kwargs)
    if pretrained:
        model,msg = load_checkpoint(model,pretrained)
        print("missing keys:")
        print(msg.missing_keys)
    return model 


@torch.no_grad()
def concat_all_gather(tensor):

    tensors_gather = [torch.ones_like(tensor)
        for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output      


class GatherLayer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        output = [torch.zeros_like(x) for _ in range(torch.distributed.get_world_size())]
        torch.distributed.all_gather(output, x)
        return tuple(output)

    @staticmethod
    def backward(ctx, *grads):
        all_gradients = torch.stack(grads)
        torch.distributed.all_reduce(all_gradients)
        return all_gradients[torch.distributed.get_rank()]


def all_gather_with_grad(tensors):

    world_size = torch.distributed.get_world_size()
    if world_size == 1:
        return tensors

    tensor_all = GatherLayer.apply(tensors)

    return torch.cat(tensor_all, dim=0)
