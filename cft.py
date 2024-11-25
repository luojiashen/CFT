

import torch
from torch import nn
import torch.nn.functional as F

import dns
from cft_setting import CFTSettings
from functions import attention_batch, sparse_softmax
class Attention_SIGFormer(nn.Module):
    def __init__(self, sample_hop, model_arc):
        super(Attention_SIGFormer, self).__init__()
        self.model_arc = model_arc

        self.spec_lambda = torch.zeros(1)
        self.path_emb = nn.Embedding(2**(sample_hop+1)-2, 1)
        nn.init.zeros_(self.path_emb.weight)
    
    def model_to_cuda(self):
        self.spec_lambda = self.spec_lambda.cuda() # 谱域注意力系数
        self.path_emb = self.path_emb.cuda() # 路径编号对应的嵌入

    def forward(self, embs, SSE, SPE):
        embs = F.layer_norm(embs, normalized_shape=(embs.shape[-1],))
        embs_attn = self.attention(
            embs, embs, embs,
            SSE,
            SPE)
        return embs_attn
        
    def attention(self, q, k, v, SSE, SPE):
        attn_emb, attn_spec, attn_path = [], [], []

        pathes = SPE[:, :2]

        attn_emb = attention_batch(q, k, pathes[:, 0], pathes[:, 1], with_dim_norm=True)
        
        if "sse" in self.model_arc:
            attn_spec = attention_batch(SSE, SSE, pathes[:, 0], pathes[:, 1])
            attn_emb_spec = attn_emb + self.spec_lambda * attn_spec
        else:
            attn_emb_spec = attn_emb

        attn_emb = sparse_softmax(pathes.T, attn_emb_spec, q.shape[0])

        if "spe" in self.model_arc:
            attn_path = self.path_emb(SPE[:, 2]).view(-1)
            attn_path = sparse_softmax(pathes.T, attn_path, q.shape[0])
            attn_emb = attn_emb + attn_path
        
        sp_graph = torch.sparse_coo_tensor(pathes.T, attn_emb, 
                                           torch.Size([q.shape[0], q.shape[0]]))
        sp_graph = sp_graph.coalesce()
        sp_graph.detach_()
        return torch.sparse.mm(sp_graph, v)

class CFT(dns.DNS):
    def __init__(self, user_num, item_num, sse, spe,
                 params: CFTSettings):
        super(CFT, self).__init__(user_num, item_num, params)
        self.params = params
        self.sse = sse
        self.spe = spe

        self.trans_layers = []
        for _ in range(params.layer_num):
            layer = Attention_SIGFormer(params.walk_len, 
                                  params.model_arc)
            self.trans_layers.append(layer)
    
    def encoder(self):
        users_emb = self.user_embs.weight
        items_emb = self.item_embs.weight
        all_embs = torch.cat([users_emb, items_emb]) 

        embs = [all_embs]
        for i in range(self.params.layer_num):
            all_embs = self.trans_layers[i](all_embs, self.sse, self.spe)
            embs.append(all_embs)
        
        embs = torch.stack(embs, dim=1)
        light_out = torch.mean(embs, dim=1)
        users, items = torch.split(light_out, [self.user_num, self.item_num])

        return users, items
    
    def save(self, path):
        super().save(path)
        torch.save({
            'model': self.state_dict(),
            'spe': self.spe,
            }, path)
    
    def load(self, load_path):
        params = torch.load(load_path, map_location=torch.device('cpu'))
        self.load_state_dict(params['model'])
        self.spe = params['spe']
    
    def model_to_cuda(self):
        super().model_to_cuda()
        self.sse = self.sse.cuda()
        self.spe = self.spe.cuda()
        for i in range(self.params.layer_num):
            self.trans_layers[i].model_to_cuda()
        