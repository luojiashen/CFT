from torch import nn, diag
import torch
from torch.nn import functional as F

class Losses:
    @staticmethod
    def loss_BPR(user_embs, pos_item_embs, neg_item_embs):

        pos_score = (user_embs * pos_item_embs).sum(dim=-1)
        neg_score = (user_embs * neg_item_embs).sum(dim=-1)
        
        loss = - (pos_score - neg_score).sigmoid().log().mean()
        return loss
    
    @staticmethod
    def InfoNCE(view1, view2, temp):
        '''InfoNCE loss'''
        view1, view2 = nn.functional.normalize(view1, dim=1), F.normalize(view2, dim=1)
        pos_score = (view1 @ view2.T) / temp
        score = diag(F.log_softmax(pos_score, dim=1))
        return -score.mean()


def attention_batch(q, k, path0, path1, with_dim_norm = False):

    attn_bs = 20480
    embs_dim = torch.tensor(q.shape[-1]) if with_dim_norm else torch.tensor(1.)
    attn, path_num = [], path0.shape[0]
    for i in range(path_num//attn_bs+1):
        p0_bs = path0[i*attn_bs:min(path_num, (i+1)*attn_bs)]
        p1_bs = path1[i*attn_bs:min(path_num, (i+1)*attn_bs)]
        attn.append(torch.mul(q[p0_bs], k[p1_bs]).sum(dim=-1) * 1./torch.sqrt(embs_dim))
    return torch.concat(attn, dim=0)

def sum_norm(indices, values, n):
    s = torch.zeros(n, device=values.device).scatter_add(0, indices[0], values)
    s[s == 0.] = 1.
    return values/s[indices[0]]

def sparse_softmax(indices, values, n):
    return sum_norm(indices, torch.clamp(torch.exp(values), min=-5, max=5), n)


class Similarities:
    @staticmethod
    def sim_ip(user_embs, item_embs):
        """Inner Product Similarity"""
        return (user_embs * item_embs).sum(dim=-1)
