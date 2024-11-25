from tqdm import tqdm

from torch import nn, concat

from functions import Losses, Similarities
from abstract import AbstractCFRecommender


class BPR(nn.Module, AbstractCFRecommender):
    def __init__(self, user_num, item_num, 
                 params):
        super(BPR, self).__init__()
        self.user_num = user_num
        self.item_num = item_num
        self.params = params

        self.user_embs = nn.Embedding(user_num, params.embs_dim)
        self.item_embs = nn.Embedding(item_num, params.embs_dim)
        nn.init.normal_(self.user_embs.weight, std=0.01)
        nn.init.normal_(self.item_embs.weight, std=0.01)

    def model_to_cuda(self):
        self.user_embs = self.user_embs.cuda()
        self.item_embs = self.item_embs.cuda()

    def encoder_predict(self):
        return self.encoder()

    def encoder(self):
        return self.user_embs.weight, self.item_embs.weight

    def loss(self, data_batch) -> dict:
        uids, iids, jids = data_batch
        uids = uids.long()
        iids = iids.long()
        jids = jids.squeeze().long()# 单一负样本则压缩第二个维度

        user_embs, item_embs = self.encoder()

        uids_embs = user_embs[uids]
        iids_embs = item_embs[iids]
        jids_embs = item_embs[jids]

        loss = Losses.loss_BPR(uids_embs, iids_embs, jids_embs)
        return {"loss_BPR": loss}
    
    def predict(self, topk, uid_rated_items):
        user_embs, item_embs = self.encoder_predict()
        item_embs = item_embs.data
        user_embs = user_embs.data
        R_preds = []
        for uid in tqdm(range(self.user_num), desc='R_preds'):
            uid_embs = user_embs[uid].view(1, -1)
            uid_preds = Similarities.sim_ip(uid_embs, item_embs)
            rated_items = uid_rated_items[uid]
            uid_preds[rated_items] = -1e8
            R_preds.append(uid_preds.topk(topk)[1].view(-1, 1).cpu())
        R_preds = concat(R_preds, dim = 1).T
        return R_preds.numpy()
    
