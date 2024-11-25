from bpr import BPR
from functions import Losses
from torch import max, gather

class DNS(BPR):
    @staticmethod
    def Dynamic_Negative_Sampling(user_embs, item_embs, uids, jids):

        uid_embs, jid_embs = user_embs[uids], item_embs[jids]
        uid_embs = uid_embs.unsqueeze(dim = 1) # [batch_size, 1, embs_dim]
        scores = (uid_embs * jid_embs).sum(dim = -1) # [batch_size, neg_c]

        indices = max(scores, dim = -1)[1]
        ng_jids = gather(jids, dim = 1, index = indices.unsqueeze(-1)).squeeze()
        return item_embs[ng_jids]
    
    def loss(self, data_batch) -> dict:
        uids, iids, jids = data_batch
        uids = uids.long()
        iids = iids.long()
        jids = jids.long()
        
        user_embs, item_embs = self.encoder()

        uids_embs = user_embs[uids]
        iids_embs = item_embs[iids]
        jids_embs = self.Dynamic_Negative_Sampling(user_embs, item_embs, uids, jids)

        loss = Losses.loss_BPR(uids_embs, iids_embs, jids_embs)/uids.shape[0]

        return {"loss_dns": loss}