from data import DataBasic
from random import shuffle
from torch.utils.data import Dataset, DataLoader
from torch import randperm, zeros_like, nonzero
from tqdm import tqdm

class Dataset_NS_Batch(Dataset):
    def __init__(self, ratings, user_neg_dict, item_num, user_num, neg_c):
        super(Dataset_NS_Batch, self).__init__()
        self.ratings = ratings.tolist()
        self.item_num = item_num
        self.user_neg_dict = user_neg_dict
        self.user_num = user_num
        self.neg_c = neg_c
        self.ng_times = [0] * self.user_num
        self.neg_item_num_min = min([i.shape[0] for i in self.user_neg_dict.values()])
        self.init_shuffle()
        
    def shuffle(self):
        pass

    def init_shuffle(self):
        self.ng_times = [0] * self.user_num
        shuffle(self.ratings)
        for uid in tqdm(range(self.user_num), desc = '[Shuffle]'):
            rand_idx = randperm(self.user_neg_dict[uid].shape[0])
            self.user_neg_dict[uid] = self.user_neg_dict[uid][rand_idx].long()

    def __len__(self):
        return len(self.ratings)

    def __getitem__(self, idx):
        ratings = self.ratings
        user = ratings[idx][0]
        item_i = ratings[idx][1]
        item_j = self.user_neg_dict[user][self.ng_times[user] * self.neg_c: (self.ng_times[user] + 1)* self.neg_c]
        self.ng_times[user] += 1
        if (self.ng_times[user] + 1) * self.neg_c > self.neg_item_num_min:
            self.ng_times[user] = 0
        return user, item_i, item_j
    
class DataBPR(DataBasic):
    def __init__(self, data_name):
        super(DataBPR, self).__init__(data_name)

    def Load_User_Neg_Dict(self):
        R = self.Load_R('train', "dense_tensor")
        R_inv = zeros_like(R)
        R_inv[R == 0] = 1

        user_neg_dict = dict()
        for uid in tqdm(range(self.user_num), desc = "[UNI]"):
            neg_items = nonzero(R_inv[uid]).squeeze()
            user_neg_dict[uid] = neg_items
        return user_neg_dict
    
    def Get_Train_Loader(self, batch_size = 64, neg_c = 1):
        ratings = self.Load_Data(use = 'train', type = 'tensor')
        user_neg_dict = self.Load_User_Neg_Dict()
        dataset = Dataset_NS_Batch(ratings, user_neg_dict, self.item_num, 
                                   self.user_num, neg_c)
        train_loader = DataLoader(dataset, batch_size = batch_size, shuffle= True)

        return train_loader
    