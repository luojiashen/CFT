from data_bpr import DataBPR
from tools import Tools
import os
import torch
import numpy as np
import scipy.sparse as sp
from torch import svd_lowrank


class DataLightGCN(DataBPR):
    def __init__(self, data_name):
        super(DataLightGCN, self).__init__(data_name)

    def L_eig(self, svd_k = 64):
        I = torch.ones(size=(self.num_rating, ))
        A_norm = self.A_norm
        I = torch.sparse.FloatTensor(self.ratings.T, I, 
                                     (self.user_num+self.item_num, self.user_num+self.item_num))

        L = I - A_norm
        U, S, V = svd_lowrank(L, q = svd_k)
        return U

    @property
    def L(self):
        """Laplacian matrix of the graph."""
        I = torch.ones(size=(self.num_rating, ))
        A_norm = self.A_norm
        I = torch.sparse.FloatTensor(self.ratings.T, I, 
                                     (self.user_num+self.item_num, self.user_num+self.item_num))

        return I - A_norm


    @property
    def A_norm(self):
        return self.Load_A_norm()

    @property
    def A(self):
        return self.A()
    
    def Load_A(self):
        save_path = os.path.join(self.data_root, 'A_sparse.npz')
        
        data_tr_df = self.Load_Data(use = 'train')
        users = torch.tensor(data_tr_df['uid']).view(1,-1)
        items = torch.tensor(data_tr_df['iid']).view(1,-1)
        adjacency_matrix = torch.concat([users,items],dim=0)

        values=torch.ones(adjacency_matrix.shape[1])

        R = sp.csr_matrix((values,adjacency_matrix)
                                    ,(self.user_num,self.item_num))
        adj_mat = sp.dok_matrix((self.user_num + self.item_num, 
                                self.user_num + self.item_num)
                                , dtype=np.float32)
        adj_mat = adj_mat.tolil()
        R = R.tolil()        
        
        adj_mat[:self.user_num, self.user_num:] = R
        adj_mat[self.user_num:, :self.user_num] = R.T
        adj_mat = adj_mat.tocsr()

        print('save norm_adj to:', save_path)
        sp.save_npz(save_path, adj_mat)

        Graph = Tools.sp_mat_to_sp_tensor(adj_mat)
        Graph = Graph.coalesce().cuda()
        return Graph
    
    def Load_A_norm(self):
        save_path = os.path.join(self.data_root, 'A_norm_sparse.npz')
        
        if os.path.exists(save_path): # 读取
            norm_adj = sp.load_npz(save_path)
            graph = Tools.sp_mat_to_sp_tensor(norm_adj)
            print("Load norm_adj from ", save_path)
            graph = graph.coalesce()
            return graph
        
        data_tr_df = self.Load_Data(use = 'train')
        users = torch.tensor(data_tr_df['uid']).view(1,-1)
        items = torch.tensor(data_tr_df['iid']).view(1,-1)
        adjacency_matrix = torch.concat([users,items],dim=0)

        values=torch.ones(adjacency_matrix.shape[1])
        R = sp.csr_matrix((values,adjacency_matrix)
                                    ,(self.user_num,self.item_num))
        adj_mat = sp.dok_matrix((self.user_num + self.item_num, 
                                self.user_num + self.item_num)
                                , dtype=np.float32)
        adj_mat = adj_mat.tolil()
        R = R.tolil()        
        
        adj_mat[:self.user_num, self.user_num:] = R
        adj_mat[self.user_num:, :self.user_num] = R.T
        adj_mat = adj_mat.todok()

        rowsum = np.array(adj_mat.sum(axis=1))
        d_inv = np.power(rowsum, -0.5).flatten()
        d_inv[np.isinf(d_inv)] = 0.
        d_mat = sp.diags(d_inv)
        
        norm_adj = d_mat.dot(adj_mat)
        norm_adj = norm_adj.dot(d_mat)
        norm_adj = norm_adj.tocsr()
        print('save norm_adj to:', save_path)
        sp.save_npz(save_path, norm_adj)

        Graph = Tools.sp_mat_to_sp_tensor(norm_adj)
        Graph = Graph.coalesce()
        return Graph