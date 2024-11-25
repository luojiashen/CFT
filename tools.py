import torch
from numpy import float32
import random
import numpy as np
import scipy.sparse as sp

class Tools:
    @staticmethod
    def sp_mat_to_sp_tensor(X):
        '''sp_csr to tensor'''
        coo = X.tocoo().astype(float32)
        row = torch.Tensor(coo.row).long()
        col = torch.Tensor(coo.col).long()
        index = torch.stack([row, col])
        data = torch.FloatTensor(coo.data)
        return torch.sparse.FloatTensor(index, data, torch.Size(coo.shape))
    
    @staticmethod
    def A_norm_to_L(A_norm):
        """Normalized adjacency matrix to laplacian matrix."""
        rate_num, ratings, shape = A_norm.values().shape[0], A_norm.indices(), A_norm.shape

        I = torch.ones(size=(rate_num, ))
        I = torch.sparse.FloatTensor(ratings, I, shape)

        return I - A_norm
    
    @staticmethod
    def rating_to_A_norm(ratings, user_num, item_num):
        values=torch.ones(ratings.shape[1])

        R = sp.csr_matrix((values, ratings)
                                    ,(user_num, item_num))
        adj_mat = sp.dok_matrix((user_num + item_num, 
                                user_num + item_num)
                                , dtype=np.float32)
        adj_mat = adj_mat.tolil()
        R = R.tolil()        
        
        adj_mat[:user_num, user_num:] = R
        adj_mat[user_num:, :user_num] = R.T
        adj_mat = adj_mat.todok()
        # normalize
        rowsum = np.array(adj_mat.sum(axis=1))
        d_inv = np.power(rowsum, -0.5).flatten()
        d_inv[np.isinf(d_inv)] = 0.
        d_mat = sp.diags(d_inv)
        
        norm_adj = d_mat.dot(adj_mat)
        norm_adj = norm_adj.dot(d_mat)
        norm_adj = norm_adj.tocsr()

        Graph = Tools.sp_mat_to_sp_tensor(norm_adj)
        Graph = Graph.coalesce()
        
        return Graph
    
    @staticmethod
    def random_walk(start_node, walk_length, adj_table, with_neg_edges = True):
        path = [[start_node]]
        path_types = []
        for i in range(walk_length):
            last_node = path[-1][-1]

            if last_node not in adj_table:
                return None

            p_or_n = random.randint(0, 1) if with_neg_edges else 1# pos or neg
            path_type = path_types[-1] + p_or_n * 2**i if path_types else p_or_n

            if p_or_n == 1:
                neighbor = random.choice(list(adj_table[last_node]))
            else:
                neighbor = random.randint(0, len(adj_table)-1)
                while neighbor in adj_table[last_node]:
                    neighbor = random.randint(0, len(adj_table))

            path[-1].append(neighbor)
            path_types.append(path_type)
            path.append([neighbor])

        return [p+[pt] for p, pt in zip(path[:-1], path_types)]
    
    @staticmethod
    def random_walk_sigformer(start_node, walk_length, adj_table, neg_adj_table):
        path = [[start_node]]
        path_types = []
        for i in range(walk_length):
            last_node = path[-1][-1]

            if last_node not in adj_table:
                return None

            p_or_n = random.randint(0, 1)
            path_type = path_types[-1] + p_or_n * 2**i if path_types else p_or_n

            neighbor_choice = adj_table if p_or_n else neg_adj_table
            neighbor = random.choice(list(neighbor_choice[last_node]))

            path[-1].append(neighbor)
            path_types.append(path_type)
            path.append([neighbor])

        return [p+[pt] for p, pt in zip(path[:-1], path_types)]