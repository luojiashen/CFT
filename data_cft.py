from data_lightgcn import DataLightGCN
import tqdm
from tools import Tools
from torch import tensor, concat, svd_lowrank

class DataCFT(DataLightGCN):
    def __init__(self, data_name):
        super(DataCFT, self).__init__(data_name)
        self.path_types = []
    
    def sse(self, sse_dim, sse_type):
        if sse_type == "L":
            print("[L Spectral Encoding]")
            L = self.L
            eig_vec = svd_lowrank(L, q = sse_dim)[0]
            return eig_vec
        
        print("[R Spectral Encoding]")
        R = self.Load_R(R_type="sparse_tenor")
        U, _, V = svd_lowrank(R, q = sse_dim)
        eig_vec = concat([U, V], dim=0)

        return eig_vec

    def pathes_ptypes(self, sample_hop, path_num_per_node = 5, with_neg_edges = True):
        adj_table = self.adj_table
        pathes_ptypes = []
        for i in tqdm.tqdm(range(self.user_num+self.item_num), desc="[Perference Path Sampling]"):
            for j in range(path_num_per_node):

                ppt = Tools.random_walk(i, sample_hop, adj_table, with_neg_edges)
                if ppt is not None:
                    pathes_ptypes.append(tensor(ppt))
        pathes_ptypes = concat(pathes_ptypes, dim=0)
        print("Path Number", pathes_ptypes.shape)

        return pathes_ptypes