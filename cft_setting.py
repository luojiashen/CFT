from basic_settings import BasicSettings
"""
Collaborative Filtering Transformer (CFT)
"""
class CFTSettings(BasicSettings):
    def __init__(self, data_name):
        super(CFTSettings, self).__init__('cft', data_name)
        self.param_init()
        self.test_epoch = 5
        self.embs_dim = 64
        self.layer_num = 1
        self.neg_c = self.neg_c_map[data_name]

        self.walk_len = 4
        self.path_num = self.path_num_map[data_name]
        self.with_neg_edges = True

        self.model_arc = self.model_arc_map[data_name]
        self.sse_dim = 64
        self.sse_type = "R"
    
    def param_init(self):
        self.neg_c_map = {"ml_100k": 5, "ml_1m": 10, "douban_book": 50
                          , "yelp2018": 50, "gowalla": 50}
        self.path_num_map = {"ml_100k": 60, "ml_1m": 80, "douban_book": 40
                             , "yelp2018": 16, "gowalla": 16}
        self.model_arc_map = {"ml_100k": " ", "ml_1m": "spe sse", "douban_book": "sse"
                              , "yelp2018": "sse", "gowalla": ' '}
    


model_parameters:list = ['embs_dim',
                          "sse_dim", "walk_len","path_num","sse_type",
                          "model_arc"
                          ]