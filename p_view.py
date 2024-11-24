"""performance viewer"""
from results import Results, Set_Rec, Setting
from typing import List
import pandas as pd
from tqdm import tqdm
import pprint
from collections import defaultdict
from functools import singledispatch

def sep_line(original_function):
    def wrapper(*args, **kwargs):
        # 这里是在调用原始函数前添加的新功能
        print("\n", "++++"*25, "\n")
        result = original_function(*args, **kwargs)
        # 这里是在调用原始函数后添加的新功能
        print("\n", "++++"*25, "\n")
       
        return result
    return wrapper

@singledispatch
def rss_show(rss:None):
    print("模型不存在训练记录")

@rss_show.register(list)
@sep_line
def rss_show_list_sr(srs):
    if len(srs) == 1:# 单模型
        rss_show(srs[0])
        return None
    settings:List[Setting] = [sr.setting for sr in srs]
    model, data = settings[0].model_name, settings[0].data_name

    set_sta = Results.settings_to_setting_statistic(settings, keep_single=False)
    if len(set_sta) == 1: # 单超参
        print("<单超参>")
        performances = defaultdict(list)
        print(set_sta)
        p, vs = list(set_sta.keys())[0], list(set_sta.values())[0]
        srs = []
        for v in vs:
            filter_dict = {p:v, "model_name":model, "data_name":data}
            srs.extend(Results.load_rs_by_filter(filter_dict))

        # 读取模型性能指标
        for sr in srs:
            for k, v in sr.record.get_best_performance().items():
                performances[k].append(v)

        for k, v in performances.items():
            print(k, v)
    elif len(set_sta) == 2:
        print("<双超参>")
        p1, p2 = set_sta.keys()
        v1, v2 = set_sta.values()
        v1, v2 = list(v1), list(v2)
        v1.sort()
        v2.sort()
        hyper_table = pd.DataFrame(columns = [f"{p1}:{v}" for v in v1], 
                                   index=[f"{p2}:{v}" for v in v2])
        for v1i in v1:
            for v2i in v2:
                filterd = {p1:v1i, p2:v2i, "model_name":model, "data_name":data}
                sr:Set_Rec = Results.load_rs_by_filter(filterd)
                print("filterd", filterd,f"模型个数：{len(sr)}")
                if len(sr)>0:
                    perdict = sr[0].record.get_best_performance()
                    hyper_table[f"{p1}:{v1i}"][f"{p2}:{v2i}"] = list(perdict.values())[0]
                
        print(hyper_table)
    else:
        print("<多超参>")
        for sr in srs:
            # 过滤不在set_sta中的参数
            filter_dict = {k:v for k,v in sr.setting.get_params().items() if k in set_sta}
            print("[Params]", filter_dict)
            print("[Performance]", sr.record.get_best_performance())
            print("[Training Time Cost]:", sr.record.training_time_cost, end = "\n\n")

@rss_show.register(Set_Rec)
def rss_show_sr(sr):
    print("仅有一个模型")
    sr.setting.show()
    pprint.pprint(sr.record.get_best_performance(), width=50)
    print("Training Time Cost:", sr.record.training_time_cost)

@sep_line
def model_data_best_setting():
    from datas import ALL_DATAS
    from models import ALL_MODELS
    print(f"[datasets]{ALL_DATAS}\n[models]{ALL_MODELS}")
    command = input("model_name data_name|quit: ")
    if command == "quit":
        return None
    else:# 最优参数        
        model_name, data_name = command.split(' ')[:2]
        sr:Set_Rec = Results.load_rs_model_data(model_name, data_name, True)
        if sr:
            sr.setting.show()
            pprint.pprint(sr.record.get_best_performance())
            print("Training Time Cost:", sr.record.training_time_cost)

@sep_line
def show_all():
    '''展示所有模型在所有数据集上的最优性能'''
    from datas import ALL_DATAS
    from models import ALL_MODELS
    
    p_table = pd.DataFrame(columns=pd.MultiIndex.from_product([ALL_DATAS, 
                                                               ["recall@20","recall@40"]])
                           , index=ALL_MODELS)
    # 加载所有记录
    all_srs = Results.load_all_set_res()
    md_srs = defaultdict(list)
    # 按模型和数据集分组
    for sr in all_srs:
        md_srs[(sr.setting.model_name, sr.setting.data_name)].append(sr)
    # 读取每个模型在每个数据集上的最优性能
    for data in tqdm(ALL_DATAS, desc = "reading"):
        for model in ALL_MODELS:
            if (model, data) in md_srs:
                md_s = md_srs[(model, data)]

                pers = [sr.record.get_best_performance()["recall@20"] for sr in md_s]
                p_table[data, "recall@20"][model] = max(pers)

                pers = [sr.record.get_best_performance()["recall@40"] for sr in md_s]
                p_table[data, "recall@40"][model] = max(pers)

    print(p_table)

def hyperparameter_analyis():
    '''展示所有模型在所有数据集上的最优性能'''
    from datas import ALL_DATAS
    from models import ALL_MODELS
    print(f"[datasets]{ALL_DATAS}\n[models]{ALL_MODELS}")
    data, model = input("Input(dataname model_name):").split(' ')
    rss: list[Set_Rec] = Results.load_rs_model_data(model, data, False)
    rss_show(rss)

def delete_model():
    res = Results()

    # 提取当前所有记录
    all_rec_set = Results.load_all_set_res()

    def show_statistic(set_recs):
        statistices = dict(Results.set_recs_to_statistic(set_recs))
        # 过滤数量为1的参数
        new_key = [k for k, vs in statistices.items() if len(vs)!=1]
        new_vs = [vs for _, vs in statistices.items() if len(vs)!=1]
        statistices = dict(zip(new_key, new_vs))
        print("**"*10,f"{len(set_recs)} models Founded","**"*10)
        pprint.pprint(statistices, compact=True, width=120)
        print("**"*20)
        
    while 1:
        show_statistic(all_rec_set)
        command = input("Filter（1），quit（quit），reset（0），delete（delete）")
        if command == "quit":
            break
        elif command == '1':
            command = input("Enter key value pair（such as：model_name bpr）")
            command = command.split(' ')
            # 列表转字典，奇数位置为参数名，偶数位置为参数值
            command_d = dict(zip([x for x in command[::2]], [x for x in command[1::2]]))
            all_rec_set = Results.filter_dict(all_rec_set, command_d)
        elif command == '0':
            all_rec_set = Results.load_all_set_res()
        elif command == "delete":
            for setrec in all_rec_set:
                Results.Delete_Result(setrec.timestamp)
            # 重置当前未删除result
            all_rec_set = res.load_all_set_res()

def train_curve():
    from models import ALL_MODELS
    from datas import ALL_DATAS
    print(f"datas:   {ALL_DATAS} \nmodels:  {ALL_MODELS}")
    model_names = ALL_MODELS # input("请输入模型（一个或多个）名称：").split(' ')
    if "all" in model_names:
        model_names = ALL_MODELS
    data_name = input("Enter data name:")
    
    filter_dict = {"model_name":model_names, "data_name":data_name}
    srs = Results.load_rs_by_filter(filter_dict)
    sr_dict = defaultdict(list)
    for sr in srs: # 按模型和数据集分组
        sr_dict[(sr.setting.model_name, sr.setting.data_name)].append(sr)
    for model_data, srs in sr_dict.items(): # 保留最大值
        sr_dict[model_data] = max(srs)
    # 持久化存储
    data_model_dict = {"data_name":data_name,
                       "train_performance":{},
                       "training_time":{}}
    # 绘制曲线

    import matplotlib.pyplot as plt
    
    plt.style.use('ggplot')
    plt.figure(figsize=(10, 6))
    # 构建两个子图
    # ax1 = plt.subplot(311)
    ax2 = plt.subplot(211)
    ax3 = plt.subplot(212)
    for model_data, sr in sr_dict.items():
        # 绘制性能指标
        for per_name, y in sr.record.training_performance.items():
            # 保存训练性能曲线
            data_model_dict["train_performance"][model_data[0]] = y
            # 绘制
            ax2.plot(y, label=f"{model_data[0]}")
            ax2.set_title(f"Training Performance of {model_names} on {data_name}")
            ax2.set_xlabel('Epoch')
            ax2.set_ylabel(per_name)
            ax2.legend()
            break
        # 训练时间
        # 保存训练时间
        data_model_dict["training_time"][model_data[0]] = sr.record.training_time_cost

        time_cost = sr.record.training_time_cost#  * (epoch_num-80)/epoch_num
        ax3.bar(model_data[0], time_cost, label=f"{model_data[0]}")
        ax3.set_ylabel('Training Time(s)')
    # 保存数据
    import json
    with open(f"visual_data/5_{data_name}.json", 'w') as f:
        json.dump(data_model_dict, f)
    # 设置matplotlib窗口出现位置
    plt.show()
    
    
command_list = [
                "1. best setting of model",
                "2. hyperparameter analysis ",
                "3. show all ",
                "4. delete model record ",
                "5. training curve show"
                ]
@sep_line
def show_commands():
    for c in command_list:
        print(c)

Results()
while 1:
    show_commands()
    command = input("command number or quit: ")
    if command == "quit":
        break
    elif command == '1':
        model_data_best_setting()
    elif command == '2':
        hyperparameter_analyis()
    elif command == '3':
        show_all()
    elif command == '4':
        delete_model()
    elif command == '5':
        train_curve()
        