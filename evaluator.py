import math

class Evaluator:
    @staticmethod
    def Recall_K(uids, predictions, test_labels, topk):
        user_num = 0
        all_recall = 0
        for i in range(len(uids)):
            uid = uids[i]
            prediction = list(predictions[uid][:topk])
            label = test_labels[uid]
            if len(label)>0:
                hit = 0
                for item in label:
                    if item in prediction:
                        hit += 1
                all_recall += hit/len(label)
                user_num += 1
        
        return all_recall
    
    @staticmethod
    def NDCG_K(uids, predictions, test_labels, topk):
        user_num = 0
        all_ndcg = 0
        idcg = sum([1/math.log2(i+2) for i in range(topk)])
        for i in range(len(uids)):
            uid = uids[i]
            prediction = list(predictions[uid][:topk])
            label = set(test_labels[uid])
            if len(label)>0:
                dcg = 0
                for j in range(topk):
                    if prediction[j] in label:
                        dcg += 1/math.log2(j+2)
                all_ndcg += dcg/idcg
                user_num += 1
        
        return all_ndcg/user_num
    
    @staticmethod
    def Training_Evaluate(predictions, test_labels):
        uids = list(range(len(predictions)))
        recall_20 = Evaluator.Recall_K(uids, predictions, test_labels, 20)/len(uids)
        ndcg_20 = Evaluator.NDCG_K(uids, predictions, test_labels, 20)
        return {'recall@20':recall_20, 'ndcg@20':ndcg_20}
