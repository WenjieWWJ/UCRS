import numpy as np 
import torch
import math
import time

def RMSE(model, model_name, dataloader):
    RMSE = np.array([], dtype=np.float32)
    for features, feature_values, label in dataloader:
        features = features.cuda()
        feature_values = feature_values.cuda()
        label = label.cuda()

        prediction = model(features, feature_values)
        prediction = prediction.clamp(min=-1.0, max=1.0)
        SE = (prediction - label).pow(2)
        RMSE = np.append(RMSE, SE.detach().cpu().numpy())

    return np.sqrt(RMSE.mean())

def pre_ranking(item_feature):
    '''prepare for the ranking: construct item_feature data'''

    features = []
    feature_values = []
    
    for itemID in item_feature: # for itemID in range(len(item_feature))
        features.append(np.array(item_feature[itemID][0]))
        feature_values.append(np.array(item_feature[itemID][1], dtype=np.float32))

    features = torch.tensor(features).cuda()
    feature_values = torch.tensor(feature_values).cuda()
    
    return features, feature_values

def selected_concat(user_feature, all_item_features, all_item_feature_values, userID, batch_item_idx):

    item_num = len(batch_item_idx)

    test = user_feature[userID][0]
    user_feat = torch.tensor(user_feature[userID][0]).cuda()

    user_feat = user_feat.expand(item_num, -1)
    user_feat_values = torch.tensor(np.array(user_feature[userID][1], dtype=np.float32)).cuda()
    user_feat_values = user_feat_values.expand(item_num, -1)

    batch_item_idx = torch.tensor(batch_item_idx).cuda()
    batch_item_features = all_item_features[batch_item_idx]
    batch_item_feature_values = all_item_feature_values[batch_item_idx]

    features = torch.cat([user_feat, batch_item_features], 1).cuda()
    feature_values = torch.cat([user_feat_values, batch_item_feature_values], 1).cuda()

    return features, feature_values


def Ranking(model, valid_dict, test_dict, train_dict, user_feature, all_item_features, all_item_feature_values, \
            batch_size, topN, item_map_dict, return_pred=False, is_test = False):
    """evaluate the performance of top-n ranking by recall, precision, and ndcg"""
    user_gt_test = []
    user_gt_valid = []
    user_pred = []
    user_pred_dict = {}
    user_item_top1k = {}
    item_map_dict_reverse = {v: k for k, v in item_map_dict.items()}

    for userID in test_dict:
        # features, feature_values, mask = user_rank_feature[userID]

        batch_num = all_item_features.size(0) // batch_size
        item_idx = list(range(all_item_features.size(0)))
        mask = torch.zeros(all_item_features.size(0)).cuda()
        his_items = torch.tensor(train_dict[userID]).cuda()
        mask[his_items] = -999
        if is_test == True:
            mask_valid_item = [item_map_dict[itemID] for itemID in valid_dict[userID]]
            if len(mask_valid_item) > 0:
                mask_valid_item = torch.tensor(mask_valid_item).cuda()
                mask[mask_valid_item] = -999
                
        st, ed = 0, batch_size
        for i in range(batch_num):
            batch_item_idx = item_idx[st: ed]
            batch_feature, batch_feature_values = selected_concat(user_feature, all_item_features, \
                                                                  all_item_feature_values, userID, batch_item_idx)

            prediction = model(batch_feature, batch_feature_values)
            prediction = prediction
            if i == 0:
                all_predictions = prediction
            else:
                all_predictions = torch.cat([all_predictions, prediction], 0)

            st, ed = st + batch_size, ed + batch_size

        # prediction for the last batch
        batch_item_idx = item_idx[st:]
        batch_feature, batch_feature_values = selected_concat(user_feature, all_item_features, \
                                                              all_item_feature_values, userID, batch_item_idx)

        prediction = model(batch_feature, batch_feature_values)
        if batch_num == 0:
            all_predictions = prediction
        else:
            all_predictions = torch.cat([all_predictions, prediction], 0)
        all_predictions = all_predictions + mask

        user_gt_valid.append(valid_dict[userID])
        user_gt_test.append(test_dict[userID])
        top_values, indices = torch.topk(all_predictions, topN[-1])
        pred_items = torch.tensor(item_idx)[indices].cpu().numpy().tolist()
        ### map into real index
        for i in range(len(pred_items)):
            pred_items[i] = item_map_dict_reverse[pred_items[i]]

        user_item_top1k[userID] = pred_items
        user_pred_dict[userID] = all_predictions.detach().cpu().numpy()
        user_pred.append(pred_items)

    valid_results = computeTopNAccuracy(user_gt_valid, user_pred, topN)
    test_results = computeTopNAccuracy(user_gt_test, user_pred, topN)

    if return_pred:
        return valid_results, test_results, user_pred_dict, user_item_top1k
    return valid_results, test_results

##################################end#####################################

def sigmoid(x):
    s = 1 / (1 + np.exp(-x))
    return s
    
    
def computeTopNAccuracy(GroundTruth, predictedIndices, topN, GroundTruthTarget=None):
    precision = [] 
    recall = [] 
    NDCG = [] 
    MRR = []
    WNDCG = []
    WNDCG_return = False
    
    if GroundTruthTarget is None:
        GroundTruthTarget = [[] for i in GroundTruth]
    else:
        WNDCG_return = True
    
    for index in range(len(topN)):
        sumForPrecision = 0
        sumForRecall = 0
        sumForNdcg = 0
        sumForMRR = 0
        sumForWndcg = 0
        
        for i in range(len(predictedIndices)):  # for a user,
            if len(GroundTruth[i]) != 0:
                mrrFlag = True
                userHit = 0
                userMRR = 0
                
                dcg = 0
                idcg = 0
                idcgCount = len(GroundTruth[i])
                ndcg = 0
                
                wdcg = 0
                widcg = 0
                widcgCount = len(GroundTruthTarget[i])
                wndcg = 0
                
                hit = []
                for j in range(topN[index]):
                    if predictedIndices[i][j] in GroundTruth[i]:
                        # if Hit!
                        dcg += 1.0/math.log2(j + 2)
                        if predictedIndices[i][j] in GroundTruthTarget[i]:
                            wdcg += 2.0/math.log2(j + 2)
                        else:
                            wdcg += 1.0/math.log2(j + 2)
                            
                        if mrrFlag:
                            userMRR = (1.0/(j+1.0))
                            mrrFlag = False
                        userHit += 1.0
                
                    if idcgCount > 0:
                        idcg += 1.0/math.log2(j + 2)
                        idcgCount = idcgCount-1
                        if widcgCount > 0:
                            widcg += 2.0/math.log2(j + 2)
                            widcgCount = widcgCount-1
                        else:
                            widcg += 1.0/math.log2(j + 2)
                            
                if(idcg != 0):
                    ndcg += (dcg/idcg)
                if widcg > 0:
                    wndcg += (wdcg/widcg)
                    
                sumForPrecision += userHit / topN[index]
                sumForRecall += userHit / len(GroundTruth[i])               
                sumForNdcg += ndcg
                sumForMRR += userMRR
                sumForWndcg += wndcg
        
        precision.append(round(sumForPrecision / len(predictedIndices), 4))
        recall.append(round(sumForRecall / len(predictedIndices), 4))
        NDCG.append(round(sumForNdcg / len(predictedIndices), 4))
        MRR.append(round(sumForMRR / len(predictedIndices), 4))
        WNDCG.append(round(sumForWndcg / len(predictedIndices), 4))
        
    if WNDCG_return:
        return precision, recall, NDCG, MRR, WNDCG
    return precision, recall, NDCG, MRR



def print_results(train_RMSE, valid_result, test_result):
    """output the evaluation results."""
    if train_RMSE is not None:
        print("[Train]: RMSE: {:.4f}".format(train_RMSE))
    if valid_result is not None: 
        print("[Valid]: Precision: {} Recall: {} NDCG: {} MRR: {}".format(
                            '-'.join([str(x) for x in valid_result[0]]), 
                            '-'.join([str(x) for x in valid_result[1]]), 
                            '-'.join([str(x) for x in valid_result[2]]), 
                            '-'.join([str(x) for x in valid_result[3]])))
    if test_result is not None: 
        print("[Test]: Precision: {} Recall: {} NDCG: {} MRR: {}".format(
                            '-'.join([str(x) for x in test_result[0]]), 
                            '-'.join([str(x) for x in test_result[1]]), 
                            '-'.join([str(x) for x in test_result[2]]), 
                            '-'.join([str(x) for x in test_result[3]])))

