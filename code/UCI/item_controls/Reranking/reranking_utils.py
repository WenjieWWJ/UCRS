import random
import numpy as np
import torch
import math
import time

def get_group_distribution(user_list, interaction_dict, item_feature, category_len, is_category_avg = True):
    distribution = [0] * category_len
    distribution_user = [0] * category_len
    for user in user_list:
        distribution = [0] * category_len
        for item in interaction_dict[user]:
            for cate in item_feature[item]:
                if is_category_avg == True:
                    distribution[cate] += 1/ len(item_feature[item])
                else:
                    distribution[cate] += 1
        distribution_user = [distribution_user[i] + distribution[i] / len(interaction_dict[user]) for i in
                             range(category_len)]
    distribution_avg = [i / len(user_list) for i in distribution_user]

    return distribution_avg


def get_interest_group(train_dict, item_category, category_list, threshold_favorite_proportion, threshold_group_number):
    interest_group = {}

    for user in train_dict:
        distribution = get_group_distribution([user], train_dict, item_category, len(category_list), is_category_avg=False)
        if max(distribution) > threshold_favorite_proportion:
            interest = distribution.index(max(distribution))
            if interest not in interest_group:
                interest_group[interest] = [user]
            else:
                interest_group[interest].append(user)

    drop_key = []
    for key in interest_group:
        if len(interest_group[key]) < threshold_group_number:
            drop_key.append(key)

    for key in drop_key:
        interest_group.pop(key)

    interest_group_size = {}
    for interest in interest_group:
        interest_group_size[interest] = len(interest_group[interest])

    interest_group_size = dict(sorted(interest_group_size.items(), key=lambda e: e[1], reverse=True))
    sorted_interest_group = {}
    for interest in interest_group_size:
        sorted_interest_group[interest] = interest_group[interest]

    return sorted_interest_group


def reranking(pred_dict, train_dict, item_category, category_list, threshold_proportion,
                                        threshold_group_number, random_seed, alpha, filter_category = 3):
    threshold_proportion = 0
    threshold_group_number = 0
    # interest_group = get_interest_group(train_dict, item_category, category_list, threshold_proportion,
    #                                     threshold_group_number)

    target_interest = list(range(len(category_list)))
    random.seed(random_seed)
    target_list = random.choices(target_interest, k=2*len(train_dict))

    count = 0
    user_target = {}
    weight_dict = {}

    for user in train_dict:
        user_distribution = get_group_distribution([user], train_dict, item_category, len(category_list), \
                                                   is_category_avg=False)
        _, index = torch.topk(torch.tensor(user_distribution), filter_category)
        while target_list[count] in index:
            count += 1
        user_target[user] = target_list[count]
        count += 1
        weight_dict[user] = [0] * len(pred_dict[user])
        for item in pred_dict[user]:
            if user_target[user] in item_category[item]:
                weight_dict[user][pred_dict[user].index(item)] = alpha

    return weight_dict, user_target

def Ranking(pred_dict, valid_dict, test_dict, train_dict, all_item_features, all_item_feature_values, \
             topN, item_map_dict, reranking_score, return_pred=False):
    """evaluate the performance of top-n ranking by recall, precision, and ndcg"""
    user_gt_test = []
    user_gt_valid = []
    test_user_pred = []
    valid_user_pred = []
    user_pred_dict = {}
    user_item_top1k = {}
    item_map_dict_reverse = {v: k for k, v in item_map_dict.items()}

    for userID in test_dict:
        # features, feature_values, mask = user_rank_feature[userID]
        # item_idx = list(range(all_item_features.size(0)))
        item_idx = pred_dict[userID]

        # prediction = model(batch_feature, batch_feature_values)
        prediction = torch.tensor(reranking_score[userID])

        user_gt_valid.append(valid_dict[userID])
        user_gt_test.append(test_dict[userID])

        ## valid
        all_predictions = prediction
        _, indices = torch.topk(all_predictions, topN[-1])
        pred_items = torch.tensor(item_idx)[indices].cpu().numpy().tolist()
        user_item_top1k[userID] = pred_items
        user_pred_dict[userID] = all_predictions.detach().cpu().numpy()
        valid_user_pred.append(pred_items)

        ## test with mask valid
        mask = torch.zeros(len(pred_dict[userID]))
        his_items = [pred_dict[userID].index(item) for item in pred_dict[userID] if item in valid_dict[userID]]
        if his_items != []:
            mask[his_items] = -999
        all_predictions = prediction + mask
        _, indices = torch.topk(all_predictions, topN[-1])
        pred_items = torch.tensor(item_idx)[indices].cpu().numpy().tolist()
        test_user_pred.append(pred_items)

    valid_results = computeTopNAccuracy(user_gt_valid, valid_user_pred, topN)
    test_results = computeTopNAccuracy(user_gt_test, test_user_pred, topN)

    if return_pred:  # used in the inference.py
        return valid_results, test_results, user_pred_dict, user_item_top1k
    return valid_results, test_results

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


def computeTopNAccuracy(GroundTruth, predictedIndices, topN):
    precision = []
    recall = []
    NDCG = []
    MRR = []

    for index in range(len(topN)):
        sumForPrecision = 0
        sumForRecall = 0
        sumForNdcg = 0
        sumForMRR = 0
        for i in range(len(predictedIndices)):  # for a user,
            if len(GroundTruth[i]) != 0:
                mrrFlag = True
                userHit = 0
                userMRR = 0
                dcg = 0
                idcg = 0
                idcgCount = len(GroundTruth[i])
                ndcg = 0
                hit = []
                for j in range(topN[index]):
                    if predictedIndices[i][j] in GroundTruth[i]:
                        # if Hit!
                        dcg += 1.0 / math.log2(j + 2)
                        if mrrFlag:
                            userMRR = (1.0 / (j + 1.0))
                            mrrFlag = False
                        userHit += 1

                    if idcgCount > 0:
                        idcg += 1.0 / math.log2(j + 2)
                        idcgCount = idcgCount - 1

                if (idcg != 0):
                    ndcg += (dcg / idcg)

                sumForPrecision += userHit / topN[index]
                sumForRecall += userHit / len(GroundTruth[i])
                sumForNdcg += ndcg
                sumForMRR += userMRR

        precision.append(round(sumForPrecision / len(predictedIndices), 4))
        recall.append(round(sumForRecall / len(predictedIndices), 4))
        NDCG.append(round(sumForNdcg / len(predictedIndices), 4))
        MRR.append(round(sumForMRR / len(predictedIndices), 4))

    return precision, recall, NDCG, MRR

