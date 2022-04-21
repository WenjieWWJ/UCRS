import os
import heapq
import random
import argparse

import torch
import evaluate
import data_utils
from item_side_utils import *

import numpy as np
import pandas as pd
import seaborn as sns
import torch.utils.data as data
import torch.backends.cudnn as cudnn

from operator import itemgetter
from matplotlib import pyplot as plt
from matplotlib.pyplot import MultipleLocator

sns.set()
random_seed = 1
pd.set_option('display.max_rows', None)

parser = argparse.ArgumentParser()

parser.add_argument("--dataset",
    type=str,
    default="ml_1m",
    help="dataset option: 'ml_1m', 'amazon_book' ")
parser.add_argument("--model", 
    type=str,
    default="NFM",
    help="model option: 'NFM' or 'FM'")
parser.add_argument("--data_path",
    type=str,
    default="../../../../data/",
    help="load data path")
parser.add_argument("--model_path",
    type=str,
    default="../../../FM_NFM/best_models/",
    help="saved rec result path")
parser.add_argument("--file_head",
    type=str,
    default="",
    help="saved file name with hyper-parameters")
parser.add_argument("--topN", 
    default='[10, 20]',  
    help="the recommended item num")
parser.add_argument("--gpu",
    type=str,
    default='0',
    help="gpu")
args = parser.parse_args()
print("args:", args)
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
cudnn.benchmark = True
seed = 1

####
# load data
if args.model == 'FM':
    if args.dataset == 'ml_1m':
        file_head = "FM_ml_1m_64hidden_[32]layer_0.05lr_1024bs_[0.3,0.3]dropout_0.1lamda_1bn_1500epoch"
    elif args.dataset == 'amazon_book':
        file_head = "FM_amazon_book_64hidden_[64]layer_0.05lr_1024bs_[0.5,0.2]dropout_0.1lamda_1bn_1000epoch"
elif args.model == 'NFM':
    if args.dataset == 'ml_1m':
        file_head = "NFM_ml_1m_64hidden_[16]layer_0.01lr_1024bs_[0.3,0.3]dropout_0.1lamda_1bn_500epoch"
    elif args.dataset == 'amazon_book':
        file_head = "NFM_amazon_book_64hidden_[32]layer_0.05lr_1024bs_[0.3,0.3]dropout_0.0lamda_1bn_500epoch"
else:
    print('not implement')
print("file_head", file_head)

if args.dataset == 'ml_1m':
    user_mask_main = np.load(args.data_path+args.dataset+'/user_mask_main.npy', allow_pickle=True).item()
    user_target_main = np.load(args.data_path+args.dataset+'/user_target_main.npy', allow_pickle=True).item()
elif args.dataset == 'amazon_book':
    user_mask_main = np.load(args.data_path+args.dataset+'/user_mask_main.npy', allow_pickle=True).item()
    user_target_main = np.load(args.data_path+args.dataset+'/user_target_main.npy', allow_pickle=True).item()

category_list = np.load(args.data_path+args.dataset+'/category_list.npy', allow_pickle=True).tolist()
item_category = np.load(args.data_path+args.dataset+'/item_category.npy', allow_pickle=True).tolist()

train_path = args.data_path+args.dataset+'/training_list.npy'
test_path = args.data_path+args.dataset+'/testing_dict.npy'
valid_path = args.data_path+args.dataset+'/validation_dict.npy'

user_feature_path = args.data_path + args.dataset + '/user_feature_file.npy'
item_feature_path = args.data_path + args.dataset + '/item_feature_file.npy'

train_list = np.load(train_path, allow_pickle=True).tolist()
user_feature, item_feature, num_features, user_map_dict, item_map_dict = data_utils.map_features(user_feature_path, item_feature_path)
all_item_features, all_item_feature_values = evaluate.pre_ranking(item_feature)
valid_dict = data_utils.loadData(valid_path)
test_dict = data_utils.loadData(test_path)

train_dict_all = {}
for pair in train_list:
    userID, itemID = pair
    if userID not in train_dict_all:
        train_dict_all[userID] = []
    train_dict_all[userID].append(itemID)

train_dataset = data_utils.FMData(train_path, user_feature, item_feature, "log_loss", user_map_dict, item_map_dict)
item_map_dict_reverse = {v: k for k, v in item_map_dict.items()}

print('All predicted users\' number is ' + str(len(user_mask_main)))

# before reranking. used to obtain the user-item scores 
model = torch.load('{}{}_best.pth'.format(args.model_path, file_head))
model.cuda()
model.eval()

_, test_result, user_pred_dict, user_item_top1k = evaluate.Ranking(model, valid_dict, test_dict,\
                                         train_dataset.train_dict, user_feature, all_item_features, all_item_feature_values,\
                                         10000, eval(args.topN), item_map_dict, True, is_test = True)
print('---'*18)

# score_name = '{}{}_top{}_score.npy'.format(args.model_path, file_head, eval(args.topN)[-1])
# rec_result_file = '{}{}_top{}_result.npy'.format(args.model_path, file_head, eval(args.topN)[-1])
# np.save(score_name, user_pred_dict)
# np.save(rec_result_file, user_item_top1k)


# 1. get the score of each user over all items
# 2. for the users in user target, add a reg term with weight args.alpha. rerank and get top-K items
# 3. calc metrics: recall ndcg, GA GD, GA-T GD-T

# 1. get the score of each user over all items
# user_pred_dict = np.load(score_name, allow_pickle=True).item()
# user_item_top1k = np.load(rec_result_file, allow_pickle=True).item()



def calc_metric(gt_test_list, gt_target_list, pred_list, pred_dict, train_dict):
    
    test_results = evaluate.computeTopNAccuracy(gt_test_list, pred_list, eval(args.topN), gt_target_list)
    if test_results is not None: 
        print("[Test] Recall: {} NDCG: {} WNDCG: {}".format(
                            '-'.join([str(x) for x in test_results[1][:2]]), 
                            '-'.join([str(x) for x in test_results[2][:2]]),
                            '-'.join([str(x) for x in test_results[4][:2]])))

    # used to find the users with extremely biased interests. Do not affect overall performance.
    threshold_proportion_list = [0, 0.3] # only users with the main category proportion larger than threshold are considered.
    threshold_group_number = 100 # group with userNum less than threshold is not returned.
    top_K_category_list = [1, 2, 3, 4, 5, 10] # consider top_K categories to calculate the proportion.
    threshold_category_list = [0]

    interest_GA, interest_GD, \
    target_GA_avg, target_GD_avg, \
    GA_all_avg, GD_all_avg, \
    GA_extreme_avg, GD_extreme_avg, \
    target_extreme_GA_avg, target_extreme_GD_avg, \
    all_user_num, extreme_user_num = get_GA_GD_all_threshold(threshold_proportion_list,
                                                             top_K_category_list, train_dict, pred_dict,
                                                             item_category, category_list,
                                                             threshold_group_number, user_target_main)
    category_A, category_D, \
    category_A_avg, category_D_avg = get_category_A_D_threshold(train_dict, pred_dict,
                                                                item_category, threshold_category_list)

    print('coverage_A: {} coverage: {}'.format(round(category_A_avg[0], 4), round(category_D_avg[0], 4)))
    print("GA_all_avg: {} GD_all_avg(MCD): {} target_GA_avg: {} target_GD_avg(TCD): {}".format(
                        '-'.join([str(round(GA_all_avg[0][x], 4)) for x in [1]]), 
                        '-'.join([str(round(GD_all_avg[0][x], 4)) for x in [1]]), 
                        str(round(target_GA_avg[0], 4)), str(round(target_GD_avg[0], 4))))

print('\n'+'-'*36)
print('Reranking')
print('-'*36+'\n')

# beta=0: FM/NFM; beta>0: UCI-Reranking
beta_list = [0, 0.05, 0.07, 0.1, 0.5, 0.75, 1]
for beta in beta_list:
    print('-'*30)
    print(f'beta {beta}')
    
    # 2. for the users in user target, add a reg term with weight beta. rerank and get top-K items
    gt_test_list = []
    gt_target_list = []
    pred_list = []
    pred_dict = {}
    train_dict = {}

    for userID in user_mask_main:
        if userID not in test_dict:
            continue
            
        predictions = user_pred_dict[userID]
        assert len(predictions) == len(item_category)
        reg_list = []
        for itemID in item_feature: # use the item order in item_feature for inference (evaluate.py)
            mask = False
            for cate in item_category[itemID]:
                if cate in user_mask_main[userID]:
                    mask = True
            if mask:
                reg_list.append(0)
            else:
                reg_list.append(beta)
            
        predictions = torch.sigmoid(torch.tensor(predictions).cuda())
        reg_list = torch.tensor(reg_list).cuda()
        predictions = predictions + reg_list
        _, indices = torch.topk(predictions, eval(args.topN)[-1])
        indices = indices.cpu().numpy().tolist()
        pred_items = [item_map_dict_reverse[index] for index in indices]
        gt_test_list.append(test_dict[userID])
        item_target = []
        for item in test_dict[userID]:
            for cate in item_category[item]:
                if cate in user_target_main[userID]:
                    item_target.append(item)
                    break
        gt_target_list.append(item_target)
        pred_list.append(pred_items)
        pred_dict[userID] = pred_items[:eval(args.topN)[0]] # only top-10 items are used for GA metrics
        train_dict[userID] = train_dict_all[userID]


    # 3. calc metrics: recall ndcg, GA GD, GA-T GD-T
    calc_metric(gt_test_list, gt_target_list, pred_list, pred_dict, train_dict)
