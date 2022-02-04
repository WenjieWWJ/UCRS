import os
import copy
import heapq
import random
import argparse

import torch
import evaluate
import data_utils
from item_side_utils import *
from user_side_utils import *

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
    default="video",
    help="dataset option: 'ml_1m', 'amazon_book' ")
parser.add_argument("--model", 
    type=str,
    default="FM",
    help="model option: 'NFM' or 'FM'")
parser.add_argument("--data_path",
    type=str,
    default="/storage/wjwang/filter_bubbles/data/",
    help="load data path")
parser.add_argument("--model_path",
    type=str,
    default="../../FM_NFM/models/",
    help="saved rec result path")
parser.add_argument("--file_head",
    type=str,
    default="",
    help="saved file name with hyper-parameters")
parser.add_argument("--topN", 
    default='[10, 20]',  
    help="the recommended item num")
parser.add_argument("--alpha",
    type=float,
    default=0,
    help="alpha")
parser.add_argument("--threshold_proportion",
    type=float,
    default=0.0,
    help="threshold_proportion")
parser.add_argument("--threshold_group_number",
    type=float,
    default=0.0,
    help="threshold_group_number")
parser.add_argument("--gpu",
    type=str,
    default='0',
    help="gpu")
args = parser.parse_args()
print("args:", args)
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
cudnn.benchmark = True
seed = 1
# FM video
# FM_video_64hidden_[64]layer_0.01lr_1024bs_[0.5,0.2]dropout_0lamda_1bn_500epoch_UF_min
# if args.model == 'FM':
#     file_head = "FM_video_64hidden_[64]layer_0.01lr_1024bs_[0.5,0.2]dropout_0lamda_1bn_500epoch_UF_min"
# elif args.model == 'NFM':
# #     file_head = "NFM_video_64hidden_[4]layer_0.01lr_1024bs_[0.5,0.3]dropout_0.2lamda_1bn_500epoch_UF_min"
#     file_head = "NFM_video_64hidden_[4]layer_0.01lr_1024bs_[0.5,0.3]dropout_0lamda_1bn_500epoch_UF_min"
# else:
#     print('not implement')
file_head = args.file_head
    
####
# load data
file_head = file_head.replace('_0lamda', '_0.0lamda')
topN = eval(args.topN)

user_feature_dict = np.load(args.data_path+args.dataset+'/user_feature.npy', allow_pickle=True).item()
category_list = np.load(args.data_path+args.dataset+'/category_list.npy', allow_pickle=True).tolist()
item_category = np.load(args.data_path+args.dataset+'/item_category.npy', allow_pickle=True).tolist()
user_features_list = np.load('{}{}/user_features_list.npy'.format(args.data_path, args.dataset), allow_pickle=True).tolist()

train_path = args.data_path+args.dataset+'/training_list.npy'
test_path = args.data_path+args.dataset+'/testing_dict.npy'
valid_path = args.data_path+args.dataset+'/validation_dict.npy'
train_list = np.load(train_path, allow_pickle=True).tolist()

if 'wo_age' in file_head:
    user_feature_path = args.data_path + '{}/user_feature_min_woA_file.npy'.format(args.dataset)
else:
    user_feature_path = args.data_path+args.dataset + '/user_feature_min_file.npy'
    
item_feature_path = args.data_path + '{}/item_feature_file.npy'.format(args.dataset)
user_feature, item_feature, num_features, user_map_dict, item_map_dict, features_map = data_utils.map_features(user_feature_path, item_feature_path, True)

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

test_user_list = np.load(args.data_path+args.dataset+'/close_user_0.5_list.npy', allow_pickle=True).tolist()
# test_user_list = list(test_dict.keys())

print('All predicted users\' number is ' + str(len(test_user_list)))

# before reranking. used to obtain the user-item scores 
model = torch.load('{}{}_best.pth'.format(args.model_path, file_head))
model.cuda()
model.eval()
_, test_result, user_pred_dict, user_item_top1k = evaluate.Ranking(model, valid_dict, test_dict,\
                         train_dataset.train_dict, user_feature, all_item_features, all_item_feature_values,\
                         10000, eval(args.topN), item_map_dict, True, is_test = True)
print('---'*18)
print('brefore reranking')
evaluate.print_results(None, None, test_result)

# score_name = '{}{}_top{}_score.npy'.format(args.model_path, file_head, eval(args.topN)[-1])
# rec_result_file = '{}{}_top{}_result.npy'.format(args.model_path, file_head, eval(args.topN)[-1])
# np.save(score_name, user_pred_dict)
# np.save(rec_result_file, user_item_top1k)


pred_dict = {}
test_user_pred_dict = {}
test_user_train_dict = {}
test_user_feature_dict = {}
for user in user_item_top1k:
    if user in test_dict:
        pred_dict[user] = user_item_top1k[user][:topN[0]]
    if user in test_user_list:
        test_user_pred_dict[user] = user_item_top1k[user][:topN[0]]
        test_user_train_dict[user] = train_dict_all[user]
        test_user_feature_dict[user] = user_feature_dict[user]
        
## get acc metric
pred_test_list = []
gt_test_list = []
for user in user_item_top1k:
    if user in test_user_list:
        pred_test_list.append(user_item_top1k[user][:topN[-1]])
        gt_test_list.append(test_dict[user])
train_dict = train_dict_all

feature_index = 0
normlize = False


test_result = computeTopNAccuracy(gt_test_list, pred_test_list, topN)
if test_result is not None: 
    print("[Test] Recall: {} NDCG: {}".format(
                        '-'.join([str(x) for x in test_result[1][:2]]), 
                        '-'.join([str(x) for x in test_result[2][:2]])))

feature_group = get_user_group(user_feature_dict, train_dict, item_category, category_list)
test_feature_group = get_user_group(test_user_feature_dict, train_dict, item_category, category_list)

training_group_vector, pred_group_vector, training_group_vector_top_10, pred_group_vector_top_10, \
        index_list_top10 = get_group_vestor(feature_group, train_dict, pred_dict, item_category, \
                                            category_list, user_features_list, normlize=normlize)

# print('history user ~ history group')
euc, KL, euc_top10, KL_top10 = calculate_DIS_user(test_user_list, feature_index, user_feature_dict, train_dict, item_category, category_list, user_features_list, training_group_vector, training_group_vector_top_10, index_list_top10, normlize=normlize)
print('his EUC/KL {:.4f} {:.4f} {:.4f} {:.4f}'.format(euc, euc_top10, KL, KL_top10))

# print('predict user ~ predict group')
euc, KL, euc_top10, KL_top10 = calculate_DIS_user(test_user_list, feature_index, user_feature_dict, pred_dict, item_category, category_list, user_features_list, pred_group_vector, pred_group_vector_top_10, index_list_top10, normlize=normlize)
print('pred EUC/KL {:.4f} {:.4f} {:.4f} {:.4f}'.format(euc, euc_top10, KL, KL_top10))

threshold_category_list = [0,1]
category_A, category_D, category_A_avg, category_D_avg = get_category_A_D_threshold(test_user_train_dict, test_user_pred_dict, item_category, threshold_category_list)
ss = 'Cov_A {:.4f} Cov_D {:.4f}'.format(category_A_avg[threshold_category_list[0]], category_D_avg[threshold_category_list[0]])
print(ss)

history_isolation = multi_isolation_index(test_feature_group[feature_index], train_dict, item_category)
prediction_isolation = multi_isolation_index(test_feature_group[feature_index], test_user_pred_dict, item_category)
ss = 'his-pred isolation {:.4f} {:.4f}'.format(history_isolation, prediction_isolation)
print(ss)
# history_isolation = multi_isolation_index(feature_group[feature_index], train_dict, item_category)
# prediction_isolation = multi_isolation_index(feature_group[feature_index], pred_dict, item_category)
# ss = 'his-pred isolation {:.4f} {:.4f}'.format(history_isolation, prediction_isolation)
# print(ss)
