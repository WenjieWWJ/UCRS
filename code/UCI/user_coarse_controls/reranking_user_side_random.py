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
    default="../FM_NFM/models/",
    help="saved rec result path")
parser.add_argument("--file_head",
    type=str,
    default="FM_video_64hidden_[64]layer_0.01lr_1024bs_[0.5,0.2]dropout_0lamda_1bn_500epoch_UF_min",
    help="saved file name with hyper-parameters")
parser.add_argument("--topN", 
    default='[10, 20, 50, 100]',  
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

####
# load data
file_head = args.file_head.replace('_0lamda', '_0.0lamda')
topN = eval(args.topN)

user_feature_dict = np.load(args.data_path+args.dataset+'/user_feature.npy', allow_pickle=True).item()
# user_target = np.load(args.data_path+args.dataset+'/user_mask_main_0.5.npy', allow_pickle=True).item()
category_list = np.load(args.data_path+args.dataset+'/category_list.npy', allow_pickle=True).tolist()
item_category = np.load(args.data_path+args.dataset+'/item_category.npy', allow_pickle=True).tolist()
user_features_list = np.load('{}{}/user_features_list.npy'.format(args.data_path, args.dataset), allow_pickle=True).tolist()

train_path = args.data_path+args.dataset+'/training_list.npy'
test_path = args.data_path+args.dataset+'/testing_dict.npy'
valid_path = args.data_path+args.dataset+'/validation_dict.npy'

user_feature_path = args.data_path+args.dataset + '/user_feature_min_file.npy'
item_feature_path = args.data_path + '{}/item_feature_file.npy'.format(args.dataset)

train_list = np.load(train_path, allow_pickle=True).tolist()
user_feature, item_feature, num_features, _, item_map_dict, features_map = data_utils.map_features(user_feature_path, item_feature_path, True)
user_feature_flip_gender = copy.deepcopy(user_feature)
for user in user_feature_flip_gender:
    gender = user_feature_flip_gender[user][0][2]
    if user == 451 or user == 7642:
        print('before', user_feature_flip_gender[user])
    if gender == features_map['UF2_0']:
        user_feature_flip_gender[user][0][2] = features_map['UF2_1']
    else:
        user_feature_flip_gender[user][0][2] = features_map['UF2_0']
    if user == 451 or user == 7642:
        print('after', user_feature_flip_gender[user])

all_item_features, all_item_feature_values = evaluate.pre_ranking(item_feature)
valid_dict = data_utils.loadData(valid_path)
test_dict = data_utils.loadData(test_path)

train_dict_all = {}
for pair in train_list:
    userID, itemID = pair
    if userID not in train_dict_all:
        train_dict_all[userID] = []
    train_dict_all[userID].append(itemID)

# train_dataset = data_utils.FMData(train_path, user_feature_flip_gender, item_feature, "log_loss", user_map_dict, item_map_dict)
item_map_dict_reverse = {v: k for k, v in item_map_dict.items()}

print('All predicted users\' number is ' + str(len(test_dict)))


## get acc metric
pred_dict = {}
pred_test_list = []
gt_test_list = []
for user in test_dict:
    indices = random.sample(range(15526), 1000)
    pred_items = [item_map_dict_reverse[index] for index in indices]
    pred_dict[user] = pred_items[:10]
    pred_test_list.append(pred_items)
    gt_test_list.append(test_dict[user])

random.shuffle(pred_test_list)
for i in range(1000):
    for user in pred_dict:
        u = random.sample(list(pred_dict.keys()), 1)[0]
        pred_dict[user] = pred_dict[u]
    
train_dict = train_dict_all
feature_index = 1
normlize = False


test_result = computeTopNAccuracy(gt_test_list, pred_test_list, topN)
if test_result is not None: 
    print("[Test] Recall: {} NDCG: {}".format(
                        '-'.join([str(x) for x in test_result[1][:2]]), 
                        '-'.join([str(x) for x in test_result[2][:2]])))

feature_group = get_user_group(user_feature_dict, train_dict, item_category, category_list)
change_gender_feature_group = copy.deepcopy(feature_group)
change_gender_feature_group[1][0] = feature_group[1][1]
change_gender_feature_group[1][1] = feature_group[1][0]

training_group_vector, pred_group_vector, training_group_vector_top_10, pred_group_vector_top_10, \
        index_list_top10 = get_group_vestor(feature_group, train_dict, pred_dict, item_category, \
                                            category_list, user_features_list, normlize=normlize)

# print('history user ~ history group')
euc, KL, euc_top10, KL_top10 = calculate_DIS(feature_index, feature_group, train_dict, item_category, category_list, user_features_list, training_group_vector, training_group_vector_top_10, index_list_top10, normlize=normlize)
print('his EUC/KL {:.4f} {:.4f} {:.4f} {:.4f}'.format(euc, euc_top10, KL, KL_top10))
his_dis = [euc, KL, euc_top10, KL_top10]

# print('predict user ~ predict group')
euc, KL, euc_top10, KL_top10 = calculate_DIS(feature_index, feature_group, pred_dict, item_category, category_list, user_features_list, pred_group_vector, pred_group_vector_top_10, index_list_top10, normlize=normlize)
print('pred EUC/KL {:.4f} {:.4f} {:.4f} {:.4f}'.format(euc, euc_top10, KL, KL_top10))
pred_dis = [euc, KL, euc_top10, KL_top10]

# print('history user ~ history target group')
euc, KL, euc_top10, KL_top10 = calculate_DIS(feature_index, change_gender_feature_group, train_dict, item_category, category_list, user_features_list, training_group_vector, training_group_vector_top_10, index_list_top10, normlize=normlize)
print('his-T EUC/KL {:.4f} {:.4f} {:.4f} {:.4f}'.format(euc, euc_top10, KL, KL_top10))
his_dis_target = [euc, KL, euc_top10, KL_top10]

#     print('predict user ~ predict target group')
euc, KL, euc_top10, KL_top10 = calculate_DIS(feature_index, change_gender_feature_group, pred_dict, item_category, category_list, user_features_list, pred_group_vector, pred_group_vector_top_10, index_list_top10, normlize=normlize)
print('pred-T EUC/KL {:.4f} {:.4f} {:.4f} {:.4f}'.format(euc, euc_top10, KL, KL_top10))
pred_dis_target = [euc, KL, euc_top10, KL_top10]

his_diff = [his_dis_target[i] - his_dis[i] for i in range(len(his_dis))]
ss = 'his diff EUC/KL {:.4f} {:.4f} {:.4f} {:.4f}'.format(his_diff[0], his_diff[1], his_diff[2], his_diff[3])
print(ss)
pred_diff = [pred_dis_target[i] - pred_dis[i] for i in range(len(pred_dis))]
ss = 'pred diff EUC/KL {:.4f} {:.4f} {:.4f} {:.4f}'.format(pred_diff[0], pred_diff[1], pred_diff[2], pred_diff[3])
print(ss)

threshold_category_list = [0,1]
category_A, category_D, category_A_avg, category_D_avg = get_category_A_D_threshold(train_dict, pred_dict, item_category, threshold_category_list)
ss = 'Cov_A {:.4f} Cov_D {:.4f}'.format(category_A_avg[threshold_category_list[0]], category_D_avg[threshold_category_list[0]])
print(ss)

history_isolation = get_item_gender_interaction(feature_group[feature_index], train_dict, item_category)
prediction_isolation = get_item_gender_interaction(feature_group[feature_index], pred_dict, item_category)
ss = 'his-pred isolation {:.4f} {:.4f}'.format(history_isolation, prediction_isolation)
print(ss)

for gender in feature_group[feature_index]:
    print(gender, len(feature_group[feature_index][gender]))

