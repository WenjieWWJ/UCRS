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
    default="DIGIX",
    help="dataset option: 'DIGIX'")
parser.add_argument("--model", 
    type=str,
    default="FM",
    help="model option: 'NFM' or 'FM'")
parser.add_argument("--data_path",
    type=str,
    default="../../../data/",
    help="load data path")
parser.add_argument("--model_path",
    type=str,
    default="../../FM_NFM/best_models/",
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

if args.model == 'FM':
    file_head = "FM_DIGIX_64hidden_[64]layer_0.01lr_1024bs_[0.5,0.2]dropout_0.0lamda_1bn_500epoch_UF_min"
elif args.model == 'NFM':
    file_head = "NFM_DIGIX_64hidden_[4]layer_0.01lr_1024bs_[0.5,0.3]dropout_0.0lamda_1bn_500epoch_UF_min"
else:
    print('not implement')
topN = eval(args.topN)

user_feature_dict = np.load(args.data_path+args.dataset+'/user_feature.npy', allow_pickle=True).item()
category_list = np.load(args.data_path+args.dataset+'/category_list.npy', allow_pickle=True).tolist()
item_category = np.load(args.data_path+args.dataset+'/item_category.npy', allow_pickle=True).tolist()
user_features_list = np.load(args.data_path+args.dataset+'/user_features_list.npy', allow_pickle=True).tolist()

train_path = args.data_path+args.dataset+'/training_list.npy'
test_path = args.data_path+args.dataset+'/testing_dict.npy'
valid_path = args.data_path+args.dataset+'/validation_dict.npy'

user_feature_path = args.data_path+args.dataset + '/user_feature_min_file.npy'
item_feature_path = args.data_path+args.dataset + '/item_feature_file.npy'

train_list = np.load(train_path, allow_pickle=True).tolist()
user_feature, item_feature, num_features, _, item_map_dict, features_map = data_utils.map_features(user_feature_path, item_feature_path, True)

all_item_features, all_item_feature_values = evaluate.pre_ranking(item_feature)
valid_dict = data_utils.loadData(valid_path)
test_dict = data_utils.loadData(test_path)

train_dict_all = {}
for pair in train_list:
    userID, itemID = pair
    if userID not in train_dict_all:
        train_dict_all[userID] = []
    train_dict_all[userID].append(itemID)

model = torch.load('{}{}_best.pth'.format(args.model_path, file_head))
model.cuda()
model.eval()

alpha_list = [0.8, 0.9, 1.0] # if alpha=1.0, FM/NFM-changeUF; else, UCI.
for alpha in alpha_list:
    print('--'*16)
    print(f'alpha {alpha}')
    
    user_feature_flip_gender = copy.deepcopy(user_feature)
    for user in user_feature_flip_gender:
        # user_feature: [keys, values]. keys: [id, age, gender]. Only keep id(0) and gender (2)
        gender = user_feature_flip_gender[user][0][2]
        # change gender value
        if gender == features_map['UF2_0']: # 'UF2_0' and 'UF2_1'is the key of male and female.
            user_feature_flip_gender[user][0][2] = features_map['UF2_1']
        else:
            user_feature_flip_gender[user][0][2] = features_map['UF2_0']
        # use alpha to mitigate the effect of user ID representations
        user_feature_flip_gender[user][1][0] = alpha * float(user_feature_flip_gender[user][1][0])
        
    print('All predicted users\' number is ' + str(len(test_dict)))

    # use user_feature_flip_gender to inference 
    _, test_result, user_pred_dict, user_item_top1k = evaluate.Ranking(model, valid_dict, test_dict,\
                             train_dict_all, user_feature_flip_gender, all_item_features, all_item_feature_values,\
                             10000, eval(args.topN), item_map_dict, True, is_test = True)
    
    pred_dict = {}
    for user in user_item_top1k:
        if user in train_dict_all:
            pred_dict[user] = user_item_top1k[user][:topN[0]]

    ## get acc metric
    pred_test_list = []
    gt_test_list = []
    for user in user_item_top1k:
        if user in test_dict:
            pred_test_list.append(user_item_top1k[user][:topN[-1]])
            gt_test_list.append(test_dict[user])
    train_dict = train_dict_all
    
    # use gender(1) to divide user groups
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
    his_dis = [euc, KL, euc_top10, KL_top10]

    # print('predict user ~ predict group')
    euc, KL, euc_top10, KL_top10 = calculate_DIS(feature_index, feature_group, pred_dict, item_category, category_list, user_features_list, pred_group_vector, pred_group_vector_top_10, index_list_top10, normlize=normlize)
    pred_dis = [euc, KL, euc_top10, KL_top10]

    # print('history user ~ history target group')
    euc, KL, euc_top10, KL_top10 = calculate_DIS(feature_index, change_gender_feature_group, train_dict, item_category, category_list, user_features_list, training_group_vector, training_group_vector_top_10, index_list_top10, normlize=normlize)
    his_dis_target = [euc, KL, euc_top10, KL_top10]

    #     print('predict user ~ predict target group')
    euc, KL, euc_top10, KL_top10 = calculate_DIS(feature_index, change_gender_feature_group, pred_dict, item_category, category_list, user_features_list, pred_group_vector, pred_group_vector_top_10, index_list_top10, normlize=normlize)
    pred_dis_target = [euc, KL, euc_top10, KL_top10]

    his_diff = [his_dis_target[i] - his_dis[i] for i in range(len(his_dis))]
    ss = 'his diff EUC/KL {:.4f} {:.4f} {:.4f} {:.4f}'.format(his_diff[0], his_diff[1], his_diff[2], his_diff[3])
#     print(ss)

    pred_diff = [pred_dis_target[i] - pred_dis[i] for i in range(len(pred_dis))]
    ss = 'pred diff EUC/KL {:.4f} {:.4f} {:.4f} {:.4f}'.format(pred_diff[0], pred_diff[1], pred_diff[2], pred_diff[3])
    print(ss)
    
    # threshold to calculate the coverage. only covered categories with interactions more than the threshold count
    threshold_category_list = [0]
    category_A, category_D, category_A_avg, category_D_avg = get_category_A_D_threshold(train_dict, pred_dict, item_category, threshold_category_list)
    ss = 'Cov_A {:.4f} Cov {:.4f}'.format(category_A_avg[threshold_category_list[0]], category_D_avg[threshold_category_list[0]])
    print(ss)

    history_isolation = get_item_gender_interaction(feature_group[feature_index], train_dict, item_category)
    prediction_isolation = get_item_gender_interaction(feature_group[feature_index], pred_dict, item_category)
    ss = 'his-pred isolation {:.4f} {:.4f}'.format(history_isolation, prediction_isolation)
    print(ss)

