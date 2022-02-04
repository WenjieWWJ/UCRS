import os
import time
import copy
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torch.backends.cudnn as cudnn
# from tensorboardX import SummaryWriter

import model
import evaluate
import data_utils
from item_side_utils import *

import random
random_seed = 1
torch.manual_seed(random_seed) # cpu
torch.cuda.manual_seed(random_seed) #gpu
np.random.seed(random_seed) #numpy
random.seed(random_seed) #random and transforms
torch.backends.cudnn.deterministic=True # cudnn


parser = argparse.ArgumentParser()
parser.add_argument("--dataset",
    type=str,
    default="ml_1m",
    help="dataset option: 'ml_1m, amazon_book_only_first'")
parser.add_argument("--model",
    type=str,
    default="FM",
    help="model option: 'FM' or 'NFM'")
parser.add_argument("--optimizer",
    type=str,
    default="Adagrad",
    help="optimizer option: 'Adagrad', 'Adam', 'SGD', 'Momentum'")
parser.add_argument("--data_path",
    type=str,
    default="/storage/wjwang/filter_bubbles/data/",  
    help="load data path")
parser.add_argument("--model_path", 
    type=str,
    default="./models",
    help="saved model path")
parser.add_argument("--act_function",
    type=str,
    default="sigmoid",
    help="activation_function option: 'relu', 'sigmoid', 'tanh', 'identity'")
parser.add_argument("--lr", 
    type=float, 
    default=0.01,
    help="learning rate")
parser.add_argument("--dropout", 
    default='0.1',  
    type=float,
    help="dropout rate for MLP")
parser.add_argument("--lamda", 
    default='0.2',
    type=float, 
    help="L2 norm")
parser.add_argument("--batch_size", 
    type=int, 
    default=256, 
    help="batch size for training")
parser.add_argument("--epochs", 
    type=int,
    default=1000, 
    help="training epochs")
parser.add_argument("--topN", 
    default='[10, 20]',  
    help="the recommended item num")
parser.add_argument("--layers", 
    default='[128]', 
    help="size of layers in MLP model")
parser.add_argument("--batch_norm", 
    type=int,
    default=1,
    help="use batch_norm or not. option: {1, 0}")
parser.add_argument("--log_name", 
    type=str,
    default="",
    help="special name for logs")
parser.add_argument("--gpu", 
    type=str,
    default="0",
    help="gpu card ID")
args = parser.parse_args()
print("args:", args)

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
cudnn.benchmark = True


user_distribution_dict_2P = np.load(args.data_path+args.dataset+'/user_distribution_dict_2P.npy', allow_pickle=True).item()
user_distribution_list = np.load(args.data_path+args.dataset+'/user_distribution_list.npy', allow_pickle=True).tolist()

# user_mask_main = np.load(args.data_path+args.dataset+'/user_mask_main.npy', allow_pickle=True).item()
if args.dataset == 'ml_1m':
    user_mask_main = np.load(args.data_path+args.dataset+'/user_mask_main_od.npy', allow_pickle=True).item()
    user_target_main = np.load(args.data_path+args.dataset+'/user_target_main_od.npy', allow_pickle=True).item()
    user_target_weights = np.load(args.data_path+args.dataset+'/user_target_weights_od.npy', allow_pickle=True).item()
else:
    user_mask_main = np.load(args.data_path+args.dataset+'/user_mask_main_small_od.npy', allow_pickle=True).item()
    user_target_main = np.load(args.data_path+args.dataset+'/user_target_main_small_od.npy', allow_pickle=True).item()
    user_target_weights = np.load(args.data_path+args.dataset+'/user_target_weights_small_od.npy', allow_pickle=True).item()

user_num = len(user_distribution_list)
cate_num = len(user_distribution_list[0])
print(f'user_num {user_num}; cate_num {cate_num}')

user_list = []
for user in user_mask_main:
    user_list.append(user_distribution_dict_2P[user][0])

# user_list = user_distribution_list
user_list = np.array(user_list, dtype=np.float32)
data = torch.tensor(user_list).cuda()
sample_num = len(user_list)

test_user_list = []
user_train_weights = {}
for user in user_mask_main:
    his_dis = user_distribution_dict_2P[user][1]
#     his_dis = user_distribution_dict[user]
    for mask in user_mask_main[user]:
        his_dis[mask] = 0
    user_train_weights[user] = his_dis
    test_user_list.append(his_dis)
test_user_list = torch.tensor(test_user_list).cuda()


if args.dataset == 'ml_1m':
    file_head = "MLP_ml_1m_[32]layer_0.01lr_512bs_0.2dropout_0.1lamda_Adagrad_tanh_1bn_3000epoch_od_log"
elif args.dataset == 'amazon_book_only_first':
    file_head = "MLP_amazon_book_only_first_[32]layer_0.01lr_1024bs_0.3dropout_0.1lamda_Adagrad_sigmoid_1bn_2000epoch_small_od_log"
    
file_head = file_head.replace('_0dropout', '_0.0dropout')

category_list = np.load(args.data_path+args.dataset+'/category_list.npy', allow_pickle=True).tolist()
item_category = np.load(args.data_path+args.dataset+'/item_category.npy', allow_pickle=True).tolist()

train_path = args.data_path+args.dataset+'/training_list.npy'
test_path = args.data_path+args.dataset+'/testing_dict.npy'
valid_path = args.data_path+args.dataset+'/validation_dict.npy'

user_feature_path = args.data_path+args.dataset + '/user_feature_file.npy'
item_feature_path = args.data_path + '{}/item_feature_file.npy'.format(args.dataset)
train_list = np.load(train_path, allow_pickle=True).tolist()
user_feature, item_feature, num_features, user_map_dict, item_map_dict = data_utils.map_features(user_feature_path, item_feature_path)
train_dataset = data_utils.FMData(train_path, user_feature, item_feature, 'log_loss', user_map_dict, item_map_dict)
all_item_features, all_item_feature_values = evaluate.pre_ranking(item_feature)

valid_dict = data_utils.loadData(valid_path)
test_dict = data_utils.loadData(test_path)

train_dict_all = {}
for pair in train_list:
    userID, itemID = pair
    if userID not in train_dict_all:
        train_dict_all[userID] = []
    train_dict_all[userID].append(itemID)
item_map_dict_reverse = {v: k for k, v in item_map_dict.items()}

print('All predicted users\' number is ' + str(len(user_mask_main)))

model_path = "../../FM_NFM/best_models/"
if args.dataset == 'ml_1m':
    if args.model == 'FM':
        FM_file_head = "FM_ml_1m_64hidden_[32]layer_0.05lr_1024bs_[0.3,0.3]dropout_0.1lamda_1bn_1500epoch"
    elif args.model == 'NFM':
        FM_file_head = "NFM_ml_1m_64hidden_[16]layer_0.01lr_1024bs_[0.3,0.3]dropout_0.1lamda_1bn_500epoch"
    else:
        print('not implement')
elif args.dataset == 'amazon_book_only_first':
    if args.model == 'FM':
        FM_file_head = "FM_amazon_book_only_first_64hidden_[64]layer_0.05lr_1024bs_[0.5,0.2]dropout_0.1lamda_1bn_1000epoch"
    elif args.model == 'NFM':
        FM_file_head = "NFM_amazon_book_only_first_64hidden_[32]layer_0.05lr_1024bs_[0.3,0.3]dropout_0.0lamda_1bn_500epoch"
    else:
        print('not implement')
else:
    print('not implement')

FM_model = torch.load('{}{}_best.pth'.format(model_path, FM_file_head))
FM_model.cuda()
FM_model.eval()

test_model = torch.load('{}/{}.pth'.format(args.model_path, file_head))
test_model.cuda()
test_model.eval()
X_pred = test_model(test_user_list).cpu().detach().numpy()


def calc_metric(gt_test_list, gt_target_list, pred_list, pred_dict, train_dict):
    
    test_results = evaluate.computeTopNAccuracy(gt_test_list, pred_list, eval(args.topN), gt_target_list)
    if test_results is not None: 
        print("[Test] Recall: {} NDCG: {} WNDCG: {}".format(
                            '-'.join([str(x) for x in test_results[1][:2]]), 
                            '-'.join([str(x) for x in test_results[2][:2]]),
                            '-'.join([str(x) for x in test_results[4][:2]])))

    if args.dataset == 'ml_1m':
        threshold_proportion_list = [0, 0.4, 0.5, 0.6]
    elif args.dataset == 'amazon_book_only_first':
        threshold_proportion_list = [0, 0.3, 0.4, 0.5]
    else:
        threshold_proportion_list = [0]
    threshold_group_number = 100
    top_K_category_list = [1, 2, 3, 4, 5, 10]
    threshold_category_list = [0, 1]

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

    print('coverage_A: {} coverage_D: {}'.format(round(category_A_avg[0], 4), round(category_D_avg[0], 4)))
    print("GA_all_avg: {} GD_all_avg: {} target_GA_avg: {} target_GD_avg: {}".format(
                        '-'.join([str(round(GA_all_avg[0][x], 4)) for x in [1,2,3]]), 
                        '-'.join([str(round(GD_all_avg[0][x], 4)) for x in [1,2,3]]), 
                        str(round(target_GA_avg[0], 4)), str(round(target_GD_avg[0], 4))))

start_time = time.time()    

if args.dataset == 'amazon_book_only_first':
    item_cate = []
    for itemID in item_feature: # use the item order in item_feature for inference (evaluate.py)
        if len(item_category[itemID])>1:
            print('Error! more than one label in amazon book')
        item_cate.append(item_category[itemID][0])
    item_cate = torch.tensor(item_cate).cuda().view(len(item_feature), 1)
    item_cate_one_hot = torch.zeros(len(item_feature), cate_num).cuda().scatter_(1, item_cate, 1)

k_list = [1, 2, 3, 4, 5]
alpha_list = [0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.075, 0.1, 0.3, 0.5, 1, 2, 5, 10]
beta_list = [0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.6, 1.0]

for beta in beta_list:
    print('--'*16)
    print(f'beta {beta}')
    
    user_feature_changed = copy.deepcopy(user_feature)
    for user in user_feature_changed:
        user_feature_changed[user][1][0] = beta * float(user_feature_changed[user][1][0])
        
    _, test_result, user_pred_dict, user_item_top1k = evaluate.Ranking(FM_model, valid_dict, test_dict,\
                             train_dataset.train_dict, user_feature_changed, all_item_features, all_item_feature_values,\
                             100000, eval(args.topN), item_map_dict, True, is_test = True)
    print('---'*18)
    print('brefore reranking')
    evaluate.print_results(None, None, test_result)
    print('costs ' + time.strftime("%H: %M: %S", time.gmtime(time.time()-start_time)))
    start_time = time.time()
    
    # 1. get the score of each user over all items    
    # score_name = '{}{}_top{}_score.npy'.format(model_path, FM_file_head, 100)
    # rec_result_file = '{}{}_top{}_result.npy'.format(model_path, FM_file_head, 100)
    # user_pred_dict = np.load(score_name, allow_pickle=True).item()
    # user_item_top1k = np.load(rec_result_file, allow_pickle=True).item()

    for k in k_list:
        print('-'*60)
        print(f'k: {k}')
        user_estimate_weights = {}
        hit = 0
#         X_pred = test_model(test_user_list).cpu().detach().numpy()
        X_prediction = copy.deepcopy(X_pred)
        for i, user in enumerate(user_mask_main):
            for mask in user_mask_main[user]:
                X_prediction[i][mask] = -999
            pred_indices = set(np.argsort(-np.array(X_prediction[i]))[:k])

            gt_indices = set(np.argsort(-np.array(user_target_weights[user]))[:1])

            
            if len(pred_indices.intersection(gt_indices)) > 0:
                hit += 1
            for j in range(len(X_prediction[i])):
                if j in user_mask_main[user]:
                    X_prediction[i][j] = 0
                elif j in pred_indices:
                    X_prediction[i][j] = 2
                else:
                    X_prediction[i][j] = 1
            user_estimate_weights[user] = X_prediction[i]
                
        acc = 1.0 * hit / len(user_mask_main)
        print(f'acc {acc}')
        print('costs ' + time.strftime("%H: %M: %S", time.gmtime(time.time()-start_time)))
        start_time = time.time()
        
        for alpha in alpha_list:
            print('-'*30)
            print(f'alpha {alpha}')

            # 2. for the users in user target, add a reg term with weight args.alpha. rerank and get top-K items
            gt_test_list = []
            gt_target_list = []
            pred_list = []
            pred_dict = {}
            train_dict = {}
            
            for userID in user_mask_main:
                if userID not in test_dict:
                    continue
                predictions = user_pred_dict[userID]
#                 assert len(predictions) == len(item_category)

                if args.dataset == 'amazon_book_only_first':
                    predictions = torch.sigmoid(torch.tensor(predictions).cuda())
                    cate_weights = torch.tensor(user_estimate_weights[userID]).cuda()
                    weights = torch.matmul(item_cate_one_hot, cate_weights).view(-1)
                    predictions = predictions + alpha * weights                    
                else:
                    reg_list = []
                    for itemID in item_feature: # use the item order in item_feature for inference (evaluate.py)
                        weights = 0
                        for cate in item_category[itemID]:
                            weights += user_estimate_weights[userID][cate]
                        reg_list.append(alpha*weights/len(item_category[itemID]))
            #             reg_list.append(alpha*weights)
                    predictions = torch.sigmoid(torch.tensor(predictions).cuda())
                    reg_list = torch.tensor(reg_list).cuda()
                    predictions = predictions + reg_list
                
                _, indices = torch.topk(predictions, eval(args.topN)[1])
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