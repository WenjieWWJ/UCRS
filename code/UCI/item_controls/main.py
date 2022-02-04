import os
import time
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
    default="amazon_book_only_first",
    help="dataset option: 'ml_1m'")
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

user_target_list = []
for user in user_mask_main:
    user_target_list.append(user_distribution_dict_2P[user][1])
user_target_list = np.array(user_target_list, dtype=np.float32)
data_label = torch.tensor(user_target_list).cuda()

sample_num = len(user_list)

test_user_list = []
for user in user_mask_main:
    his_dis = user_distribution_dict_2P[user][1]
    for mask in user_mask_main[user]:
        his_dis[mask] = 0
    test_user_list.append(his_dis)
test_user_list = torch.tensor(test_user_list).cuda()

hidden_dims = [cate_num] + eval(args.layers) + [1]
model = model.MLP(cate_num=cate_num, hidden_dims=hidden_dims, \
                  batch_norm=args.batch_norm, drop_prob=args.dropout, act_function=args.act_function)
model.cuda()

if args.optimizer == 'Adagrad':
    optimizer = optim.Adagrad(model.parameters(), lr=args.lr, initial_accumulator_value=1e-8)
elif args.optimizer == 'Adam':
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
elif args.optimizer == 'SGD':
    optimizer = optim.SGD(model.parameters(), lr=args.lr)
elif args.optimizer == 'Momentum':
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.95)

criterion = nn.MSELoss(reduction='sum')


file_head = '{}_{}_{}layer_{}lr_{}bs_{}dropout_{}lamda_{}_{}_{}bn_{}epoch_{}'.format(
                        'MLP', args.dataset, args.layers, args.lr, args.batch_size, args.dropout, \
                        args.lamda, args.optimizer, args.act_function, args.batch_norm, args.epochs, args.log_name)

best_acc = -100
for epoch in range(args.epochs):
    model.train() # Enable dropout and batch_norm
    start_time = time.time()    
    st, ed = 0, args.batch_size
    batch_num = sample_num // args.batch_size
    for i in range(batch_num+1):
        batch_data = data[st:ed]
        batch_label = data_label[st:ed]
        
        model.zero_grad()
        prediction = model(batch_data)
        loss = criterion(prediction, batch_label)
        norm_loss = model.l2_norm()
        loss += args.lamda * norm_loss
        loss.backward()
        optimizer.step()
        
        if i == batch_num:
            st += args.batch_size
            ed = sample_num-1
        else:
            st += args.batch_size
            ed += args.batch_size
            
    if epoch % 100 == 0:
        model.eval()
        X_pred = model(test_user_list).cpu().detach().numpy()
        hit = 0
        for i, user in enumerate(user_mask_main):
            for mask in user_mask_main[user]:
                X_pred[i][mask] = 0
            pred_indices = set(np.argsort(-np.array(X_pred[i]))[:1])
            gt_indices = set(np.argsort(-np.array(user_target_weights[user]))[:1])
            if len(pred_indices.intersection(gt_indices)) > 0:
                hit += 1
        acc = 1.0 * hit / len(user_mask_main)
        print(f' epoch: {epoch}; hit@1:{acc}')
        if acc > best_acc:
            best_acc = acc
            torch.save(model, '{}/{}.pth'.format(args.model_path, file_head))

print(f'End. best_acc {best_acc}')

# # np.save('{}/user_estimate_weights_{}.npy'.format(args.model_path, file_head), user_estimate_weights)

# category_list = np.load(args.data_path+args.dataset+'/category_list.npy', allow_pickle=True).tolist()
# item_category = np.load(args.data_path+args.dataset+'/item_category.npy', allow_pickle=True).tolist()

# train_path = args.data_path+args.dataset+'/training_list.npy'
# test_path = args.data_path+args.dataset+'/testing_dict.npy'
# valid_path = args.data_path+args.dataset+'/validation_dict.npy'

# user_feature_path = args.data_path+args.dataset + '/user_feature_file.npy'
# item_feature_path = args.data_path + '{}/item_feature_file.npy'.format(args.dataset)
# train_list = np.load(train_path, allow_pickle=True).tolist()
# user_feature, item_feature, num_features, user_map_dict, item_map_dict = data_utils.map_features(user_feature_path, item_feature_path)
# valid_dict = data_utils.loadData(valid_path)
# test_dict = data_utils.loadData(test_path)

# train_dict_all = {}
# for pair in train_list:
#     userID, itemID = pair
#     if userID not in train_dict_all:
#         train_dict_all[userID] = []
#     train_dict_all[userID].append(itemID)
# item_map_dict_reverse = {v: k for k, v in item_map_dict.items()}

# print('All predicted users\' number is ' + str(len(user_mask_main)))

# model_path = "../../FM_NFM/best_models/"
# if args.dataset == 'ml_1m':
#     FM_file_head = "FM_ml_1m_64hidden_[32]layer_0.05lr_1024bs_[0.3,0.3]dropout_0.1lamda_1bn_1500epoch"
# elif args.dataset == 'amazon_book_only_first':
#     FM_file_head = "FM_amazon_book_only_first_64hidden_[64]layer_0.05lr_1024bs_[0.5,0.2]dropout_0.1lamda_1bn_1000epoch"
# else:
#     print('not implement')
    
# score_name = '{}{}_top{}_score.npy'.format(model_path, FM_file_head, 100)
# rec_result_file = '{}{}_top{}_result.npy'.format(model_path, FM_file_head, 100)

# # 1. get the score of each user over all items
# user_pred_dict = np.load(score_name, allow_pickle=True).item()
# user_item_top1k = np.load(rec_result_file, allow_pickle=True).item()

# def calc_metric(gt_test_list, gt_target_list, pred_list, pred_dict, train_dict):
    
#     test_results = evaluate.computeTopNAccuracy(gt_test_list, pred_list, eval(args.topN), gt_target_list)
#     if test_results is not None: 
#         print("[Test] Recall: {} NDCG: {} WNDCG: {}".format(
#                             '-'.join([str(x) for x in test_results[1][:2]]), 
#                             '-'.join([str(x) for x in test_results[2][:2]]),
#                             '-'.join([str(x) for x in test_results[4][:2]])))

#     if args.dataset == 'ml_1m':
#         threshold_proportion_list = [0, 0.4, 0.5, 0.6]
#     elif args.dataset == 'amazon_book_only_first':
#         threshold_proportion_list = [0, 0.3, 0.4, 0.5]
#     else:
#         threshold_proportion_list = [0]
#     threshold_group_number = 100
#     top_K_category_list = [1, 2, 3, 4, 5, 10]
#     threshold_category_list = [0, 1]

#     interest_GA, interest_GD, \
#     target_GA_avg, target_GD_avg, \
#     GA_all_avg, GD_all_avg, \
#     GA_extreme_avg, GD_extreme_avg, \
#     target_extreme_GA_avg, target_extreme_GD_avg, \
#     all_user_num, extreme_user_num = get_GA_GD_all_threshold(threshold_proportion_list,
#                                                              top_K_category_list, train_dict, pred_dict,
#                                                              item_category, category_list,
#                                                              threshold_group_number, user_target_main)
#     category_A, category_D, \
#     category_A_avg, category_D_avg = get_category_A_D_threshold(train_dict, pred_dict,
#                                                                 item_category, threshold_category_list)

#     print('coverage_A: {} coverage_D: {}'.format(round(category_A_avg[0], 4), round(category_D_avg[0], 4)))
#     print("GA_all_avg: {} GD_all_avg: {} target_GA_avg: {} target_GD_avg: {}".format(
#                         '-'.join([str(round(GA_all_avg[0][x], 4)) for x in [1,2,3]]), 
#                         '-'.join([str(round(GD_all_avg[0][x], 4)) for x in [1,2,3]]), 
#                         str(round(target_GA_avg[0], 4)), str(round(target_GD_avg[0], 4))))

    
# test_model = torch.load('{}/{}.pth'.format(args.model_path, file_head))
# test_model.cuda()
# test_model.eval()
# X_pred = test_model(test_user_list).cpu().detach().numpy()

# k_list = [1, 2, 3]
# alpha_list = [0.01, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.075, 0.08, 0.09, 0.1, 0.5, 1] 
# # alpha_list = [0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.075, 0.08, 0.09, 0.1, 0.2, 0.3, 0.4, 0.5] # 0.75, 1, 2, 10 

# for k in k_list:
#     print('-'*60)
#     print(f'k: {k}')
#     user_estimate_weights = {}
#     for i, user in enumerate(user_mask_main):
#         for mask in user_mask_main[user]:
#             X_pred[i][mask] = -999
#         pred_indices = set(np.argsort(-np.array(X_pred[i]))[:k]) 
#         for j in range(len(X_pred[i])):
#             if j in user_mask_main[user]:
#                 X_pred[i][j] = 0
#             elif j in pred_indices:
#                 X_pred[i][j] = 2
#             else:
#                 X_pred[i][j] = 1
#         user_estimate_weights[user] = X_pred[i]
        
#     for alpha in alpha_list:
#         print('-'*30)
#         print(f'alpha {alpha}')

#         # 2. for the users in user target, add a reg term with weight args.alpha. rerank and get top-K items
#         gt_test_list = []
#         gt_target_list = []
#         pred_list = []
#         pred_dict = {}
#         train_dict = {}

#         for userID in user_mask_main:
#             if userID not in test_dict:
#                 continue
#             predictions = user_pred_dict[userID]
#             assert len(predictions) == len(item_category)
#             reg_list = []
#             for itemID in item_feature: # use the item order in item_feature for inference (evaluate.py)
#                 weights = 0
#                 for cate in item_category[itemID]:
#                     weights += user_estimate_weights[userID][cate]
#                 reg_list.append(alpha*weights/len(item_category[itemID]))
#     #             reg_list.append(alpha*weights)

#             predictions = torch.sigmoid(torch.tensor(predictions).cuda())
#             reg_list = torch.tensor(reg_list).cuda()
#             predictions = predictions + reg_list
#             _, indices = torch.topk(predictions, eval(args.topN)[1])
#             indices = indices.cpu().numpy().tolist()
#             pred_items = [item_map_dict_reverse[index] for index in indices]
#             gt_test_list.append(test_dict[userID])
#             item_target = []
#             for item in test_dict[userID]:
#                 for cate in item_category[item]:
#                     if cate in user_target_main[userID]:
#                         item_target.append(item)
#                         break
#             gt_target_list.append(item_target)
#             pred_list.append(pred_items)
#             pred_dict[userID] = pred_items[:eval(args.topN)[0]] # only top-10 items are used for GA metrics
#             train_dict[userID] = train_dict_all[userID]

#         # 3. calc metrics: recall ndcg, GA GD, GA-T GD-T
#         calc_metric(gt_test_list, gt_target_list, pred_list, pred_dict, train_dict)
        
        
        
# print('\n'+'-'*36)
# print('user no mask')
# print('-'*36+'\n')

# for k in k_list:
#     print('-'*60)
#     print(f'k: {k}')
#     user_estimate_weights = {}
#     for i, user in enumerate(user_mask_main):
# #         for mask in user_mask_main[user]:
# #             X_pred[i][mask] = -999
#         pred_indices = set(np.argsort(-np.array(X_pred[i]))[:k]) 
#         for j in range(len(X_pred[i])):
#             if j in user_mask_main[user]:
#                 X_pred[i][j] = 0
#             elif j in pred_indices:
#                 X_pred[i][j] = 2
#             else:
#                 X_pred[i][j] = 1
#         user_estimate_weights[user] = X_pred[i]
        
#     for alpha in alpha_list:
#         print('-'*30)
#         print(f'alpha {alpha}')

#         # 2. for the users in user target, add a reg term with weight args.alpha. rerank and get top-K items
#         gt_test_list = []
#         gt_target_list = []
#         pred_list = []
#         pred_dict = {}
#         train_dict = {}

#         for userID in user_mask_main:
#             if userID not in test_dict:
#                 continue
#             predictions = user_pred_dict[userID]
#             assert len(predictions) == len(item_category)
#             reg_list = []
#             for itemID in item_feature: # use the item order in item_feature for inference (evaluate.py)
#                 weights = 0
#                 for cate in item_category[itemID]:
#                     weights += user_estimate_weights[userID][cate]
#                 reg_list.append(alpha*weights/len(item_category[itemID]))
#     #             reg_list.append(alpha*weights)

#             predictions = torch.sigmoid(torch.tensor(predictions).cuda())
#             reg_list = torch.tensor(reg_list).cuda()
#             predictions = predictions + reg_list
#             _, indices = torch.topk(predictions, eval(args.topN)[1])
#             indices = indices.cpu().numpy().tolist()
#             pred_items = [item_map_dict_reverse[index] for index in indices]
#             gt_test_list.append(test_dict[userID])
#             item_target = []
#             for item in test_dict[userID]:
#                 for cate in item_category[item]:
#                     if cate in user_target_main[userID]:
#                         item_target.append(item)
#                         break
#             gt_target_list.append(item_target)
#             pred_list.append(pred_items)
#             pred_dict[userID] = pred_items[:eval(args.topN)[0]] # only top-10 items are used for GA metrics
#             train_dict[userID] = train_dict_all[userID]

#         # 3. calc metrics: recall ndcg, GA GD, GA-T GD-T
#         calc_metric(gt_test_list, gt_target_list, pred_list, pred_dict, train_dict)     
            
