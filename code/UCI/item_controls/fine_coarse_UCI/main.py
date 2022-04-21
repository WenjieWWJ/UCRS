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

user_mask_main = np.load(args.data_path+args.dataset+'/user_mask_main.npy', allow_pickle=True).item()
user_target_main = np.load(args.data_path+args.dataset+'/user_target_main.npy', allow_pickle=True).item()
user_target_weights = np.load(args.data_path+args.dataset+'/user_target_weights.npy', allow_pickle=True).item()

user_num = len(user_distribution_list)
cate_num = len(user_distribution_list[0])
print(f'user_num {user_num}; cate_num {cate_num}')

user_list = []
for user in user_mask_main:
    user_list.append(user_distribution_dict_2P[user][0])
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

            
