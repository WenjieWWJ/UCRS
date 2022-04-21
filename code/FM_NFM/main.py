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
import pandas as pd


import random
random_seed = 1
torch.manual_seed(random_seed) # cpu
torch.cuda.manual_seed(random_seed) #gpu
np.random.seed(random_seed) #numpy
random.seed(random_seed) #random and transforms
torch.backends.cudnn.deterministic=True # cudnn
def worker_init_fn(worker_id):
    np.random.seed(random_seed + worker_id)

parser = argparse.ArgumentParser()
parser.add_argument("--dataset",
    type=str,
    default="DIGIX",
    help="dataset option: 'ml_1m', 'amazon_book' ")
parser.add_argument("--model", 
    type=str,
    default="NFM",
    help="model option: 'NFM' or 'FM'")
parser.add_argument("--loss_type", 
    type=str,
    default="log_loss",
    help="loss option: 'square_loss' or 'log_loss'")
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
    default="./models/",
    help="saved model path")
parser.add_argument("--rec_result_path",
    type=str,
    default="./rec_results/",
    help="saved rec result path")
parser.add_argument("--activation_function",
    type=str,
    default="relu",
    help="activation_function option: 'relu', 'sigmoid', 'tanh', 'identity'")
parser.add_argument("--lr", 
    type=float, 
    default=0.05, 
    help="learning rate")
parser.add_argument("--dropout", 
    default='[0.5,0.2]',
    help="dropout rate for FM and MLP")
parser.add_argument("--batch_size", 
    type=int, 
    default=128, 
    help="batch size for training")
parser.add_argument("--epochs", 
    type=int,
    default=500,
    help="training epochs")
parser.add_argument("--hidden_factor", 
    type=int,
    default=64, 
    help="predictive factors numbers in the model")
parser.add_argument("--layers", 
    default='[64]', 
    help="size of layers in MLP model, '[]' is NFM-0")
parser.add_argument("--lamda", 
    type=float, 
    default=0.0, 
    help="regularizer for bilinear layers")
parser.add_argument("--topN", 
    default='[10, 20, 50, 100]',  
    help="the recommended item num")
parser.add_argument("--batch_norm", 
    type=int,
    default=1,   
    help="use batch_norm or not. option: {1, 0}")
parser.add_argument("--log_name", 
    type=str,
    default="",
    help="log_name")
parser.add_argument("--pre_train", 
    action='store_true', 
    default=False,
    help="whether use the pre-train or not")
parser.add_argument("--pre_train_model_path", 
    type=str,
    default="./models/",
    help="pre_trained model_path")
parser.add_argument("--out", 
    default=True,
    help="save model or not")
parser.add_argument("--gpu", 
    type=str,
    default="0",
    help="gpu card ID")


args = parser.parse_args()
print("args:", args)

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
cudnn.benchmark = True

############################  PREPARE DATASET #########################
start_time = time.time()


train_path = args.data_path + '{}/training_list.npy'.format(args.dataset)
valid_path = args.data_path + '{}/validation_dict.npy'.format(args.dataset)
test_path = args.data_path + '{}/testing_dict.npy'.format(args.dataset)
if args.dataset == 'DIGIX':
    user_feature_path = args.data_path + '{}/user_feature_min_file.npy'.format(args.dataset)
else:
    user_feature_path = args.data_path + '{}/user_feature_file.npy'.format(args.dataset)
    
item_feature_path = args.data_path + '{}/item_feature_file.npy'.format(args.dataset)

user_feature, item_feature, num_features, user_map_dict, item_map_dict = data_utils.map_features(user_feature_path, item_feature_path)

train_dataset = data_utils.FMData(train_path, user_feature, item_feature, args.loss_type, user_map_dict, item_map_dict)
valid_dict = data_utils.loadData(valid_path)
test_dict = data_utils.loadData(test_path)

train_loader = data.DataLoader(train_dataset, drop_last=True,
            batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True, worker_init_fn=worker_init_fn)
all_item_features, all_item_feature_values = evaluate.pre_ranking(item_feature)


print('data ready. costs ' + time.strftime("%H: %M: %S", time.gmtime(time.time()-start_time)))

##############################  CREATE MODEL ###########################
if args.pre_train and args.model=='NFM':
    assert os.path.exists(args.pre_train_model_path), 'lack of FM model'
    assert args.model == 'NFM', 'only support NFM for now'
    FM_model = torch.load(args.pre_train_model_path)
else:
    FM_model = None

if args.model == 'FM':
    if args.pre_train: # pre-trained model on iid
        model = torch.load(args.pre_train_model_path)
    else:
        model = model.FM(num_features, args.hidden_factor,
                    args.batch_norm, eval(args.dropout))
elif args.model == 'NFM':
    model = model.NFM(num_features, args.hidden_factor, 
                    args.activation_function, eval(args.layers), 
                    args.batch_norm, eval(args.dropout), FM_model)
else:
    raise Exception('model not implemented!')
    
model.cuda()
if args.optimizer == 'Adagrad':
    optimizer = optim.Adagrad(
        model.parameters(), lr=args.lr, initial_accumulator_value=1e-8)
elif args.optimizer == 'Adam':
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
elif args.optimizer == 'SGD':
    optimizer = optim.SGD(model.parameters(), lr=args.lr)
elif args.optimizer == 'Momentum':
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.95)

if args.loss_type == 'square_loss':
    criterion = nn.MSELoss(reduction='sum')
elif args.loss_type == 'log_loss':
    criterion = nn.BCEWithLogitsLoss(reduction='sum')
else:
    raise Exception('loss type not implemented!')

# writer = SummaryWriter() # for visualization

###############################  TRAINING ############################

count, best_recall = 0, -100
best_test_result = []
file_head = '{}_{}_{}hidden_{}layer_{}lr_{}bs_{}dropout_{}lamda_{}bn_{}epoch_{}'.format(
                        args.model, args.dataset, args.hidden_factor, args.layers, args.lr, \
                        args.batch_size, args.dropout, args.lamda, args.batch_norm, args.epochs, args.log_name)

print('begin training')
for epoch in range(args.epochs):
    model.train() # Enable dropout and batch_norm
    start_time = time.time()
    train_loader.dataset.ng_sample()
    
    for features, feature_values, label in train_loader:
        features = features.cuda()
        feature_values = feature_values.cuda()
        label = label.cuda()

        model.zero_grad()
        prediction = model(features, feature_values)
        loss = criterion(prediction, label)
        loss += args.lamda * model.embeddings.weight.norm()
        loss.backward()
        optimizer.step()
        # writer.add_scalar('data/loss', loss.item(), count)
        count += 1

    
    if epoch % 10 == 0:
        if epoch < 100:
            continue
            
        model.eval()
        train_RMSE = evaluate.RMSE(model, args.model, train_loader)

        valid_result, _, _, _, _  = evaluate.Ranking(model, valid_dict, test_dict, train_dataset.train_dict,\
                    user_feature, all_item_features, all_item_feature_values, 40000, eval(args.topN), item_map_dict, True)
        _ , test_result, user_pred_dict, user_item_top1k, top_values = evaluate.Ranking(model, valid_dict, test_dict,\
                    train_dataset.train_dict, user_feature, all_item_features, all_item_feature_values, \
                    40000, eval(args.topN), item_map_dict, True, is_test=True)
        print('---'*18)
        print("Runing Epoch {:03d} ".format(epoch) + 'loss {:.4f}'.format(loss) + " costs " + time.strftime(
                            "%H: %M: %S", time.gmtime(time.time()-start_time)))
        evaluate.print_results(train_RMSE, valid_result, test_result)

#         if epoch %100 == 0:
#             torch.save(model, '{}{}_nowepoch_{}.pth'.format(args.model_path, file_head, epoch))

        if valid_result[1][0] > best_recall: # recall@10 for selection
            best_recall, best_epoch = valid_result[1][0], epoch
            best_test_result = test_result
            best_user_item_top1k = user_item_top1k
            best_top_values = top_values
            print("------------Best model, saving...------------")
            if args.out:
                if not os.path.exists(args.model_path):
                    os.mkdir(args.model_path)
                torch.save(model, '{}{}_best.pth'.format(args.model_path, file_head))

print("End. Best epoch {:03d}".format(best_epoch))
evaluate.print_results(None, None, best_test_result)


rec_result_file = '{}{}_top{}_result.npy'.format(args.rec_result_path, file_head, eval(args.topN)[-1])
score_name = '{}{}_top{}_score.npy'.format(args.rec_result_path, file_head, eval(args.topN)[-1])

np.save(rec_result_file, best_user_item_top1k)
np.save(score_name, np.array(best_top_values))

print('----------------------')
print('score_file:')
print(score_name)
print('----------------------')
print('rec result file:')
print(rec_result_file)
print('----------------------')
print('best model file:')
print('{}{}_best_opt.pth'.format(args.model_path, file_head))
print('----------------------')
print('finish saving')