import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, cate_num, hidden_dims, batch_norm, drop_prob, act_function):
        super(MLP, self).__init__()
        """
        cate_num: number of categories,
        hidden_dims: dims of hidden layers,
        batch_norm: bool type, whether to use batch norm or not,
        drop_prob: dropout ratio
        """
        self.cate_num = cate_num
        self.hidden_dims = hidden_dims
        self.batch_norm = batch_norm
        self.drop_prob = drop_prob
        
        self.act_function = None
        if act_function == 'tanh':
            self.act_function = F.tanh
        elif act_function == 'relu':
            self.act_function = F.relu
        elif act_function == 'sigmoid':
            self.act_function = F.sigmoid

        self.mask = torch.unsqueeze(torch.ones(cate_num, cate_num) - torch.eye(cate_num, cate_num), 1).cuda()

        MLP_module = []
        in_dim = self.hidden_dims[0]
        assert self.hidden_dims[0] == cate_num
        assert self.hidden_dims[-1] == 1
        for dim in self.hidden_dims[1:]:
            out_dim = dim
            MLP_module.append(nn.Parameter(torch.randn(cate_num, in_dim, out_dim, requires_grad=True)))
            in_dim = out_dim
        self.MLP_module = nn.ParameterList(MLP_module)
        
        self.drop = nn.Dropout(drop_prob)
        self.init_weights()
        
    def forward(self, data):

        # 1*b*d
        data = torch.unsqueeze(data, 0)
        # d*b*d
        h = data * self.mask
        for i, matrix in enumerate(self.MLP_module):
            h = torch.matmul(h, matrix)
            if self.drop_prob > 0:
                h = self.drop(h)
            if i != len(self.MLP_module) - 1:
                if self.act_function is not None:
                    h = self.act_function(h)
            else:
                h = torch.squeeze(h)
                h = h.T

        return h

    def l2_norm(self):
        norm = 0
        for matrix in self.MLP_module:
            norm += matrix.data.norm()
        return norm
        
    def init_weights(self):
        for matrix in self.MLP_module:
            # Xavier Initialization for weights
            size = matrix.data.size()
            fan_out = size[0]
            fan_in = size[1]
            std = np.sqrt(2.0/(fan_in + fan_out))
            matrix.data.normal_(0.0, std)
            
            
class NFM(nn.Module):
    def __init__(self, num_features, num_factors, 
        act_function, layers, batch_norm, drop_prob, pretrain_FM):
        super(NFM, self).__init__()
        """
        num_features: number of features,
        num_factors: number of hidden factors,
        act_function: activation function for MLP layer,
        layers: list of dimension of deep layers,
        batch_norm: bool type, whether to use batch norm or not,
        drop_prob: list of the dropout rate for FM and MLP,
        pretrain_FM: the pre-trained FM weights.
        """
        self.num_features = num_features
        self.num_factors = num_factors
        self.act_function = act_function
        self.layers = layers
        self.batch_norm = batch_norm
        self.drop_prob = drop_prob
        self.pretrain_FM = pretrain_FM

        self.embeddings = nn.Embedding(num_features, num_factors)
        self.biases = nn.Embedding(num_features, 1)
        self.bias_ = nn.Parameter(torch.tensor([0.0]))

        FM_modules = []
        if self.batch_norm:
            FM_modules.append(nn.BatchNorm1d(num_factors))        
        FM_modules.append(nn.Dropout(drop_prob[0]))
        self.FM_layers = nn.Sequential(*FM_modules)

        MLP_module = []
        in_dim = num_factors
        for dim in self.layers:
            out_dim = dim
            MLP_module.append(nn.Linear(in_dim, out_dim))
            in_dim = out_dim

            if self.batch_norm:
                MLP_module.append(nn.BatchNorm1d(out_dim))
            if self.act_function == 'relu':
                MLP_module.append(nn.ReLU())
            elif self.act_function == 'sigmoid':
                MLP_module.append(nn.Sigmoid())
            elif self.act_function == 'tanh':
                MLP_module.append(nn.Tanh())

            MLP_module.append(nn.Dropout(drop_prob[-1]))
        self.deep_layers = nn.Sequential(*MLP_module)

        predict_size = layers[-1] if layers else num_factors
        self.prediction = nn.Linear(predict_size, 1, bias=False)

        self._init_weight_()

    def _init_weight_(self):
        """ Try to mimic the original weight initialization. """
        if self.pretrain_FM:
            self.embeddings.weight.data.copy_(
                            self.pretrain_FM.embeddings.weight)
            self.biases.weight.data.copy_(
                            self.pretrain_FM.biases.weight)
            self.bias_.data.copy_(self.pretrain_FM.bias_)
        else:
            nn.init.normal_(self.embeddings.weight, std=0.01)
            nn.init.constant_(self.biases.weight, 0.0)

        # for deep layers
        if len(self.layers) > 0:
            for m in self.deep_layers:
                if isinstance(m, nn.Linear):
                    nn.init.xavier_normal_(m.weight)
            nn.init.xavier_normal_(self.prediction.weight)
        else:
            nn.init.constant_(self.prediction.weight, 1.0)

    def forward(self, features, feature_values):
        nonzero_embed = self.embeddings(features)
        feature_values = feature_values.unsqueeze(dim=-1)
        nonzero_embed = nonzero_embed * feature_values

        # Bi-Interaction layer
        sum_square_embed = nonzero_embed.sum(dim=1).pow(2)
        square_sum_embed = (nonzero_embed.pow(2)).sum(dim=1)

        # FM model
        FM = 0.5 * (sum_square_embed - square_sum_embed)
        FM = self.FM_layers(FM)
        if self.layers: # have deep layers
            FM = self.deep_layers(FM)
        FM = self.prediction(FM)

        # bias addition
        feature_bias = self.biases(features)
        feature_bias = (feature_bias * feature_values).sum(dim=1)
        FM = FM + feature_bias + self.bias_
        return FM.view(-1)


    
class FM(nn.Module):
    def __init__(self, num_features, num_factors, batch_norm, drop_prob):
        super(FM, self).__init__()
        """
        num_features: number of features,
        num_factors: number of hidden factors,
        batch_norm: bool type, whether to use batch norm or not,
        drop_prob: list of the dropout rate for FM and MLP,
        """
        self.num_features = num_features
        self.num_factors = num_factors
        self.batch_norm = batch_norm
        self.drop_prob = drop_prob

        self.embeddings = nn.Embedding(num_features, num_factors)
        self.biases = nn.Embedding(num_features, 1)
        self.bias_ = nn.Parameter(torch.tensor([0.0]))

        FM_modules = []
        if self.batch_norm:
            FM_modules.append(nn.BatchNorm1d(num_factors))        
        FM_modules.append(nn.Dropout(drop_prob[0]))
        self.FM_layers = nn.Sequential(*FM_modules)

        nn.init.normal_(self.embeddings.weight, std=0.01)
        nn.init.constant_(self.biases.weight, 0.0)


    def forward(self, features, feature_values):
        nonzero_embed = self.embeddings(features)
        feature_values = feature_values.unsqueeze(dim=-1)
        nonzero_embed = nonzero_embed * feature_values

        # Bi-Interaction layer
        sum_square_embed = nonzero_embed.sum(dim=1).pow(2)
        square_sum_embed = (nonzero_embed.pow(2)).sum(dim=1)

        # FM model
        FM = 0.5 * (sum_square_embed - square_sum_embed)
        FM = self.FM_layers(FM).sum(dim=1, keepdim=True)
        
        # bias addition
        feature_bias = self.biases(features)
        feature_bias = (feature_bias * feature_values).sum(dim=1)
        FM = FM + feature_bias # + self.bias_
        return FM.view(-1)   
            