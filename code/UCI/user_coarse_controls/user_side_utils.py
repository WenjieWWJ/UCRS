import pandas as pd
import openpyxl
from  openpyxl import load_workbook
import numpy as np
import scipy.stats


def get_user_group(user_feature, train_dict, item_category, category_list, normlize = True, threshold = 0):
    group = {}
    
    feature_num = len(user_feature[list(user_feature.keys())[0]])    
    for i in range(feature_num):
        group[i] = {}
        for user in user_feature:
            if threshold != 0:
                distribution = get_group_distribution([user], train_dict, item_category, len(category_list), normlize = normlize)
                if max(distribution) < threshold:
                    continue
                    
            if user_feature[user][i] not in group[i]:
                group[i][user_feature[user][i]] = [user]
            else:
                group[i][user_feature[user][i]].append(user)        
        group[i] = dict(sorted(group[i].items(), key = lambda item: item[0]))
    group = dict(sorted(group.items(), key = lambda item: item[0]))
    
    return group

    
def get_group_distribution(user_list, interaction_dict, item_feature, category_len, normlize = True):
    distribution = [0] * category_len
    distribution_user = [0] * category_len
    for user in user_list:
        distribution = [0] * category_len
        for item in interaction_dict[user]:
            for cate in item_feature[item]:
                if normlize == True:
                    distribution[cate] += 1/len(item_feature[item])
                else:
                    distribution[cate] += 1
        distribution_user = [distribution_user[i] + distribution[i]/len(interaction_dict[user]) for i in range(category_len)]
    distribution_avg = [i/len(user_list) for i in distribution_user]
    
    return distribution_avg

def get_distribution_df(feature_index, feature_group, train_dict, item_feature, second_class_list, normlize):
    df_list = []
    for cate in feature_group[feature_index]:
        if cate in feature_group[feature_index]:
            distribution = get_group_distribution(feature_group[feature_index][cate], train_dict, item_feature, len(second_class_list), normlize = normlize)
            df_list.append(distribution)

    if df_list == []:
        return None
    print('df_list', df_list)
    distribution_df = pd.DataFrame(df_list).sort_values(by=[0], axis = 1, ascending=False)
    print("distribution_df", distribution_df)
    
#     # Convert category name into Chinese
#     for i in range(len(second_class_list)):
#         distribution_df.rename(columns={i: str(i) + '_' +id_second_class_map[i]},inplace=True)
#     for i in range(len(feature_group[feature_index].keys())):
#         index = list(feature_group[feature_index].keys())[i]
#         distribution_df.rename(index={i: str(i) + '_' + str(len(feature_group[feature_index][index]))},inplace=True)

    feature_values = feature_group[feature_index].keys() # eg, [0,1] for gender
    for i in range(len(feature_values)):
        index = list(feature_values)[i]
        distribution_df.rename(index={i: str(i) + '_' + str(len(feature_group[feature_index][index]))},inplace=True) # the number of users
    return distribution_df

def get_KL(x, y):
    return scipy.stats.entropy(x, y)
def get_dis(x, y):
    return np.sqrt(np.sum(np.square(np.array(x) - np.array(y))))

def get_group_vestor(feature_group, train_dict, pred_dict, item_category, category_list, user_features_list, normlize=False):
    training_group_vector = {}
    pred_group_vector = {}
    # user_features_list: ['age', 'gender', 'country', 'province', 'city', 'city_level', 'device_name']
    for feature_index in range(len(user_features_list)):
        if feature_index != 1 and feature_index != 0: # only for 'age', 'gender'
            continue
        
        training_group_vector[feature_index] = {}
        pred_group_vector[feature_index] = {}

#         train_distribution_df = get_distribution_df(feature_index, feature_group, train_dict, item_category, category_list, normlize=normlize).T
#         pred_distribution_df = get_distribution_df(feature_index, feature_group, pred_dict, item_category, category_list, normlize=normlize).T
#         print("train_distribution_df.columns", train_distribution_df.columns)
        
#         for column in train_distribution_df.columns:
#             index = list(train_distribution_df.columns).index(column)
#             print('index', index)

        for index in feature_group[feature_index]:
            training_group_vector[feature_index][index] = get_group_distribution(feature_group[feature_index][index], train_dict, item_category, len(category_list), normlize = normlize)
            pred_group_vector[feature_index][index] = get_group_distribution(feature_group[feature_index][index], pred_dict, item_category, len(category_list), normlize = normlize)
    
    training_group_vector_top_10 = {}
    pred_group_vector_top_10 = {}
    index_list_top10 = {}
    
    for feature_index in training_group_vector: 
        training_group_vector_top_10[feature_index] = {}
        pred_group_vector_top_10[feature_index] = {}
        index_list_top10[feature_index] = {}
        for index in training_group_vector[feature_index]:
            # get the top10 item categories w.r.t the interaction prob in train_dict
            index_list=[i[0] for i in sorted(enumerate(training_group_vector[feature_index][index]), key=lambda x:x[1])]
            index_list_top10[feature_index][index] = index_list[-10:][::-1]
            training_group_vector_top_10[feature_index][index] = [training_group_vector[feature_index][index][i] for i in index_list[-10:][::-1]]
            pred_group_vector_top_10[feature_index][index] = [pred_group_vector[feature_index][index][i] for i in index_list[-10:][::-1]]
            
    return training_group_vector, pred_group_vector, training_group_vector_top_10, pred_group_vector_top_10, index_list_top10



def calculate_self_DIS(feature_index, feature_group, train_dict, item_category, category_list, user_features_list, training_group_vector, training_group_vector_top_10, index_list_top10, normlize=False):
    ## self bubble
    self_dis_all = {}
    self_dis_top10_all = {}
    self_KL_all = {}
    self_KL_top10_all = {}
    for index in feature_group[feature_index]:
        for user in feature_group[feature_index][index]:
            user_distribution = get_group_distribution([user], train_dict, item_category, len(category_list), normlize=normlize)
            user_distribution_top10 = [user_distribution[i] for i in index_list_top10[feature_index][index]]

            self_dis = get_dis(user_distribution, training_group_vector[feature_index][index])
            self_KL = get_KL(user_distribution, training_group_vector[feature_index][index])
            self_dis_top10 = get_dis(user_distribution_top10, training_group_vector_top_10[feature_index][index])
            self_KL_top10 = get_KL(user_distribution_top10, training_group_vector_top_10[feature_index][index])

            other_dis = 0
            other_KL = 0
            other_dis_top10 = 0
            other_KL_top10 = 0

            for other_index in feature_group[feature_index]:
                if other_index != index:
                    other_distribution_top10 = [training_group_vector[feature_index][other_index][i] for i in index_list_top10[feature_index][index]]

                    other_dis += get_dis(user_distribution, training_group_vector[feature_index][other_index])
                    other_KL += get_KL(user_distribution, training_group_vector[feature_index][other_index])
                    other_dis_top10 += get_dis(user_distribution_top10, other_distribution_top10)
                    other_KL_top10 += get_KL(user_distribution_top10, other_distribution_top10)

            other_dis /= len(feature_group[feature_index])-1
            other_KL /= len(feature_group[feature_index])-1
            other_dis_top10 /= len(feature_group[feature_index])-1
            other_KL_top10 /= len(feature_group[feature_index])-1

            self_dis_all[user] = other_dis - self_dis
            self_KL_all[user] = other_KL - self_KL
            self_dis_top10_all[user] = other_dis_top10 - self_dis_top10
            self_KL_top10_all[user] =  other_KL_top10 - self_KL_top10

    self_dis_all_avg = sum(self_dis_all.values())/len(self_dis_all)
    self_KL_all_avg = sum(self_KL_all.values())/len(self_KL_all)
    self_dis_top10_all_avg = sum(self_dis_top10_all.values())/len(self_dis_top10_all)
    self_kl_top10_all_avg = sum(self_KL_top10_all.values())/len(self_KL_top10_all)

    return self_dis_all_avg, self_KL_all_avg, self_dis_top10_all_avg, self_kl_top10_all_avg


def calculate_DIS(feature_index, feature_group, train_dict, item_category, category_list, user_features_list, training_group_vector, training_group_vector_top_10, index_list_top10, normlize=False):
    ## self bubble

    self_dis_all = {}
    self_dis_top10_all = {}
    self_KL_all = {}
    self_KL_top10_all = {}
    for index in feature_group[feature_index]:
        for user in feature_group[feature_index][index]:
            user_distribution = get_group_distribution([user], train_dict, item_category, len(category_list), normlize=normlize)
            user_distribution_top10 = [user_distribution[i] for i in index_list_top10[feature_index][index]]

            if np.sum(np.array(user_distribution_top10)) == 0:
                print('ignore user ', user)
                continue

            self_dis = get_dis(user_distribution, training_group_vector[feature_index][index])
            self_KL = get_KL(user_distribution, training_group_vector[feature_index][index])
            self_dis_top10 = get_dis(user_distribution_top10, training_group_vector_top_10[feature_index][index])
            self_KL_top10 = get_KL(user_distribution_top10, training_group_vector_top_10[feature_index][index])

            self_dis_all[user] = self_dis
            self_KL_all[user] = self_KL
            self_dis_top10_all[user] = self_dis_top10
            self_KL_top10_all[user] =  self_KL_top10
            
    self_dis_all_avg = sum(self_dis_all.values())/len(self_dis_all)
    self_KL_all_avg = sum(self_KL_all.values())/len(self_KL_all)
    self_dis_top10_all_avg = sum(self_dis_top10_all.values())/len(self_dis_top10_all)
    self_kl_top10_all_avg = sum(self_KL_top10_all.values())/len(self_KL_top10_all)
    
    return self_dis_all_avg, self_KL_all_avg, self_dis_top10_all_avg, self_kl_top10_all_avg


def calculate_DIS_user(user_list, feature_index, user_feature_dict, train_dict, item_category, category_list, user_features_list, training_group_vector, training_group_vector_top_10, index_list_top10, normlize=False):
    ## self bubble

    self_dis_all = {}
    self_dis_top10_all = {}
    self_KL_all = {}
    self_KL_top10_all = {}
    for user in user_list:
        feat = user_feature_dict[user][feature_index]
        user_distribution = get_group_distribution([user], train_dict, item_category, len(category_list), normlize=normlize)
        user_distribution_top10 = [user_distribution[i] for i in index_list_top10[feature_index][feat]]
        if np.sum(np.array(user_distribution_top10)) == 0:
#             print('ignore user ', user)
            continue
        
        self_dis = get_dis(user_distribution, training_group_vector[feature_index][feat])
        self_KL = get_KL(user_distribution, training_group_vector[feature_index][feat])
        self_dis_top10 = get_dis(user_distribution_top10, training_group_vector_top_10[feature_index][feat])
        self_KL_top10 = get_KL(user_distribution_top10, training_group_vector_top_10[feature_index][feat])

        self_dis_all[user] = self_dis
        self_KL_all[user] = self_KL
        self_dis_top10_all[user] = self_dis_top10
        self_KL_top10_all[user] =  self_KL_top10
            
    self_dis_all_avg = sum(self_dis_all.values())/len(self_dis_all)
    self_KL_all_avg = sum(self_KL_all.values())/len(self_KL_all)
    self_dis_top10_all_avg = sum(self_dis_top10_all.values())/len(self_dis_top10_all)
    self_kl_top10_all_avg = sum(self_KL_top10_all.values())/len(self_KL_top10_all)
    
    return self_dis_all_avg, self_KL_all_avg, self_dis_top10_all_avg, self_kl_top10_all_avg

def get_category_num(user_dict, item_category, threshold):
    user_category_num = {}
    for user in user_dict:
        category_num = {}
        for item in user_dict[user]:
            for cate in item_category[item]:
                if cate not in category_num:
                    category_num[cate] = 1
                else:
                    category_num[cate] += 1
        user_category_num[user] = sum([1 for i in category_num.values() if i > threshold])
    return user_category_num


def get_category_A_D(train_dict, pred_dict, item_category, threshold):
    train_category = {}
    pred_category = {}
    category_A = {}
    category_D = {}
    train_category = get_category_num(train_dict, item_category, threshold)
    pred_category = get_category_num(pred_dict, item_category, threshold)

    for user in train_category:
        category_A[user] = pred_category[user] - train_category[user]
    category_D = pred_category

    return category_A, category_D


def get_category_A_D_threshold(train_dict, pred_dict, item_category, threshold_category_list):
    category_A = {}
    category_D = {}
    category_A_avg = {}
    category_D_avg = {}

    for threshold in threshold_category_list:
        category_A[threshold], category_D[threshold] = get_category_A_D(train_dict, pred_dict, item_category, threshold)
        category_A_avg[threshold] = sum([i for i in category_A[threshold].values()]) / len(category_A[threshold])
        category_D_avg[threshold] = sum([i for i in category_D[threshold].values()]) / len(category_D[threshold])

    return category_A, category_D, category_A_avg, category_D_avg

def get_item_gender_interaction(user_gender_group, interaction_dict, item_category):
    m_i = {}
    f_i = {}
    m_num = 0
    f_num = 0

    for user in user_gender_group[0]:
        for item in interaction_dict[user]:
            m_num += 1
            if item not in m_i:
                m_i[item] = 1
            else:
                m_i[item] += 1

    for user in user_gender_group[1]:
        for item in interaction_dict[user]:
            f_num += 1
            if item not in f_i:
                f_i[item] = 1
            else:
                f_i[item] += 1

    S_m = 0
    S_f = 0
    for item in item_category:
        if item not in m_i and item not in f_i:
            continue
        if item not in m_i:
            m_i[item] = 0
        if item not in f_i:
            f_i[item] = 0

        S_m += m_i[item] / m_num * m_i[item] /(m_i[item] + f_i[item])
        S_f += f_i[item] / f_num * m_i[item] / (m_i[item] + f_i[item])

    S = S_m-S_f

    return S




def multi_isolation_index(user_age_group, interaction_dict, item_category):
    
    group_num = len(user_age_group)
    group_index = list(user_age_group.keys())
    cnt = 0
    S_total = 0
    for a in range(group_num):
        for b in range(a+1, group_num):
            cnt += 1    
            feat_a = group_index[a]
            feat_b = group_index[b]
            m_i = {}
            f_i = {}
            m_num = 0
            f_num = 0

            for user in user_age_group[feat_a]:
                for item in interaction_dict[user]:
                    m_num += 1
                    if item not in m_i:
                        m_i[item] = 1
                    else:
                        m_i[item] += 1

            for user in user_age_group[feat_b]:
                for item in interaction_dict[user]:
                    f_num += 1
                    if item not in f_i:
                        f_i[item] = 1
                    else:
                        f_i[item] += 1
            S_m = 0
            S_f = 0
            for item in item_category:
                if item not in m_i and item not in f_i:
                    continue
                if item not in m_i:
                    m_i[item] = 0
                if item not in f_i:
                    f_i[item] = 0

                S_m += m_i[item] / m_num * m_i[item] /(m_i[item] + f_i[item])
                S_f += f_i[item] / f_num * m_i[item] / (m_i[item] + f_i[item])

            S = S_m - S_f
            S_total += S
#             print(f'a {feat_a}; b {feat_b}', S)
            
    S_avg = S_total / cnt
    
    return S_avg