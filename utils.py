import json
import math
import os
import random

import numpy as np
import pandas as pd
import torch
from scipy.sparse import csr_matrix


def set_seed(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # some cudnn methods can be random even after fixing the seed
    # unless you tell it to be deterministic
    torch.backends.cudnn.deterministic = True


def check_path(path):
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"{path} created")


def neg_sample(item_set, item_size):
    item = random.randint(1, item_size - 1)
    while item in item_set:
        item = random.randint(1, item_size - 1)
    return item


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(self, checkpoint_path, patience=7, verbose=False, delta=0):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        self.checkpoint_path = checkpoint_path
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.delta = delta

    def compare(self, score):
        for i in range(len(score)):

            if score[i] > self.best_score[i] + self.delta:
                return False
        return True

    def __call__(self, score, model):

        if self.best_score is None:
            self.best_score = score
            self.score_min = np.array([0] * len(score))
            self.save_checkpoint(score, model)
        elif self.compare(score):
            self.counter += 1
            print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(score, model)
            self.counter = 0

    def save_checkpoint(self, score, model):
        """Saves model when the performance is better."""
        if self.verbose:
            print(f"Better performance. Saving model ...")
        torch.save(model.state_dict(), self.checkpoint_path)
        self.score_min = score


def kmax_pooling(x, dim, k):
    index = x.topk(k, dim=dim)[1].sort(dim=dim)[0]
    return x.gather(dim, index).squeeze(dim)


def avg_pooling(x, dim):
    return x.sum(dim=dim) / x.size(dim)


def generate_rating_matrix_valid(user_seq, num_users, num_items):
    # three lists are used to construct sparse matrix
    row = []  # user의 id만 저장
    col = []  # item(영화)의 id만 저장 (중복 포함)
    data = []
    for user_id, item_list in enumerate(user_seq):
        for item in item_list[:-2]:  # 마지막 2개 제외
            row.append(user_id)
            col.append(item)
            data.append(1)

    row = np.array(row)
    col = np.array(col)
    data = np.array(data)
    rating_matrix = csr_matrix((data, (row, col)), shape=(num_users, num_items))

    return rating_matrix


def generate_rating_matrix_test(user_seq, num_users, num_items):
    # three lists are used to construct sparse matrix
    row = []
    col = []
    data = []
    for user_id, item_list in enumerate(user_seq):
        for item in item_list[:-1]:  # 마지막 1개 제외
            row.append(user_id)
            col.append(item)
            data.append(1)

    row = np.array(row)
    col = np.array(col)
    data = np.array(data)
    rating_matrix = csr_matrix((data, (row, col)), shape=(num_users, num_items))

    return rating_matrix


def generate_rating_matrix_submission(user_seq, num_users, num_items):
    # three lists are used to construct sparse matrix
    row = []
    col = []
    data = []
    for user_id, item_list in enumerate(user_seq):
        for item in item_list[:]:  #
            row.append(user_id)
            col.append(item)
            data.append(1)

    row = np.array(row)
    col = np.array(col)
    data = np.array(data)
    rating_matrix = csr_matrix((data, (row, col)), shape=(num_users, num_items))

    return rating_matrix


def generate_submission_file(data_file, preds):

    rating_df = pd.read_csv(data_file)
    users = rating_df["user"].unique()

    result = []

    for index, items in enumerate(preds):
        for item in items:
            result.append((users[index], item))

    pd.DataFrame(result, columns=["user", "item"]).to_csv(
        "output/submission.csv", index=False
    )


def get_user_seqs(data_file):
    rating_df = pd.read_csv(data_file)
    lines = rating_df.groupby("user")["item"].apply(list)  # user들이 본 영화 리스트 목록
    user_seq = []  # 각 user들이 본 영화 리스트 자체를 element로 저장한 리스트
    item_set = set()  # 이전까지의 user들에 대한 unique 영화 리스트
    for line in lines:  # 개별 user의 영화 리스트에 대해

        items = line
        user_seq.append(items)
        item_set = item_set | set(items)
    max_item = max(item_set)  # unique한 영화 리스트 중 (영화 번호 상) 가장 마지막 영화

    num_users = len(lines)  # 전체 user 인원 수
    num_items = max_item + 2

    valid_rating_matrix = generate_rating_matrix_valid(user_seq, num_users, num_items)
    test_rating_matrix = generate_rating_matrix_test(user_seq, num_users, num_items)
    submission_rating_matrix = generate_rating_matrix_submission(
        user_seq, num_users, num_items
    )
    return (
        user_seq,
        max_item,
        valid_rating_matrix,
        test_rating_matrix,
        submission_rating_matrix,
    )


def get_user_seqs_long(data_file):
    rating_df = pd.read_csv(data_file)
    lines = rating_df.groupby("user")["item"].apply(list)  # user들이 본 영화 리스트 목록 생성
    user_seq = []  # 각 user들이 본 영화 리스트 자체를 element로 저장한 리스트
    long_sequence = []  # 모든 user 들이 본 전체 영화 리스트 (중복 포함)
    item_set = set()  # 이전까지의 user들에 대한 unique 영화 리스트
    for line in lines:
        items = line
        long_sequence.extend(items)
        user_seq.append(items)
        item_set = item_set | set(items)
    max_item = max(item_set)  # unique한 영화 리스트 중 (영화 번호 상) 가장 마지막 영화 

    return user_seq, max_item, long_sequence


def get_item2attribute_json(data_file):
    item2attribute = json.loads(open(data_file).readline())
    attribute_set = set()
    for item, attributes in item2attribute.items():  # 영화 번호, 그 영화의 장르 번호(리스트)
        attribute_set = attribute_set | set(attributes)  # 전체 영화에 대한 unique 장르 set 생성
    attribute_size = max(attribute_set)
    return item2attribute, attribute_size


### 새롭게 추가한 부분!!
def deepfm_data_setting(data_file, genre_data):
    ## Rating df 생성
    raw_rating_df = pd.read_csv(data_file)
    raw_rating_df['rating'] = 1.0  # implicit feedback
    raw_rating_df.drop(['time'],axis=1,inplace=True)
    users = set(raw_rating_df.loc[:, 'user'])  # user set
    items = set(raw_rating_df.loc[:, 'item'])  # item set 


    ## Genre df 생성
    # main script 측에서 argument로 genres.tsv 경로 변수를 만들어서 이를 인자로 받아 사용해야 한다.
    # genre_data = args.data_dir + "genres.tsv"   # main script 측에서 해줘야 할 일
    raw_genre_df = pd.read_csv(genre_data, sep='\t')
    raw_genre_df = raw_genre_df.drop_duplicates(subset=['item']) # item별 하나의 장르만 남도록 drop
    genre_dict = {genre:i for i, genre in enumerate(set(raw_genre_df['genre']))}  # 장르를 number로 mapping
    raw_genre_df['genre']  = raw_genre_df['genre'].map(lambda x : genre_dict[x]) #genre id로 변경

    ## Negative instance 생성
    num_negative = 50  # 각 user당 50개 생성
    user_group_dfs = list(raw_rating_df.groupby('user')['item'])  # 각 user당 rating 남긴 영화 리스트
    first_row = True
    user_neg_dfs = pd.DataFrame()  # 전체 user들의 negative sampling 데이터 dataframe

    for u, u_items in user_group_dfs:
        u_items = set(u_items)  # 특정 user의 영화 목록
        i_user_neg_item = np.random.choice(list(items - u_items), num_negative, replace=False)  # 특정 user가 고르지 않은 영화 50개 선정
        
        i_user_neg_df = pd.DataFrame({'user': [u]*num_negative, 'item': i_user_neg_item, 'rating': [0]*num_negative})  # negative sampling dataframe 생성
        if first_row == True:
            user_neg_dfs = i_user_neg_df
            first_row = False
        else:
            user_neg_dfs = pd.concat([user_neg_dfs, i_user_neg_df], axis = 0, sort=False)

    raw_rating_df = pd.concat([raw_rating_df, user_neg_dfs], axis = 0, sort=False)  # 기존 train set + negative sampling 데이터
    
    ## Join dfs
    joined_rating_df = pd.merge(raw_rating_df, raw_genre_df, left_on='item', right_on='item', how='inner')
    

    ## user, item을 zero-based index로 mapping
    users = sorted(list(set(joined_rating_df.loc[:,'user'])))  # unique user list
    items =  sorted(list(set((joined_rating_df.loc[:, 'item']))))  # unique item list
    genres =  sorted(list(set((joined_rating_df.loc[:, 'genre']))))  # unique genre list

    # unique user list가 0-based index가 아닌 경우
    if len(users)-1 != max(users):
        users_dict = {users[i]: i for i in range(len(users))}
        joined_rating_df['user']  = joined_rating_df['user'].map(lambda x : users_dict[x])
        users = list(set(joined_rating_df.loc[:,'user']))

    # unique item list가 0-based index가 아닌 경우
    if len(items)-1 != max(items):
        items_dict = {items[i]: i for i in range(len(items))}
        joined_rating_df['item']  = joined_rating_df['item'].map(lambda x : items_dict[x])
        items =  list(set((joined_rating_df.loc[:, 'item'])))

    joined_rating_df = joined_rating_df.sort_values(by=['user'])  # 합친 dataframe을 user를 기준으로 정렬
    joined_rating_df.reset_index(drop=True, inplace=True)

    data = joined_rating_df

    n_user, n_item, n_genre = len(users), len(items), len(genres)


    # user, item, genre를 각각의 tensor로 변경
    user_col = torch.tensor(data.loc[:,'user'])
    item_col = torch.tensor(data.loc[:,'item'])
    genre_col = torch.tensor(data.loc[:,'genre'])

    # 각 tensor에 offset을 더한다.
    offsets = [0, n_user, n_user+n_item]
    for col, offset in zip([user_col, item_col, genre_col], offsets):  # user_col : 0, item_col : n_user, genre_col : n_user+n_item
        col += offset


    ### return value가 될 것들    
    X = torch.cat([user_col.unsqueeze(1), item_col.unsqueeze(1), genre_col.unsqueeze(1)], dim=1)  # shape:[6722471, 3]
    y = torch.tensor(list(data.loc[:,'rating']))  # rating 내역 존재 여부

    return X, y, n_user, n_item, n_genre


def get_metric(pred_list, topk=10):
    NDCG = 0.0
    HIT = 0.0
    MRR = 0.0
    # [batch] the answer's rank
    for rank in pred_list:
        MRR += 1.0 / (rank + 1.0)
        if rank < topk:
            NDCG += 1.0 / np.log2(rank + 2.0)
            HIT += 1.0
    return HIT / len(pred_list), NDCG / len(pred_list), MRR / len(pred_list)


def precision_at_k_per_sample(actual, predicted, topk):
    num_hits = 0
    for place in predicted:
        if place in actual:
            num_hits += 1
    return num_hits / (topk + 0.0)


def precision_at_k(actual, predicted, topk):
    sum_precision = 0.0
    num_users = len(predicted)
    for i in range(num_users):
        act_set = set(actual[i])
        pred_set = set(predicted[i][:topk])
        sum_precision += len(act_set & pred_set) / float(topk)

    return sum_precision / num_users


def recall_at_k(actual, predicted, topk):
    sum_recall = 0.0
    num_users = len(predicted)
    true_users = 0
    for i in range(num_users):
        act_set = set(actual[i])
        pred_set = set(predicted[i][:topk])
        if len(act_set) != 0:
            sum_recall += len(act_set & pred_set) / float(len(act_set))
            true_users += 1
    return sum_recall / true_users


def apk(actual, predicted, k=10):
    """
    Computes the average precision at k.
    This function computes the average precision at k between two lists of
    items.
    Parameters
    ----------
    actual : list
             A list of elements that are to be predicted (order doesn't matter)
    predicted : list
                A list of predicted elements (order does matter)
    k : int, optional
        The maximum number of predicted elements
    Returns
    -------
    score : double
            The average precision at k over the input lists
    """
    if len(predicted) > k:
        predicted = predicted[:k]

    score = 0.0
    num_hits = 0.0

    for i, p in enumerate(predicted):
        if p in actual and p not in predicted[:i]:
            num_hits += 1.0
            score += num_hits / (i + 1.0)

    if not actual:
        return 0.0

    return score / min(len(actual), k)


def mapk(actual, predicted, k=10):
    """
    Computes the mean average precision at k.
    This function computes the mean average prescision at k between two lists
    of lists of items.
    Parameters
    ----------
    actual : list
             A list of lists of elements that are to be predicted
             (order doesn't matter in the lists)
    predicted : list
                A list of lists of predicted elements
                (order matters in the lists)
    k : int, optional
        The maximum number of predicted elements
    Returns
    -------
    score : double
            The mean average precision at k over the input lists
    """
    return np.mean([apk(a, p, k) for a, p in zip(actual, predicted)])


def ndcg_k(actual, predicted, topk):
    res = 0
    for user_id in range(len(actual)):
        k = min(topk, len(actual[user_id]))
        idcg = idcg_k(k)
        dcg_k = sum(
            [
                int(predicted[user_id][j] in set(actual[user_id])) / math.log(j + 2, 2)
                for j in range(topk)
            ]
        )
        res += dcg_k / idcg
    return res / float(len(actual))


# Calculates the ideal discounted cumulative gain at k
def idcg_k(k):
    res = sum([1.0 / math.log(i + 2, 2) for i in range(k)])
    if not res:
        return 1.0
    else:
        return res
