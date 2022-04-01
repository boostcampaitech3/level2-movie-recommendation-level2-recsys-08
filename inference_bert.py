import argparse

from collections import defaultdict
import pandas as pd
import numpy as np
from tqdm import tqdm

import torch

from models.BERT4REC import BERT4Rec

from utils import (
    
    set_seed,
)

def generate_submission_file(data_file, preds):
    
    rating_df = pd.read_csv(data_file)
    users = rating_df["user"].unique()

    result = []

    for index, items in enumerate(preds):
        for item in items:
            result.append((users[index], item))

    pd.DataFrame(result, columns=["user", "item"]).to_csv(
        '/opt/ml/input/code/output/bert_submission.csv', index=False
    )

def main(args, parser):

    set_seed(args.seed)
        
    df = pd.read_csv(args.data_path)

    item_ids = df['item'].unique()
    user_ids = df['user'].unique()
    num_item, num_user = len(item_ids), len(user_ids)
    num_batch = num_user // args.batch_size    
    
    parser.add_argument("--num_item", type=int, default=num_item, help="num_item")
    parser.add_argument("--num_user", type=int, default=num_user, help="num_user")
    parser.add_argument("--num_batch", type=int, default=num_batch, help="num_batch")

    args = parser.parse_args()

    # user, item indexing
    item2idx = pd.Series(data=np.arange(len(item_ids))+1, index=item_ids) # item re-indexing (1~num_item), num_item+1: mask idx
    user2idx = pd.Series(data=np.arange(len(user_ids)), index=user_ids) # user re-indexing (0~num_user-1)

    # dataframe indexing
    df = pd.merge(df, pd.DataFrame({'item': item_ids, 'item_idx': item2idx[item_ids].values}), on='item', how='inner')
    df = pd.merge(df, pd.DataFrame({'user': user_ids, 'user_idx': user2idx[user_ids].values}), on='user', how='inner')
    df.sort_values(['user_idx', 'time'], inplace=True)
    del df['item'], df['user'] 

    # train set, valid set 생성
    users = defaultdict(list) # defaultdict은 dictionary의 key가 없을때 default 값을 value로 반환
    
    for u, i, t in zip(df['user_idx'], df['item_idx'], df['time']):
        users[u].append(i)

    model = BERT4Rec(args=args)
    model.to(args.device)
    model.load_state_dict(torch.load(args.load_path))
    model.eval()
    
    preds = []

    for u in tqdm(users):
        seq = (users[u] + [num_item + 1])[-args.max_len:]               

        with torch.no_grad():
            predictions = - model(np.array([seq]))
            predictions = predictions[0][-1]
            pred = predictions.argsort().tolist()[1:11]
            preds.append(item_ids[pred])
            

    generate_submission_file(args.data_path, preds) 

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_len", type=int, default=50, help="max_len")
    parser.add_argument("--hidden_units", type=int, default=50, help="hidden_units")
    parser.add_argument("--num_heads", type=int, default=1, help="num_heads")
    parser.add_argument("--num_layers", type=int, default=2, help="num_layers")
    parser.add_argument("--dropout_rate", type=float, default=0.5, help="dropout_rate")
    parser.add_argument("--num_workers", type=int, default=1, help="max_len")
    parser.add_argument("--device", type=str, default="cuda", help="cuda")
    parser.add_argument("--lr", type=float, default=1e-3, help="learning_rate")
    parser.add_argument("--batch_size", type=int, default=128, help="batch_size")
    parser.add_argument("--epochs", type=int, default=200, help="epochs")
    parser.add_argument("--mask_prob", type=float, default=0.15, help="mask_prob")
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--no_cuda", action="store_true")
    
    parser.add_argument(
        "--weight_decay", type=float, default=0.0, help="weight_decay of adam"
    )
    parser.add_argument(
        "--adam_beta1", type=float, default=0.9, help="adam first beta value"
    )
    parser.add_argument(
        "--adam_beta2", type=float, default=0.999, help="adam second beta value"
    )
    parser.add_argument("--gpu_id", type=str, default="0", help="gpu_id")
    
    parser.add_argument("--data_path", type=str, default='/opt/ml/input/data/train/train_ratings.csv', help="data_path")
    parser.add_argument("--load_path", type=str, default='/opt/ml/input/code/output/best_bert.pt', help='load_best_bert')


    args = parser.parse_args()
    
    main(args, parser)
