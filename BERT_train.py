import math
import pandas as pd
from tqdm import tqdm
from collections import defaultdict

import torch.nn as nn
import torch.nn.functional as F
import argparse
import os

import numpy as np
import torch
from torch.utils.data import Dataset ,DataLoader

# Dataset

class SeqDataset(Dataset):
    def __init__(self, user_train, num_user, num_item, max_len, mask_prob):
        self.user_train = user_train
        self.num_user = num_user
        self.num_item = num_item
        self.max_len = max_len
        self.mask_prob = mask_prob

    def __len__(self):
        # 총 user의 수 = 학습에 사용할 sequence의 수
        return self.num_user

    def __getitem__(self, user): 
        # iterator를 구동할 때 사용됩니다.
        seq = self.user_train[user]
        tokens = []
        labels = []
        for s in seq:
            prob = np.random.random() # TODO1: numpy를 사용해서 0~1 사이의 임의의 값을 샘플링하세요.
            if prob < self.mask_prob:
                prob /= self.mask_prob

                # BERT 학습
                if prob < 0.8:
                    # masking
                    tokens.append(self.num_item + 1)  # mask_index: num_item + 1, 0: pad, 1~num_item: item index
                elif prob < 0.9:
                    tokens.append(np.random.randint(1, self.num_item+1))  # item random sampling
                else:
                    tokens.append(s)
                labels.append(s)  # 학습에 사용
            else:
                tokens.append(s)
                labels.append(0)  # 학습에 사용 X, trivial
        tokens = tokens[-self.max_len:]
        labels = labels[-self.max_len:]
        mask_len = self.max_len - len(tokens)

        # zero padding
        tokens = [0] * mask_len + tokens
        labels = [0] * mask_len + labels
        return torch.LongTensor(tokens), torch.LongTensor(labels)

class ScaledDotProductAttention(nn.Module):
    def __init__(self, hidden_units, dropout_rate):
        super(ScaledDotProductAttention, self).__init__()
        self.hidden_units = hidden_units
        self.dropout = nn.Dropout(dropout_rate) # dropout rate

    def forward(self, Q, K, V, mask):
        attn_score = torch.matmul(Q, K.transpose(2, 3)) / math.sqrt(self.hidden_units)
        attn_score = attn_score.masked_fill(mask == 0, -1e9)  # 유사도가 0인 지점은 -infinity로 보내 softmax 결과가 0이 되도록 함
        attn_dist = self.dropout(F.softmax(attn_score, dim=-1))  # attention distribution
        output = torch.matmul(attn_dist, V)  # dim of output : batchSize x num_head x seqLen x hidden_units
        return output, attn_dist

# modules

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, hidden_units, dropout_rate):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads # head의 수
        self.hidden_units = hidden_units
        
        # query, key, value, output 생성을 위해 Linear 모델 생성
        self.W_Q = nn.Linear(hidden_units, hidden_units, bias=False)
        self.W_K = nn.Linear(hidden_units, hidden_units, bias=False)
        self.W_V = nn.Linear(hidden_units, hidden_units, bias=False)
        self.W_O = nn.Linear(hidden_units, hidden_units, bias=False)

        self.attention = ScaledDotProductAttention(hidden_units, dropout_rate) # scaled dot product attention module을 사용하여 attention 계산
        self.dropout = nn.Dropout(dropout_rate) # dropout rate
        self.layerNorm = nn.LayerNorm(hidden_units, 1e-6) # layer normalization

    def forward(self, enc, mask):
        residual = enc # residual connection을 위해 residual 부분을 저장
        batch_size, seqlen = enc.size(0), enc.size(1)
        
        # Query, Key, Value를 (num_head)개의 Head로 나누어 각기 다른 Linear projection을 통과시킴
        Q = self.W_Q(enc).view(batch_size, seqlen, self.num_heads, self.hidden_units) 
        K = self.W_K(enc).view(batch_size, seqlen, self.num_heads, self.hidden_units)
        V = self.W_V(enc).view(batch_size, seqlen, self.num_heads, self.hidden_units)

        # Head별로 각기 다른 attention이 가능하도록 Transpose 후 각각 attention에 통과시킴
        Q, K, V = Q.transpose(1, 2), K.transpose(1, 2), V.transpose(1, 2)
        output, attn_dist = self.attention(Q, K, V, mask)

        # 다시 Transpose한 후 모든 head들의 attention 결과를 합칩니다.
        output = output.transpose(1, 2).contiguous() 
        output = output.view(batch_size, seqlen, -1)

        # Linear Projection, Dropout, Residual sum, and Layer Normalization
        output = self.layerNorm(self.dropout(self.W_O(output)) + residual)
        return output, attn_dist
    
class PositionwiseFeedForward(nn.Module): # encoder decoder를 layer 쌓아주는 그런 역할?
    def __init__(self, hidden_units, dropout_rate):
        super(PositionwiseFeedForward, self).__init__()
        
        # SASRec과의 dimension 차이가 있습니다.
        self.W_1 = nn.Linear(hidden_units, 4 * hidden_units) 
        self.W_2 = nn.Linear(4 * hidden_units, hidden_units)
        self.dropout = nn.Dropout(dropout_rate)
        self.layerNorm = nn.LayerNorm(hidden_units, 1e-6) # layer normalization

    def forward(self, x):
        residual = x
        output = self.W_2(F.gelu(self.dropout(self.W_1(x)))) # activation: relu -> gelu
        output = self.layerNorm(self.dropout(output) + residual)
        return output
    
class BERT4RecBlock(nn.Module):
    def __init__(self, num_heads, hidden_units, dropout_rate):
        super(BERT4RecBlock, self).__init__()
        self.attention = MultiHeadAttention(num_heads, hidden_units, dropout_rate)
        self.pointwise_feedforward = PositionwiseFeedForward(hidden_units, dropout_rate)

    def forward(self, input_enc, mask):
        output_enc, attn_dist = self.attention(input_enc, mask)
        output_enc = self.pointwise_feedforward(output_enc)
        return output_enc, attn_dist


class BERT4Rec(nn.Module):
    def __init__(self, args):
        super(BERT4Rec, self).__init__()

        self.num_user = args.num_user
        self.num_item = args.num_item
        self.hidden_units =args.hidden_units
        self.num_heads = args.num_heads
        self.num_layers =args.num_layers 
        self.device =args.device
        
        self.item_emb = nn.Embedding(args.num_item + 2, args.hidden_units, padding_idx=0) # TODO2: mask와 padding을 고려하여 embedding을 생성해보세요.
        self.pos_emb = nn.Embedding(args.max_len, args.hidden_units) # learnable positional encoding
        self.dropout = nn.Dropout(args.dropout_rate)
        self.emb_layernorm = nn.LayerNorm(args.hidden_units, eps=1e-6)
        
        self.blocks = nn.ModuleList([BERT4RecBlock(args.num_heads, args.hidden_units, args.dropout_rate) for _ in range(args.num_layers)])
        self.out = nn.Linear(args.hidden_units, args.num_item + 1) # TODO3: 예측을 위한 output layer를 구현해보세요. (num_item 주의)
        
    def forward(self, log_seqs):
        seqs = self.item_emb(torch.LongTensor(log_seqs).to(self.device))
        positions = np.tile(np.array(range(log_seqs.shape[1])), [log_seqs.shape[0], 1])
        seqs += self.pos_emb(torch.LongTensor(positions).to(self.device))
        seqs = self.emb_layernorm(self.dropout(seqs))

        mask = torch.BoolTensor(log_seqs > 0).unsqueeze(1).repeat(1, log_seqs.shape[1], 1).unsqueeze(1).to(self.device) # mask for zero pad
        for block in self.blocks:
            seqs, attn_dist = block(seqs, mask)
        out = self.out(seqs)
        return out

def main(args,parser):
    

    data_path = '/opt/ml/input/data/train/train_ratings.csv'
    df = pd.read_csv(data_path)

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
    user_train = {}
    user_valid = {}
    for u, i, t in zip(df['user_idx'], df['item_idx'], df['time']):
        users[u].append(i)

    for user in users:
        user_train[user] = users[user][:-1]
        user_valid[user] = [users[user][-1]]

    model = BERT4Rec(args)
    model.to(args.device)
    criterion = nn.CrossEntropyLoss(ignore_index=0) # label이 0인 경우 무시
    seq_dataset = SeqDataset(user_train, num_user, num_item, args.max_len, args.mask_prob)
    data_loader = DataLoader(seq_dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True) # TODO4: pytorch의 DataLoader와 seq_dataset을 사용하여 학습 파이프라인을 구현해보세요.
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)


    best_loss = 1e4
    for epoch in range(1, args.epochs + 1):
        tbar = tqdm(data_loader)
        for step, (log_seqs, labels) in enumerate(tbar):
            logits = model(log_seqs)
            
            # size matching
            logits = logits.view(-1, logits.size(-1))
            labels = labels.view(-1).to(args.device)
            
            optimizer.zero_grad()
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            
            if loss < best_loss:
                torch.save(model.state_dict(), args.save_path)        
                tbar.set_description(f'Epoch: {epoch:3d}| Step: {step:3d}| Train loss: {loss:.5f} | save best model!')
            else:
                tbar.set_description(f'Epoch: {epoch:3d}| Step: {step:3d}| Train loss: {loss:.5f}')
        
    
    
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
    
    parser.add_argument("--data_path", type=str, default='/opt/ml/input/data/train/train_ratings.csv', help="data_path")
    parser.add_argument("--save_path", type=str, default='/opt/ml/input/code/output/best_bert.pt', help='save path best model')
            

    args = parser.parse_args()
    
    main(args, parser)
