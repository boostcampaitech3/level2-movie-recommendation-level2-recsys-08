import argparse
import os

import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler

# 사용자 정의 모듈 import
import arguments

from datasets.PretrainDataset import PretrainDataset
from models.S3Rec import S3RecModel
from trainers.PretrainTrainer import PretrainTrainer
from utils import (
    EarlyStopping,
    check_path,
    get_item2attribute_json,
    get_user_seqs_long,
    set_seed,
)


def main():
    args = arguments.parse_args()
    pretrain_sasrec(args)


# ================================= 여기서부터 분할하기!!!
def pretrain_sasrec(args):
    set_seed(args.seed)  # argument로 준 seed값 세팅
    check_path(args.output_dir)  # output_dir 경로 존재여부 확인

    args.checkpoint_path = os.path.join(args.output_dir, "Pretrain.pt")  # checkpoint 파일 저장 경로 지정

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id  # 사용할 GPU 정보 환경변수화
    args.cuda_condition = torch.cuda.is_available() and not args.no_cuda

    # args.data_file = args.data_dir + args.data_name + '.txt'
    args.data_file = args.data_dir + "train_ratings.csv"  # 학습 데이터 경로 지정
    item2attribute_file = args.data_dir + args.data_name + "_item2attributes.json"
    # concat all user_seq get a long sequence, from which sample neg segment for SP
    user_seq, max_item, long_sequence = get_user_seqs_long(args.data_file)  # user가 조회한 영화들 정보 얻어내기

    item2attribute, attribute_size = get_item2attribute_json(item2attribute_file)

    args.item_size = max_item + 2
    args.mask_id = max_item + 1
    args.attribute_size = attribute_size + 1

    args.item2attribute = item2attribute

    model = S3RecModel(args=args)
    trainer = PretrainTrainer(model, None, None, None, None, args)

    early_stopping = EarlyStopping(args.checkpoint_path, patience=10, verbose=True)

    for epoch in range(args.pre_epochs):

        pretrain_dataset = PretrainDataset(args, user_seq, long_sequence)
        pretrain_sampler = RandomSampler(pretrain_dataset)
        pretrain_dataloader = DataLoader(
            pretrain_dataset, sampler=pretrain_sampler, batch_size=args.pre_batch_size
        )

        losses = trainer.pretrain(epoch, pretrain_dataloader)

        ## comparing `sp_loss_avg``
        early_stopping(np.array([-losses["sp_loss_avg"]]), trainer.model)
        if early_stopping.early_stop:
            print("Early stopping")
            break


if __name__ == "__main__":
    main()
