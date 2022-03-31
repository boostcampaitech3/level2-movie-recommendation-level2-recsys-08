import argparse
import os

import numpy as np
import torch

### 새롭게 추가한 부분!! (random_split)
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, random_split

import arguments

from datasets.SASRecDataset import SASRecDataset

### 새롭게 추가한 부분!!
from datasets.DeepFMDataset import DeepFMDataset


from models.S3Rec import S3RecModel

### 새롭게 추가한 부분!!
from models.DeepFM import DeepFM

from trainers.FinetuneTrainer import FinetuneTrainer

### 새롭게 추가한 부분!!
from trainers.DeepFMTrainer import DeepFMTrainer

### 새롭게 추가한 부분!! (deepfm_data_setting)
from utils import (
    EarlyStopping,
    check_path,
    get_item2attribute_json,
    get_user_seqs,
    set_seed,
    deepfm_data_setting
)


def main():
    args = arguments.parse_args()

    ### 추후 이 부분도 arguments.py로 연결시켜야 할 것 같다.
    # train_sasrec(args)
    train_deepfm(args)


def train_sasrec(args):
    set_seed(args.seed)  # seed setting
    check_path(args.output_dir)  # output 파일 저장할 디렉토리 확인 후 생성

    # GPU setting
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    args.cuda_condition = torch.cuda.is_available() and not args.no_cuda

    args.data_file = args.data_dir + "train_ratings.csv"
    item2attribute_file = args.data_dir + args.data_name + "_item2attributes.json"

    # 데이터 파일 로드
    user_seq, max_item, valid_rating_matrix, test_rating_matrix, _ = get_user_seqs(
        args.data_file
    )

    item2attribute, attribute_size = get_item2attribute_json(item2attribute_file)

    args.item_size = max_item + 2
    args.mask_id = max_item + 1
    args.attribute_size = attribute_size + 1

    # save model args
    args_str = f"{args.model_name}-{args.data_name}"
    args.log_file = os.path.join(args.output_dir, args_str + ".txt")
    print(str(args))

    args.item2attribute = item2attribute
    # set item score in train set to `0` in validation
    args.train_matrix = valid_rating_matrix

    # save model
    checkpoint = args_str + ".pt"
    args.checkpoint_path = os.path.join(args.output_dir, checkpoint)

    train_dataset = SASRecDataset(args, user_seq, data_type="train")
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(
        train_dataset, sampler=train_sampler, batch_size=args.batch_size
    )

    eval_dataset = SASRecDataset(args, user_seq, data_type="valid")
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(
        eval_dataset, sampler=eval_sampler, batch_size=args.batch_size
    )

    test_dataset = SASRecDataset(args, user_seq, data_type="test")
    test_sampler = SequentialSampler(test_dataset)
    test_dataloader = DataLoader(
        test_dataset, sampler=test_sampler, batch_size=args.batch_size
    )

    model = S3RecModel(args=args)

    trainer = FinetuneTrainer(
        model, train_dataloader, eval_dataloader, test_dataloader, None, args
    )

    print(args.using_pretrain)
    if args.using_pretrain:
        pretrained_path = os.path.join(args.output_dir, "Pretrain.pt")
        try:
            trainer.load(pretrained_path)
            print(f"Load Checkpoint From {pretrained_path}!")

        except FileNotFoundError:
            print(f"{pretrained_path} Not Found! The Model is same as SASRec")
    else:
        print("Not using pretrained model. The Model is same as SASRec")

    early_stopping = EarlyStopping(args.checkpoint_path, patience=10, verbose=True)
    for epoch in range(args.epochs):
        trainer.train(epoch)

        scores, _ = trainer.valid(epoch)

        early_stopping(np.array(scores[-1:]), trainer.model)
        if early_stopping.early_stop:
            print("Early stopping")
            break

    trainer.args.train_matrix = test_rating_matrix
    print("---------------Change to test_rating_matrix!-------------------")
    # load the best model
    trainer.model.load_state_dict(torch.load(args.checkpoint_path))
    scores, result_info = trainer.test(0)
    print(result_info)


#### 새롭게 추가한 부분!!!
def train_deepfm(args):
    set_seed(args.seed)  # seed setting
    check_path(args.output_dir)  # output 파일 저장할 디렉토리 확인 후 생성

    # GPU setting
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    args.cuda_condition = torch.cuda.is_available() and not args.no_cuda


    args.data_file = args.data_dir + "train_ratings.csv"
    args.genre_data = args.data_dir + "genres.tsv"

    ### 이걸 사용하는가?
    item2attribute_file = args.data_dir + args.data_name + "_item2attributes.json"
    item2attribute, attribute_size = get_item2attribute_json(item2attribute_file)
    args.item2attribute = item2attribute


    # 데이터 파일 로드
    x, y, n_user, n_item, n_genre = deepfm_data_setting(args.data_file,args.genre_data)

    # save model args
    args_str = f"{args.model_name}-{args.data_name}"
    args.log_file = os.path.join(args.output_dir, args_str + ".txt")
    print(str(args))

    # save model
    checkpoint = args_str + ".pt"
    args.checkpoint_path = os.path.join(args.output_dir, checkpoint)



    ### 이걸 사용하는가?
    # args.item_size = max_item + 2
    # args.mask_id = max_item + 1
    # args.attribute_size = attribute_size + 1

    args.train_matrix = valid_rating_matrix  # set item score in train set to `0` in validation

    
    # dataset & dataloader setting
    train_ratio = 0.8  # default : 0.9

    train_size = int(train_ratio * len(x))
    test_size = len(x) - train_size
    
    dataset = DeepFMDataset(x, y)

    ### 의문점 : evaluation dataset을 만들어야 하는가?
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size]) 

    train_sampler = RandomSampler(train_dataset)
    test_sampler = SequentialSampler(test_dataset)

    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=args.batch_size, shuffle=False)
    

    ### 의문점 : argument로 받아줄 것인가? 그렇다면 arguments.py에 작성해야 한다.
    input_dims = [n_user, n_item, n_genre]
    embedding_dim = 10  
    mlp_dims=[30, 20, 10]
    model = DeepFM(input_dims, embedding_dim, mlp_dims=mlp_dims)



    # ===================== 여기부터 수정 필요!!!!
    trainer = DeepFMTrainer(model, train_dataloader)
    # trainer = FinetuneTrainer(
    #     model, train_dataloader, eval_dataloader, test_dataloader, None, args
    # )


    ### 지워도 되는 부분...?
    # print(args.using_pretrain)
    # if args.using_pretrain:
    #     pretrained_path = os.path.join(args.output_dir, "Pretrain.pt")
    #     try:
    #         trainer.load(pretrained_path)
    #         print(f"Load Checkpoint From {pretrained_path}!")

    #     except FileNotFoundError:
    #         print(f"{pretrained_path} Not Found! The Model is same as SASRec")
    # else:
    #     print("Not using pretrained model. The Model is same as SASRec")

    early_stopping = EarlyStopping(args.checkpoint_path, patience=10, verbose=True)
    for epoch in range(args.epochs):
        trainer.train(epoch)

        scores, _ = trainer.valid(epoch)

        early_stopping(np.array(scores[-1:]), trainer.model)
        if early_stopping.early_stop:
            print("Early stopping")
            break

    trainer.args.train_matrix = test_rating_matrix
    print("---------------Change to test_rating_matrix!-------------------")
    # load the best model
    trainer.model.load_state_dict(torch.load(args.checkpoint_path))
    scores, result_info = trainer.test(0)
    print(result_info)


if __name__ == "__main__":
    main()
