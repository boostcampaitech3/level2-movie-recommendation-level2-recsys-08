import argparse
import os

import torch
from torch.utils.data import DataLoader, SequentialSampler

import arguments

from datasets.SASRecDataset import SASRecDataset
from models.S3Rec import S3RecModel
from trainers.FinetuneTrainer import FinetuneTrainer
from utils import (
    check_path,
    generate_submission_file,
    get_item2attribute_json,
    get_user_seqs,
    set_seed,
)


def main():
    args = arguments.parse_args()
    inference_sasrec(args)


def inference_sasrec(args):
    set_seed(args.seed)
    check_path(args.output_dir)

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    args.cuda_condition = torch.cuda.is_available() and not args.no_cuda

    args.data_file = args.data_dir + "train_ratings.csv"
    item2attribute_file = args.data_dir + args.data_name + "_item2attributes.json"

    user_seq, max_item, _, _, submission_rating_matrix = get_user_seqs(args.data_file)

    item2attribute, attribute_size = get_item2attribute_json(item2attribute_file)

    args.item_size = max_item + 2
    args.mask_id = max_item + 1
    args.attribute_size = attribute_size + 1

    # save model args
    args_str = f"{args.model_name}-{args.data_name}"

    print(str(args))

    args.item2attribute = item2attribute

    args.train_matrix = submission_rating_matrix

    checkpoint = args_str + ".pt"
    args.checkpoint_path = os.path.join(args.output_dir, checkpoint)

    submission_dataset = SASRecDataset(args, user_seq, data_type="submission")
    submission_sampler = SequentialSampler(submission_dataset)
    submission_dataloader = DataLoader(
        submission_dataset, sampler=submission_sampler, batch_size=args.batch_size
    )

    model = S3RecModel(args=args)

    trainer = FinetuneTrainer(model, None, None, None, submission_dataloader, args)

    trainer.load(args.checkpoint_path)
    print(f"Load model from {args.checkpoint_path} for submission!")
    preds = trainer.submission(0)

    generate_submission_file(args.data_file, preds)


if __name__ == "__main__":
    main()
