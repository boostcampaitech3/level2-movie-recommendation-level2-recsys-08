import argparse
import glob
import json
import os
import random
import re
from models.ImcGae import models

def train(args):
    pass

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42, help='random seed (default: 42)')
    parser.add_argument('--epochs', type=int, default=300, help='number of epochs to train (default: 1)')

    args = parser.parse_args()
    print(args)

    train(args)
