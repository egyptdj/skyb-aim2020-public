import os
import csv
import argparse


def parse():
    parser = argparse.ArgumentParser(description='KAWS-AIM2020-MOBILEISP')
    parser.add_argument('-n', '--exp_name', type=str, default='pynetca', help='')
    parser.add_argument('-s', '--seed', type=int, default=0)
    parser.add_argument('-e', '--num_epochs', type=int, default=16, help='')
    parser.add_argument('-l', '--lr', type=float, default=0.00005, help='')
    parser.add_argument('-ds', '--source_dir', type=str, default='./data', help='')
    parser.add_argument('-dt', '--target_dir', type=str, default='./result', help='')
    parser.add_argument('--checkpoint_step', type=int, default=1000, help='')
    parser.add_argument('--train_from_level', type=int, default=None)
    parser.add_argument('--perceptual', action='store_true', help='')
    parser.add_argument('--skip_train', action='store_true', help='')
    parser.add_argument('--test_dir', type=str, default=None, help='')
    args = parser.parse_args()
    args.target_dir = os.path.join(args.target_dir, args.exp_name)
    if not args.skip_train:
        os.makedirs(args.target_dir)
        with open(os.path.join(args.target_dir, 'argv.csv'), 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerows(vars(args).items())
    return args
