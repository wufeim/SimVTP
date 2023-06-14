import argparse
import os
import time

import torch

from build_datasets import build_pretraining_dataset
import utils


def parse_args():
    parser = argparse.ArgumentParser(description='Test video loading speed')
    # parser.add_argument('--data_path', type=str, default='/data/home/wufeim/research/SimVTP/data/webvid_train.json')
    parser.add_argument('--data_path', type=str, default='/data/home/wufeim/research/SimVTP/webvid/webvid_train_tiny.json')
    parser.add_argument('--num_frames', type=int, default=16)
    parser.add_argument('--sampling_rate', type=int, default=4)
    parser.add_argument('--mask_type', type=str, default='tube')
    parser.add_argument('--input_size', type=str, default=224)
    parser.add_argument('--pin_mem', type=bool, default=True)
    parser.add_argument('--num_workers', type=int, default=16)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--mask_ratio', type=float, default=0.9)
    args = parser.parse_args()

    args.patch_size = (16, 16)
    args.window_size = (args.num_frames // 2, args.input_size // args.patch_size[0], args.input_size // args.patch_size[1])
    return args


def main():
    args = parse_args()

    dataset_train = build_pretraining_dataset(args)

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
        worker_init_fn=utils.seed_worker
    )
    print('Dataset length:', len(dataset_train))
    print('Dataloader length:', len(data_loader_train))
    input('Continue...')

    start_time = time.time()
    try:
        for n, batch in enumerate(data_loader_train):
            videos, bool_masked_pos,text_str = batch['process_data'], batch['mask'], batch['text']

            t = time.time() - start_time
            print(f'Loaded {n+1} batches (avg {t/(n+1):.3f} sec/batch)')
    except KeyboardInterrupt:
        print('\n' + '='*8)
        print(f'Loaded {n} batches (avg {t/n:.3f} sec/batch)')


if __name__ == '__main__':
    main()
