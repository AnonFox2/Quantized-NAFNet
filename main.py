import argparse
import datetime
import math
import random
import time
import torch
from os import path as osp
from dataloader import Dataloader
from deblurring_model import DeblurringModel
from utils import set_random_seed

from torch.utils.data.sampler import Sampler
import torch
from torch.utils.data import DataLoader

class CPUPrefetcher():
    def __init__(self, loader):
        self.ori_loader = loader
        self.loader = iter(loader)

    def next(self):
        try:
            return next(self.loader)
        except StopIteration:
            return None

    def reset(self):
        self.loader = iter(self.ori_loader)

class EnlargedSampler(Sampler):
    def __init__(self, dataset, num_replicas, rank, ratio=1):
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.num_samples = len(self.dataset) 
        self.total_size = self.num_samples * self.num_replicas

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.epoch)
        indices = torch.randperm(self.total_size, generator=g).tolist()
        dataset_size = len(self.dataset)
        indices = [v % dataset_size for v in indices]
        indices = indices[self.rank:self.total_size:self.num_replicas]
        assert len(indices) == self.num_samples
        return iter(indices)

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch

def parse_options():
    parser = argparse.ArgumentParser(description='Image Restoration Model Training/Validation')
    
    # Dataset paths
    parser.add_argument('--dataset_root', type=str, required=True,
                        help='Root directory of the dataset')
    
    # Mode selection
    parser.add_argument('--mode', type=str, choices=['train', 'val'],default='train',
                        help='Run mode: train or val')
    
    # Model weights
    parser.add_argument('--weights', type=str, default=None,
                        help='Path to the pretrained weights (.pth file)')
    
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Training batch size')
    parser.add_argument('--val_batch_size', type=int, default=1,
                        help='Validation batch size')
    parser.add_argument('--num_workers', type=int, default=8,
                        help='Number of data loading workers')
    parser.add_argument('--total_iters', type=int, default=400000,
                        help='Total training iterations')
    parser.add_argument('--seed', type=int, default=10,
                        help='Random seed')
    parser.add_argument('--im_size', type=int, default=128,
                        help='Image size for training')

    args = parser.parse_args()
    return args

def create_train_val_dataloader(args):
    train_loader, val_loader = None, None
    
    # Construct dataset paths
    train_gt_path = osp.join(args.dataset_root, 'REDS/train/train_sharp.lmdb')
    train_lq_path = osp.join(args.dataset_root, 'REDS/train/train_blur_jpeg.lmdb')
    val_gt_path = osp.join(args.dataset_root, 'REDS/val/sharp_300.lmdb')
    val_lq_path = osp.join(args.dataset_root, 'REDS/val/blur_300.lmdb')

    if args.mode == 'train':
        train_set = Dataloader(train_gt_path, train_lq_path, 
                                     im_size=args.im_size, train=True)
        train_sampler = EnlargedSampler(train_set, 1, 0, 1)

        train_loader = DataLoader(
            dataset=train_set,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            drop_last=True,
            persistent_workers=True
        )

        num_iter_per_epoch = math.ceil(len(train_set) / args.batch_size)
        total_epochs = math.ceil(args.total_iters / num_iter_per_epoch)
    
    # Always create validation loader regardless of mode
    val_set = Dataloader(val_gt_path, val_lq_path, 
                                im_size=args.im_size, train=False)
    val_loader = DataLoader(
        dataset=val_set,
        batch_size=args.val_batch_size,
        shuffle=False,
        num_workers=0,
    )

    if args.mode == 'train':
        return train_loader, train_sampler, val_loader, total_epochs, args.total_iters
    else:
        return None, None, val_loader, 0, 0

def main():
    args = parse_options()
    set_random_seed(args.seed)
    torch.backends.cudnn.benchmark = True

    # Create model
    model = DeblurringModel(training=args.mode ,load_path=args.weights)
    

    # Get dataloaders
    result = create_train_val_dataloader(args)
    train_loader, train_sampler, val_loader, total_epochs, total_iters = result

    if args.mode == 'train':
        start_epoch = 0
        current_iter = 0
        prefetcher = CPUPrefetcher(train_loader)
        
        data_time, iter_time = time.time(), time.time()
        start_time = time.time()

        epoch = start_epoch
        while current_iter <= total_iters:
            train_sampler.set_epoch(epoch)
            prefetcher.reset()
            train_data = prefetcher.next()

            while train_data is not None:
                data_time = time.time() - data_time

                current_iter += 1
                if current_iter > total_iters:
                    break
                
                model.update_learning_rate(current_iter, -1)
                model.feed_data(train_data, is_val=False)
                result_code = model.optimize_parameters(current_iter)

                iter_time = time.time() - iter_time
                
                if current_iter % 200 == 0:
                    log_vars = {
                        'epoch': epoch, 
                        'iter': current_iter, 
                        'total_iter': total_iters
                    }
                    print(log_vars)

                if current_iter % 1000 == 0:
                    model.save(epoch, current_iter)

                if current_iter % 500 == 0 or current_iter == 1000:
                    model.validate(val_loader, current_iter)
                    log_vars = {
                        'epoch': epoch, 
                        'iter': current_iter, 
                        'total_iter': total_iters
                    }
                    print(log_vars)

                data_time = time.time()
                iter_time = time.time()
                train_data = prefetcher.next()
            epoch += 1
    
    else:  # Validation mode
        print("Running validation...")
        model.validate(val_loader, 0)

if __name__ == '__main__':

    main()