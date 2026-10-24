import sys
import time
import argparse
import json

import ipdb
import numpy as np
from torch import optim
import torchvision.transforms as T
from torch.utils.data import DataLoader
from utils.data_loading import BasicDataset
from utils.path_hyperparameter import ph
import torch
from utils.losses import FCCDN_loss_without_seg
import os
import logging
import random
import wandb
from models.Models import DPCD
from torchmetrics import MetricCollection, Accuracy, Precision, Recall, F1Score
from utils.utils import train_val
from utils.dataset_process import compute_mean_std
from utils.dataset_process import image_shuffle, split_image
from torch.utils.data import DataLoader
from prefetch_generator import BackgroundGenerator


class DataLoaderX(DataLoader):
    """Using prefetch_generator to accelerate data loading

    原本 PyTorch 默认的 DataLoader 会创建一些 worker 线程来预读取新的数据，但是除非这些线程的数据全部都被清空，这些线程才会读下一批数据。
    使用 prefetch_generator，我们可以保证线程不会等待，每个线程都总有至少一个数据在加载。

    Parameter:
        DataLoader(class): torch.utils.data.DataLoader.
    """

    def __iter__(self):
        return BackgroundGenerator(super().__iter__())


def random_seed(SEED):
    random.seed(SEED)
    os.environ['PYTHONHASHSEED'] = str(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    torch.backends.cudnn.deterministic = True  # keep convolution algorithm deterministic
    # torch.backends.cudnn.benchmark = False  # using fixed convolution algorithm to accelerate training
    # if model and input are fixed, set True to search better convolution algorithm
    torch.backends.cudnn.benchmark = True

def auto_experiment(dataset_root, image_dir_a, image_dir_b, label_dir, work_dir, num_workers):
    random_seed(SEED=ph.random_seed)
    try:
        train_net(dataset_name=dataset_root, image_dir_a=image_dir_a,
                  image_dir_b=image_dir_b, label_dir=label_dir,
                  work_dir=work_dir, num_workers=num_workers)
    except KeyboardInterrupt:
        logging.info('Interrupt')
        sys.exit(0)


def train_net(dataset_name, image_dir_a='A', image_dir_b='B', label_dir='label',
              work_dir='./FHRB-Net_work_dirs/default', num_workers=8):
    """
    This is the workflow of training model and evaluating model,
    note that the dataset should be organized as
    :obj:`dataset_name`/`train` or `val`/`t1` or `t2` or `label`

    Parameter:
        dataset_name(str): name of dataset

    Return:
        return nothing
    """
    # 1. Create dataset, checkpoint and best model path

    # compute mean and std of train dataset to normalize train/val dataset
    ph.dataset_name = dataset_name
    ph.image_dir_a = image_dir_a
    ph.image_dir_b = image_dir_b
    ph.label_dir = label_dir
    ph.log_path = os.path.join(work_dir, 'features') + os.sep
    os.makedirs(work_dir, exist_ok=True)
    with open(os.path.join(work_dir, 'config.json'), 'w', encoding='utf-8') as config_file:
        json.dump(ph.state_dict(), config_file, indent=2)

    t1_mean, t1_std = compute_mean_std(images_dir=os.path.join(dataset_name, 'train', image_dir_a))
    t2_mean, t2_std = compute_mean_std(images_dir=os.path.join(dataset_name, 'train', image_dir_b))

    # dataset path should be dataset_name/train or val/t1 or t2 or label
    dataset_args = dict(t1_mean=t1_mean, t1_std=t1_std, t2_mean=t2_mean, t2_std=t2_std)
    train_dataset = BasicDataset(t1_images_dir=os.path.join(dataset_name, 'train', image_dir_a),
                                 t2_images_dir=os.path.join(dataset_name, 'train', image_dir_b),
                                 labels_dir=os.path.join(dataset_name, 'train', label_dir),
                                 train=True, **dataset_args)
    val_dataset = BasicDataset(t1_images_dir=os.path.join(dataset_name, 'val', image_dir_a),
                               t2_images_dir=os.path.join(dataset_name, 'val', image_dir_b),
                               labels_dir=os.path.join(dataset_name, 'val', label_dir),
                               train=False, **dataset_args)

    # 2. Markdown dataset size
    n_train = len(train_dataset)
    n_val = len(val_dataset)

    # 3. Create data loaders

    loader_args = dict(num_workers=num_workers, persistent_workers=num_workers > 0)
    if num_workers > 0:
        loader_args['prefetch_factor'] = 5
    train_loader = DataLoaderX(train_dataset, shuffle=True, drop_last=False, batch_size=ph.batch_size, **loader_args)
    val_loader = DataLoaderX(val_dataset, shuffle=False, drop_last=False,
                             batch_size=ph.batch_size * ph.inference_ratio, **loader_args)

    # 4. Initialize logging

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # working device
    logging.basicConfig(level=logging.INFO)
    localtime = time.asctime(time.localtime(time.time()))
    hyperparameter_dict = ph.state_dict()
    hyperparameter_dict['time'] = localtime
    # using wandb to log hyperparameter, metrics and output
    # resume=allow means if the id is identical with the previous one, the run will resume
    # (anonymous=must) means the id will be anonymous
    log_wandb = wandb.init(project=ph.log_wandb_project,mode="offline", resume='allow', anonymous='must',
                           settings=wandb.Settings(start_method='thread'),
                           config=hyperparameter_dict)
    logging.info(f'''Starting training:
        Epochs:          {ph.epochs}
        Batch size:      {ph.batch_size}
        Learning rate:   {ph.learning_rate}
        Training size:   {n_train}
        Validation size: {n_val}
        Checkpoints:     {ph.save_checkpoint}
        save best model: {ph.save_best_model}
        Device:          {device.type}
        Mixed Precision: {ph.amp}
    ''')

    # 5. Set up model, optimizer, warm_up_scheduler, learning rate scheduler, loss function and other things

    net = DPCD()  # change detection model
    net = net.to(device=device)
    optimizer = optim.AdamW(net.parameters(), lr=ph.learning_rate,
                            weight_decay=ph.weight_decay)  # optimizer
    warmup_lr = np.arange(1e-7, ph.learning_rate,
                          (ph.learning_rate - 1e-7) / ph.warm_up_step)  # warm up learning rate
    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=ph.patience,
    #                                                  factor=ph.factor)  # learning rate scheduler
    grad_scaler = torch.cuda.amp.GradScaler()  # loss scaling for amp

    # load model and optimizer
    if ph.load:
        checkpoint = torch.load(ph.load, map_location=device)
        net.load_state_dict(checkpoint['net'])
        logging.info(f'Model loaded from {ph.load}')
        if 'optimizer' in checkpoint.keys():
            optimizer.load_state_dict(checkpoint['optimizer'])
            for g in optimizer.param_groups:
                g['lr'] = ph.learning_rate
            optimizer.param_groups[0]['capturable'] = True

    total_step = 0  # logging step
    lr = ph.learning_rate  # learning rate

    criterion = FCCDN_loss_without_seg  # loss function

    best_metrics = dict.fromkeys(['best_f1score', 'lowest loss'], 0)  # best evaluation metrics
    metric_collection = MetricCollection({
        'accuracy': Accuracy().to(device=device),
        'precision': Precision().to(device=device),
        'recall': Recall().to(device=device),
        'f1score': F1Score().to(device=device)
    })  # metrics calculator

    to_pilimg = T.ToPILImage()  # convert to PIL image to log in wandb

    # model saved path
    # checkpoint_path = f'./{dataset_name}_checkpoint/'
    # best_f1score_model_path = f'./{dataset_name}_best_f1score_model/'
    # best_loss_model_path = f'./{dataset_name}_best_loss_model/'

    checkpoint_path = os.path.join(work_dir, 'checkpoints') + os.sep
    best_f1score_model_path = os.path.join(work_dir, 'best_f1score') + os.sep
    best_loss_model_path = os.path.join(work_dir, 'best_loss') + os.sep

    non_improved_epoch = 0  # adjust learning rate when non_improved_epoch equal to patience

    # 5. Begin training

    for epoch in range(ph.epochs):

        log_wandb, net, optimizer, grad_scaler, total_step, lr = \
            train_val(
                mode='train', dataset_name=dataset_name,
                dataloader=train_loader, device=device, log_wandb=log_wandb, net=net,
                optimizer=optimizer, total_step=total_step, lr=lr, criterion=criterion,
                metric_collection=metric_collection, to_pilimg=to_pilimg, epoch=epoch,
                warmup_lr=warmup_lr, grad_scaler=grad_scaler,
                image_dir_a=image_dir_a, image_dir_b=image_dir_b, label_dir=label_dir
            )
        # 6. Begin evaluation

        # starting validation from evaluate epoch to minimize time
        if epoch >= ph.evaluate_epoch:
            with torch.no_grad():
                log_wandb, net, optimizer, total_step, lr, best_metrics, non_improved_epoch = \
                    train_val(
                        mode='val', dataset_name=dataset_name,
                        dataloader=val_loader, device=device, log_wandb=log_wandb, net=net,
                        optimizer=optimizer, total_step=total_step, lr=lr, criterion=criterion,
                        metric_collection=metric_collection, to_pilimg=to_pilimg, epoch=epoch,
                        best_metrics=best_metrics, checkpoint_path=checkpoint_path,
                        best_f1score_model_path=best_f1score_model_path, best_loss_model_path=best_loss_model_path,
                        non_improved_epoch=non_improved_epoch,
                        image_dir_a=image_dir_a, image_dir_b=image_dir_b, label_dir=label_dir
                    )

    wandb.finish()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset-root', default=ph.dataset_name)
    parser.add_argument('--image-dir-a', default='A')
    parser.add_argument('--image-dir-b', default='B')
    parser.add_argument('--label-dir', default='label')
    parser.add_argument('--work-dir', default='./FHRB-Net_work_dirs/default')
    parser.add_argument('--num-workers', type=int, default=8)
    args = parser.parse_args()
    auto_experiment(args.dataset_root, args.image_dir_a, args.image_dir_b,
                    args.label_dir, args.work_dir, args.num_workers)

