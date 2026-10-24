import argparse
import logging
import os

import torch
from torch.utils.data import DataLoader
from torchmetrics import MetricCollection, Accuracy, Precision, Recall, F1Score

from models.Models import DPCD
from utils.data_loading import BasicDataset
from utils.dataset_process import compute_mean_std
from utils.path_hyperparameter import ph
from tqdm import tqdm


def run_inference(dataset_root, checkpoint, image_dir_a, image_dir_b, label_dir,
                  checkpoint_format, work_dir, num_workers):
    ph.dataset_name = dataset_root
    ph.image_dir_a = image_dir_a
    ph.image_dir_b = image_dir_b
    ph.label_dir = label_dir
    ph.load = checkpoint
    ph.log_path = os.path.join(work_dir, 'features') + os.sep
    os.makedirs(work_dir, exist_ok=True)

    t1_mean, t1_std = compute_mean_std(
        images_dir=os.path.join(dataset_root, 'train', image_dir_a)
    )
    t2_mean, t2_std = compute_mean_std(
        images_dir=os.path.join(dataset_root, 'train', image_dir_b)
    )
    dataset_args = dict(t1_mean=t1_mean, t1_std=t1_std, t2_mean=t2_mean, t2_std=t2_std)
    test_dataset = BasicDataset(
        t1_images_dir=os.path.join(dataset_root, 'test', image_dir_a),
        t2_images_dir=os.path.join(dataset_root, 'test', image_dir_b),
        labels_dir=os.path.join(dataset_root, 'test', label_dir),
        train=False,
        **dataset_args,
    )
    loader_args = dict(num_workers=num_workers, persistent_workers=num_workers > 0)
    if num_workers > 0:
        loader_args['prefetch_factor'] = 5
    test_loader = DataLoader(
        test_dataset,
        shuffle=False,
        drop_last=False,
        batch_size=ph.batch_size * ph.inference_ratio,
        **loader_args,
    )

    logging.basicConfig(level=logging.INFO)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info('Using device %s', device)
    net = DPCD().to(device=device)
    load_model = torch.load(checkpoint, map_location=device)
    if checkpoint_format == 'checkpoint':
        net.load_state_dict(load_model['net'])
    else:
        net.load_state_dict(load_model)
    logging.info('Model loaded from %s', checkpoint)
    torch.save(net.state_dict(), os.path.join(work_dir, 'test_model_state_dict.pth'))

    metric_collection = MetricCollection({
        'accuracy': Accuracy().to(device=device),
        'precision': Precision().to(device=device),
        'recall': Recall().to(device=device),
        'f1score': F1Score().to(device=device),
    })
    net.eval()
    with torch.no_grad():
        for batch_img1, batch_img2, labels, names in tqdm(test_loader):
            batch_img1 = batch_img1.float().to(device)
            batch_img2 = batch_img2.float().to(device)
            labels = labels.float().to(device)
            cd_preds = net(batch_img1, batch_img2, log=True, img_name=names)
            cd_preds = torch.sigmoid(cd_preds[0]).float()
            labels = labels.int().unsqueeze(1)
            metric_collection.update(cd_preds, labels)

        test_metrics = metric_collection.compute()
        print(f'Metrics on all data: {test_metrics}')
        metric_collection.reset()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset-root', required=True)
    parser.add_argument('--image-dir-a', default='A')
    parser.add_argument('--image-dir-b', default='B')
    parser.add_argument('--label-dir', default='label')
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--checkpoint-format', choices=['state_dict', 'checkpoint'], default='state_dict')
    parser.add_argument('--work-dir', required=True)
    parser.add_argument('--num-workers', type=int, default=8)
    args = parser.parse_args()
    run_inference(
        args.dataset_root,
        args.checkpoint,
        args.image_dir_a,
        args.image_dir_b,
        args.label_dir,
        args.checkpoint_format,
        args.work_dir,
        args.num_workers,
    )
