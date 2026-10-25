# FHRB-Net Server Actions

Server checkout: `/home/xxj/code/DINOCLIP_comparative_test/FHRB-Net`  
Dataset root: `/home/xxj/Data`

## Pull reproduce branch

First time:

```bash
cd /home/xxj/code/DINOCLIP_comparative_test/FHRB-Net
git fetch origin
git switch --track -c reproduce origin/reproduce
```

Later updates:

```bash
cd /home/xxj/code/DINOCLIP_comparative_test/FHRB-Net
git switch reproduce
git pull --ff-only
```

## Environment and weights

```bash
bash setup_server_env.sh
conda activate fhrbnet-rs
bash prepare_server_data.sh /home/xxj/Data
```

To run inference with an author-provided trained model, download the `weights` archive from [Baidu Netdisk](https://pan.baidu.com/s/1JKW8Btz_esnTO18PtuZ5mw), use extraction code `au2c`, and put the desired `.pth` file in `weights/`. Training defaults to `load=False` and does not require this FHRB-Net checkpoint. The environment setup uses the separate `requirements_server.txt`; the original file contains empty `onnx==` and `netron==` pins.

## Train

```bash
bash run_train.sh levircd
bash run_train.sh levircdplus
bash run_train.sh cdd
bash run_train.sh whucd
bash run_train.sh sysucd
```

If LEVIR-CD training runs out of GPU memory at the default batch size:

```bash
BATCH_SIZE=4 bash run_train.sh levircd
```

Restored defaults are 200 epochs, batch 8, AdamW, lr `2e-4`, weight decay `1e-3`, seed 42, 500-step warmup, AMP, and validation from epoch 100. GPU 0 and 8 workers are the launcher defaults; override with `GPU_ID=1` or `NUM_WORKERS=4`. If GPU memory is insufficient, use `BATCH_SIZE=4`; validation batch size remains twice the training batch size.

## Inference

For a downloaded author state dict:

```bash
bash run_inference.sh levircd weights/<downloaded-weight>.pth
bash run_inference.sh levircdplus weights/<downloaded-weight>.pth
bash run_inference.sh cdd weights/<downloaded-weight>.pth
bash run_inference.sh whucd weights/<downloaded-weight>.pth
bash run_inference.sh sysucd weights/<downloaded-weight>.pth
```

For a best-F1 state dict from training, use the file created below `FHRB-Net_work_dirs/<dataset>/best_f1score/` with the same command. Inference reports accuracy, precision, recall, and F1; the source inference entry does not report IoU. Outputs are written to a fresh timestamped directory.
