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

## Model complexity

After activating `fhrbnet-rs`, install the profiling dependency and count parameters/FLOPs for one paired 256x256 input. No trained checkpoint is needed:

```bash
python -m pip install -r requirements_profile.txt
CUDA_VISIBLE_DEVICES=0 python profile_model.py --device cuda:0 \
  --output FHRB-Net_work_dirs/profile_256.json
```

The JSON lists unsupported operators. The model contains FFT operations; if fvcore lists them as unsupported, the FLOP total excludes FFT work and is only partial. Add `--debug-flops` to re-raise tracing failures with their full traceback.

## Train

```bash
bash run_train.sh levircd
bash run_train.sh levircdplus
bash run_train.sh cdd
bash run_train.sh whucd
bash run_train.sh sysucd
```

LEVIR-CD and LEVIR-CD-PLUS images are tiled into aligned, non-overlapping 256x256 patches in memory by the launcher. This is required for 1024x1024 source images and does not write or alter dataset files. Already-cropped 256x256 data remains one sample per file. Other datasets retain their existing read behavior.

Defaults are 200 epochs, batch 8, AdamW, lr `2e-4`, weight decay `1e-3`, seed 42, 500-step warmup, AMP, and validation from epoch 100. GPU 0 and 8 workers are the launcher defaults; override with `GPU_ID=1` or `NUM_WORKERS=4`. Validation batch size is twice the training batch size. For LEVIR datasets on a 32 GB GPU, start with batch 1 or 2:

```bash
GPU_ID=1 BATCH_SIZE=1 NUM_WORKERS=2 bash run_train.sh levircdplus
```

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

LEVIR-CD and LEVIR-CD-PLUS inference applies the same 256x256 in-memory tiling. Inference batch defaults to 1; adjust it with `INFERENCE_BATCH_SIZE` if memory allows.
