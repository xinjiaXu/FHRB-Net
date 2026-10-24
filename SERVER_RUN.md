# FHRB-Net server setup

FHRB-Net reads the original dataset folders under `/home/xxj/Data`; no dataset copy or per-model adapter tree is created. Checkpoints, W&B offline files, feature logs, and inference outputs are stored under `FHRB-Net_work_dirs/<dataset>/`.

## Environment

From `/home/xxj/code/DINOCLIP_comparative_test/FHRB-Net`:

```bash
bash setup_server_env.sh
conda activate fhrbnet-rs
```

The setup uses Python 3.10 and PyTorch 2.7.1/CUDA 12.8 for RTX 5090 support. The environment script installs the valid project dependencies from `requirements_server.txt`. The checked-in `requirements.txt` contains empty version pins (`onnx==`, `netron==`); those imports were unused by training and are omitted from the server dependency list.

## Restored defaults

The latest checkout had an empty `utils/path_hyperparameter.py`. Its previous nonempty version is present in Git history, so the reproduce branch restores those project defaults and sets `load=False` for training from scratch:

- 200 epochs, batch size 8, AdamW, learning rate `2e-4`, weight decay `1e-3`
- Seed 42, 500-step warmup, AMP enabled
- Validation starts at epoch 100; best-F1 and periodic checkpoints are saved
- Training augmentation/noise defaults retained (`noise_p=0.3`)

The paper reports the same epoch count, batch size, optimizer, seed, learning rate and weight decay. The runnable defaults are recovered from the repository's earlier config commit.

## Data layout

Check the five dataset paths:

```bash
bash prepare_server_data.sh /home/xxj/Data
```

The command checks folders only. Training and inference read each split directly. Filenames must match by stem across the two image folders and label folder.

| Key | Dataset root | Image A | Image B | Label |
| --- | --- | --- | --- | --- |
| `levircd` | `/home/xxj/Data/LEVIR_CD` | `A` | `B` | `label` |
| `levircdplus` | `/home/xxj/Data/LEVIR_CD_PLUS/cut_data` | `A` | `B` | `label` |
| `cdd` | `/home/xxj/Data/CDD` | `A` | `B` | `OUT` |
| `whucd` | `/home/xxj/Data/WHUCD` | `A` | `B` | `label` |
| `sysucd` | `/home/xxj/Data/SYSU_CD` | `time1` | `time2` | `label` |

The server layout uses `A/B`, while the README describes `t1/t2`; the wrappers pass the actual directory names directly. No rename or preprocessing copy is needed.

## Train

```bash
bash run_train.sh levircd
bash run_train.sh levircdplus
bash run_train.sh cdd
bash run_train.sh whucd
bash run_train.sh sysucd
```

The default GPU is 0 and workers default to 8. Override runtime choices with `GPU_ID=1` or `NUM_WORKERS=4`. Checkpoints are separated into `checkpoints/`, `best_f1score/`, and `best_loss/` below each dataset work directory. Offline W&B logs stay within that same work directory.

The source initializes `best_metrics['lowest loss']` to zero and only saves a best-loss model when a later loss is below that value; positive losses therefore normally will not produce a file in `best_loss/`. Best-F1 and periodic checkpoint saving are unaffected.

## Inference

The weights README links an author-provided Baidu Drive archive (extraction code `au2c`). Download the desired weight file into `weights/` yourself. The upstream inference entry uses a raw state dictionary by default:

```bash
bash run_inference.sh levircd weights/<downloaded-weight-file>.pth
```

For a checkpoint created by this training code, pass its checkpoint path and format:

```bash
bash run_inference.sh levircd FHRB-Net_work_dirs/levircd/checkpoints/<checkpoint-file>.pth checkpoint
```

Inference reports the project's TorchMetrics accuracy, precision, recall and F1 over the test split. It does not report IoU although the paper reports IoU as a metric. Feature outputs and the loaded model state are placed in a fresh timestamped work directory.
