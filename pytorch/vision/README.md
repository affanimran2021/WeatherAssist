# Image Classification with PyTorch

This example trains a small CNN. Run all commands below from `pytorch/vision`
after installing the root requirements.

## Try the code without a download

```bash
python make_toy_dataset.py --output_dir data/toy
python train.py --data_dir data/toy --model_dir experiments/base_model --device cpu
python evaluate.py --data_dir data/toy --model_dir experiments/base_model --split val --device cpu
python predict.py --image data/toy/val_signs/0_val_000.png --model_dir experiments/base_model --device cpu
```

The toy images are synthetic colour/stripe patterns for checking the pipeline
only. They are not hand signs or a suitable project dataset.

## Use the hand-sign image format

The original image example uses RGB hand-sign images for six classes (0–5).
The [SIGNS dataset link](https://drive.google.com/file/d/1ufiR6hUKhXoAyiBNsySPkUwlvE_wfEHC/view)
is provided as an optional starting point; check availability and reuse rights.
Expected structure and filename labels:

```text
data/SIGNS/train_signs/0_IMG_5864.jpg
data/SIGNS/test_signs/0_IMG_5942.jpg
```

```bash
python build_dataset.py --data_dir data/SIGNS --output_dir data/64x64_SIGNS --seed 4110
python train.py --data_dir data/64x64_SIGNS --model_dir experiments/base_model
```

Data preparation resizes images and creates a reproducible train/validation split
from the raw training partition. It preserves the supplied test partition and
writes a split manifest. Use a new output directory for another preparation run.
Check duplicates and subject/source leakage for your actual dataset.

## Run experiments

Edit `experiments/base_model/params.json` before training. Copy its configuration
into a new experiment folder for another run. Supported optimizers are Adam,
AdamW and SGD. `--device auto` selects CUDA, then MPS, then CPU; pass `--device cpu`
to force CPU. Use `--overwrite` only to explicitly restart an existing run.

```bash
python search_hyperparams.py --data_dir data/toy --parent_dir experiments/learning_rate
python synthesize_results.py --parent_dir experiments
```

Each run saves `history.csv`, `train.log`, resolved settings, environment details,
validation metrics, and best/last checkpoints. Use the CSV for learning curves.
Once choices are fixed, evaluate the selected model on the held-out test split:

```bash
python evaluate.py --data_dir data/toy --model_dir experiments/base_model --split test
```

## Adapt to your project

- `model/data_loader.py`: input format, labels, preprocessing and batching.
- `model/net.py`: architecture and loss. The example returns raw class logits.
- `train.py`: training choices and validation-based checkpoint selection.
- `utils.py`: epoch metrics; the example reports mean loss and image accuracy.
- `evaluate.py` and `predict.py`: task-specific evaluation and inference.

Set `num_classes`, preprocessing and augmentation appropriately. The example
selects checkpoints by validation accuracy. Change that criterion for tasks such
as regression or detection. The CNN supports partial batches, including one image.
