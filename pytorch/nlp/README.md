# Token Tagging with PyTorch

This example predicts one named-entity tag per word using an embedding layer,
a unidirectional LSTM and a linear output layer. Run commands from `pytorch/nlp`
after installing the root requirements.

## Quick start

The supplied `data/small` contains 10 sentences per split, for checking the code.

```bash
python build_vocab.py --data_dir data/small
python train.py --data_dir data/small --model_dir experiments/base_model --device cpu
python evaluate.py --data_dir data/small --model_dir experiments/base_model --split val --device cpu
python predict.py --text "John lives in New York" --model_dir experiments/base_model --device cpu
```

Words are learned from the training split only. `label_schema.json` defines the
task's possible tags independently of evaluation observations. If absent, tags
are taken from training data. Evaluation and prediction use the exact vocabulary
and tag mapping saved beside the trained checkpoint.

## Data format

Each of `train`, `val` and `test` contains `sentences.txt` and `labels.txt`.
One line is one sentence, with space-separated tokens and a matching tag per token.

```text
Sentence: John lives in New York
Labels:   B-per O O B-geo I-geo
```

Do not include the `Sentence:` or `Labels:` prefixes in the actual data files.
Unknown words map to UNK. Padding labels are -1 and are excluded from loss and
accuracy. All sentences, including the final partial batch, are processed.

For the original [NER CSV corpus](https://www.kaggle.com/datasets/abhinavwalia95/entity-annotated-corpus),
check access and reuse rights. To convert `ner_dataset.csv` after downloading it:

```bash
python build_kaggle_dataset.py --csv data/kaggle/ner_dataset.csv --output_dir data/kaggle
python build_vocab.py --data_dir data/kaggle
```

The converter preserves sentence order with a 70/15/15 split and records a
manifest. Check duplicate/document leakage and adapt the split for your project.

## Experiments and evaluation

Edit `experiments/base_model/params.json` before training. Use a fresh experiment
directory per configuration and seed. `--overwrite` explicitly restarts a run.

```bash
python search_hyperparams.py --data_dir data/small --parent_dir experiments/learning_rate
python synthesize_results.py --parent_dir experiments
```

Runs save configurations, checkpoints, logs and `history.csv` for learning curves.
Select with validation data, then evaluate the held-out test split:

```bash
python evaluate.py --data_dir data/small --model_dir experiments/base_model --split test
```

## Adapt to your project

Replace data loading, architecture, losses and metrics for your task. Sentence
classification needs a sentence-level output and target; this example produces
token-level outputs. Pretrained models require their own tokenizer/preprocessing.

The example selects by validation token accuracy. For real named-entity
recognition, add entity-level precision, recall and F1; token accuracy can be
dominated by the O tag. Use the tiny bundled data only as a software check.
