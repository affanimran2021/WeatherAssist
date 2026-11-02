# ARTE 4110U — PyTorch Code Examples

These examples show how to organize a deep learning project in PyTorch.
Choose the example closest to your task and adapt its data loader, model, loss
and evaluation metrics to your approved project.

- [Vision example](pytorch/vision/README.md): image classification with a small CNN.
- [NLP example](pytorch/nlp/README.md): token tagging with an embedding layer and LSTM.

Both examples include data preparation, JSON experiment configurations, training
with validation, checkpoint saving/loading, test evaluation, hyperparameter search
and prediction on a new input. Each run saves its configuration, epoch history
and metrics in its experiment directory.

## Setup

Use Python 3.10–3.12 and a virtual environment. Install a compatible PyTorch and
torchvision pair for your hardware using https://pytorch.org/get-started/locally/.
Then run from the repository root:

```bash
python -m venv .venv
# macOS / Linux
source .venv/bin/activate
python -m pip install -r requirements.txt
```

On Windows PowerShell, activate with `.venv\Scripts\Activate.ps1` instead.

Run each example's commands from its own folder. See its README for the exact steps.
The small supplied/generated datasets are for checking the code, not final
project evaluation. Follow the course guide for project and submission requirements.

## Experiment structure

```text
pytorch/
    vision/
        model/
        experiments/base_model/params.json
        train.py
        evaluate.py
        predict.py
        search_hyperparams.py
        synthesize_results.py
    nlp/
        model/
        data/small/
        experiments/base_model/params.json
        train.py
        evaluate.py
        predict.py
        search_hyperparams.py
        synthesize_results.py
```

Keep one shared repository per project group. Use fresh experiment directories
for new configurations and seeds. Select models with validation data and reserve
test data for final evaluation. Adapt the example's accuracy-based checkpoint
selection when another metric is appropriate.

Keep data, large checkpoints and credentials out of Git. Provide clear commands
to reproduce your own project and run its demonstration.
