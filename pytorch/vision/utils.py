"""Configuration, reproducibility, checkpoints and correctly weighted metrics.

See the source license in ../../LICENSE.
"""
import csv
import importlib.metadata
import json
import logging
import os
from pathlib import Path
import platform
import random
import shutil
import subprocess
import sys

import numpy as np
import torch


class Params:
    def __init__(self, json_path):
        self.update(json_path)

    def update(self, json_path):
        with open(json_path, encoding='utf-8') as handle:
            self.__dict__.update(json.load(handle))

    def save(self, json_path):
        save_dict_to_json(self.dict, json_path)

    @property
    def dict(self):
        return self.__dict__


def save_dict_to_json(values, json_path):
    path = Path(json_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as handle:
        json.dump(values, handle, indent=2, allow_nan=False)
        handle.write('\n')


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    # Seeds aid repeatability; identical results across devices/versions are not guaranteed.


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def get_device(request='auto'):
    if request == 'auto':
        if torch.cuda.is_available():
            return torch.device('cuda')
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return torch.device('mps')
        return torch.device('cpu')
    return torch.device(request)


def set_logger(log_path):
    Path(log_path).parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)s %(message)s',
                        handlers=[logging.FileHandler(log_path, encoding='utf-8'),
                                  logging.StreamHandler()], force=True)


def make_optimizer(model, params):
    options = dict(lr=params.learning_rate,
                   weight_decay=getattr(params, 'weight_decay', 0.0))
    name = getattr(params, 'optimizer', 'adam').lower()
    if name == 'adam':
        return torch.optim.Adam(model.parameters(), **options)
    if name == 'adamw':
        return torch.optim.AdamW(model.parameters(), **options)
    if name == 'sgd':
        return torch.optim.SGD(model.parameters(),
                               momentum=getattr(params, 'momentum', 0.9), **options)
    raise ValueError('optimizer must be adam, adamw or sgd')


def prepare_run(model_dir, params, data_dir, overwrite=False):
    directory = Path(model_dir)
    directory.mkdir(parents=True, exist_ok=True)
    if not overwrite and any((directory / name).exists()
                             for name in ['history.csv', 'last.pth.tar', 'best.pth.tar']):
        raise FileExistsError('Run already exists. Use a new model_dir, or --overwrite to restart it.')
    # --overwrite is an explicit restart, not an exact training-resume mechanism.
    if overwrite:
        for pattern in ['*.pth.tar', 'metrics_*.json', 'history.csv', 'train.log',
                        'evaluate.log', 'environment.json', 'params_resolved.json']:
            for path in directory.glob(pattern):
                path.unlink()
    resolved = dict(params.dict)
    resolved['data_dir'] = str(Path(data_dir).resolve())
    save_dict_to_json(resolved, directory / 'params_resolved.json')
    packages = {d.metadata['Name']: d.version for d in importlib.metadata.distributions()
                if d.metadata['Name']}
    try:
        commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'],
                                         stderr=subprocess.DEVNULL, text=True).strip()
        dirty = bool(subprocess.check_output(['git', 'status', '--porcelain'],
                                            stderr=subprocess.DEVNULL, text=True).strip())
    except (OSError, subprocess.CalledProcessError):
        commit, dirty = None, None
    save_dict_to_json({'python': sys.version, 'platform': platform.platform(),
                       'torch': str(torch.__version__), 'device': params.device,
                       'git_commit': commit, 'git_dirty': dirty,
                       'packages': packages}, directory / 'environment.json')
    set_logger(directory / 'train.log')


def save_checkpoint(state, is_best, checkpoint):
    directory = Path(checkpoint)
    directory.mkdir(parents=True, exist_ok=True)
    torch.save(state, directory / 'last.pth.tar')
    if is_best:
        shutil.copyfile(directory / 'last.pth.tar', directory / 'best.pth.tar')


def load_checkpoint(checkpoint, model, optimizer=None):
    if not Path(checkpoint).is_file():
        raise FileNotFoundError(checkpoint)
    device = next(model.parameters()).device
    state = torch.load(checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(state['state_dict'])
    if optimizer is not None:
        optimizer.load_state_dict(state['optim_dict'])
    return state


def append_history(model_dir, row):
    path = Path(model_dir) / 'history.csv'
    exists = path.exists()
    with path.open('a', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def run_epoch(model, loss_fn, batches, device, optimizer=None):
    """Aggregate over every image/valid token, including the last partial batch.

    This example reports accuracy and mean loss. Replace/add metrics for your task.
    Token labels use -1 for padding; these positions never count in the metrics.
    """
    model.train(optimizer is not None)
    total_loss, correct, count = 0.0, 0, 0
    with torch.set_grad_enabled(optimizer is not None):
        for inputs, labels in batches:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)
            if not torch.isfinite(loss):
                raise ValueError('Non-finite loss: check inputs, labels and learning rate.')
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
            flat_labels = labels.reshape(-1)
            mask = flat_labels >= 0
            n = int(mask.sum().item())
            if n == 0:
                continue
            predicted = outputs.argmax(dim=-1).reshape(-1)
            correct += int(((predicted == flat_labels) & mask).sum().item())
            total_loss += float(loss.item()) * n
            count += n
    if not count:
        raise ValueError('Empty split or no valid labels.')
    return {'loss': total_loss / count, 'accuracy': correct / count, 'count': count}
