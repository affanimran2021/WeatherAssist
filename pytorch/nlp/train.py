"""Train on train; select the best checkpoint on validation accuracy only."""
import argparse
import logging
import shutil
from pathlib import Path
import torch
import utils
from model.net import Net, loss_fn
from model.data_loader import DataLoader


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data_dir', default='data/small')
    parser.add_argument('--model_dir', default='experiments/base_model')
    parser.add_argument('--device', default=None, help='auto, cpu, cuda or mps; overrides params.json')
    parser.add_argument('--overwrite', action='store_true', help='Explicitly restart an existing run')
    args = parser.parse_args()
    params = utils.Params(Path(args.model_dir) / 'params.json')
    if params.num_epochs < 1 or params.batch_size < 1:
        raise ValueError('num_epochs and batch_size must be positive')
    params.device = str(utils.get_device(args.device or getattr(params, 'device', 'auto')))
    utils.seed_everything(params.seed)
    loader = DataLoader(args.data_dir, params)
    data = loader.load_data(['train', 'val'], args.data_dir)
    train_batches = lambda: loader.data_iterator(data['train'], params, shuffle=True)
    val_batches = lambda: loader.data_iterator(data['val'], params)
    utils.prepare_run(args.model_dir, params, args.data_dir, args.overwrite)
    for name in ['words.txt', 'tags.txt', 'dataset_params.json']:
        shutil.copyfile(Path(args.data_dir)/name, Path(args.model_dir)/name)
    model = Net(params).to(params.device)
    optimizer = utils.make_optimizer(model, params)
    best_accuracy = float('-inf')
    for epoch in range(1, params.num_epochs + 1):
        train_metrics = utils.run_epoch(model, loss_fn, train_batches(), params.device, optimizer)
        val_metrics = utils.run_epoch(model, loss_fn, val_batches(), params.device)
        logging.info('Epoch %d/%d train=%s val=%s', epoch, params.num_epochs, train_metrics, val_metrics)
        is_best = val_metrics['accuracy'] > best_accuracy
        if is_best:
            best_accuracy = val_metrics['accuracy']
            utils.save_dict_to_json(dict(val_metrics, epoch=epoch),
                                    Path(args.model_dir) / 'metrics_val_best_weights.json')
        utils.save_dict_to_json(dict(val_metrics, epoch=epoch),
                                Path(args.model_dir) / 'metrics_val_last_weights.json')
        utils.save_checkpoint({'epoch': epoch, 'state_dict': model.state_dict(),
                               'optim_dict': optimizer.state_dict(), 'params': params.dict,
                               'best_val_accuracy': best_accuracy}, is_best, args.model_dir)
        utils.append_history(args.model_dir, {'epoch': epoch,
            'train_loss': train_metrics['loss'], 'train_accuracy': train_metrics['accuracy'],
            'train_count': train_metrics['count'], 'val_loss': val_metrics['loss'],
            'val_accuracy': val_metrics['accuracy'], 'val_count': val_metrics['count']})
    logging.info('Finished. Best validation accuracy: %.4f', best_accuracy)


if __name__ == '__main__':
    main()
