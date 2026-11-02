"""Predict one image with the saved checkpoint. Adapt outputs to your task."""
import argparse
import json
from pathlib import Path
from PIL import Image
import torch
import utils
from model.net import Net
from model.data_loader import make_transform


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--image', required=True)
    parser.add_argument('--model_dir', default='experiments/base_model')
    parser.add_argument('--device', default='auto')
    args = parser.parse_args()
    device = utils.get_device(args.device)
    state = torch.load(Path(args.model_dir)/'best.pth.tar', map_location='cpu', weights_only=True)
    params = utils.Params(Path(args.model_dir)/'params.json')
    params.__dict__.update(state['params'])
    model = Net(params).to(device)
    model.load_state_dict(state['state_dict'])
    model.eval()
    with Image.open(args.image) as image:
        tensor = make_transform(params)(image.convert('RGB')).unsqueeze(0).to(device)
    with torch.no_grad():
        probabilities = model(tensor).softmax(dim=-1)[0].cpu()
    index = int(probabilities.argmax())
    print(json.dumps({'class_id': index, 'probability': float(probabilities[index]),
                      'probabilities': probabilities.tolist()}))


if __name__ == '__main__':
    main()
