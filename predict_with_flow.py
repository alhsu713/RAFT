import argparse
import glob
import os
import torch
import numpy as np


def load_image(imfile):
    img = np.array(Image.open(imfile)).astype(np.uint8)
    img = torch.from_numpy(img).permute(2, 0, 1).float()
    return img[None].to(DEVICE)


def predict_with_flow(args):

    images = glob.glob(os.path.join(args.image, '*.png')) + \
        glob.glob(os.path.join(args.image, '*.jpg'))

    images = sorted(images)
    torch.nn.functional.grid_sample(images, args.flow)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', help="folder of images")
    parser.add_argument('--flow', help="Flow array shape (N, H, W, 2)")

    args = parser.parse_args()

    predict_with_flow(args)
