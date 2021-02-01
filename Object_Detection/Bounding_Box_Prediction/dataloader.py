import torch
import torch.nn as nn
import torch.utils.data as data
from torch.autograd import Variable
import torchvision
import matplotlib.pyplot as plt
import matplotlib.patches as pt
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
np.random.seed(40)
IMG_SIZE = 8
MIN_OBJECT_SIZE = 1
MAX_OBJECT_SIZE = 4
TRAIN_SAMPLES = 10000
VAL_SAMPLES = 500
BATCH_SIZE = 500


def DataGenerator(NUM_IMGS, train=False):
    dataset = np.zeros((NUM_IMGS, IMG_SIZE, IMG_SIZE))
    bboxes = []

    for i in range(NUM_IMGS):
        w, h = np.random.randint(MIN_OBJECT_SIZE, MAX_OBJECT_SIZE + 1, size=2)
        x = np.random.randint(0, IMG_SIZE - w)
        y = np.random.randint(0, IMG_SIZE - h)
        dataset[i, y: y + h, x: x + w] = 1
        if train:
            bboxes.append([x, y, w, h])

    if train:
        return dataset, bboxes
    return dataset


trainset, train_bboxes = DataGenerator(TRAIN_SAMPLES, True)
valset, val_bboxes = DataGenerator(VAL_SAMPLES, True)


def plot_data(dataset, bboxes, figsize):
    a = np.random.randint(200)
    fig = plt.figure(figsize=figsize)
    for id, data in enumerate(dataset[a:a + 12]):
        fig.add_subplot(3, 4, id + 1)
        bbox = bboxes[a + id]
        plt.imshow(data, cmap="binary", interpolation='none',
                   origin='lower', extent=[0, IMG_SIZE, 0, IMG_SIZE])
        plt.gca().add_patch(pt.Rectangle((bbox[0], bbox[1]), bbox[2], bbox[3],
                                         ec="r", fc="none", lw=5))
    plt.show()


plot_data(trainset, train_bboxes, (20, 15))
