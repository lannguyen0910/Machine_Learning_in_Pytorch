import torch
import torch.utils.data as data
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
np.random.seed(40)
IMG_SIZE = 8
MIN_OBJECT_SIZE = 1
MAX_OBJECT_SIZE = 4
TRAIN_SAMPLES = 10000
VAL_SAMPLES = 500
TEST_SAMPLES = 10
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
testset, testbboxes = DataGenerator(TEST_SAMPLES, True)

trainset_tensor = torch.Tensor(trainset).view(-1, IMG_SIZE*IMG_SIZE)
valset_tensor = torch.Tensor(valset).view(-1, IMG_SIZE*IMG_SIZE)
testset_tensor = torch.Tensor(testset).view(-1, IMG_SIZE*IMG_SIZE)

train_bboxes = torch.Tensor(train_bboxes).view(-1, 4)

trainset_tensor = data.TensorDataset(trainset_tensor, train_bboxes)
trainloader = data.DataLoader(
    trainset_tensor, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
valloader = data.DataLoader(
    valset_tensor, batch_size=BATCH_SIZE, num_workers=4, pin_memory=True)
