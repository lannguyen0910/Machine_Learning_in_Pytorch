import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

df = pd.read_csv('train_masks.csv')
print(len(df))

train_df, val_df = train_test_split(df, train_size=0.8, random_state=40)
print(df.head())


def denormalize(img, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    """
    From tensor to numpy to plot images
    """

    mean = np.array(mean)
    std = np.array(std)

    img_show = img.clone()
    if img_show.shape[0] == 1:
        img_show = img_show.squeeze(0)

    img_show = img_show.numpy().transpose((1, 2, 0))
    img_show = (img_show * std) + mean
    img_show = np.clip(img_show, 0, 1)

    return img_show
