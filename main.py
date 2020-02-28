#!/usr/bin/env python3

import os
import pickle

import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from tqdm import tqdm


def resize() -> None:
    drct = 'images/'
    new = 'new/'
    dims = (200, 300)
    for i in tqdm(os.listdir(drct)):
        img = cv2.imread(drct + i)
        img_new = cv2.resize(img, dims, interpolation=cv2.INTER_AREA)
        cv2.imwrite(drct + new + i, img_new)


def main() -> None:
    drct = 'images/new/'
    imgs = []
    for i in os.listdir(drct):
        img = cv2.imread(drct + i)
        img = img.astype('float32')
        img = (img - 127.5) / 127.5
        imgs.append(img)
        print(np.max(img), np.min(img))

    pickle.dump(np.array(imgs), open('train.pkl', 'wb'))
    # train = tf.data.Dataset.from_tensor_slices(imgs)
    # pickle.dump(train, open('train_data.pkl', 'wb'))


if __name__ == '__main__':
    main()
