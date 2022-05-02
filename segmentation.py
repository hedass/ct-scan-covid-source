import tensorflow as tf
import glob
import cv2
import os
import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow import keras
from skimage import morphology
from natsort import natsorted
import imageio
from pathlib import Path
import sys
from numba import cuda


class Segmentation():
    def __init__(self, src, dst):
        self.model_path = './models/segmentasi_lession_resnet34_fold_3-03-0.9583.hdf5'
        self.src = src
        self.dst = dst
        self.SIZE_X = 512
        self.SIZE_Y = 512

    def combine_image(self, ori, pred):
        # create a structuring element
        se_disk_1 = morphology.disk(2)

        # erosi
        eroded_mask = morphology.erosion(pred[:, :, 0], se_disk_1)

        # Substract with original mask
        edge_mask = pred[:, :, 0] - eroded_mask
        edge_mask[edge_mask < 0.9] = 0

        temp = ori.copy()[:, :, 0]

        edge_mask = cv2.cvtColor(edge_mask, cv2.COLOR_GRAY2RGB)

        temp = cv2.cvtColor(temp, cv2.COLOR_GRAY2RGB)

        for i in range(temp.shape[0]):
            for j in range(temp.shape[0]):
                if edge_mask[i, j].all() > 0:
                    temp[i, j] = [255, 0, 0]

        return temp

    def run(self):
        model = keras.models.load_model(
            './models/segmentasi_lession_resnet34_fold_3-03-0.9583.hdf5', compile=False)

        source = self.src
        dest = self.dst

        classesDir = glob.glob(source + "*/")

        file_paths = glob.glob(os.path.join(source, "*.png"))
        file_paths = natsorted(file_paths)
        for i in range(0, len(file_paths)):
            try:
                ori = cv2.imread(file_paths[i])

                # convert to gray
                tmp = cv2.cvtColor(ori, cv2.COLOR_BGR2GRAY)
                tmp = cv2.resize(tmp, (self.SIZE_Y, self.SIZE_X))
                tmp = np.expand_dims(tmp, axis=0)
                pred = model.predict(tmp)
                pred = pred[0]
                pred[pred < 0.9] = 0
                result = self.combine_image(ori, pred)
                dest_dir = dest

                # create a dir
                if (not os.path.exists(dest_dir)):
                    tmp = Path(dest_dir)
                    tmp.mkdir(parents=True)
                imageio.imwrite(os.path.join(
                    dest_dir, "{:05d}.png".format(i + 1)), result)
            except KeyboardInterrupt:
                sys.exit(0)
            except Exception as e:
                del model
                print(e)
                print("error :" + file_paths[i])
                sys.exit(0)
        del model
