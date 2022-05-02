import os
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import cv2
import pydicom
import imageio
from skimage.segmentation import clear_border, mark_boundaries
from skimage.morphology import closing
from skimage.morphology import disk
from natsort import natsorted


class PreProcessing():
    def __init__(self, src, dst):
        self.src = src
        self.dst = dst
        self.classes = ["COVID", "Normal", "Pneumonia"]

    def compute_area(self, mask, pixdim):
        """
        Computes the area (number of pixels) of a binary mask and multiplies the pixels
        with the pixel dimension of the acquired CT image
        Args:
            lung_mask: binary lung mask
            pixdim: list or tuple with two values

        Returns: the lung area in mm^2
        """
        mask[mask >= 1] = 1
        lung_pixels = np.sum(mask)
        return lung_pixels * pixdim[0] * pixdim[1]

    def transform_to_hu(self, medical_image, image):
        intercept = medical_image.RescaleIntercept
        slope = medical_image.RescaleSlope
        hu_image = image * slope + intercept

        return hu_image

    def window_image(self, image, window_center, window_width):
        img_min = window_center - window_width // 2
        img_max = window_center + window_width // 2
        window_image = image.copy()
        window_image[window_image < img_min] = img_min
        window_image[window_image > img_max] = img_max

        return window_image

    def get_segmented_lungs(self, raw_im, plot=False):

        im = raw_im.copy()
        binary = im < -400
        cleared = clear_border(binary)

        return cleared

    def get_segmented_lungs_2(self, raw_im):

        im = raw_im.copy()
        binary = im < -400
        footprint = disk(3)
        opened = closing(binary, footprint)
        cleared = clear_border(opened)

        return cleared

    def isAxial(self, ds):
        return all(np.round(ds.ImageOrientationPatient, 0) == [1, 0, 0, 0, 1, 0])

    def run(self):
        dataset_path = self.src
        dest_dir = self.dst
        classes = self.classes

        # lung_areas = {}
        patients_folders = os.listdir(dataset_path)

        all_dcms = []
        curr_dcm = []
        for i, path in enumerate(patients_folders):
            try:
                if('.dcm' not in path):
                    continue
                tmp_path = os.path.join(
                    dataset_path, path)
                tmp_dcm_read = pydicom.dcmread(tmp_path, force=True)
                if(i != 0 and int(tmp_dcm_read.SliceLocation) == 0):
                    all_dcms.append(curr_dcm)
                    curr_dcm = []
                curr_dcm.append(path)
            except:
                pass
        all_dcms.append(curr_dcm)

        for i, dcm in enumerate(all_dcms):
            tmp_dest_dir = dest_dir

            if (not os.path.exists(tmp_dest_dir)):
                temp = Path(tmp_dest_dir)
                temp.mkdir(parents=True)
            count = 0

            for path in dcm:
                try:
                    if('.dcm' not in path):
                        continue
                    tmp_path = os.path.join(
                        dataset_path, path)
                    medical_image = pydicom.dcmread(
                        tmp_path, force=True)

                    if(not self.isAxial(medical_image)):
                        continue
                    image = medical_image.pixel_array
                    hu_image = self.transform_to_hu(
                        medical_image, image)
                    lung_image = self.window_image(hu_image, 0, 1600)
                    lung_seg = self.get_segmented_lungs_2(hu_image)
                    lung_area = self.compute_area(
                        lung_seg, medical_image.PixelSpacing)
                    if(lung_area > 30000 or lung_area < 10000):
                        continue
                    fixed = cv2.normalize(
                        src=lung_image, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                    tmp_1 = (fixed.shape[0] - 512) // 2
                    tmp_2 = (fixed.shape[1] - 512) // 2
                    fixed = fixed[tmp_1:(512+tmp_1), tmp_2:(512+tmp_2)]
                    if(fixed.size < 1000):
                        continue
                    count += 1
                    imageio.imwrite(os.path.join(
                        tmp_dest_dir, "{:05d}".format(count)) + ".png", fixed)
                except:
                    print("DCM ERROR", tmp_path)
                    pass
