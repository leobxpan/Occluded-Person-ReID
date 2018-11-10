import cv2
import numpy as np
from keras.preprocessing import image
import matplotlib.pyplot as plt
import os
import cv2
from scipy.misc import imsave


def occlusion_simulator(image_path, height, width):
    """
    @param image_path: The path to the image to edit
    @param coords: A tuple of x/y coordinates (x1, y1, x2, y2)
    @param saved_location: Path to save the cropped image
    """
    img = image.load_img (image_path)
    img = image.img_to_array(img)
    org_shape= img.shape
    mask = np.ones(img.shape)

    start_height = np.random.randint(org_shape[0] - height)
    start_width = np.random.randint (org_shape[1] - width)
    mask[start_height: start_height + height, start_width: start_width + width] = np.zeros([height, width, 3])

    transformed_img = np.multiply(img.flatten(), mask.flatten())
    transformed_img = transformed_img.astype('int').reshape(org_shape)

    # Code to save the transformed image to folder /transformed/
    save_dir = os.path.join('/'.join(image_path.split('/')[:-1]), "transformed/")
    save_path = os.path.join('/'.join(image_path.split('/')[:-1]), "transformed/%s" %(image_path.split('/')[-1]))
    if not os.path.exists (save_dir):
        os.mkdir (save_dir)
    imsave(save_path, transformed_img)
