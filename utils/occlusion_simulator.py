import numpy as np
from keras.preprocessing import image
import os
from scipy.misc import imsave


def occlusion_simulator(image_path, height, width):
    """
    @param image_path: (absolute path) The path to the image to edit
    @param height: (int) Height of the black box in pixexls
    @param width: (int) Width of the black box in pixexls
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
