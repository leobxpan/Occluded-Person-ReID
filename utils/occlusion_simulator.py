import numpy as np
from keras.preprocessing import image
import os
from scipy.misc import imsave
import sys
from tqdm import tqdm
import argparse

def occlusion_simulator(image_path, height, width):
    """
    @param image_path: (absolute path) The path to the image to edit
    @param height: (int) Height of the black box in pixexls
    @param width: (int) Width of the black box in pixexls
    """
    # Read the image into python
    if image_path.split("/")[-1].startswith("."):
        return "Exiting as its not an image"
    img = image.load_img (image_path)
    img = image.img_to_array(img)
    org_shape= img.shape
    mask = np.ones(img.shape)

    # Get the bounds for the mask
    if org_shape[0] > height:
        start_height = np.random.randint(org_shape[0] - height)
    else:
        start_height = 0
        height = org_shape[0]

    if org_shape[1] > width:
        start_width = np.random.randint (org_shape[1] - width)
    else:
        start_width = 0
        width = org_shape[1]

    # Create the mast
    mask[start_height: start_height + height, start_width: start_width + width] = np.zeros([height, width, 3])

    # Transform the image
    transformed_img = np.multiply(img.flatten(), mask.flatten())
    transformed_img = transformed_img.astype('int').reshape(org_shape)

    # Save the transformed image to folder /transformed/
    save_dir = os.path.join('/'.join(image_path.split('/')[:-1]), "transformed/")
    save_path = os.path.join('/'.join(image_path.split('/')[:-1]), "transformed/%s" %(image_path.split('/')[-1]))
    if not os.path.exists (save_dir):
        os.mkdir (save_dir)
    imsave(save_path, transformed_img)


if __name__ == "__main__":
    parser = argparse.ArgumentParser (description='Transform the images provided the directory')
    parser.add_argument ('--d', type=str, help='Path of the directory where images are kept')
    parser.add_argument ('--h', type=int, help='Height of the mask')
    parser.add_argument ('--w', type=int, help='Width of the mask')
    args = parser.parse_args ()

    print args.d
    files = os.listdir(args.d)
    for f in tqdm(files):
        occlusion_simulator(os.path.join(args.d, f), args.h, args.w)
