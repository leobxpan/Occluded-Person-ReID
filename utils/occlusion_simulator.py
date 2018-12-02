import numpy as np
from keras.preprocessing import image
import os
from scipy.misc import imsave
from tqdm import tqdm
import argparse
import shutil

def occlusion_simulator(image_path, height, width, save_dir, method = 'random', area = 'top'):
    """
    @param image_path: (absolute path) The path to the image to edit
    @param height: (int) Height of the black box in pixexls
    @param width: (int) Width of the black box in pixexls
    @param method: (str) takes 'random' or 'fixed' as a way of occlusion
    @param area: (str) if
    """
    # Read the image into python
    if not image_path.split("/")[-1].endswith(".jpg"):
        return "Exiting as its not an image"
    img = image.load_img (image_path)
    img = image.img_to_array(img)
    org_shape= img.shape

    # Init the mask to image size
    mask = np.ones (img.shape)

    # Randomly create a height x width mask and apply on image
    if method == 'random':
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

    elif method == 'fixed':
        # Create mask based on top, bottom or middle areas of size 1/3*height : width
        if area == 'top':
            start_height = 0
            start_width = 0
            mask[start_height: int(org_shape[0]*0.33), start_width:org_shape[1]] = np.zeros([int(org_shape[0]*0.33), org_shape[1], 3])

        elif area == 'bottom':
            start_height = int (org_shape[0] * 0.66)
            start_width = 0
            mask[start_height: int (org_shape[0]), start_width:org_shape[1]] = np.zeros ([int (abs(start_height - org_shape[0])), org_shape[1], 3])

        elif area == 'middle':
            start_height = int (org_shape[0] * 0.33)
            start_width = 0
            mask[start_height: int (org_shape[0]*0.66), start_width:org_shape[1]] = np.zeros([int (abs(start_height - org_shape[0]*0.66)), org_shape[1], 3])

        elif area == 'top50':
            start_height = 0
            end_height = int(org_shape[0]*0.33)
            mean_width = abs(np.mean([0, org_shape[1]]))
            start_width = mean_width - int(org_shape[1]*0.66/2)
            end_width = mean_width + int(org_shape[1]*0.66/2)
            mask[start_height:end_height, int (start_width):int (end_width)] = np.zeros ([abs (start_height - end_height),
                                                                                          abs (int (end_width - start_width)), 3])

        elif area == 'middle50':
            start_height = int(org_shape[0]*0.33)
            end_height = int(org_shape[0]*0.66)
            mean_width = abs(np.mean([0, org_shape[1]]))
            start_width = mean_width - int(org_shape[1]*0.66/2)
            end_width = mean_width + int(org_shape[1]*0.66/2)
            mask[start_height:end_height, int(start_width):int(end_width)] = np.zeros([abs(start_height-end_height),
                                                                                                   abs(int(end_width - start_width)), 3])

        elif area == 'bottom50':
            start_height = int (org_shape[0] * 0.66)
            end_height = int (org_shape[0])
            mean_width = abs(np.mean([0, org_shape[1]]))
            start_width = mean_width - int(org_shape[1]*0.66/2)
            end_width = mean_width + int(org_shape[1]*0.66/2)
            mask[start_height:end_height, int (start_width):int (end_width)] = np.zeros ([abs (start_height - end_height),
                                                                                          abs (int (end_width - start_width)), 3])
    # Transform the image using mask created
    transformed_img = np.multiply (img.flatten (), mask.flatten ())
    transformed_img = transformed_img.astype ('int').reshape (org_shape)

    # Save the transformed image to folder /occludedimages/
    save_path = os.path.join(save_dir, image_path.split('/')[-1])
    if not os.path.exists (save_dir):
        os.mkdir (save_dir)
    imsave(save_path, transformed_img)



if __name__ == "__main__":
    parser = argparse.ArgumentParser (description='Transform the images provided the directory')
    parser.add_argument ('--d', type=str, help='Path of the directory where images are kept')
    parser.add_argument ('--h', type=int, help='Height of the mask')
    parser.add_argument ('--w', type=int, help='Width of the mask')
    parser.add_argument ('--m', type=str, help='Method of excecution', choices = ['random', 'fixed'])
    parser.add_argument ('--a', type=str, help='Area to be masked', choices=['top', 'bottom', 'middle', 'top50',
                                                                             'middle50', 'bottom50'])
    #parser.add_argument('--save_dir',type = str, help = 'Directory to save the images')

    args = parser.parse_args()
    folders = os.listdir(args.d)

    if not os.path.exists(os.path.join(args.d, "occludedimages")):
        os.mkdir(os.path.join(args.d, "occludedimages"))

    for folder in folders:
        print folder
        if folder not in ['.DS_Store', 'Thumbs.db', 'readme.txt', 'occludedimages']:
            if folder != 'gt_query':
                files = os.listdir(os.path.join(args.d, folder))
                for f in tqdm(files):
                    if f not in ['.DS_Store', 'Thumbs.db']:
                        save_dir = os.path.join(args.d, "occludedimages", folder)
                        occlusion_simulator(os.path.join(args.d, folder, f), args.h, args.w, save_dir, args.m, args.a)
            else:
                save_dir = os.path.join (args.d, "occludedimages")
                if not os.path.exists (save_dir):
                    os.mkdir (save_dir)
                    os.system('cp -r %s %s' %(os.path.join(args.d, folder), save_dir))
                else:
                    pass
