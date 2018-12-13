import numpy as np
from keras.preprocessing import image
import os
from scipy.misc import imsave
from tqdm import tqdm
import argparse
import pickle
from shutil import copyfile

def occlusion_simulator(image_path, height, width, method = 'random', area = 'top'):
    """
    @param image_path: (absolute path) The path to the image to edit
    @param height: (int) Height of the black box in pixexls
    @param width: (int) Width of the black box in pixexls
    @param method: (str) takes 'random' or 'fixed' as a way of occlusion
    @param area: (str) if
    """
    # Read the image into python
    if image_path.split("/")[-1].startswith(".") or image_path.split("/")[-1].startswith("transformed"):
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
        average = img.mean ( axis=0 ).mean ( axis=0 )
        mask_tmp = np.ones ( [height, width, 3] )
        for i in range(mask_tmp.shape[2]):
            tmp = mask_tmp[:, :, i]
            tmp = tmp * average[i]/255.0
            mask_tmp[:,:,i] = tmp

        #print mask_tmp
        mask[start_height: start_height + height, start_width: start_width + width] = mask_tmp

    elif method == 'fixed':
        # Create mask based on top, bottom or middle areas of size 1/3*height : width
        if area == 'top':
            start_height = 0
            start_width = 0
            mask[start_height: int(img.shape[0]*0.33), start_width:img.shape[1]] = np.zeros ([int(img.shape[0]*0.33), img.shape[1], 3])
        elif area == 'bottom':
            start_height = int (img.shape[0] * 0.66)
            start_width = 0
            mask[start_height: int (img.shape[0]), start_width:img.shape[1]] = np.zeros (
                [int (img.shape[0]), img.shape[1], 3])
        elif area == 'middle':
            start_height = int (img.shape[0] * 0.33)
            start_width = 0
            mask[start_height: int (img.shape[0]*0.66), start_width:img.shape[1]] = np.zeros (
                [int (img.shape[0]*0.66), img.shape[1], 3])

    # Transform the image using mask created
    transformed_img = np.multiply (img.flatten (), mask.flatten ())
    transformed_img = transformed_img.astype ('int').reshape (org_shape)

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
    parser.add_argument ('--m', type=str, help='Method of excecution', choices = ['random', 'fixed'])
    parser.add_argument ('--a', type=str, help='Area to be masked', choices=['top', 'bottom', 'middle'])
    parser.add_argument ('--partition', type=str, help='Path to the partition.pkl file')

    args = parser.parse_args ()
    # print args
    # files = os.listdir(args.d)

    partition = pickle.load(open(args.partition))
    print partition.keys()
    #print partition['val_marks']
    #print partition['val_im_names']
    im_names_group = ['train_im_names', 'trainval_im_names', 'val_im_names', 'test_im_names']
    marks = ['test_marks', 'val_marks']
    for im_group in im_names_group:
        im_names = partition[im_group]

        # Check if marks exists or not for im_group
        if im_group.startswith('test'):
            marks_group = 'test_marks'
        elif im_group.startswith('val'):
            marks_group = 'val_marks'
        else:
            marks_group = None

        for i, im_name in tqdm(enumerate(im_names)):
            if marks_group is not None:
                mark = partition[marks_group][i]
                if mark != 1:
                    occlusion_simulator ( os.path.join ( args.d, im_name ), args.h, args.w, args.m, args.a )
                else:
                    if not os.path.exists(os.path.join ( args.d, 'transformed')):
                        os.mkdir(os.path.join ( args.d, 'transformed'))
                    copyfile ( os.path.join ( args.d, im_name ), os.path.join ( args.d, 'transformed', im_name ) )
            else:
                occlusion_simulator ( os.path.join ( args.d, im_name ), args.h, args.w, args.m, args.a )