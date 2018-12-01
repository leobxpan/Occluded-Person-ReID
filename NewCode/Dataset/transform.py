import glob
import os.path
import os
import numpy as np
import re
from collections import defaultdict
import shutil
import pickle


imagename_re = re.compile ('(?P<id>\d+)_c(?P<cam>\d)s(\d)')
new_image_name_template = '{:08d}_{:04d}_{:08d}_{:01d}.jpg'

def parse_original_image_name(image_name, parse_type='id'):
    """
    Get the person id or cam from an image name.
    :param image_name: Name of the image
    :param parse_type: Whether to parse the image id or image cam id
    :return: Parsed value (either cam id or image id)
    """
    global parsed
    assert parse_type in ('id', 'cam')
    imagename_search = imagename_re.search (image_name)
    if imagename_search is not None:
        group_dict = imagename_search.groupdict ()
        if parse_type == 'id':
            parsed = group_dict['id']
        else:
            parsed = group_dict['cam']
    return int(parsed)


def get_image_names(image_dir, pattern='*.jpg', return_np=True, return_path=False):
    """
    Get the image names in a dir. Optional to return numpy array, paths.
    :param image_dir: Directory of the images
    :param pattern: type of image (e.g.: *.jpg, *.png etc)
    :param return_np: Boolean value denoting whether to return a numpy array
    :param return_path: Boolean value denoting whether to return image path or image names
    :return: image path or image names
    """
    image_paths = glob.glob (os.path.join (image_dir, pattern))
    image_names = [os.path.basename (path) for path in image_paths]
    return_path = image_paths if return_path else image_names
    if return_np:
        return_path = np.array (return_path)
    return return_path

def move_images(original_image_paths, new_image_dir, parse_original_image_name, new_image_name_tmplate, occluded_images = False):
    """
    Rename and move images to new directory.
    :param original_image_paths: Original image path
    :param new_image_dir: New directory of images
    :param parse_original_image_name: Function parse_original_image_name
    :param new_image_name_tmplate: Template of image name
    :param occluded_images: Boolean to identify if images are occluded or not
    :return: new_image_names: New image names
    """
    cnt = defaultdict(int)
    new_image_names = []

    # Get an occluded id (1 = occluded, 0 = non-occluded)
    if occluded_images:
        occluded_id = 1
    else:
        occluded_id = 0

    # Change name and save image
    for image_path in original_image_paths:
        image_name = os.path.basename (image_path)
        image_id = parse_original_image_name (image_name, 'id')
        cam_id = parse_original_image_name (image_name, 'cam')
        cnt[(image_id, cam_id)] += 1
        new_image_name = new_image_name_tmplate.format (image_id, cam_id, cnt[(image_id, cam_id)] - 1, occluded_id)
        # Check if new_image_dir exists else create
        if not os.path.exists(new_image_dir):
            os.mkdir(new_image_dir)

        shutil.copy (image_path, os.path.join (new_image_dir, new_image_name))
        new_image_names.append (new_image_name)
    return new_image_names


def transform(base_path, new_image_dir, occluded_images=False, train_test_split_file=None):
    """
    Takes a directory and transforms the market data to proper labels and saves the train/test/validate split
    :param base_path: Path where the images
    :param occluded: Boolean value if the directory has occluded images or non-occluded images
    :param new_image_dir: Path for new images
    :param train_test_split_file: File path for saving train/test split file
    :return: None
    """
    image_paths = []
    num_of_images = []

    # ----
    # Get all image paths
    # ----

    # Bounding box test
    _image_paths_ = get_image_names (os.path.join (base_path, 'bounding_box_test'), return_path=True, return_np=False)
    _image_paths_.sort ()
    image_paths += list (_image_paths_)
    num_of_images.append (len (_image_paths_))

    # Bounding box train
    _image_paths_ = get_image_names (os.path.join (base_path, 'bounding_box_train'), return_path=True, return_np=False)
    _image_paths_.sort ()
    image_paths += list (_image_paths_)
    num_of_images.append (len (_image_paths_))

    # Query
    _image_paths_ = get_image_names (os.path.join (base_path, 'query'), return_path=True, return_np=False)
    _image_paths_.sort ()
    image_paths += list (_image_paths_)
    num_of_images.append (len (_image_paths_))

    # Gather image_id and cam_id for images used in testing/querying
    query_image_id_cam = set ([(parse_original_image_name (os.path.basename (p), 'id'),
                                parse_original_image_name (os.path.basename (p), 'cam')) for p in _image_paths_])

    # gt_bbox
    _image_paths_ = get_image_names (os.path.join (base_path, 'gt_bbox'), return_path=True, return_np=False)
    _image_paths_.sort ()
    # Get only images from gt_bbox that are used for testing/querying
    _image_paths_ = [path for path in _image_paths_
                     if (parse_original_image_name (os.path.basename (path), 'id'),
                         parse_original_image_name (os.path.basename (path), 'cam'))
                     in query_image_id_cam]

    image_paths += list (_image_paths_)
    num_of_images.append (len (_image_paths_))

    # Save images with new name
    image_names = move_images (image_paths, new_image_dir, parse_original_image_name, new_image_name_template,
                               occluded_images)

    # Split images into train, test, validation sets
    split = dict ()
    keys = ['trainval_im_names', 'gallery_im_names', 'q_im_names', 'mq_im_names']
    inds = [0] + num_of_images
    inds = np.cumsum (np.array (inds))
    for i, k in enumerate (keys):
        split[k] = image_names[inds[i]:inds[i + 1]]

    with open (train_test_split_file, 'wb') as f:
        pickle.dump (split, f, protocol=2)

    print('Saving images done.')
    return split


if __name__ == "__main__":
    transform ('/Users/soutikchakraborty/Downloads/Market-1501',
               new_image_dir = '/Users/soutikchakraborty/Downloads/Market-1501/image',
               train_test_split_file = '/Users/soutikchakraborty/Downloads/Market-1501/train_test_split.pkl',
               occluded_images=True)
