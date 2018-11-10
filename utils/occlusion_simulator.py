import cv2
import numpy as np
from keras.preprocessing import image
import matplotlib.pyplot as plt

def occlusion_simulator(image_path, height, width):
    """
    @param image_path: The path to the image to edit
    @param coords: A tuple of x/y coordinates (x1, y1, x2, y2)
    @param saved_location: Path to save the cropped image
    """
    img = image.load_img ('/Users/soutikchakraborty/Downloads/CS230Data/cuhk03/labeled/images/00000000_0000_00000001.jpg')
    img = image.img_to_array(img)
    org_shape= img.shape
    mask = np.ones(img.shape)

    start_height = np.random.randint(org_shape[0] - height)
    start_width = np.random.randint (org_shape[1] - width)
    mask[start_height: start_height + height, start_width: start_width + width] = np.zeros([height, width, 3])

    transformed_img = np.multiply(img.flatten(), mask.flatten())
    transformed_img = transformed_img.astype('int').reshape(org_shape)
    plt.imshow(transformed_img)
    plt.show()

