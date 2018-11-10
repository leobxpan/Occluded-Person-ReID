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
    image = cv2.imread(image_path)
    org_shape= image.shape
    mask = np.ones(image.shape)

    start_height = np.random.randint(org_shape[0] - height)
    start_width = np.random.randint (org_shape[1] - width)
    mask[start_height: start_height + height, start_width: start_width + width] = np.zeros([height, width, 3])

    transformed_img = np.multiply(image.flatten(), mask.flatten())
    transformed_img = transformed_img.astype('int').reshape(org_shape)
    plt.imshow(transformed_img)
    plt.show()

