from scipy.spatial.distance import cosine
from keras.preprocessing import image

def cosine_distance(i1, i2):
    """
    Function to calculate the cosine distance between images

    Input:
    i1, i2 -- images to get the distance between

    Output:
    distance -- distance between i1 & i2
    """
    # Convert to arrays
    i1 = image.img_to_array (i1)
    i2 = image.img_to_array (i2)

    # Get the shapes
    i1_shape = i1.shape
    i2_shape = i2.shape

    shape_1 = i1_shape[0] * i1_shape[1] * i1_shape[2]
    shape_2 = i2_shape[0] * i2_shape[1] * i2_shape[2]

    # Flatten the dist
    i1 = i1.reshape(shape_1)
    i2 = i2.reshape(shape_2)

    return cosine (i1, i2)
