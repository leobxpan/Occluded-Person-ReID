from utils.cosine_distance import cosine_distance
from utils.occlusion_simulator import occlusion_simulator
import pandas as pd
import numpy as np
import keras
from keras.preprocessing import image
import matplotlib.pyplot as plt

i1 = image.load_img('/Users/soutikchakraborty/Downloads/CS230Data/cuhk03/labeled/images/00000000_0000_00000000.jpg',
                        target_size=(100, 100, 3))
i2 = image.load_img('/Users/soutikchakraborty/Downloads/CS230Data/cuhk03/labeled/images/00000000_0000_00000001.jpg',
                    target_size=(100, 100, 3))


path = '/Users/soutikchakraborty/Downloads/CS230Data/cuhk03/labeled/images/00000000_0000_00000001.jpg'
occlusion_simulator(path,  50, 50)
print cosine_distance(i1, i2)
