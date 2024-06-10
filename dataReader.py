import numpy as np

from PIL import Image, ImageTk


import os, struct

class DataReader:
    def __init__(self, filename: str):
        self.filename = filename

    
    def get_data(self):
        return np.fromfile(self.filename, dtype=np.uint16)
    

    def reshape(self, data, height, width):
        return data.reshape(data.shape[0] // height // width, height, width).astype(np.float32)


    def convert_bits(self, data):
        return np.divide(data, 256)
    
    def convert_to_pixels(self, data):
        return data[0]


    def img_to_pil(self, img_arr):
        return Image.fromarray(img_arr)
        

