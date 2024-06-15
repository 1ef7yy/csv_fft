import dataReader as dr
from tkinter import filedialog

from imageHandler import *


def choose_file(elem):
    filename = filedialog.askopenfilename()

    if filename:
        reader = dr.DataReader(filename)
        img = reader.readFile()

        imagehandler = ImageHandler(img)

        imagehandler.resize_img(320, 256)

        img = imagehandler.get_img()

        elem.configure(image=img)
        elem.image = img
