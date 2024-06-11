import dataReader as dr
from PIL import Image, ImageTk
from tkinter import filedialog


def choose_file(elem):
    filename = filedialog.askopenfilename()

    if filename:
        reader = dr.DataReader(filename)
        img = reader.readFile()

        
        elem.configure(image=img)
        elem.image = img
