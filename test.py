from tkinter import *
from tkinter import ttk
from tkinter import filedialog

from PIL import Image, ImageTk


import dataReader as dr


reader = dr.DataReader("static/3_off_med_dde_claagc.data")


data = reader.get_data()

reshaped = reader.reshape(data, 512, 640)

converted = reader.convert_bits(reshaped)


pixels = reader.convert_to_pixels(converted)

img = reader.img_to_pil(pixels)


root = Tk()


frm = ttk.Frame(root, padding=10)
frm.grid()

image = ImageTk.PhotoImage(img)
img_original = ttk.Label(frm, image=image)

img_original.grid(column=0, row=0)


root.mainloop()
