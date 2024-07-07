from PIL import Image, ImageTk


class ImageHandler:
    def __init__(self, img: ImageTk.PhotoImage):
        self.img = img

    def resize_img(self, width: int, height: int) -> None:
        # conversion to Image
        resized_img = ImageTk.getimage(self.img).resize((width, height), Image.AFFINE)
        self.img = ImageTk.PhotoImage(resized_img)

    def get_img(self) -> ImageTk.PhotoImage:
        return self.img
