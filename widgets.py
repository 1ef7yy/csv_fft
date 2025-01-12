from PyQt5.QtWidgets import QWidget
import pyqtgraph as pg
import numpy as np

class OriginalImageWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        plot_widget = pg.PlotWidget()
        self.widget = plot_widget
        


    def getWidget(self):
        return self.widget
    

    def updateImage(self, image, filetype=None):
        self.widget.clear()
        img = pg.ImageItem(image)

        # images rotating is an unknown behavior, needs fixing
        rotations = {
            "data": 0,
            "jpg": 270,
            "csv": 270,
            None: 0,
        }
        img.setRotation(rotations[filetype])
        self.widget.addItem(img)

        


class OriginalFFTWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        plot_widget = pg.PlotWidget()
        self.widget = plot_widget
        

    def getWidget(self):
        return self.widget
    

    def makeFFT(self, image):
        fft = np.fft.fft2(image)
        fft_shift = np.fft.fftshift(fft)
        magnitude_spectrum = 20 * np.log(np.abs(fft_shift))
        return magnitude_spectrum
    

    def updateImage(self, image):
        self.widget.clear()
        self.widget.addItem(pg.ImageItem(self.makeFFT(image)))