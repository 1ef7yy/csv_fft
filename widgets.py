from PyQt5.QtWidgets import (
    QWidget,
    QTabWidget,
    QSpinBox,
    QHBoxLayout,
    QVBoxLayout,
)
import pyqtgraph as pg
import numpy as np
from scipy import interpolate

currentMode = "radial"


def createInterpolator(self, x, y):
    return interpolate.Akima1DInterpolator(x, y)


class OriginalImageWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        plot_widget = pg.PlotWidget()
        self.widget = plot_widget
        self.widget.setTitle("Original image")

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
        self.widget.setTitle("Original FFT")

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


class EQWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.mainWidget = QWidget()

        self.layout = QVBoxLayout()

        self.mainWidget.setLayout(self.layout)

        self.widget = pg.PlotWidget()
        self.widget.setTitle("EQ")

        # default mode
        global currentMode
        currentMode = "radial"

        self.radEqualizer = QWidget()
        self.vertEqualizer = QWidget()
        self.horEqualizer = QWidget()

        self.tabs = QTabWidget(self.widget)

        self.tabs.addTab(self.radEqualizer, "Radial")
        self.tabs.addTab(self.vertEqualizer, "Vertical")
        self.tabs.addTab(self.horEqualizer, "Horizontal")
        self.tabs.currentChanged.connect(self.changeEQ)

        self.radialValues = [1.0] * 10
        self.verticalValues = [1.0] * 10
        self.horizontalValues = [1.0] * 10
        self.spinboxes = []
        self.spinboxLayout = QHBoxLayout()

        for i in range(10):
            spinbox = QSpinBox(parent=self.tabs)
            spinbox.setRange(0, 1000)
            spinbox.setSingleStep(1)
            spinbox.setValue(int(self.radialValues[i] * 100))
            spinbox.setStyleSheet("color: black; background-color: white")
            spinbox.valueChanged.connect(self.spinboxChanged)
            self.spinboxes.append(spinbox)
            self.spinboxLayout.addWidget(spinbox)

        self.tabs.show()

        self.pen = pg.mkPen(width=5)

        self.layout.addWidget(self.tabs)
        self.layout.addWidget(self.widget)
        self.layout.addLayout(self.spinboxLayout)

        self.updatePlot()

    def getWidget(self):
        return self.mainWidget

    def getValues(self) -> dict[str, list[int]]:
        return {
            "radial": self.radialValues,
            "horizontal": self.horizontalValues,
            "vertical": self.verticalValues,
        }

    def changeRadial(self, values: list[int]):
        self.radialValues = values

    def changeVertical(self, values: list[int]):
        self.verticalValues = values

    def changeHorizontal(self, values: list[int]):
        self.horizontalValues = values

    def updatePlot(self):
        self.widget.clear()
        x = [i for i in range(10)]
        global currentMode
        vals = [i * 100 for i in self.getValues()[currentMode]]
        self.widget.plot(x, vals, pen=self.pen, symbol="o", symbolSize=8)

    def changeEQ(self):
        global currentMode
        currentMode = ["radial", "vertical", "horizontal"][
            self.tabs.currentIndex()
        ]

        vals = self.getValues()[currentMode]

        for idx, spinbox in enumerate(self.spinboxes):
            spinbox.setValue(int(vals[idx] * 100))

        self.updatePlot()

    def spinboxChanged(self):
        values = []
        for spinbox in self.spinboxes:
            values.append(spinbox.value() / 100)

        match currentMode:
            case "radial":
                self.radialValues = values
            case "vertical":
                self.verticalValues = values
            case "horizontal":
                self.horizontalValues = values
        self.updatePlot()


class ProcessingImage(QWidget):
    def __init__(self, parent=None):
        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setTitle("Processing Image")

    def getWidget(self):
        return self.plot_widget


class ProcessingFFT(QWidget):
    def __init__(self, parent=None):
        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setTitle("Processing FFT")

    def getWidget(self):
        return self.plot_widget


class FFTMask(QWidget):
    def __init__(self, parent=None):
        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setTitle("FFT Mask")
        self.masks = {
             "radial": self.generateMask(0, "radial"),
             "vertical": self.generateMask(0, "vertical"),
             "horizontal": self.generateMask(0, "horizontal"),
         }

    def getWidget(self):
        return self.plot_widget

    def updatePlot(self):
        self.plot_widget.clear()

        mask = self.generateMask()

        self.plot_widget.addItem(mask)

    def generateMask(self, f, mode):
        if mode == "radial":
            self.Z = f((self.X**2 + self.Y**2) ** (1 / 2))
        elif mode == "vertical":
            self.Z = f(self.X)
        elif mode == "horizontal":
            self.Z = f(self.Y)

        self.Z = np.array(
            [
                [
                    0 if self.Z[i][j] < 0 else self.Z[i][j]
                    for j in range(self.Z.shape[1])
                ]
                for i in range(self.Z.shape[0])
            ]
        )
        Z1 = self.Z[:, ::-1]
        mask_half = np.concatenate((Z1, self.Z), 1)
        mask = np.concatenate((mask_half[::-1, :], mask_half), 0)
        return mask
