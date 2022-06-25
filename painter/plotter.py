import cv2
import numpy as np


class Plotter:
    def __init__(self, plot_width, plot_height):
        self.width = plot_width
        self.height = plot_height
        self.color = (255, 0, 0)
        self.color_list = [(255, 0, 0), (0, 250, 0), (0, 0, 250),
                           (0, 255, 250), (250, 0, 250), (250, 250, 0),
                           (200, 100, 200), (100, 200, 200), (200, 200, 100)]
        self.val = []
        self.plot_canvas = np.ones((self.height, self.width, 3)) * 255

    def plot(self, val, label="plot"):
        self.val.append(int(val))
        while len(self.val) > self.width:
            self.val.pop(0)
        self.show_plot(label)

    def multiplot(self, val, label="plot"):
        self.val.append(val)
        while len(self.val) > self.width:
            self.val.pop(0)
        self.show_plot(label)

    def show_plot(self, label):
        self.plot_canvas = np.ones((self.height, self.width, 3)) * 255
        cv2.line(self.plot, (0, int(self.height / 2)), (self.width, int(self.height / 2)), (0, 255, 0), 1)
        for i in range(len(self.val) - 1):
            for j in range(len(self.val[0])):
                cv2.line(self.plot, (i, int(self.height / 2) - self.val[i][j]),
                         (i + 1, int(self.height / 2) - self.val[i + 1][j]), self.color[j], 1)

        cv2.imshow(label, self.plot)
        cv2.waitKey(10)
