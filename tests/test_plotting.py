import unittest
import os, shutil
import numpy as np
from spirit_extras.plotting import Paper_Plot
import matplotlib.pyplot as plt

class Calculation_Folder_Test(unittest.TestCase):
    IMAGE_FOLDER = os.path.join( os.path.dirname(__file__), "images")

    def test(self):
        ### There is not much to test except that no exceptions get thrown, I guess

        pplot = Paper_Plot(15 * Paper_Plot.cm)

        pplot.ncols = 5
        pplot.hspace = 0.2
        pplot.wspace = 0.2
        pplot.nrows = 3
        pplot.width_ratios = np.ones(pplot.ncols)
        pplot.width_ratios[0] = 2

        pplot.height_from_aspect_ratio(1)

        fig = pplot.fig()
        gs  = pplot.gs()

        x = np.linspace(0,2*np.pi)

        # row
        for a in pplot.row(-1, slice(1,None,2)):
            a.plot(x, np.sin(x))

        # col
        for a in pplot.col(0, slice(0,1)):
            a.plot(x, np.cos(x))

        # annotate
        ax = fig.add_subplot(gs[0,1])
        ax.plot(x, x**2)

        for i in range(6):
            xy = (i, i**2)
            xy_text = pplot.xy_text_auto(ax, xy, deriv = 2*i)
            pplot.annotate_graph( ax, xy, xy_text)

        # spine axes
        pplot.spine_axis(gs[0,2], color="red", which=["left", "right"])

        # picture stuff
        image = plt.imread(os.path.join(self.IMAGE_FOLDER, "spirit.png"))
        image_crop = pplot.crop_to_content(image, replace_background_color=[0.0,0.0,0.0,0])
        plt.imsave("cropped.png", image_crop)
        print(image.shape)

        ax_img = fig.add_subplot(gs[-1,0])
        pplot.image_to_ax(ax_img, image_crop)

        fig.savefig("test_fig.png", dpi=300)

if __name__ == "__main__":
    test = Calculation_Folder_Test()
    test.test()