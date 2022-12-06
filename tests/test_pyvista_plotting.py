import unittest
import numpy as np
from spirit_extras.pyvista_plotting import Spin_Plotter
from spirit_extras import data
from spirit import state, configuration
import os


SCRIPT_DIR = os.path.dirname(__file__)


class Pyvista_Plotting_Test(unittest.TestCase):
    INPUT_FILE = os.path.join(SCRIPT_DIR, "inputs/test_pyvista_plot.cfg")
    PLOT_OUTPUTS = os.path.join(SCRIPT_DIR, "pyvista_test")

    def setUp(self) -> None:
        if not os.path.exists(self.PLOT_OUTPUTS):
            os.makedirs(self.PLOT_OUTPUTS)

    def tearDown(self) -> None:
        pass

    def test(self):
        with state.State(self.INPUT_FILE, quiet=True) as p_state:
            configuration.plus_z(p_state)
            configuration.hopfion(p_state, radius=5)
            spin_system = data.spin_system_from_p_state(p_state).deepcopy()

        plotter = Spin_Plotter(spin_system)
        plotter.resolution = (420, 420)  # low res so test is faster

        plotter.arrows()

        plotter.render_to_png(
            os.path.join(
                os.path.join(self.PLOT_OUTPUTS, "cones_with_white_background.png")
            )
        )

        plotter.background_color = "transparent"
        plotter.render_to_png(
            os.path.join(
                os.path.join(self.PLOT_OUTPUTS, "cones_with_transparent_background.png")
            )
        )

        plotter.background_color = "black"
        plotter.render_to_png(
            os.path.join(
                os.path.join(self.PLOT_OUTPUTS, "cones_with_black_background.png")
            )
        )

        plotter.axes = True
        plotter.camera_focal_point = spin_system.center()
        plotter.camera_focus(distance=120, direction="XYZ")
        plotter.render_to_png(
            os.path.join(os.path.join(self.PLOT_OUTPUTS, "cones_with_camera_focus.png"))
        )

        plotter.colormap("rgb")
        plotter.clear_meshes()
        plotter.arrows()
        plotter.render_to_png(
            os.path.join(os.path.join(self.PLOT_OUTPUTS, "cones_colormap_rgb.png"))
        )
