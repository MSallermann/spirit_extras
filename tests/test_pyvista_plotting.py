import unittest
import numpy as np
from spirit_extras.pyvista_plotting import Spin_Plotter
from spirit_extras import data
from spirit import state, configuration, geometry
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

    def test_random(self):
        with state.State(self.INPUT_FILE, quiet=True) as p_state:
            configuration.plus_z(p_state)
            configuration.hopfion(p_state, radius=5)
            spin_system = data.spin_system_from_p_state(p_state).deepcopy()

        plotter = Spin_Plotter(spin_system)
        plotter.axes = True
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

        plotter.set_camera_focus(distance=200, direction="XYZ")
        plotter.render_to_png(
            os.path.join(os.path.join(self.PLOT_OUTPUTS, "cones_with_camera_focus.png"))
        )

        plotter.colormap("rgb")
        plotter.clear_meshes()
        plotter.arrows()
        plotter.render_to_png(
            os.path.join(os.path.join(self.PLOT_OUTPUTS, "cones_colormap_rgb.png"))
        )

        plotter.colormap("rb", cardinal=np.array([1, 0, 0]))
        plotter.clear_meshes()
        plotter.arrows()
        plotter.render_to_png(
            os.path.join(os.path.join(self.PLOT_OUTPUTS, "cones_colormap_rb_x.png"))
        )

    def test_some_skyrmion_plot(self):
        import pyvista as pv

        with state.State(self.INPUT_FILE, quiet=True) as p_state:
            geometry.set_n_cells(p_state, [64, 64, 1])
            configuration.plus_z(p_state)
            configuration.skyrmion(p_state, radius=10)
            spin_system = data.spin_system_from_p_state(p_state).deepcopy()

        plotter = Spin_Plotter(spin_system)
        plotter.axes = True
        plotter.resolution = (1600, 1600)
        plotter.background_color = "lightgrey"
        plotter.colormap("hsv")
        plotter.arrows()
        plotter.set_camera_focus(50, "XZ")

        plotter.render_to_png(
            os.path.join(os.path.join(self.PLOT_OUTPUTS, "skyrmion.png"))
        )

        with state.State(self.INPUT_FILE, quiet=True) as p_state:
            geometry.set_n_cells(p_state, [64, 64, 1])
            configuration.plus_z(p_state)
            configuration.skyrmion(p_state, radius=5)
            spin_system2 = data.spin_system_from_p_state(p_state).deepcopy()

        plotter.update_spins(spin_system2)
        plotter.clear_meshes()
        plotter.arrows()
        plotter.render_to_png(
            os.path.join(os.path.join(self.PLOT_OUTPUTS, "skyrmion2.png"))
        )
