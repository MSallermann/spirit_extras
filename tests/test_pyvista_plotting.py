import sys, os
SCRIPT_DIR = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(SCRIPT_DIR, "../src") )

from spirit_extras import import_spirit, plotting, data, pyvista_plotting

choose = lambda x : x.version_major >= 2 and x.cuda==False
spirit_libs_list = import_spirit.find_and_insert("~/Coding/spirit", choose=choose)

print(spirit_libs_list[0])

def test():

    INPUT_FILE = os.path.join(SCRIPT_DIR, "inputs/test_plotting.cfg")

    from spirit import state, configuration

    with state.State(INPUT_FILE) as p_state:
        configuration.plus_z(p_state)
        configuration.hopfion(p_state, 7)
        # configuration.skyrmion(p_state, 7)

        system = data.spin_system_from_p_state(p_state)

        plotter = pyvista_plotting.Spin_Plotter(system)
        plotter.camera_position = 'yz'
        plotter.camera_azimuth   = 45
        plotter.camera_elevation = 50

        # plotter.compute_delaunay()
        # plotter.save_delaunay("delaunay.vtk")
        plotter.load_delaunay("delaunay.vtk")

        # plotter.add_preimage([1,0,0], tol=0.02)
        plotter.isosurface(0, "spins_z")
        # plotter.show()
        plotter.render_to_png("test")

test()