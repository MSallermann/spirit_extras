from .plotting import get_rgba_colors
import numpy as np

def create_point_cloud(spin_system):
    import pyvista as pv
    point_cloud = pv.PolyData(spin_system.positions)
    point_cloud["spins_x"]    = spin_system.spins[:,0]
    point_cloud["spins_y"]    = spin_system.spins[:,1]
    point_cloud["spins_z"]    = spin_system.spins[:,2]
    point_cloud["spins"]      = spin_system.spins
    point_cloud["spins_rgba"] = get_rgba_colors( spin_system.spins, opacity=1.0 )
    return point_cloud

def interpolate_point_cloud(spin_system, point_cloud=None, factor=1):
    if point_cloud is None:
        point_cloud = create_point_cloud(spin_system)

    nos       = spin_system.nos()
    nos_after = spin_system.nos() * factor**3
    n_cells   = spin_system.n_cells

    spin_system.shape()
    offset_a = spin_system.positions[0, n_cells[0]-1, 0, 0] / factor
    offset_b = spin_system.positions[0, 0, n_cells[1]-1, 0] / factor
    offset_c = spin_system.positions[0, 0, 0, n_cells[2]-1] / factor
    spin_system.flatten()

    print(offset_a)
    print(offset_b)
    print(offset_c)

    positions_interpolated = np.empty( shape = (nos_after, 3) )

    for a in range(factor):
        for b in range(factor):
            for c in range(factor):
                offset_vector = a * offset_a + b * offset_b + c * offset_c

                idx_1 = (a + factor * (b + factor * c)) * nos
                positions_interpolated[idx_1 : idx_1 + nos] = spin_system.positions / factor + offset_vector

    print(spin_system.positions)
    print(positions_interpolated)

    return pv.PolyData( positions_interpolated ).interpolate(point_cloud)

def delaunay(point_cloud):
    return point_cloud.delaunay_3d(progress_bar=True)

def isosurface_from_delaunay(delaunay, isovalue=0, scalars_key="spins_z"):
    import pyvista as pv
    # Create the contour
    isosurface = delaunay.contour([isovalue], scalars = scalars_key, progress_bar=True)
    isosurface = isosurface.smooth()
    return isosurface

def arrows_from_point_cloud(point_cloud, factor=1):
    import pyvista as pv
    geom   = pv.Cone(radius=0.25, resolution=18)
    arrows = point_cloud.glyph(orient="spins", scale=False, factor=factor, geom=geom)
    return arrows

def create_pre_image(pre_image_spin, point_cloud, angle_tolerance=0.05, n_neighbours=10):
    from .post_processing import compute_pre_image
    import pyvista as pv
    positions = compute_pre_image( point_cloud.points, point_cloud["spins"], pre_image_spin, angle_tolerance=angle_tolerance, n_neighbours=n_neighbours )[1]
    return pv.Spline(positions).tube(radius=0.3)

def save_to_png(image_file_name, mesh_list):
    import pyvista as pv

    pv.start_xvfb()
    plotter = pv.Plotter(off_screen=True, shape=(1,1))

    for m in mesh_list:
        plotter.add_mesh(m, scalars = "spins_rgba", rgb = True, specular=0.7, ambient=0.4, specular_power=5, smooth_shading=True, show_scalar_bar=False, show_edges=False, metallic=True )

    plotter.set_background("white")
    plotter.add_axes(color="black", line_width=6)

    plotter.show(screenshot=image_file_name + ".png")

def plot_color_sphere(image_file_name, spin_to_rgba_func):
    import pyvista as pv

    sphere = pv.Sphere(radius=1.0, start_theta=180, end_theta = 90, phi_resolution=60, theta_resolution=60)

    sphere["spins_rgba"] = spin_to_rgba_func(sphere.points)

    pv.start_xvfb()
    plotter = pv.Plotter( off_screen=True, shape=(1,1), lighting=None)
    light = pv.Light(light_type='headlight')
    # these don't do anything for a headlight:
    light.position = (1, 2, 3)
    light.focal_point = (4, 5, 6)
    plotter.add_light(light)
    plotter.add_mesh(sphere, scalars="spins_rgba", specular=0.7, ambient=0.4, specular_power=5, rgb=True, smooth_shading=True)
    plotter.set_background("white")
    plotter.show(screenshot=image_file_name + ".png")

import pyvista as pv
class Spin_Plotter:

    def __init__(self, system):
        self.spin_system  = system

        self._point_cloud = create_point_cloud(system)

        self._delaunay    = None

        self.background_color = "white"

        self.meshlist = []

        self.camera_position    = None
        self.camera_up          = None
        self.camera_focal_point = None
        self.camera_azimuth     = None
        self.camera_elevation   = None
        self.camera_distance    = None
        self.camera_view_angle  = None

        self._preimages       = []

        self.default_render_args = dict(scalars = "spins_rgba", rgb = True, specular=0.0, ambient=0.7, specular_power=5, smooth_shading=True, show_scalar_bar=False, show_edges=False)

    def camera_from_json(self, save_camera_file):
        with open(save_camera_file, "r") as f:
            data = json.load(f)
        self.camera_position = data["position"]
        self.camera_up = data["up"]
        self.camera_focal_point = data["focal_point"]
        self.camera_distance = data["distance"]
        self.camera_view_angle = data["view_angle"]

    def set_camera(self, plotter):
        if not self.camera_position is None:
            plotter.camera.position = self.camera_position
        if not self.camera_azimuth is None:
            plotter.camera.azimuth = self.camera_azimuth
        if not self.camera_elevation is None:
            plotter.camera.elevation = self.camera_elevation
        if not self.camera_focal_point is None:
            plotter.camera.focal_point = self.camera_focal_point
        if not self.camera_up is None:
            plotter.camera.up = self.camera_up
        if not self.camera_distance is None:
            plotter.camera.distance = self.camera_distance
        if not self.camera_view_angle is None:
            plotter.camera.view_angle = self.camera_view_angle

    def compute_delaunay(self):
        self._delaunay = delaunay(self._point_cloud)

    def save_delaunay(self, path):
        self._delaunay.save(path)

    def load_delaunay(self, path):
        self._delaunay = pv.read(path)
        self._delaunay.copy_attributes( self._point_cloud )

    def update_spins(self, spin_system):
        self._point_cloud["spins_x"]    = spin_system.spins[:,0]
        self._point_cloud["spins_y"]    = spin_system.spins[:,1]
        self._point_cloud["spins_z"]    = spin_system.spins[:,2]
        self._point_cloud["spins"]      = spin_system.spins
        self._point_cloud["spins_rgba"] = get_rgba_colors( spin_system.spins, opacity=1.0 )

        if self._delaunay:
            self._delaunay.copy_attributes( self._point_cloud )

    def add_mesh(self, mesh, render_args):
        self.meshlist.append( [mesh, render_args] )
        return self.meshlist[-1]

    def isosurface(self, isovalue, scalars_key, render_args=None):
        if not self._delaunay:
            raise Exception("No delaunay")

        if render_args is None:
            render_args = self.default_render_args.copy()

        return self.add_mesh(isosurface_from_delaunay(self._delaunay, isovalue, scalars_key), render_args)

    def arrows(self, render_args=None):

        if render_args is None:
            render_args = self.default_render_args.copy()

        return self.add_mesh(arrows_from_point_cloud(self._point_cloud), render_args)

    def add_preimage(self, spin_dir, tol=0.05, n_neighbours=10, interpolation_factor=1, render_args={"color" : "black"}):
        if not self._delaunay:
            raise Exception("No delaunay")

        if render_args is None:
            render_args = self.default_render_args.copy()

        return self.add_mesh( create_pre_image(spin_dir, self._point_cloud, angle_tolerance=tol, n_neighbours=n_neighbours), render_args )

    def show(self, save_camera_file=None):
        plotter = pv.Plotter(shape=(1,1))
        self.set_camera(plotter)

        for m, args in self.meshlist:
            try:
                plotter.add_mesh(m, **args)
            except Exception as e:
                print(f"Could not add_mesh {m}")
                print(e)

        if not self.background_color is None:
            plotter.set_background(self.background_color)

        plotter.add_axes(color="black", line_width=6)
        plotter.show()

        if save_camera_file is not None:
            camera_dict = dict(
                position = plotter.camera.position,
                up = plotter.camera.up,
                focal_point = plotter.camera.focal_point,
                distance = plotter.camera.distance,
                view_angle = plotter.camera.view_angle
            )
            with open(save_camera_file, "w") as f:
                f.write(json.dumps(camera_dict, indent=4))

    def render_to_png(self, png_path ):
        pv.start_xvfb()
        plotter = pv.Plotter(off_screen=True, shape=(1,1))
        self.set_camera(plotter)

        for m, args in self.meshlist:
            try:
                plotter.add_mesh(m, **args)
            except Exception as e:
                print(f"Could not add_mesh {m}")
                print(e)

        if not self.background_color is None:
            plotter.set_background(self.background_color)

        plotter.add_axes(color="black", line_width=6)
        plotter.show(screenshot= png_path + ".png")