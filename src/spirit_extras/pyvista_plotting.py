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

    positions_interpolated = np.empty( shape = (nos_after, 3) )

    for a in range(factor):
        for b in range(factor):
            for c in range(factor):
                offset_vector = a * offset_a + b * offset_b + c * offset_c

                idx_1 = (a + factor * (b + factor * c)) * nos
                positions_interpolated[idx_1 : idx_1 + nos] = spin_system.positions / factor + offset_vector

    return pv.PolyData( positions_interpolated ).interpolate(point_cloud)

def delaunay(point_cloud):
    return point_cloud.delaunay_3d(progress_bar=True)

def isosurface_from_delaunay(delaunay, isovalue=0, scalars_key="spins_z"):
    import pyvista as pv
    # Create the contour
    isosurface = delaunay.contour([isovalue], scalars = scalars_key, progress_bar=True)
    if isosurface.n_faces < 1:
        return None
    isosurface = isosurface.smooth()
    return isosurface

def arrows_from_point_cloud(point_cloud, factor=1, scale=False, orient="spins", geom=None):
    import pyvista as pv
    if geom is None:
        geom   = pv.Cone(radius=0.25, resolution=18)
    arrows = point_cloud.glyph(orient=orient, scale=scale, factor=factor, geom=geom)
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
    import vtk

    sphere = pv.Sphere(radius=1.0, start_theta=180, end_theta = 90, phi_resolution=60, theta_resolution=60)

    sphere["spins_rgba"] = spin_to_rgba_func(sphere.points)

    pv.start_xvfb()
    plotter = pv.Plotter( off_screen=True, shape=(1,1), lighting=None)
    plotter.window_size = 724*4, 724*4

    pv.global_theme.font.family = 'times'
    pv.global_theme.font.size   = 48

    plotter.add_mesh(sphere, scalars="spins_rgba", pbr=True, roughness=0.3, specular=0.2, ambient=0.1, specular_power=2, rgb=True, smooth_shading=True)

    directions = [ [1,0,0], [-1,0,0], [0,1,0], [0,-1,0], [0,0,1], [0,0,-1]]
    for d in directions:
        arrow = pv.Arrow(direction = d)
        plotter.add_mesh(arrow, color=spin_to_rgba_func([d])[0], specular=0.0, ambient=0.1, specular_power=2, rgb=True, smooth_shading=True)

    normals = [ [1,0,0], [0,1,0], [0,0,1]]
    for n in normals:
        disc = pv.Disc(inner=2, outer = 2.5, normal=n, c_res=48)
        disc["spins_rgba"] = spin_to_rgba_func( [p/np.linalg.norm(p) for p in disc.points] )
        plotter.add_mesh(disc, scalars="spins_rgba", show_edges=True)
    
    labels = ["x", "y", "z"]
    points = [[1.0,0,0], [0,1,0], [0,0,1]]

    for l,p in zip(labels, points):
        txt = pv.Text3D(l, depth=0.005 ).rotate_x(90).rotate_z(-45-180).scale(0.53)
        txt = txt.translate( 1.22 * np.array(p) - txt.center )

        txt2 = pv.Text3D(l, depth=0.01 ).rotate_x(90).rotate_z(-45-180).scale(0.5)
        txt2 = txt2.translate( 1.22 * np.array(p) - txt2.center)

        plotter.add_mesh(txt, color="black")
        plotter.add_mesh(txt2, color="white")

    plotter.set_background("white")
    # plotter.add_axes(color="black", line_width=6)
    plotter.camera.zoom(1.7)

    plotter.screenshot(image_file_name + ".png", transparent_background=True)

import pyvista as pv
class Spin_Plotter:

    def __init__(self, system):
        self.spin_system  = system

        self._point_cloud = create_point_cloud(system)

        self._delaunay    = None

        self.background_color = "white"
        self.axes             = False

        self.meshlist   = []
        self.resolution = (1980, 1080)

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

    def rotate_camera(self, axis, angle):
        from scipy.spatial.transform import Rotation
        R = Rotation.from_rotvec( angle * np.array(axis )).as_matrix()
        relative_pos         = self.camera_position    - self.camera_focal_point
        self.camera_position = np.dot(R, relative_pos) + self.camera_focal_point
        self.camera_up       = np.dot(R, self.camera_up)

    def align_camera_position(self, axis, align_direction, distance=None):
        if distance is None:
            distance = np.linalg.norm(self.camera_focal_point - self.camera_position)

        # Remove component along axis from align direction and from current relative position
        align_direction = np.array(align_direction)
        align_direction = align_direction - np.dot(align_direction,axis) * axis
        align_direction /= np.linalg.norm(align_direction)

        current_relative_pos = self.camera_position - self.camera_focal_point
        current_relative_pos /= np.linalg.norm(current_relative_pos)

        # Comp along axis
        axis_component = np.dot(current_relative_pos, axis) * axis

        relative_pos = align_direction + axis_component
        relative_pos /= np.linalg.norm(relative_pos)
        self.camera_position = self.camera_focal_point + relative_pos * distance

    def align_camera_up(self, axis):
        from scipy.spatial.transform import Rotation
        relative_pos    = self.camera_position  - self.camera_focal_point
        self.camera_up  = axis - relative_pos.dot(axis) / np.linalg.norm(relative_pos)**2 * relative_pos
        self.camera_up /= np.linalg.norm(self.camera_up)

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
        iso = isosurface_from_delaunay(self._delaunay, isovalue, scalars_key)

        if iso is None:
            return None, {}
        else:
            return self.add_mesh(iso, render_args)

    def arrows(self, geom=None, render_args=None):

        if render_args is None:
            render_args = self.default_render_args.copy()

        return self.add_mesh(arrows_from_point_cloud(self._point_cloud, factor=1, scale=False, orient="spins", geom=geom), render_args)

    def add_preimage(self, spin_dir, tol=0.05, n_neighbours=10, interpolation_factor=1, render_args={"color" : "black"}):
        if not self._delaunay:
            raise Exception("No delaunay")

        if render_args is None:
            render_args = self.default_render_args.copy()

        return self.add_mesh( create_pre_image(spin_dir, self._point_cloud, angle_tolerance=tol, n_neighbours=n_neighbours), render_args )

    def _axes(self, plotter):
        colors = get_rgba_colors([[1,0,0],[0,1,0],[0,0,1]])
        plotter.add_axes(color="black", line_width=6, x_color=colors[0], y_color=colors[1], z_color=colors[2])

    def show(self, save_camera_file=None):
        plotter = pv.Plotter(shape=(1,1))
        plotter.window_size = self.resolution

        self.set_camera(plotter)

        for m, args in self.meshlist:
            try:
                plotter.add_mesh(m, **args)
            except Exception as e:
                print(f"Could not add_mesh {m}")
                print(e)

        if self.axes:
            self._axes(plotter)

        transparent_background = False
        if not self.background_color is None:
            if self.background_color.lower() == "transparent":
                transparent_background = True
            else:
                plotter.set_background(self.background_color)

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
        plotter.close()

    def render_to_png(self, png_path ):
        pv.start_xvfb()
        plotter = pv.Plotter(off_screen=True, shape=(1,1), multi_samples=8)
        plotter.window_size = self.resolution
        self.set_camera(plotter)

        for m, args in self.meshlist:
            try:
                plotter.add_mesh(m, **args)
            except Exception as e:
                print(f"Could not add_mesh {m}")
                print(e)

        if self.axes:
            self._axes(plotter)

        transparent_background = False
        if not self.background_color is None:
            if self.background_color.lower() == "transparent":
                transparent_background = True
            else:
                plotter.set_background(self.background_color)

        plotter.screenshot(png_path + ".png", transparent_background=transparent_background)
        plotter.close()