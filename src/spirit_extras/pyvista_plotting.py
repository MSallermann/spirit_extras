from .plotting import (
    get_rgba_colors,
    get_rgba_colors_red_blue,
    get_rgba_colors_red_green_blue,
)
import numpy as np
import os
import json
import pyvista as pv
from .data import Spin_System


def create_point_cloud(spin_system: Spin_System) -> pv.PolyData:
    point_cloud = pv.PolyData(spin_system.positions)
    point_cloud["spins_x"] = spin_system.spins[:, 0]
    point_cloud["spins_y"] = spin_system.spins[:, 1]
    point_cloud["spins_z"] = spin_system.spins[:, 2]
    point_cloud["spins"] = spin_system.spins
    point_cloud["spins_rgba"] = get_rgba_colors(spin_system.spins, opacity=1.0)
    return point_cloud


def interpolate_point_cloud(
    spin_system: Spin_System, point_cloud: pv.PolyData = None, factor: int = 1
) -> pv.PolyData:
    if point_cloud is None:
        point_cloud = create_point_cloud(spin_system)

    nos = spin_system.nos()
    nos_after = spin_system.nos() * factor**3
    n_cells = spin_system.n_cells

    spin_system.shape()
    offset_a = spin_system.positions[0, n_cells[0] - 1, 0, 0] / factor
    offset_b = spin_system.positions[0, 0, n_cells[1] - 1, 0] / factor
    offset_c = spin_system.positions[0, 0, 0, n_cells[2] - 1] / factor
    spin_system.flatten()

    positions_interpolated = np.empty(shape=(nos_after, 3))

    for a in range(factor):
        for b in range(factor):
            for c in range(factor):
                offset_vector = a * offset_a + b * offset_b + c * offset_c

                idx_1 = (a + factor * (b + factor * c)) * nos
                positions_interpolated[idx_1 : idx_1 + nos] = (
                    spin_system.positions / factor + offset_vector
                )

    return pv.PolyData(positions_interpolated).interpolate(point_cloud)


def delaunay(point_cloud: pv.PolyData):
    return point_cloud.delaunay_3d(progress_bar=True)


def isosurface_from_delaunay(
    delaunay, isovalue: float = 0, scalars_key: str = "spins_z"
):
    # Create the contour
    isosurface = delaunay.contour([isovalue], scalars=scalars_key, progress_bar=True)

    if isosurface.n_faces < 1:
        return None

    isosurface = isosurface.smooth()
    return isosurface


def create_pre_image(
    pre_image_spin, point_cloud, angle_tolerance=0.05, n_neighbours=10
):
    from .post_processing import compute_pre_image
    import pyvista as pv

    positions = compute_pre_image(
        point_cloud.points,
        point_cloud["spins"],
        pre_image_spin,
        angle_tolerance=angle_tolerance,
        n_neighbours=n_neighbours,
    )[1]
    return pv.Spline(positions).tube(radius=0.3)


def plot_color_sphere(image_file_name, spin_to_rgba_func):
    import pyvista as pv

    sphere = pv.Sphere(
        radius=1.0,
        start_theta=180,
        end_theta=90,
        phi_resolution=60,
        theta_resolution=60,
    )

    sphere["spins_rgba"] = spin_to_rgba_func(sphere.points)

    pv.start_xvfb()
    plotter = pv.Plotter(off_screen=True, shape=(1, 1), lighting=None)
    plotter.window_size = 724 * 4, 724 * 4

    pv.global_theme.font.family = "times"
    pv.global_theme.font.size = 48

    plotter.add_mesh(
        sphere,
        scalars="spins_rgba",
        pbr=True,
        roughness=0.3,
        specular=0.2,
        ambient=0.1,
        specular_power=2,
        rgb=True,
        smooth_shading=True,
    )

    directions = [[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]]
    for d in directions:
        arrow = pv.Arrow(direction=d)
        plotter.add_mesh(
            arrow,
            color=spin_to_rgba_func([d])[0],
            specular=0.0,
            ambient=0.1,
            specular_power=2,
            rgb=True,
            smooth_shading=True,
        )

    normals = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    for n in normals:
        disc = pv.Disc(inner=2, outer=2.5, normal=n, c_res=48)
        disc["spins_rgba"] = spin_to_rgba_func(
            [p / np.linalg.norm(p) for p in disc.points]
        )
        plotter.add_mesh(disc, scalars="spins_rgba", show_edges=True)

    labels = ["x", "y", "z"]
    points = [[1.0, 0, 0], [0, 1, 0], [0, 0, 1]]

    for l, p in zip(labels, points):
        txt = pv.Text3D(l, depth=0.005).rotate_x(90).rotate_z(-45 - 180).scale(0.53)
        txt = txt.translate(1.22 * np.array(p) - txt.center)

        txt2 = pv.Text3D(l, depth=0.01).rotate_x(90).rotate_z(-45 - 180).scale(0.5)
        txt2 = txt2.translate(1.22 * np.array(p) - txt2.center)

        plotter.add_mesh(txt, color="black")
        plotter.add_mesh(txt2, color="white")

    plotter.set_background("white")
    # plotter.add_axes(color="black", line_width=6)
    plotter.camera.zoom(1.7)

    plotter.screenshot(image_file_name + ".png", transparent_background=True)


import pyvista as pv


class Spin_Plotter:
    """A class to create pyvista plots of spin systems."""

    def __init__(self, system: Spin_System):
        # The spin system
        self.spin_system: Spin_System = system

        # Point cloud representing the spin system
        self._point_cloud = create_point_cloud(system)

        self._delaunay = None
        self._xvfb_wait = 0.5

        self.background_color = "white"
        self.axes = False

        self.meshlist = []
        self.resolution = (1980, 1080)

        self.camera_dict = dict(
            position=None,
            up=None,
            focal_point=None,
            azimuth=None,
            elevation=None,
            distance=None,
            view_angle=None,
        )

        # self._render_to_png = True
        # self.shape = (1, 1)

        self._preimages = []

        self._plotter: pv.Plotter = None

        self.default_render_args = dict(
            scalars="spins_rgba",
            rgb=True,
            specular=0.0,
            ambient=0.7,
            specular_power=5,
            smooth_shading=True,
            show_scalar_bar=False,
            show_edges=False,
        )

    def camera_from_json(self, save_camera_file):
        with open(save_camera_file, "r") as f:
            data = json.load(f)
        self.camera_dict = data.copy()

    def _string_to_direction(self, input):
        """Converts strings like "+X-Z" to normalized directions, here 1/sqrt(2)[1,0,-1]
        Args:
            input (str): the string to be parsed
        """
        input = input.lower()

        res = np.zeros(3)

        for name, direction in [["x", [1, 0, 0]], ["y", [0, 1, 0]], ["z", [0, 0, 1]]]:
            idx = input.find(name)
            if idx == 0:
                res += np.array(direction, dtype=float)
            elif idx > 0:
                if input[idx - 1] == "+":
                    res += np.array(direction, dtype=float)
                elif input[idx - 1] == "-":
                    res -= np.array(direction, dtype=float)
                else:
                    res += np.array(direction, dtype=float)

        res = res / np.linalg.norm(res)
        return res

    def set_camera_focus(
        self, distance=80, direction="X", view_angle=None, look_at=None
    ):
        if look_at is None:
            look_at = self.spin_system.center()
        look_at = np.array(look_at, dtype=float)

        if type(direction) is str:
            direction = self._string_to_direction(direction)

        direction /= np.linalg.norm(direction)

        self.camera_dict["view_angle"] = view_angle
        self.camera_dict["position"] = look_at + distance * direction
        self.camera_dict["focal_point"] = look_at - 2 * distance * direction

    def rotate_camera(self, axis, angle):
        from scipy.spatial.transform import Rotation

        R = Rotation.from_rotvec(angle * np.array(axis)).as_matrix()
        relative_pos = self.camera_dict["position"] - self.camera_dict["focal_point"]
        self.camera_dict["position"] = (
            np.dot(R, relative_pos) + self.camera["focal_point"]
        )
        self.camera_dict["up"] = np.dot(R, self.camera_dict["up"])

    def align_camera_position(self, axis, align_direction, distance=None):
        if distance is None:
            distance = np.linalg.norm(
                self.camera_dict["focal_point"] - self.camera_dict["position"]
            )

        # Remove component along axis from align direction and from current relative position
        align_direction = np.array(align_direction)
        align_direction = align_direction - np.dot(align_direction, axis) * axis
        align_direction /= np.linalg.norm(align_direction)

        current_relative_pos = (
            self.camera_dict["position"] - self.camera_dict["focal_point"]
        )
        current_relative_pos /= np.linalg.norm(current_relative_pos)

        # Comp along axis
        axis_component = np.dot(current_relative_pos, axis) * axis

        relative_pos = align_direction + axis_component
        relative_pos /= np.linalg.norm(relative_pos)

        self.camera_dict["position"] = (
            self.camera_dict["focal_point"] + relative_pos * distance
        )

    def align_camera_up(self, axis):
        from scipy.spatial.transform import Rotation

        relative_pos = self.camera_dict["position"] - self.camera_dict["focal_point"]
        self.camera_dict["up"] = (
            axis
            - relative_pos.dot(axis) / np.linalg.norm(relative_pos) ** 2 * relative_pos
        )
        self.camera_dict["up"] /= np.linalg.norm(self.camera_dict["up"])

    def _set_camera(self, plotter: pv.Plotter):
        if not self.camera_dict.get("position", None) is None:
            plotter.camera.position = self.camera_dict["position"]
        if not self.camera_dict.get("azimuth", None) is None:
            plotter.camera.azimuth = self.camera_dict["azimuth"]
        if not self.camera_dict.get("elevation", None) is None:
            plotter.camera.elevation = self.camera_dict["elevation"]
        if not self.camera_dict.get("focal_point", None) is None:
            plotter.camera.focal_point = self.camera_dict["focal_point"]
        if not self.camera_dict.get("up", None) is None:
            plotter.camera.up = self.camera_dict["up"]
        if not self.camera_dict.get("distance", None) is None:
            plotter.camera.distance = self.camera_dict["distance"]
        if not self.camera_dict.get("view_angle", None) is None:
            plotter.camera.view_angle = self.camera_dict["view_angle"]

    def compute_delaunay(self):
        self._delaunay = delaunay(self._point_cloud)

    def save_delaunay(self, path):
        self._delaunay.save(path)

    def load_delaunay(self, path):
        self._delaunay = pv.read(path)
        self._delaunay.copy_attributes(self._point_cloud)

    def update_spins(self, spin_system):
        self._point_cloud["spins_x"] = spin_system.spins[:, 0]
        self._point_cloud["spins_y"] = spin_system.spins[:, 1]
        self._point_cloud["spins_z"] = spin_system.spins[:, 2]
        self._point_cloud["spins"] = spin_system.spins
        self._point_cloud["spins_rgba"] = self._colormap(spin_system.spins)

        if self._delaunay:
            self._delaunay.copy_attributes(self._point_cloud)

    def colormap(self, colormap, opacity=1.0, **kwargs):
        if type(colormap) is str:
            if colormap.lower() == "hsv":
                self._colormap = lambda spins: get_rgba_colors(spins, opacity, **kwargs)
            elif colormap.lower() == "rb":
                self._colormap = lambda spins: get_rgba_colors_red_blue(
                    spins, opacity, **kwargs
                )
            elif colormap.lower() == "rgb":
                self._colormap = lambda spins: get_rgba_colors_red_green_blue(
                    spins, opacity, **kwargs
                )

        self._point_cloud["spins_rgba"] = self._colormap(self.spin_system.spins)

    def add_mesh(self, mesh, **render_args):
        self.meshlist.append([mesh, render_args])
        return self.meshlist[-1]

    def clear_meshes(self):
        self.meshlist = []

    def isosurface(self, isovalue, scalars_key, **render_args):
        if not self._delaunay:
            raise Exception("No delaunay")

        # Create the contour
        isosurface = self._delaunay.contour(
            [isovalue], scalars=scalars_key, progress_bar=True
        )
        if isosurface.n_faces < 1:
            return None
        isosurface = isosurface.smooth_taubin(n_iter=1000)

        args = self.default_render_args.copy()
        args.update(render_args)

        return self.add_mesh(isosurface, **args)

    def arrows(self, glyph_args=dict(), **render_args):
        glyph_default_args = dict(
            orient="spins",
            factor=1,
            geom=pv.Cone(radius=0.25, resolution=9),
            scale=False,
        )

        glyph_default_args.update(glyph_args)
        arrows = self._point_cloud.glyph(**glyph_default_args)

        args = self.default_render_args.copy()
        args.update(render_args)

        self.add_mesh(arrows, **args)

    def add_preimage(
        self,
        spin_dir,
        tol=0.05,
        n_neighbours=10,
        interpolation_factor=1,
        render_args={"color": "black"},
    ):
        if not self._delaunay:
            raise Exception("No delaunay")

        if render_args is None:
            render_args = self.default_render_args.copy()

        return self.add_mesh(
            create_pre_image(
                spin_dir,
                self._point_cloud,
                angle_tolerance=tol,
                n_neighbours=n_neighbours,
            ),
            render_args,
        )

    def _axes(self, plotter):
        colors = get_rgba_colors([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        plotter.add_axes(
            color="black",
            line_width=6,
            x_color=colors[0],
            y_color=colors[1],
            z_color=colors[2],
        )

    def setup_plotter(self, render_to_png, **kwargs) -> pv.Plotter:
        if not self.plotter is None:
            return

        self._render_to_png = render_to_png

        # if render_to_png:
        #     pv.start_xvfb(wait=self._xvfb_wait)

        self._plotter = pv.Plotter(off_screen=render_to_png, **kwargs)
        self._plotter.window_size = self.resolution

        self._set_camera(self._plotter)

        if self.axes:
            self._axes(self._plotter)

        self.transparent_background = False
        if not self.background_color is None:
            if self.background_color.lower() == "transparent":
                self.transparent_background = True
            else:
                self._plotter.set_background(self.background_color)

    def clear_plotter(self):
        if self._plotter is None:
            self._plotter.close()
        self._plotter = None

    @property
    def plotter(self):
        return self._plotter

    def apply_meshes(self, plotter):
        for m, args in self.meshlist:
            try:
                plotter.add_mesh(m, **args)
            except Exception as e:
                print(f"Could not add_mesh {m}")
                print(e)

    def show(self, save_camera_file=None, persist_camera=False):
        self.setup_plotter(render_to_png=False)
        self.apply_meshes(self.plotter)
        self._plotter.show()

        camera_dict = dict(
            azimuth=self.plotter.camera.azimuth,
            position=self.plotter.camera.position,
            up=self.plotter.camera.up,
            focal_point=self.plotter.camera.focal_point,
            distance=self.plotter.camera.distance,
            view_angle=self.plotter.camera.view_angle,
        )

        if persist_camera:
            self.camera_dict = camera_dict.copy()

        if save_camera_file is not None:
            with open(save_camera_file, "w") as f:
                f.write(json.dumps(camera_dict, indent=4))

    def render_to_png(self, png_path):
        file_name, ext = os.path.splitext(png_path)
        if not (len(ext) == 0 or ext.lower() == ".png"):
            raise Exception("File extension must be either '.png' or not specified")

        self.setup_plotter(render_to_png=True)
        self.apply_meshes(self.plotter)
        self.plotter.screenshot(
            file_name + ".png", transparent_background=self.transparent_background
        )

    def __del__(self):
        if not self._plotter is None:
            self._plotter.close()
