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

def delaunay(point_cloud):
    return point_cloud.delaunay_3d(progress_bar=True)

def isosurface_from_delaunay(delaunay, isovalue=0, scalars_key="spins_z"):
    import pyvista as pv
    # Create the contour
    isosurface = delaunay.contour([isovalue], scalars = scalars_key, progress_bar=True)
    isosurface = isosurface.smooth()
    return isosurface

def arrows_from_point_cloud(point_cloud):
    import pyvista as pv
    geom   = pv.Cone(radius=0.25, resolution=18)
    arrows = point_cloud.glyph(orient="spins", scale=False, factor=1, geom=geom)
    return arrows

def create_pre_image(pre_image_spin, delaunay, tol=0.05):
    import pyvista as pv
    pre_image_spin = np.array(pre_image_spin) / np.linalg.norm(pre_image_spin)
    return delaunay.contour([pre_image_spin[0]], scalars="spins_x").contour([pre_image_spin[1]], scalars="spins_y").threshold( [pre_image_spin[2]-tol, pre_image_spin[2]+tol], scalars="spins_z" )

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