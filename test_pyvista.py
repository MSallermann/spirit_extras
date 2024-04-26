import numpy as np
from spirit_extras import configurations, data, pyvista_plotting
from pathlib import Path


def hopfion(x, y, z):
    # x,y,z = pos
    pos = [x, y, z]
    r = np.linalg.norm(pos)
    lam = 12 / 2

    def f_fun(r):
        return np.arcsin(2 * r * lam / (r**2 + lam**2))

    f = f_fun(r)

    sx = x / r * np.sin(2 * f) + y * z / r**2 * np.sin(f) ** 2
    sy = y / r * np.sin(2 * f) - x * z / r**2 * np.sin(f) ** 2
    sz = np.cos(2 * f) + 2 * z**2 / r**2 * np.sin(f) ** 2

    return [sx, sy, sz]


N_CELLS = [30, 30, 30]
NOS = np.prod(N_CELLS)

spins = np.zeros(shape=(NOS, 3))
positions = np.zeros(shape=(NOS, 3))

for a in range(N_CELLS[0]):
    for b in range(N_CELLS[1]):
        for c in range(N_CELLS[2]):
            idx = a + N_CELLS[0] * (b + c * N_CELLS[1])
            positions[idx] = [a, b, c]

DELAUNAY_PATH = Path("delaunay.vtk")

system = data.Spin_System(positions, spins, False, N_CELLS)
configurations.rectangular_configuration(system, system.center(), hopfion)
system.flatten()
plotter = pyvista_plotting.Spin_Plotter(system)
if not DELAUNAY_PATH.exists():
    plotter.compute_delaunay()
    plotter.save_delaunay(DELAUNAY_PATH)
else:
    plotter.load_delaunay(DELAUNAY_PATH)
plotter.isosurface(0.0, "spins_z")
plotter.render_to_png("hopfion.png")