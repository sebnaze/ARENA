#%%
import numpy as np
import plotly.graph_objects as go
from torch import Tensor
import einops
import itertools

#%%
# Setup
import os
import sys
from functools import partial
from pathlib import Path
from typing import Callable

import einops
import plotly.express as px
import plotly.graph_objects as go
import torch as t
from IPython.display import display
from ipywidgets import interact
from jaxtyping import Bool, Float
from torch import Tensor
from tqdm import tqdm

# Make sure exercises are in the path
chapter = "chapter0_fundamentals"
section = "part1_ray_tracing"
#root_dir = next(p for p in Path.cwd().parents if (p / chapter).exists())
root_dir = Path("/Users/sebastin/Documents/perso/ARENA_training/ARENA_3.0")
exercises_dir = root_dir / chapter / "exercises"
section_dir = exercises_dir / section
if str(exercises_dir) not in sys.path:
    sys.path.append(str(exercises_dir))

import part1_ray_tracing.tests as tests
from part1_ray_tracing.utils import (
    render_lines_with_plotly,
    setup_widget_fig_ray,
    setup_widget_fig_triangle,
)
from plotly_utils import imshow

MAIN = __name__ == "__main__"
#%%
# Fundamentals of ray generation and rendering
def make_rays_1d(num_pixels: int, y_limit: float) -> Tensor:
    """
    num_pixels: The number of pixels in the y dimension. Since there is one ray per pixel, this is
        also the number of rays.
    y_limit: At x=1, the rays should extend from -y_limit to +y_limit, inclusive of both endpoints.

    Returns: shape (num_pixels, num_points=2, num_dim=3) where the num_points dimension contains
        (origin, direction) and the num_dim dimension contains xyz.

    Example of make_rays_1d(9, 1.0): [
        [[0, 0, 0], [1, -1.0, 0]],
        [[0, 0, 0], [1, -0.75, 0]],
        [[0, 0, 0], [1, -0.5, 0]],
        ...
        [[0, 0, 0], [1, 0.75, 0]],
        [[0, 0, 0], [1, 1, 0]],
    ]
    """
    origins = np.zeros((num_pixels, 3))  # shape (num_pixels, num_dim)
    y_values = np.linspace(-y_limit, y_limit, num_pixels)  # shape (num_pixels,)
    directions = np.stack([np.ones(num_pixels), y_values, np.zeros(num_pixels)], axis=-1)  # shape (num_pixels, num_dim)
    rays = np.stack([origins, directions], axis=1)  # shape (num_pixels, num_points=2, num_dim=3)
    return Tensor(rays)



def render_lines_with_plotly(rays: Tensor) -> go.Figure:
    """
    rays: shape (num_rays, num_points=2, num_dim=3) where the num_points dimension contains
        (origin, direction) and the num_dim dimension contains xyz.
    Returns: A plotly figure with the rays rendered as lines.
    """
    num_rays = rays.shape[0]
    fig = go.Figure()
    for i in range(num_rays):
        origin = rays[i, 0]  # shape (num_dim,)
        direction = rays[i, 1]  # shape (num_dim,)
        line_x = [origin[0].item(), origin[0].item() + direction[0].item()]
        line_y = [origin[1].item(), origin[1].item() + direction[1].item()]
        line_z = [origin[2].item(), origin[2].item() + direction[2].item()]
        fig.add_trace(go.Scatter3d(x=line_x, y=line_y, z=line_z, mode='lines'))
    fig.update_layout(scene=dict(aspectmode='cube'))
    return fig

rays1d = make_rays_1d(9, 10.0)
fig = render_lines_with_plotly(rays1d)
fig.show()

#%% Closest points between two lines in 3D
def closest_points_3d(p1, d1, p2, d2):
    # p1, p2: starting points (numpy arrays)
    # d1, d2: direction vectors (numpy arrays)
    
    # Cross product of directions gives the common perpendicular
    n = np.cross(d1, d2)
    
    # If the cross product is zero, lines are parallel
    if np.linalg.norm(n) < 1e-10:
        return None, "Lines are parallel"

    # Solve the linear system for s and t
    # Using the property that the segment (L1(s) - L2(t)) is perp to d1 and d2
    # This forms a 2x2 system:
    # s(d1·d1) - t(d2·d1) = (p2-p1)·d1
    # s(d1·d2) - t(d2·d2) = (p2-p1)·d2
    
    rhs = p2 - p1
    matrix = np.array([
        [np.dot(d1, d1), -np.dot(d2, d1)],
        [np.dot(d1, d2), -np.dot(d2, d2)]
    ])
    
    s, t = np.linalg.solve(matrix, [np.dot(rhs, d1), np.dot(rhs, d2)])
    
    point1 = p1 + s * d1
    point2 = p2 + t * d2
    
    # If point1 == point2, they actually intersect!
    if np.allclose(point1, point2):
        return point1, "Perfect Intersection"
    else:
        return (point1 + point2) / 2, "Skew (Midpoint of closest approach)"

# Example
p1, d1 = np.array([0,0,0]), np.array([1,1,0])
p2, d2 = np.array([0,2,0]), np.array([1,-1,0])

point, status = closest_points_3d(p1, d1, p2, d2)
print(f"{status}: {point}")

# %%
# Ray intersection with line segments
def intersect_ray_1d(ray: Float[Tensor, "points dims"], segment: Float[Tensor, "points dims"]) -> bool:
    """
    ray: shape (n_points=2, n_dim=3)  # O, D points
    segment: shape (n_points=2, n_dim=3)  # L_1, L_2 points

    Return True if the ray intersects the segment.
    """
    O, D = ray  # Origin and Direction of the ray
    L1, L2 = segment  # Endpoints of the line segment

    # Vector from L1 to L2
    seg_vec = L2 - L1
    seg_len = t.norm(seg_vec)
    
    if seg_len < 1e-10:
        return False  # Segment is a point, treat as no intersection

    seg_dir = seg_vec / seg_len  # Normalize to get direction

    # Compute the closest points between the ray and the line defined by the segment
    closest_point, status = closest_points_3d(O.numpy(), D.numpy(), L1.numpy(), seg_dir.numpy())
    
    if status == "Lines are parallel":
        return False
    
    closest_point = t.tensor(closest_point)

    # Check if the closest point lies on the ray (t >= 0) and on the segment (0 <= s <= seg_len)
    ray_param = t.dot(closest_point - O, D) / t.dot(D, D)
    seg_param = t.dot(closest_point - L1, seg_dir)

    return ray_param >= 0 and 0 <= seg_param <= seg_len


tests.test_intersect_ray_1d(intersect_ray_1d)
tests.test_intersect_ray_1d_special_case(intersect_ray_1d)


# %% # Intersect 1d by Claude
import numpy as np

def line_crosses_segment(point: np.ndarray, direction: np.ndarray,
                         seg_a: np.ndarray, seg_b: np.ndarray,
                         tol: float = 1e-10) -> bool:
    """
    Check whether an infinite 3D line crosses a finite 3D segment.

    Args:
        point:     A point on the line (3D vector).
        direction: Direction vector of the line (need not be normalized).
        seg_a:     First endpoint of the segment (3D vector).
        seg_b:     Second endpoint of the segment (3D vector).
        tol:       Tolerance for floating-point comparisons.

    Returns:
        True if the line intersects the segment, False otherwise.
    """
    point     = np.asarray(point,     dtype=float)
    direction = np.asarray(direction, dtype=float)
    seg_a     = np.asarray(seg_a,     dtype=float)
    seg_b     = np.asarray(seg_b,     dtype=float)

    d = direction
    v = seg_b - seg_a          # segment direction
    w = seg_a - point          # vector from line point to segment start

    # Solve: point + t*d == seg_a + s*v  →  t*d - s*v = w
    # Use least-squares on the over-determined system via two cross-product equations.
    d_cross_v = np.cross(d, v)
    denom = np.dot(d_cross_v, d_cross_v)   # |d × v|²

    # If denom ≈ 0, d and v are parallel.
    if denom < tol:
        # Lines are parallel — check if they are actually collinear.
        # The segment lies on the line iff w × d == 0.
        return np.linalg.norm(np.cross(w, d)) < tol

    # Solve for s (segment parameter) using the scalar triple product.
    # s = (w × d) · (d × v) / |d × v|²
    s = np.dot(np.cross(w, d), d_cross_v) / denom

    # The line crosses the segment iff 0 <= s <= 1.
    return -tol <= s <= 1.0 + tol

#%% Ray - Segment interctive plot
fig: go.FigureWidget = setup_widget_fig_ray()
display(fig)


@interact(v=(0.0, 6.0, 0.01), seed=(0, 10, 1))
def update(v=0.0, seed=0):
    t.manual_seed(seed)
    L_1, L_2 = t.rand(2, 2)
    P = lambda v: L_1 + v * (L_2 - L_1)
    x, y = zip(P(0), P(6))
    with fig.batch_update():
        fig.update_traces({"x": x, "y": y}, 0)
        fig.update_traces({"x": [L_1[0], L_2[0]], "y": [L_1[1], L_2[1]]}, 1)
        fig.update_traces({"x": [P(v)[0]], "y": [P(v)[1]]}, 2)

# %%
# Intersects multiple rays, multiple segments
def intersect_rays_1d(
    rays: Float[Tensor, "nrays 2 3"], segments: Float[Tensor, "nsegments 2 3"]
) -> Bool[Tensor, " nrays"]:
    """
    For each ray, return True if it intersects any segment.
    """
    nrays, nsegments = rays.shape[0], segments.shape[0]
    out = np.zeros((nrays, nsegments))
    for i,j in itertools.product(range(nrays), range(nsegments)):
        out[i,j] = intersect_ray_1d(rays[i], segments[j])
    ray_out = t.tensor([o.any() for o in out], dtype=t.bool)
    return ray_out

tests.test_intersect_rays_1d(intersect_rays_1d)
tests.test_intersect_rays_1d_special_case(intersect_rays_1d)

# %% Ray Trace Traingale intersection
def raytrace_triangle_with_bug(
    rays: Float[Tensor, "nrays rayPoints=2 dims=3"],
    triangle: Float[Tensor, "trianglePoints=3 dims=3"]
) -> Bool[Tensor, " nrays"]:
    '''
    For each ray, return True if the triangle intersects that ray.
    '''
    NR = rays.size[0]

    A, B, C = einops.repeat(triangle, "pts dims -> pts NR dims", NR=NR)

    O, D = rays.unbind(-1)

    mat = t.stack([- D, B - A, C - A])

    dets = t.linalg.det(mat)
    is_singular = dets.abs() < 1e-8
    mat[is_singular] = t.eye(3)

    vec = O - A

    sol = t.linalg.solve(mat, vec)
    s, u, v = sol.unbind(dim=-1)

    return ((u >= 0) & (v >= 0) & (u + v <= 1) & ~is_singular)


intersects = raytrace_triangle_with_bug(rays2d, test_triangle)
img = intersects.reshape(num_pixels_y, num_pixels_z).int()
imshow(img, origin="lower", width=600, title="Triangle (as intersected by rays)")
# %%
