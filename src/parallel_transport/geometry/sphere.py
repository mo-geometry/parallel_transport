"""Sphere geometry — Delaunay triangulation, stereographic projection, and spherical triangles.

Pure NumPy module for constructing a triangulated unit sphere, computing
stereographic projections, great-circle arcs between points, and spherical
triangle areas via the spherical excess formula.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import numpy.typing as npt
from matplotlib.tri import Triangulation

# ---------------------------------------------------------------------------
# Type alias
# ---------------------------------------------------------------------------
Vec3 = npt.NDArray[np.floating[Any]]


# ---------------------------------------------------------------------------
# Sphere mesh
# ---------------------------------------------------------------------------


@dataclass
class SphereMesh:
    """Triangulated unit sphere with precomputed geometric data.

    Attributes:
        x: X coordinates of mesh vertices.
        y: Y coordinates of mesh vertices.
        z: Z coordinates of mesh vertices.
        triangulation: Matplotlib Triangulation for rendering.
        centroids: Triangle centroid positions, shape (3, N_tri).
        areas: Flat area of each triangle, shape (N_tri,).
    """

    x: npt.NDArray[np.float64]
    y: npt.NDArray[np.float64]
    z: npt.NDArray[np.float64]
    triangulation: Triangulation
    centroids: npt.NDArray[np.float64]
    areas: npt.NDArray[np.float64]
    normals: npt.NDArray[np.float64] = field(default_factory=lambda: np.empty(0))


def create_sphere_mesh(
    density: int = 32,
    delta: float = 1e-6,
) -> SphereMesh:
    """Build a triangulated unit sphere via Delaunay triangulation.

    Args:
        density: Number of latitude divisions (longitude gets 2x).
        delta: Small offset to avoid polar singularities.

    Returns:
        SphereMesh with vertices, triangulation, centroids, and areas.
    """
    v, u = np.meshgrid(
        np.linspace(delta, 2 * np.pi - delta, density * 2).astype(np.float64),
        np.linspace(0, np.pi - delta, density).astype(np.float64),
    )

    x = np.ravel(np.sin(u) * np.cos(v))
    y = np.ravel(np.sin(u) * np.sin(v))
    z = np.ravel(np.cos(u))

    tri = Triangulation(np.ravel(u), np.ravel(v))

    # Triangle vertex coordinates: (N_tri, 3_verts, 3_xyz)
    tri_coords = np.array([[x[t], y[t], z[t]] for t in tri.triangles])

    # Centroids: shape (3, N_tri)
    centroids = tri_coords.mean(axis=2).T

    # Flat triangle areas via cross product
    areas = np.zeros(len(tri.triangles))
    for i, verts in enumerate(tri_coords):
        v0, v1, v2 = verts.T[0], verts.T[1], verts.T[2]
        areas[i] = 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0))

    # Normalised centroid directions
    normals = centroids / np.sqrt((centroids**2).sum(axis=0, keepdims=True))

    return SphereMesh(
        x=x,
        y=y,
        z=z,
        triangulation=tri,
        centroids=centroids,
        areas=areas,
        normals=normals,
    )


# ---------------------------------------------------------------------------
# Stereographic projection
# ---------------------------------------------------------------------------


def stereo_project(
    points: Vec3,
    pole: str = "north",
    epsilon: float = 1e-12,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Stereographic projection from S2 to R2.

    Args:
        points: Points on the sphere, shape (3,) or (3, N).
        pole: Projection pole — "north" projects from (0,0,1),
              "south" projects from (0,0,-1).
        epsilon: Small offset to avoid division by zero.

    Returns:
        Tuple (X, Y) of projected coordinates.
    """
    rx, ry, rz = 0.5 * points[0], 0.5 * points[1], 0.5 * points[2]
    denom = rz + 0.5 + epsilon if pole == "north" else rz - 0.5 + epsilon
    return rx / denom, ry / denom


# ---------------------------------------------------------------------------
# Spherical polars
# ---------------------------------------------------------------------------


def to_spherical(points: Vec3) -> tuple[npt.NDArray[np.floating[Any]], ...]:
    """Convert Cartesian coordinates to spherical polars (theta, phi).

    Args:
        points: Shape (3,) or (3, N).

    Returns:
        Tuple (theta, phi) — polar and azimuthal angles.
    """
    theta = np.arccos(np.clip(points[2], -1.0, 1.0))
    phi = np.arctan2(points[1], points[0])
    return theta, phi


# ---------------------------------------------------------------------------
# Great-circle arcs and spherical triangles
# ---------------------------------------------------------------------------


def arc_between(v1: Vec3, v2: Vec3, n_points: int = 101) -> Vec3:
    """Compute the great-circle arc between two unit vectors.

    Uses Rodrigues' rotation formula to interpolate along the arc.

    Args:
        v1: Start point on S2, shape (3,).
        v2: End point on S2, shape (3,).
        n_points: Number of samples along the arc.

    Returns:
        Arc points, shape (3, n_points).
    """
    dot = np.clip(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)), -1.0, 1.0)
    theta_max = np.arccos(dot)

    cross = np.cross(v1, v2)
    cross_norm = np.linalg.norm(cross)

    if cross_norm < 1e-7:
        # Vectors are (anti-)parallel — return degenerate arc
        return np.column_stack([v1] * n_points)

    n = cross / cross_norm
    theta = np.linspace(0, theta_max, n_points)

    # Rodrigues' formula
    arc_x = v1[0] * np.cos(theta) + (n[1] * v1[2] - n[2] * v1[1]) * np.sin(theta)
    arc_y = v1[1] * np.cos(theta) + (n[2] * v1[0] - n[0] * v1[2]) * np.sin(theta)
    arc_z = v1[2] * np.cos(theta) + (n[0] * v1[1] - n[1] * v1[0]) * np.sin(theta)

    return np.array([arc_x, arc_y, arc_z])


def tangent_vectors_at_arc_endpoints(a: Vec3, b: Vec3) -> tuple[Vec3, Vec3]:
    """Compute tangent vectors at both endpoints of a great-circle arc.

    Args:
        a: Start point on S2, shape (3,).
        b: End point on S2, shape (3,).

    Returns:
        Tuple (t_ab, t_ba) — unit tangent at A pointing toward B,
        and unit tangent at B pointing toward A.
    """
    dot = np.clip(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)), -1.0, 1.0)
    theta = np.arccos(dot)

    cross = np.cross(a, b)
    cross_norm = np.linalg.norm(cross)

    if cross_norm < 1e-8:
        return np.full(3, np.nan), np.full(3, np.nan)

    n = cross / cross_norm

    # Tangent at A (derivative of Rodrigues at theta=0)
    t_ab = np.array(
        [
            -a[0] * np.sin(0) + (n[1] * a[2] - n[2] * a[1]) * np.cos(0),
            -a[1] * np.sin(0) + (n[2] * a[0] - n[0] * a[2]) * np.cos(0),
            -a[2] * np.sin(0) + (n[0] * a[1] - n[1] * a[0]) * np.cos(0),
        ]
    )
    t_ab = t_ab / np.linalg.norm(t_ab)

    # Tangent at B (derivative at theta=theta_max, reversed)
    t_ba = np.array(
        [
            -a[0] * np.sin(theta) + (n[1] * a[2] - n[2] * a[1]) * np.cos(theta),
            -a[1] * np.sin(theta) + (n[2] * a[0] - n[0] * a[2]) * np.cos(theta),
            -a[2] * np.sin(theta) + (n[0] * a[1] - n[1] * a[0]) * np.cos(theta),
        ]
    )
    t_ba_norm = np.linalg.norm(t_ba)
    t_ba = -t_ba / t_ba_norm if t_ba_norm > 1e-12 else np.full(3, np.nan)

    return t_ab, t_ba


def spherical_triangle_area(a: Vec3, b: Vec3, c: Vec3) -> float:
    """Compute the area of a spherical triangle via the spherical excess.

    The area equals the sum of interior angles minus pi (Girard's theorem).

    Args:
        a: First vertex on S2, shape (3,).
        b: Second vertex on S2, shape (3,).
        c: Third vertex on S2, shape (3,).

    Returns:
        Spherical excess (area on the unit sphere).
    """
    t_ab, _ = tangent_vectors_at_arc_endpoints(a, b)
    t_ac, _ = tangent_vectors_at_arc_endpoints(a, c)
    t_ba, _ = tangent_vectors_at_arc_endpoints(b, a)
    t_bc, _ = tangent_vectors_at_arc_endpoints(b, c)
    t_ca, _ = tangent_vectors_at_arc_endpoints(c, a)
    t_cb, _ = tangent_vectors_at_arc_endpoints(c, b)

    angle_a = np.arccos(np.clip(np.dot(t_ab, t_ac), -1.0, 1.0))
    angle_b = np.arccos(np.clip(np.dot(t_ba, t_bc), -1.0, 1.0))
    angle_c = np.arccos(np.clip(np.dot(t_ca, t_cb), -1.0, 1.0))

    return float(angle_a + angle_b + angle_c - np.pi)
