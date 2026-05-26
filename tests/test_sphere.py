"""Tests for the sphere geometry module."""

from __future__ import annotations

import numpy as np

from parallel_transport.geometry.sphere import (
    SphereMesh,
    arc_between,
    create_sphere_mesh,
    spherical_triangle_area,
    stereo_project,
    tangent_vectors_at_arc_endpoints,
    to_spherical,
)


class TestCreateSphereMesh:
    """Sphere mesh construction tests."""

    def test_returns_sphere_mesh(self) -> None:
        mesh = create_sphere_mesh(density=8)
        assert isinstance(mesh, SphereMesh)

    def test_vertices_on_unit_sphere(self) -> None:
        """All vertices should lie on (or very near) the unit sphere."""
        mesh = create_sphere_mesh(density=16)
        radii = np.sqrt(mesh.x**2 + mesh.y**2 + mesh.z**2)
        np.testing.assert_allclose(radii, 1.0, atol=1e-6)

    def test_centroids_shape(self) -> None:
        mesh = create_sphere_mesh(density=8)
        n_tri = len(mesh.triangulation.triangles)
        assert mesh.centroids.shape == (3, n_tri)

    def test_areas_non_negative(self) -> None:
        """All triangle areas should be non-negative (some degenerate near poles)."""
        mesh = create_sphere_mesh(density=8)
        assert np.all(mesh.areas >= 0)

    def test_total_area_approximates_4pi(self) -> None:
        """Sum of triangle areas should approximate 4*pi for the unit sphere."""
        mesh = create_sphere_mesh(density=32)
        total = mesh.areas.sum()
        np.testing.assert_allclose(total, 4 * np.pi, atol=0.5)

    def test_normals_unit_length(self) -> None:
        mesh = create_sphere_mesh(density=8)
        norms = np.sqrt((mesh.normals**2).sum(axis=0))
        np.testing.assert_allclose(norms, 1.0, atol=1e-6)


class TestStereoProject:
    """Stereographic projection tests."""

    def test_equator_projects_to_circle(self) -> None:
        """Points on the equator should project to a circle in the plane."""
        phi = np.linspace(0, 2 * np.pi, 100)
        points = np.array([np.cos(phi), np.sin(phi), np.zeros_like(phi)])
        x, y = stereo_project(points, pole="north")
        radii = np.sqrt(x**2 + y**2)
        # All projected radii should be equal
        np.testing.assert_allclose(radii, radii[0], atol=1e-10)

    def test_south_pole_maps_to_origin(self) -> None:
        """South pole (0, 0, -1) projected from north should map near origin."""
        point = np.array([0.0, 0.0, -1.0])
        x, y = stereo_project(point, pole="north")
        np.testing.assert_allclose([x, y], [0.0, 0.0], atol=1e-10)

    def test_north_pole_maps_to_origin_from_south(self) -> None:
        """North pole projected from south pole should map near origin."""
        point = np.array([0.0, 0.0, 1.0])
        x, y = stereo_project(point, pole="south")
        np.testing.assert_allclose([x, y], [0.0, 0.0], atol=1e-10)


class TestToSpherical:
    """Spherical coordinate tests."""

    def test_north_pole(self) -> None:
        theta, phi = to_spherical(np.array([0.0, 0.0, 1.0]))
        np.testing.assert_allclose(theta, 0.0, atol=1e-12)

    def test_south_pole(self) -> None:
        theta, phi = to_spherical(np.array([0.0, 0.0, -1.0]))
        np.testing.assert_allclose(theta, np.pi, atol=1e-12)

    def test_equator_x(self) -> None:
        theta, phi = to_spherical(np.array([1.0, 0.0, 0.0]))
        np.testing.assert_allclose(theta, np.pi / 2, atol=1e-12)
        np.testing.assert_allclose(phi, 0.0, atol=1e-12)


class TestArcBetween:
    """Great-circle arc tests."""

    def test_arc_endpoints(self) -> None:
        """Arc should start at v1 and end at v2."""
        v1 = np.array([1.0, 0.0, 0.0])
        v2 = np.array([0.0, 1.0, 0.0])
        arc = arc_between(v1, v2, n_points=101)
        np.testing.assert_allclose(arc[:, 0], v1, atol=1e-10)
        np.testing.assert_allclose(arc[:, -1], v2, atol=1e-10)

    def test_arc_on_sphere(self) -> None:
        """All arc points should lie on the unit sphere."""
        v1 = np.array([1.0, 0.0, 0.0])
        v2 = np.array([0.0, 0.0, 1.0])
        arc = arc_between(v1, v2, n_points=50)
        radii = np.sqrt((arc**2).sum(axis=0))
        np.testing.assert_allclose(radii, 1.0, atol=1e-10)

    def test_parallel_vectors(self) -> None:
        """Arc between parallel vectors should be degenerate."""
        v = np.array([1.0, 0.0, 0.0])
        arc = arc_between(v, v, n_points=10)
        assert arc.shape == (3, 10)


class TestTangentVectors:
    """Tangent vector tests."""

    def test_tangent_perpendicular_to_position(self) -> None:
        """Tangent at A should be perpendicular to A (lies in tangent plane)."""
        a = np.array([1.0, 0.0, 0.0])
        b = np.array([0.0, 1.0, 0.0])
        t_ab, t_ba = tangent_vectors_at_arc_endpoints(a, b)
        np.testing.assert_allclose(np.dot(t_ab, a), 0.0, atol=1e-10)
        np.testing.assert_allclose(np.dot(t_ba, b), 0.0, atol=1e-10)

    def test_tangent_unit_length(self) -> None:
        a = np.array([0.0, 0.0, 1.0])
        b = np.array([1.0, 0.0, 0.0])
        t_ab, t_ba = tangent_vectors_at_arc_endpoints(a, b)
        np.testing.assert_allclose(np.linalg.norm(t_ab), 1.0, atol=1e-10)
        np.testing.assert_allclose(np.linalg.norm(t_ba), 1.0, atol=1e-10)


class TestSphericalTriangleArea:
    """Spherical triangle area tests."""

    def test_octant_area(self) -> None:
        """Triangle with vertices at (1,0,0), (0,1,0), (0,0,1) is 1/8 sphere = pi/2."""
        a = np.array([1.0, 0.0, 0.0])
        b = np.array([0.0, 1.0, 0.0])
        c = np.array([0.0, 0.0, 1.0])
        area = spherical_triangle_area(a, b, c)
        np.testing.assert_allclose(area, np.pi / 2, atol=1e-10)

    def test_area_positive(self) -> None:
        """Area should be positive for a valid triangle."""
        a = np.array([1.0, 0.0, 0.0])
        b = np.array([0.0, 1.0, 0.0])
        c = np.array([0.0, 0.0, 1.0])
        assert spherical_triangle_area(a, b, c) > 0
