"""Tests for the quaternion algebra module."""

from __future__ import annotations

import numpy as np
from parallel_transport.core.quaternion import (
    conjugate,
    from_axis_angle,
    multiply,
    normalize,
    random_unit,
    to_axis_angle,
    to_rotation_matrix,
    to_rotation_matrix_dt,
)


class TestMultiply:
    """Quaternion multiplication tests."""

    def test_identity_left(self) -> None:
        """Multiplying by identity on the left returns the original."""
        identity = np.array([1.0, 0.0, 0.0, 0.0])
        q = random_unit()
        result = multiply(identity, q)
        np.testing.assert_allclose(result, q, atol=1e-12)

    def test_identity_right(self) -> None:
        """Multiplying by identity on the right returns the original."""
        identity = np.array([1.0, 0.0, 0.0, 0.0])
        q = random_unit()
        result = multiply(q, identity)
        np.testing.assert_allclose(result, q, atol=1e-12)

    def test_q_times_conjugate_gives_identity(self) -> None:
        """q * conj(q) = |q|^2 * identity."""
        q = random_unit()
        result = multiply(q, conjugate(q))
        expected = np.array([1.0, 0.0, 0.0, 0.0])
        np.testing.assert_allclose(result, expected, atol=1e-12)

    def test_non_commutative(self) -> None:
        """Quaternion multiplication is generally non-commutative."""
        q1 = np.array([0.0, 1.0, 0.0, 0.0])  # pure i
        q2 = np.array([0.0, 0.0, 1.0, 0.0])  # pure j
        # i * j = k, but j * i = -k
        ij = multiply(q1, q2)
        ji = multiply(q2, q1)
        np.testing.assert_allclose(ij, np.array([0.0, 0.0, 0.0, 1.0]), atol=1e-12)
        np.testing.assert_allclose(ji, np.array([0.0, 0.0, 0.0, -1.0]), atol=1e-12)

    def test_associative(self) -> None:
        """(q1 * q2) * q3 = q1 * (q2 * q3)."""
        q1, q2, q3 = random_unit(), random_unit(), random_unit()
        lhs = multiply(multiply(q1, q2), q3)
        rhs = multiply(q1, multiply(q2, q3))
        np.testing.assert_allclose(lhs, rhs, atol=1e-12)

    def test_broadcast_multiply(self) -> None:
        """Multiplication works with shape (4, N) arrays."""
        q1 = random_unit()
        n = 10
        q2 = np.column_stack([random_unit() for _ in range(n)])
        result = multiply(q1.reshape(4, 1) * np.ones((1, n)), q2)
        assert result.shape == (4, n)
        # Each column should equal the scalar multiplication
        for i in range(n):
            expected = multiply(q1, q2[:, i])
            np.testing.assert_allclose(result[:, i], expected, atol=1e-12)


class TestConjugate:
    """Quaternion conjugate tests."""

    def test_conjugate_negates_imaginary(self) -> None:
        q = np.array([1.0, 2.0, 3.0, 4.0])
        result = conjugate(q)
        np.testing.assert_allclose(result, [1.0, -2.0, -3.0, -4.0])

    def test_double_conjugate_is_identity(self) -> None:
        q = random_unit()
        np.testing.assert_allclose(conjugate(conjugate(q)), q, atol=1e-12)


class TestNormalize:
    """Normalization tests."""

    def test_unit_norm(self) -> None:
        q = np.array([1.0, 2.0, 3.0, 4.0])
        result = normalize(q)
        norm = np.sqrt((result * result).sum())
        np.testing.assert_allclose(norm, 1.0, atol=1e-12)


class TestAxisAngle:
    """Axis-angle <-> quaternion roundtrip tests."""

    def test_roundtrip(self) -> None:
        """from_axis_angle -> to_axis_angle should recover inputs."""
        beta, theta, phi = 1.2, 0.8, 2.1
        q = from_axis_angle(beta, theta, phi)
        beta_r, theta_r, phi_r = to_axis_angle(q)
        np.testing.assert_allclose(beta_r, beta, atol=1e-10)
        np.testing.assert_allclose(theta_r, theta, atol=1e-10)
        np.testing.assert_allclose(phi_r, phi, atol=1e-10)

    def test_zero_rotation(self) -> None:
        """Zero rotation gives identity quaternion."""
        q = from_axis_angle(0.0, 0.5, 1.0)
        np.testing.assert_allclose(q, [1.0, 0.0, 0.0, 0.0], atol=1e-12)

    def test_pi_rotation(self) -> None:
        """180-degree rotation about z-axis."""
        q = from_axis_angle(np.pi, 0.0, 0.0)  # theta=0 => axis is z
        # cos(pi/2) = 0, sin(pi/2)*cos(0) = 0, sin(pi/2)*sin(0) = 0, sin(pi/2)*1 = 1
        np.testing.assert_allclose(q[0], 0.0, atol=1e-12)
        np.testing.assert_allclose(np.abs(q[3]), 1.0, atol=1e-12)

    def test_vectorized(self) -> None:
        """Axis-angle conversion works with arrays."""
        beta = np.linspace(0, 2 * np.pi, 50)
        theta = np.ones(50) * 0.5
        phi = np.ones(50) * 1.0
        q = from_axis_angle(beta, theta, phi)
        assert q.shape == (4, 50)
        # Each quaternion should be unit
        norms = np.sqrt((q * q).sum(axis=0))
        np.testing.assert_allclose(norms, 1.0, atol=1e-12)


class TestRotationMatrix:
    """SO(3) rotation matrix tests."""

    def test_identity_quaternion_gives_identity_matrix(self) -> None:
        q = np.array([1.0, 0.0, 0.0, 0.0])
        r = to_rotation_matrix(q)
        np.testing.assert_allclose(r, np.eye(3), atol=1e-12)

    def test_orthogonal(self) -> None:
        """Rotation matrix should be orthogonal: R^T R = I."""
        q = random_unit()
        r = to_rotation_matrix(q)
        np.testing.assert_allclose(r.T @ r, np.eye(3), atol=1e-12)

    def test_determinant_one(self) -> None:
        """Rotation matrix should have determinant +1."""
        q = random_unit()
        r = to_rotation_matrix(q)
        np.testing.assert_allclose(np.linalg.det(r), 1.0, atol=1e-12)

    def test_180_about_z(self) -> None:
        """180-degree rotation about z maps (1,0,0) to (-1,0,0)."""
        q = from_axis_angle(np.pi, 0.0, 0.0)
        r = to_rotation_matrix(q)
        rotated = r @ np.array([1.0, 0.0, 0.0])
        np.testing.assert_allclose(rotated, [-1.0, 0.0, 0.0], atol=1e-12)

    def test_vectorized_shape(self) -> None:
        """Rotation matrix from shape (4, N) gives (3, 3, N)."""
        n = 20
        q = np.column_stack([random_unit() for _ in range(n)])
        r = to_rotation_matrix(q)
        assert r.shape == (3, 3, n)

    def test_derivative_matches_numerical(self) -> None:
        """Analytical R_dt should match finite-difference approximation."""
        dt = 1e-7
        q = random_unit()
        # Project perturbation onto the tangent space of S3 at q
        # so q_dt is orthogonal to q (preserves unit constraint)
        dq_raw = np.random.randn(4) * 0.1
        q_dt = dq_raw - np.dot(dq_raw, q) * q

        r0 = to_rotation_matrix(q)
        q_fwd = normalize(q + dt * q_dt)
        r1 = to_rotation_matrix(q_fwd)
        numerical = (r1 - r0) / dt

        analytical = to_rotation_matrix_dt(q, q_dt)
        np.testing.assert_allclose(analytical, numerical, atol=1e-4)
