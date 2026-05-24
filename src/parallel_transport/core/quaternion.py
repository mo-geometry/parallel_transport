"""Quaternion algebra — pure NumPy, no UI dependencies.

Provides quaternion multiplication, conjugation, axis-angle conversion,
and the map from unit quaternions to SO(3) rotation matrices. Quaternions
are represented as NumPy arrays of shape (4,) or (4, N) in the order
(a, b, c, d) = (scalar, i, j, k).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt

# ---------------------------------------------------------------------------
# Type alias for readability
# ---------------------------------------------------------------------------
Vec4 = npt.NDArray[np.floating[Any]]


# ---------------------------------------------------------------------------
# Quaternion primitives
# ---------------------------------------------------------------------------


def multiply(q1: Vec4, q2: Vec4) -> Vec4:
    """Hamilton product of two quaternions.

    Args:
        q1: First quaternion, shape (4,) or (4, N).
        q2: Second quaternion, shape (4,) or (4, N).

    Returns:
        Product q1 * q2, same shape as inputs.
    """
    a1, b1, c1, d1 = q1[0], q1[1], q1[2], q1[3]
    a2, b2, c2, d2 = q2[0], q2[1], q2[2], q2[3]
    a = a1 * a2 - b1 * b2 - c1 * c2 - d1 * d2
    b = a1 * b2 + b1 * a2 + c1 * d2 - d1 * c2
    c = a1 * c2 - b1 * d2 + c1 * a2 + d1 * b2
    d = a1 * d2 + b1 * c2 - c1 * b2 + d1 * a2
    return np.array([a, b, c, d])


def conjugate(q: Vec4) -> Vec4:
    """Quaternion conjugate: (a, -b, -c, -d).

    Args:
        q: Quaternion, shape (4,) or (4, N).

    Returns:
        Conjugate quaternion.
    """
    return np.array([q[0], -q[1], -q[2], -q[3]])


def normalize(q: Vec4) -> Vec4:
    """Normalize a quaternion to unit length.

    Args:
        q: Quaternion, shape (4,) or (4, N).

    Returns:
        Unit quaternion.
    """
    norm = np.sqrt((q * q).sum(axis=0))
    result: Vec4 = q / norm
    return result


def random_unit() -> Vec4:
    """Generate a random unit quaternion (uniform on S3).

    Returns:
        Unit quaternion, shape (4,).
    """
    q = np.random.randn(4)
    return q / np.linalg.norm(q)


# ---------------------------------------------------------------------------
# Axis-angle <-> quaternion conversion
# ---------------------------------------------------------------------------


def from_axis_angle(beta: npt.ArrayLike, theta: npt.ArrayLike, phi: npt.ArrayLike) -> Vec4:
    """Construct a quaternion from axis-angle representation.

    The rotation axis is parameterised by spherical coordinates (theta, phi)
    and the rotation angle is beta.

    Args:
        beta: Rotation angle (scalar or array).
        theta: Polar angle of rotation axis (scalar or array).
        phi: Azimuthal angle of rotation axis (scalar or array).

    Returns:
        Quaternion(s), shape (4,) or (4, N).
    """
    beta = np.asarray(beta)
    theta = np.asarray(theta)
    phi = np.asarray(phi)
    a = np.cos(beta / 2)
    b = np.sin(beta / 2) * np.sin(theta) * np.cos(phi)
    c = np.sin(beta / 2) * np.sin(theta) * np.sin(phi)
    d = np.sin(beta / 2) * np.cos(theta)
    return np.array([a, b, c, d])


def to_axis_angle(q: Vec4) -> tuple[npt.NDArray[np.floating[Any]], ...]:
    """Extract axis-angle representation from a quaternion.

    Args:
        q: Unit quaternion, shape (4,) or (4, N).

    Returns:
        Tuple (beta, theta, phi) — rotation angle and spherical coordinates
        of the rotation axis.
    """
    a, b, c, d = q[0], q[1], q[2], q[3]
    beta = 2 * np.arccos(np.clip(a, -1.0, 1.0))
    norm = np.sqrt(b**2 + c**2 + d**2) + 1e-12
    x = b / norm
    y = c / norm
    z = d / norm
    phi = np.arctan2(y, x)
    theta = np.arccos(np.clip(z, -1.0, 1.0))
    return np.asarray(beta), np.asarray(theta), np.asarray(phi)


# ---------------------------------------------------------------------------
# Quaternion -> SO(3) rotation matrix
# ---------------------------------------------------------------------------


def to_rotation_matrix(q: Vec4) -> npt.NDArray[np.floating[Any]]:
    """Convert a unit quaternion to a 3x3 SO(3) rotation matrix.

    Args:
        q: Unit quaternion, shape (4,) or (4, N).

    Returns:
        Rotation matrix, shape (3, 3) or (3, 3, N).
    """
    a, b, c, d = q[0], q[1], q[2], q[3]
    r11 = a**2 + b**2 - c**2 - d**2
    r12 = 2 * (b * c - a * d)
    r13 = 2 * (b * d + a * c)
    r21 = 2 * (b * c + a * d)
    r22 = a**2 - b**2 + c**2 - d**2
    r23 = 2 * (c * d - a * b)
    r31 = 2 * (b * d - a * c)
    r32 = 2 * (c * d + a * b)
    r33 = a**2 - b**2 - c**2 + d**2
    return np.array(
        [
            [r11, r12, r13],
            [r21, r22, r23],
            [r31, r32, r33],
        ]
    )


def to_rotation_matrix_dt(q: Vec4, q_dt: Vec4) -> npt.NDArray[np.floating[Any]]:
    """Time derivative of the SO(3) rotation matrix.

    Args:
        q: Unit quaternion, shape (4,) or (4, N).
        q_dt: Time derivative of q, same shape.

    Returns:
        dR/dt, shape (3, 3) or (3, 3, N).
    """
    a, b, c, d = q[0], q[1], q[2], q[3]
    a_dt, b_dt, c_dt, d_dt = q_dt[0], q_dt[1], q_dt[2], q_dt[3]
    r11 = 2 * a * a_dt + 2 * b * b_dt - 2 * c * c_dt - 2 * d * d_dt
    r12 = 2 * (b * c_dt + b_dt * c - a * d_dt - a_dt * d)
    r13 = 2 * (b * d_dt + b_dt * d + a * c_dt + a_dt * c)
    r21 = 2 * (b * c_dt + b_dt * c + a * d_dt + a_dt * d)
    r22 = 2 * (a * a_dt - b * b_dt + c * c_dt - d * d_dt)
    r23 = 2 * (c * d_dt + c_dt * d - a * b_dt - a_dt * b)
    r31 = 2 * (b * d_dt + b_dt * d - a * c_dt - a_dt * c)
    r32 = 2 * (c * d_dt + c_dt * d + a * b_dt + a_dt * b)
    r33 = 2 * (a * a_dt - b * b_dt - c * c_dt + d * d_dt)
    return np.array(
        [
            [r11, r12, r13],
            [r21, r22, r23],
            [r31, r32, r33],
        ]
    )
