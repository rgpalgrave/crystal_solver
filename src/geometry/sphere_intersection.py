"""
Core Geometry Engine for Sphere Intersections in 3D Space.

This module provides fundamental geometric calculations for finding
intersection points of spheres, which is central to predicting anion
positions in ionic crystal structures.

Mathematical Foundation:
    - CN-4: 4 spheres → 1 point (Voronoi vertex)
    - CN-3: 3 spheres → 2 points (above/below triangle plane)
    - CN-6: 6 spheres → octahedral center (overdetermined)

Units:
    All distances are in Angstroms (Å) unless otherwise specified.
    Coordinates are Cartesian unless labeled as 'fractional'.

Author: Crystal Structure Solver Project
"""

import numpy as np
from typing import Tuple, Optional


def is_collinear(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray,
                 tolerance: float = 1e-10) -> bool:
    """
    Check if three points are collinear (lie on the same line).

    Uses cross product magnitude normalized by edge lengths to be
    scale-invariant.

    Parameters
    ----------
    p1 : np.ndarray
        First point, shape (3,).
    p2 : np.ndarray
        Second point, shape (3,).
    p3 : np.ndarray
        Third point, shape (3,).
    tolerance : float, optional
        Relative tolerance for collinearity test.
        Default is 1e-10.

    Returns
    -------
    bool
        True if points are collinear within tolerance, False otherwise.

    Examples
    --------
    >>> p1 = np.array([0, 0, 0])
    >>> p2 = np.array([1, 0, 0])
    >>> p3 = np.array([2, 0, 0])
    >>> is_collinear(p1, p2, p3)
    True

    >>> p3 = np.array([1, 1, 0])
    >>> is_collinear(p1, p2, p3)
    False
    """
    v1 = np.asarray(p2, dtype=np.float64) - np.asarray(p1, dtype=np.float64)
    v2 = np.asarray(p3, dtype=np.float64) - np.asarray(p1, dtype=np.float64)
    cross = np.cross(v1, v2)
    cross_norm = np.linalg.norm(cross)
    
    # Use relative tolerance based on edge lengths for scale invariance
    v1_norm = np.linalg.norm(v1)
    v2_norm = np.linalg.norm(v2)
    scale = v1_norm * v2_norm
    
    if scale < 1e-30:
        # Degenerate case: at least one edge is essentially zero length
        return True
    
    # Compare cross product magnitude to product of edge lengths
    # For non-collinear points, |cross| = |v1| * |v2| * sin(angle)
    return cross_norm < tolerance * scale


def unit_normal_to_plane(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> Optional[np.ndarray]:
    """
    Calculate unit normal vector to plane defined by 3 points.

    The normal direction follows the right-hand rule with respect to
    the ordering p1 → p2 → p3.

    Parameters
    ----------
    p1 : np.ndarray
        First point, shape (3,).
    p2 : np.ndarray
        Second point, shape (3,).
    p3 : np.ndarray
        Third point, shape (3,).

    Returns
    -------
    np.ndarray or None
        Unit normal vector, shape (3,).
        Returns None if points are collinear (no unique plane).

    Examples
    --------
    >>> p1 = np.array([0, 0, 0])
    >>> p2 = np.array([1, 0, 0])
    >>> p3 = np.array([0, 1, 0])
    >>> normal = unit_normal_to_plane(p1, p2, p3)
    >>> np.allclose(normal, [0, 0, 1])
    True
    """
    v1 = np.asarray(p2, dtype=np.float64) - np.asarray(p1, dtype=np.float64)
    v2 = np.asarray(p3, dtype=np.float64) - np.asarray(p1, dtype=np.float64)
    cross = np.cross(v1, v2)
    norm = np.linalg.norm(cross)
    
    # Use relative tolerance based on edge lengths for scale invariance
    v1_norm = np.linalg.norm(v1)
    v2_norm = np.linalg.norm(v2)
    scale = v1_norm * v2_norm
    
    if scale < 1e-40:
        return None  # Degenerate: essentially zero-length edges
    
    if norm < 1e-10 * scale:
        return None  # Effectively collinear
    
    return cross / norm


def circumcenter_3d(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> Optional[np.ndarray]:
    """
    Calculate circumcenter of triangle formed by 3 points in 3D space.

    The circumcenter is equidistant from all three vertices and lies
    in the plane of the triangle.

    CRITICAL: Returns None if points are collinear (degenerate triangle).

    Parameters
    ----------
    p1 : np.ndarray
        First vertex of triangle, shape (3,).
    p2 : np.ndarray
        Second vertex of triangle, shape (3,).
    p3 : np.ndarray
        Third vertex of triangle, shape (3,).

    Returns
    -------
    np.ndarray or None
        Circumcenter coordinates, shape (3,).
        Returns None if points are collinear.

    Notes
    -----
    Algorithm:
        The circumcenter is found by solving for O = p1 + s*a + t*b where
        a = p2 - p1, b = p3 - p1, and O is equidistant from all vertices.
        
        This yields the 2x2 system:
        [a·a  a·b] [s]   [|a|²/2]
        [a·b  b·b] [t] = [|b|²/2]

    Examples
    --------
    >>> # Equilateral triangle centered at origin
    >>> p1 = np.array([1, 0, 0])
    >>> p2 = np.array([-0.5, np.sqrt(3)/2, 0])
    >>> p3 = np.array([-0.5, -np.sqrt(3)/2, 0])
    >>> center = circumcenter_3d(p1, p2, p3)
    >>> np.allclose(center, [0, 0, 0], atol=1e-10)
    True
    """
    p1 = np.asarray(p1, dtype=np.float64)
    p2 = np.asarray(p2, dtype=np.float64)
    p3 = np.asarray(p3, dtype=np.float64)
    
    # Check for collinearity first
    if is_collinear(p1, p2, p3):
        return None
    
    # Vectors from p1
    a = p2 - p1
    b = p3 - p1
    
    # Dot products
    a_dot_a = np.dot(a, a)
    b_dot_b = np.dot(b, b)
    a_dot_b = np.dot(a, b)
    
    # Build and solve the 2x2 system:
    # [a·a  a·b] [s]   [|a|²/2]
    # [a·b  b·b] [t] = [|b|²/2]
    det = a_dot_a * b_dot_b - a_dot_b * a_dot_b
    
    # Scale-invariant tolerance: det should be compared relative to the scale
    # det has units of length^4, so tolerance should scale with (a_dot_a * b_dot_b)
    scale_factor = a_dot_a * b_dot_b
    if scale_factor < 1e-40:
        return None  # Degenerate: effectively zero-length edges
    
    if np.abs(det) < 1e-10 * scale_factor:
        return None  # Effectively singular (collinear or nearly so)
    
    s = (b_dot_b * a_dot_a / 2 - a_dot_b * b_dot_b / 2) / det
    t = (a_dot_a * b_dot_b / 2 - a_dot_b * a_dot_a / 2) / det
    
    circumcenter = p1 + s * a + t * b
    
    return circumcenter


def solve_3_sphere_intersection(M1: np.ndarray, M2: np.ndarray, M3: np.ndarray,
                                 R: float) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Find both intersection points of 3 spheres of equal radius R.

    Given three sphere centers and a common radius, finds the two points
    (if they exist) that lie on all three spheres simultaneously.

    CRITICAL FIRST STEP: Check if M1, M2, M3 are collinear.
    If collinear, circumcenter is undefined → return (None, None)

    Parameters
    ----------
    M1 : np.ndarray
        Center of first sphere, shape (3,).
    M2 : np.ndarray
        Center of second sphere, shape (3,).
    M3 : np.ndarray
        Center of third sphere, shape (3,).
    R : float
        Common radius of all three spheres (in Å).

    Returns
    -------
    tuple of (np.ndarray or None, np.ndarray or None)
        (solution1, solution2): Both points where spheres intersect.
        (None, None): If no valid intersection or collinear points.

    Notes
    -----
    Algorithm:
        1. Check collinearity: ||cross(M2-M1, M3-M1)|| > 1e-10
        2. Calculate circumcenter of triangle M1-M2-M3
        3. Calculate height: h = sqrt(R² - circumradius²)
        4. Return: circumcenter ± h × normal

    The two solutions are symmetric about the plane containing the
    three sphere centers. For crystal structure applications, typically
    only one solution lies within the unit cell or has physical meaning.

    Examples
    --------
    >>> M1 = np.array([0, 0, 0])
    >>> M2 = np.array([1, 0, 0])
    >>> M3 = np.array([0.5, np.sqrt(3)/2, 0])
    >>> R = 0.6
    >>> sol1, sol2 = solve_3_sphere_intersection(M1, M2, M3, R)
    >>> # Both solutions at distance R from all centers
    >>> np.isclose(np.linalg.norm(sol1 - M1), R)
    True
    """
    M1 = np.asarray(M1, dtype=np.float64)
    M2 = np.asarray(M2, dtype=np.float64)
    M3 = np.asarray(M3, dtype=np.float64)
    
    # CRITICAL: Check collinearity first
    if is_collinear(M1, M2, M3):
        return (None, None)
    
    # Negative radius is invalid
    if R < 0:
        return (None, None)
    
    # Find circumcenter of the triangle
    circumcenter = circumcenter_3d(M1, M2, M3)
    if circumcenter is None:
        return (None, None)
    
    # Calculate circumradius (distance from circumcenter to any vertex)
    circumradius = np.linalg.norm(M1 - circumcenter)
    
    # Check if R is large enough for intersection
    # R must be >= circumradius for real solutions
    R_squared = R * R
    circumradius_squared = circumradius * circumradius
    
    if R_squared < circumradius_squared - 1e-10:
        # R too small - no intersection
        return (None, None)
    
    # Handle edge case where R ≈ circumradius (tangent case)
    h_squared = R_squared - circumradius_squared
    if h_squared < 0:
        h_squared = 0  # Numerical cleanup
    
    h = np.sqrt(h_squared)
    
    # Get unit normal to the plane
    normal = unit_normal_to_plane(M1, M2, M3)
    if normal is None:
        return (None, None)
    
    # Two solutions: above and below the plane
    solution1 = circumcenter + h * normal
    solution2 = circumcenter - h * normal
    
    return (solution1, solution2)


def solve_4_sphere_intersection(M1: np.ndarray, M2: np.ndarray, M3: np.ndarray,
                                 M4: np.ndarray, R: float) -> Optional[np.ndarray]:
    """
    Find intersection point of 4 spheres of equal radius (Voronoi vertex).

    This is the standard case for CN-4 tetrahedral coordination in crystals.
    Four spheres generically intersect at either 0 or 2 points; this function
    returns the intersection point that is valid for the given radius.

    Parameters
    ----------
    M1 : np.ndarray
        Center of first sphere, shape (3,).
    M2 : np.ndarray
        Center of second sphere, shape (3,).
    M3 : np.ndarray
        Center of third sphere, shape (3,).
    M4 : np.ndarray
        Center of fourth sphere, shape (3,).
    R : float
        Common radius of all four spheres (in Å).

    Returns
    -------
    np.ndarray or None
        The unique intersection point, shape (3,).
        None if no solution exists or system is degenerate.

    Notes
    -----
    Algorithm:
        Solves the 3×3 linear system from radical plane equations:
        
        2(M₂-M₁)·x = ||M₂||² - ||M₁||²
        2(M₃-M₁)·x = ||M₃||² - ||M₁||²
        2(M₄-M₁)·x = ||M₄||² - ||M₁||²

        This is derived from the condition ||x - Mᵢ|| = R for all i,
        which when squared and subtracted pairwise gives linear equations.

    For equal radii, the solution is the circumcenter of the tetrahedron,
    which is equidistant from all four vertices.

    Examples
    --------
    >>> # Regular tetrahedron
    >>> M1 = np.array([0, 0, 0])
    >>> M2 = np.array([1, 0, 0])
    >>> M3 = np.array([0.5, np.sqrt(3)/2, 0])
    >>> M4 = np.array([0.5, np.sqrt(3)/6, np.sqrt(2/3)])
    >>> R = 0.6
    >>> center = solve_4_sphere_intersection(M1, M2, M3, M4, R)
    >>> # center is equidistant from all vertices
    """
    M1 = np.asarray(M1, dtype=np.float64)
    M2 = np.asarray(M2, dtype=np.float64)
    M3 = np.asarray(M3, dtype=np.float64)
    M4 = np.asarray(M4, dtype=np.float64)
    
    if R < 0:
        return None
    
    # Build the linear system Ax = b
    # Row i: 2(M_{i+1} - M1) · x = ||M_{i+1}||² - ||M1||²
    A = np.zeros((3, 3))
    b = np.zeros(3)
    
    M1_sq = np.dot(M1, M1)
    
    for i, M in enumerate([M2, M3, M4]):
        A[i, :] = 2 * (M - M1)
        b[i] = np.dot(M, M) - M1_sq
    
    # Check if the system is solvable (non-degenerate)
    det = np.linalg.det(A)
    if np.abs(det) < 1e-10:
        # Degenerate configuration (e.g., coplanar centers)
        return None
    
    # Solve the linear system
    try:
        solution = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        return None
    
    # Verify that the solution is actually at distance R from all centers
    # (the linear system finds the circumcenter; check if R matches)
    distances = [np.linalg.norm(solution - M) for M in [M1, M2, M3, M4]]
    
    # All distances should be equal (circumcenter property)
    circumradius = distances[0]
    if not all(np.isclose(d, circumradius, rtol=1e-6) for d in distances):
        # System solved but not a valid circumcenter (numerical issues)
        return None
    
    # Check if the requested R matches the actual circumradius
    # Allow some tolerance for valid solutions
    if not np.isclose(circumradius, R, rtol=0.1, atol=0.05):
        # R doesn't match the circumradius - no valid intersection at this R
        return None
    
    return solution


def circumradius_of_tetrahedron(M1: np.ndarray, M2: np.ndarray, 
                                 M3: np.ndarray, M4: np.ndarray) -> Optional[float]:
    """
    Calculate the circumradius of a tetrahedron (distance from circumcenter to vertices).

    This is a utility function to find the natural R value for a given
    tetrahedral configuration.

    Parameters
    ----------
    M1, M2, M3, M4 : np.ndarray
        Vertices of the tetrahedron, each shape (3,).

    Returns
    -------
    float or None
        Circumradius of the tetrahedron (in same units as input).
        None if the tetrahedron is degenerate.

    Examples
    --------
    >>> # Regular tetrahedron with edge length 1
    >>> M1 = np.array([0, 0, 0])
    >>> M2 = np.array([1, 0, 0])
    >>> M3 = np.array([0.5, np.sqrt(3)/2, 0])
    >>> M4 = np.array([0.5, np.sqrt(3)/6, np.sqrt(2/3)])
    >>> r = circumradius_of_tetrahedron(M1, M2, M3, M4)
    >>> np.isclose(r, np.sqrt(3/8))  # Known result for regular tetrahedron
    True
    """
    M1 = np.asarray(M1, dtype=np.float64)
    M2 = np.asarray(M2, dtype=np.float64)
    M3 = np.asarray(M3, dtype=np.float64)
    M4 = np.asarray(M4, dtype=np.float64)
    
    # Build the linear system to find circumcenter
    A = np.zeros((3, 3))
    b = np.zeros(3)
    
    M1_sq = np.dot(M1, M1)
    
    for i, M in enumerate([M2, M3, M4]):
        A[i, :] = 2 * (M - M1)
        b[i] = np.dot(M, M) - M1_sq
    
    # Check if solvable
    det = np.linalg.det(A)
    if np.abs(det) < 1e-10:
        return None
    
    try:
        circumcenter = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        return None
    
    return np.linalg.norm(circumcenter - M1)


def find_circumcenter_of_tetrahedron(M1: np.ndarray, M2: np.ndarray,
                                      M3: np.ndarray, M4: np.ndarray) -> Optional[np.ndarray]:
    """
    Find the circumcenter of a tetrahedron (equidistant from all 4 vertices).

    Parameters
    ----------
    M1, M2, M3, M4 : np.ndarray
        Vertices of the tetrahedron, each shape (3,).

    Returns
    -------
    np.ndarray or None
        Circumcenter coordinates, shape (3,).
        None if the tetrahedron is degenerate.
    """
    M1 = np.asarray(M1, dtype=np.float64)
    M2 = np.asarray(M2, dtype=np.float64)
    M3 = np.asarray(M3, dtype=np.float64)
    M4 = np.asarray(M4, dtype=np.float64)
    
    A = np.zeros((3, 3))
    b = np.zeros(3)
    
    M1_sq = np.dot(M1, M1)
    
    for i, M in enumerate([M2, M3, M4]):
        A[i, :] = 2 * (M - M1)
        b[i] = np.dot(M, M) - M1_sq
    
    det = np.linalg.det(A)
    if np.abs(det) < 1e-10:
        return None
    
    try:
        return np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        return None
