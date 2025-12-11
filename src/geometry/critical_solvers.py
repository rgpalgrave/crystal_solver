"""
Critical Parameter Solvers for Analytical Crystal Structure Prediction.

This module provides analytical solvers for finding critical coordination
sphere radii and lattice parameters where coordination transitions occur.

The key innovation is solving constraint equations directly rather than
parameter scanning, which provides exact solutions and reveals the
mathematical structure of coordination transitions.

Mathematical Foundation:
    For a CN-3 site transitioning to CN-4, we solve:
    ||position(R) - M4|| = (1 + gap) × R
    
    Where position(R) = circumcenter + sqrt(R² - r_circ²) × normal
    
    This becomes a quadratic equation in R² with analytical solutions.

Units:
    All distances are in Angstroms (Å) unless otherwise specified.
    Gap thresholds are relative (dimensionless).

Author: Crystal Structure Solver Project
"""

import numpy as np
from typing import Optional, List, Tuple, Callable, Dict, Any
from .sphere_intersection import (
    circumcenter_3d,
    unit_normal_to_plane,
    is_collinear,
    solve_3_sphere_intersection,
)


def solve_critical_radius(M1: np.ndarray, M2: np.ndarray, M3: np.ndarray,
                          M4_constraint: np.ndarray,
                          M5_next_shell: Optional[np.ndarray] = None,
                          gap_threshold: float = 0.15) -> Optional[Dict[str, Any]]:
    """
    Solve for critical R where CN transitions from 3 to 4.

    This function analytically solves for the coordination sphere radius R
    at which a 4th neighbor enters the coordination shell with exactly
    the specified gap threshold.

    Mathematical Constraint:
        ||position(R) - M4|| = (1 + gap_threshold) × R

    Where position(R) = circumcenter ± sqrt(R² - r_circ²) × normal

    Parameters
    ----------
    M1, M2, M3 : np.ndarray
        Triangle of coordinating atoms, each shape (3,).
    M4_constraint : np.ndarray
        4th neighbor defining the constraint, shape (3,).
    M5_next_shell : np.ndarray, optional
        5th neighbor for stability validation, shape (3,).
    gap_threshold : float, optional
        Minimum relative gap between 3rd and 4th neighbor.
        Default is 0.15 (15%).

    Returns
    -------
    dict or None
        Dictionary containing:
        - 'R_critical': float - Solved radius
        - 'position': np.ndarray - Anion position at R_critical
        - 'gap_to_4th': float - By construction equals gap_threshold
        - 'gap_to_5th': float - Calculated if M5 provided, else None
        - 'stability_metric': float - gap_to_5th / gap_threshold if M5 provided
        - 'is_stable': bool - True if gap_to_5th > 0.12
        
        Returns None if no valid solution exists.

    Notes
    -----
    **Mathematical Derivation:**
    
    Position as function of R (two cases for ± h):
        x±(R) = C ± h·n̂  where h = sqrt(R² - r²), C = circumcenter, r = circumradius
    
    Constraint (M4 at distance (1+g)R from position):
        ||x±(R) - M4|| = (1+g)R
    
    Let Δ = M4 - C, v = Δ·n̂:
        Case +: ||h·n̂ - Δ||² = (1+g)²R²  →  h² - 2hv + ||Δ||² = (1+g)²R²
        Case -: ||-h·n̂ - Δ||² = (1+g)²R²  →  h² + 2hv + ||Δ||² = (1+g)²R²
    
    With h² = R² - r², these become:
        Case +: 2·sqrt(R²-r²)·v = Δ_r_sq - g_factor·R²
        Case -: 2·sqrt(R²-r²)·v = g_factor·R² - Δ_r_sq
    
    where Δ_r_sq = ||Δ||² - r², g_factor = (1+g)² - 1
    
    Squaring either case gives:
        4v²(R² - r²) = (Δ_r_sq - g_factor·R²)²
    
    Expanding to quadratic in u = R²:
        g_factor²·u² - (2·g_factor·Δ_r_sq + 4v²)·u + (Δ_r_sq² + 4v²r²) = 0
    """
    M1 = np.asarray(M1, dtype=np.float64)
    M2 = np.asarray(M2, dtype=np.float64)
    M3 = np.asarray(M3, dtype=np.float64)
    M4 = np.asarray(M4_constraint, dtype=np.float64)

    # Check for degenerate triangle
    if is_collinear(M1, M2, M3):
        return None

    # Calculate circumcenter and circumradius
    C = circumcenter_3d(M1, M2, M3)
    if C is None:
        return None

    r = np.linalg.norm(M1 - C)  # circumradius
    r_sq = r * r

    # Get unit normal to the triangle plane
    n_hat = unit_normal_to_plane(M1, M2, M3)
    if n_hat is None:
        return None

    # Vector from circumcenter to M4
    Delta = M4 - C
    Delta_sq = np.dot(Delta, Delta)
    
    # Projection of Delta onto normal
    v = np.dot(Delta, n_hat)

    # Gap factor
    g = gap_threshold
    g_factor = (1 + g) ** 2 - 1  # = 2g + g²

    # Key quantities for quadratic
    Delta_r_sq = Delta_sq - r_sq  # ||Δ||² - r²

    # Quadratic coefficients: A·u² + B·u + C = 0 where u = R²
    # From: 4v²(u - r²) = (Δ_r_sq - g_factor·u)²
    A = g_factor * g_factor - 4 * v * v
    B = -2 * g_factor * Delta_r_sq + 4 * v * v * r_sq / r_sq * 2 * r_sq  # Simplify
    
    # Let me redo this expansion more carefully:
    # 4v²u - 4v²r² = Δ_r_sq² - 2·g_factor·Δ_r_sq·u + g_factor²·u²
    # g_factor²·u² - (2·g_factor·Δ_r_sq + 4v²)·u + (Δ_r_sq² + 4v²r²) = 0
    
    A = g_factor * g_factor
    B = -(2 * g_factor * Delta_r_sq + 4 * v * v)
    C_coef = Delta_r_sq * Delta_r_sq + 4 * v * v * r_sq

    valid_solutions = []

    # Handle special case where A ≈ 0
    if np.abs(A) < 1e-12:
        if np.abs(B) < 1e-12:
            # Degenerate - check if C_coef is also ~0
            if np.abs(C_coef) < 1e-12:
                # Any R works - degenerate, skip
                pass
            return None
        u = -C_coef / B
        if u >= r_sq - 1e-10:
            u_candidates = [u]
        else:
            u_candidates = []
    else:
        # Solve quadratic
        discriminant = B * B - 4 * A * C_coef

        if discriminant < -1e-10:
            return None

        if discriminant < 0:
            discriminant = 0

        sqrt_disc = np.sqrt(discriminant)
        u1 = (-B + sqrt_disc) / (2 * A)
        u2 = (-B - sqrt_disc) / (2 * A)
        u_candidates = [u1, u2]

    # Check each candidate solution
    for u in u_candidates:
        if u < r_sq - 1e-10:
            continue  # R must be >= circumradius
        
        R = np.sqrt(max(u, r_sq))
        h = np.sqrt(max(R * R - r_sq, 0))
        
        # Check both sides of the plane
        for sign in [1, -1]:
            pos = C + sign * h * n_hat
            d4 = np.linalg.norm(pos - M4)
            expected_d4 = (1 + g) * R
            
            # Check if this is a valid solution (not an extraneous root from squaring)
            if np.isclose(d4, expected_d4, rtol=0.02):
                # Also verify that we have the sign constraint satisfied
                # Case +: 2hv = Δ_r_sq - g_factor·R² (need same sign)
                # Case -: 2hv = g_factor·R² - Δ_r_sq (need same sign)
                lhs = 2 * h * v
                if sign == 1:
                    rhs = Delta_r_sq - g_factor * R * R
                else:
                    rhs = g_factor * R * R - Delta_r_sq
                
                # Check sign compatibility (both same sign or both ~0)
                if (lhs * rhs >= -1e-10) or np.abs(lhs) < 1e-10 or np.abs(rhs) < 1e-10:
                    valid_solutions.append({
                        'R': R,
                        'position': pos.copy(),
                        'd4': d4,
                        'sign': sign
                    })

    if not valid_solutions:
        return None

    # Select the smallest valid R (most physically meaningful)
    valid_solutions.sort(key=lambda x: x['R'])
    best = valid_solutions[0]
    
    R_critical = best['R']
    position = best['position']
    d4_actual = best['d4']

    # Calculate actual gap to 4th
    gap_to_4th = d4_actual / R_critical - 1

    # Build result dictionary
    result = {
        'R_critical': R_critical,
        'position': position,
        'gap_to_4th': gap_to_4th,
        'gap_to_5th': None,
        'stability_metric': None,
        'is_stable': True  # Default, updated if M5 provided
    }

    # Calculate gap to 5th neighbor if provided
    if M5_next_shell is not None:
        M5 = np.asarray(M5_next_shell, dtype=np.float64)
        d5 = np.linalg.norm(position - M5)
        gap_to_5th = d5 / R_critical - 1
        result['gap_to_5th'] = gap_to_5th
        result['stability_metric'] = gap_to_5th / gap_threshold if gap_threshold > 0 else np.inf
        result['is_stable'] = gap_to_5th > 0.12

    return result


def solve_critical_radius_both_sides(M1: np.ndarray, M2: np.ndarray, M3: np.ndarray,
                                      M4_constraint: np.ndarray,
                                      gap_threshold: float = 0.15) -> List[Dict[str, Any]]:
    """
    Solve for critical R considering both sides of the triangle plane.

    This is useful when searching for all possible CN-3 sites, as anions
    can be positioned either above or below the metal triangle.

    Parameters
    ----------
    M1, M2, M3 : np.ndarray
        Triangle of coordinating atoms, each shape (3,).
    M4_constraint : np.ndarray
        4th neighbor defining the constraint, shape (3,).
    gap_threshold : float, optional
        Minimum relative gap between 3rd and 4th neighbor.
        Default is 0.15 (15%).

    Returns
    -------
    list of dict
        List of valid solutions (0, 1, or 2), each containing:
        - 'R_critical': float
        - 'position': np.ndarray
        - 'gap_to_4th': float
        - 'side': str ('above' or 'below')
    """
    M1 = np.asarray(M1, dtype=np.float64)
    M2 = np.asarray(M2, dtype=np.float64)
    M3 = np.asarray(M3, dtype=np.float64)
    M4 = np.asarray(M4_constraint, dtype=np.float64)

    if is_collinear(M1, M2, M3):
        return []

    C = circumcenter_3d(M1, M2, M3)
    if C is None:
        return []

    r = np.linalg.norm(M1 - C)
    r_sq = r * r

    n_hat = unit_normal_to_plane(M1, M2, M3)
    if n_hat is None:
        return []

    Delta = M4 - C
    Delta_sq = np.dot(Delta, Delta)
    v = np.dot(Delta, n_hat)

    g = gap_threshold
    g_factor = (1 + g) ** 2 - 1

    Delta_r_sq = Delta_sq - r_sq

    A = g_factor * g_factor - 4 * v * v
    B = -2 * g_factor * Delta_r_sq - 4 * v * v
    C_coef = Delta_r_sq * Delta_r_sq + 4 * v * v * r_sq

    solutions = []

    if np.abs(A) < 1e-12:
        if np.abs(B) < 1e-12:
            return []
        u = -C_coef / B
        u_candidates = [u]
    else:
        discriminant = B * B - 4 * A * C_coef
        if discriminant < -1e-12:
            return []
        if discriminant < 0:
            discriminant = 0
        sqrt_disc = np.sqrt(discriminant)
        u_candidates = [(-B + sqrt_disc) / (2 * A), (-B - sqrt_disc) / (2 * A)]

    for u in u_candidates:
        if u < r_sq - 1e-10:
            continue
        R = np.sqrt(max(u, r_sq))
        h = np.sqrt(max(R * R - r_sq, 0))

        for sign, side_name in [(1, 'above'), (-1, 'below')]:
            pos = C + sign * h * n_hat
            d4 = np.linalg.norm(pos - M4)
            expected_d4 = (1 + g) * R

            if np.isclose(d4, expected_d4, rtol=0.02):
                gap_actual = d4 / R - 1
                solutions.append({
                    'R_critical': R,
                    'position': pos,
                    'gap_to_4th': gap_actual,
                    'side': side_name
                })

    # Remove duplicates (same position)
    unique_solutions = []
    for sol in solutions:
        is_duplicate = False
        for existing in unique_solutions:
            if np.allclose(sol['position'], existing['position'], atol=1e-6):
                is_duplicate = True
                break
        if not is_duplicate:
            unique_solutions.append(sol)

    return unique_solutions


def verify_coordination_shell(position: np.ndarray, all_metals: np.ndarray,
                               R: float, expected_CN: int,
                               tolerance: float = 0.15) -> Dict[str, Any]:
    """
    Verify that position has expected coordination number with clear shell separation.

    This function validates that a proposed anion position has the expected
    number of coordinating metal atoms at approximately distance R, with
    a clear gap to the next coordination shell.

    Parameters
    ----------
    position : np.ndarray
        Anion position to validate, shape (3,).
    all_metals : np.ndarray
        All metal atom positions, shape (N, 3).
    R : float
        Expected coordination distance (in Å).
    expected_CN : int
        Expected coordination number (typically 3, 4, or 6).
    tolerance : float, optional
        Relative tolerance for distance matching (default 0.15 = 15%).

    Returns
    -------
    dict
        Dictionary containing:
        - 'actual_CN': int - Number of atoms at approximately R
        - 'gap_to_next': float - Relative gap (d_next - d_coord) / d_coord
        - 'is_valid': bool - True if actual_CN == expected_CN
        - 'is_well_separated': bool - True if gap_to_next > 0.12
        - 'nearest_distances': list - First expected_CN+2 distances
        - 'coordination_distances': list - Distances to coordinating atoms

    Examples
    --------
    >>> metals = np.array([[1, 0, 0], [-1, 0, 0], [0, 1, 0], 
    ...                    [0, -1, 0], [0, 0, 1], [0, 0, -1]])
    >>> position = np.array([0, 0, 0])
    >>> result = verify_coordination_shell(position, metals, R=1.0, expected_CN=6)
    >>> result['actual_CN']
    6
    >>> result['is_valid']
    True
    """
    position = np.asarray(position, dtype=np.float64)
    all_metals = np.asarray(all_metals, dtype=np.float64)

    if all_metals.ndim == 1:
        all_metals = all_metals.reshape(1, -1)

    # Calculate all distances
    distances = np.linalg.norm(all_metals - position, axis=1)

    # Sort distances
    sorted_distances = np.sort(distances)

    # Count atoms within tolerance of R
    R_min = R * (1 - tolerance)
    R_max = R * (1 + tolerance)
    coordination_mask = (distances >= R_min) & (distances <= R_max)
    actual_CN = np.sum(coordination_mask)
    coordination_distances = sorted(distances[coordination_mask].tolist())

    # Calculate gap to next shell
    if actual_CN > 0 and actual_CN < len(sorted_distances):
        # Average distance of coordinating atoms
        coord_avg = np.mean(coordination_distances)
        # Distance to next atom after coordination shell
        next_idx = actual_CN
        if next_idx < len(sorted_distances):
            d_next = sorted_distances[next_idx]
            gap_to_next = (d_next - coord_avg) / coord_avg
        else:
            gap_to_next = np.inf  # No atoms beyond coordination shell
    else:
        gap_to_next = 0.0 if actual_CN == 0 else np.inf

    # Get nearest distances for reporting
    n_report = min(expected_CN + 2, len(sorted_distances))
    nearest_distances = sorted_distances[:n_report].tolist()

    return {
        'actual_CN': int(actual_CN),
        'gap_to_next': float(gap_to_next),
        'is_valid': actual_CN == expected_CN,
        'is_well_separated': gap_to_next > 0.12,
        'nearest_distances': nearest_distances,
        'coordination_distances': coordination_distances
    }


def find_gap_to_neighbor(position: np.ndarray, neighbor: np.ndarray,
                          R_reference: float) -> float:
    """
    Calculate the relative gap between reference distance and a neighbor.

    Parameters
    ----------
    position : np.ndarray
        Reference position, shape (3,).
    neighbor : np.ndarray
        Neighbor position, shape (3,).
    R_reference : float
        Reference coordination distance.

    Returns
    -------
    float
        Relative gap: (d_neighbor - R_reference) / R_reference
    """
    position = np.asarray(position, dtype=np.float64)
    neighbor = np.asarray(neighbor, dtype=np.float64)

    d = np.linalg.norm(position - neighbor)
    return (d - R_reference) / R_reference


def solve_critical_ca_ratio(lattice_generator: Callable[[float], Tuple[np.ndarray, ...]],
                             R_target: float,
                             ca_range: Tuple[float, float] = (0.4, 1.0),
                             precision: float = 0.001) -> Optional[float]:
    """
    For parameterized lattices, solve for critical c/a ratio.

    This function finds the c/a ratio at which a particular coordination
    site becomes accessible for a given target radius. Uses bisection
    for robustness with numerical lattice generation.

    Parameters
    ----------
    lattice_generator : callable
        Function that takes c/a ratio and returns (M1, M2, M3, M4) as
        numpy arrays defining a triangle and constraint atom.
    R_target : float
        Target coordination sphere radius.
    ca_range : tuple of float, optional
        Range of c/a ratios to search. Default is (0.4, 1.0).
    precision : float, optional
        Precision for c/a ratio. Default is 0.001.

    Returns
    -------
    float or None
        c/a ratio where site becomes accessible at R_target, or None
        if no solution exists in the given range.

    Examples
    --------
    >>> def rutile_triangle(ca):
    ...     a = 1.0
    ...     c = ca * a
    ...     M1 = np.array([0, 0, 0])
    ...     M2 = np.array([a, 0, 0])
    ...     M3 = np.array([0.5*a, 0.5*a, 0.5*c])
    ...     M4 = np.array([0.5*a, 0.5*a, -0.5*c])
    ...     return M1, M2, M3, M4
    >>> ca_crit = solve_critical_ca_ratio(rutile_triangle, R_target=0.42)
    """
    ca_low, ca_high = ca_range

    def get_R_critical(ca: float) -> Optional[float]:
        """Get R_critical for a given c/a ratio."""
        try:
            M1, M2, M3, M4 = lattice_generator(ca)
            result = solve_critical_radius(M1, M2, M3, M4, gap_threshold=0.15)
            if result is not None:
                return result['R_critical']
        except Exception:
            pass
        return None

    # Check endpoints
    R_low = get_R_critical(ca_low)
    R_high = get_R_critical(ca_high)

    if R_low is None and R_high is None:
        return None

    # Simple bisection search
    # We're looking for ca where R_critical ≈ R_target
    for _ in range(50):  # Max iterations
        ca_mid = (ca_low + ca_high) / 2
        R_mid = get_R_critical(ca_mid)

        if R_mid is None:
            # Try to find valid region
            if R_low is not None:
                ca_high = ca_mid
            elif R_high is not None:
                ca_low = ca_mid
            else:
                return None
            continue

        if np.isclose(R_mid, R_target, rtol=precision):
            return ca_mid

        # Determine search direction based on monotonicity
        if R_low is not None and R_high is not None:
            if (R_low < R_high and R_mid < R_target) or \
               (R_low > R_high and R_mid > R_target):
                ca_low = ca_mid
                R_low = R_mid
            else:
                ca_high = ca_mid
                R_high = R_mid
        elif R_low is not None:
            if R_mid < R_target:
                ca_low = ca_mid
                R_low = R_mid
            else:
                ca_high = ca_mid
                R_high = R_mid
        else:
            if R_mid > R_target:
                ca_low = ca_mid
                R_low = R_mid
            else:
                ca_high = ca_mid
                R_high = R_mid

        if ca_high - ca_low < precision:
            return ca_mid

    return (ca_low + ca_high) / 2


def analyze_coordination_transition(M1: np.ndarray, M2: np.ndarray, M3: np.ndarray,
                                     neighbors: np.ndarray,
                                     R_range: Tuple[float, float] = (0.3, 0.8),
                                     gap_threshold: float = 0.15) -> List[Dict[str, Any]]:
    """
    Analyze all coordination transitions for a triangle as R varies.

    This function identifies all critical R values where the coordination
    number of the site changes due to additional neighbors entering
    the coordination shell.

    Parameters
    ----------
    M1, M2, M3 : np.ndarray
        Triangle of coordinating atoms, each shape (3,).
    neighbors : np.ndarray
        Array of potential neighbor positions, shape (N, 3).
    R_range : tuple of float, optional
        Range of R values to consider. Default is (0.3, 0.8).
    gap_threshold : float, optional
        Gap threshold for coordination transitions. Default is 0.15.

    Returns
    -------
    list of dict
        List of transition points, each containing:
        - 'R_critical': float
        - 'position': np.ndarray
        - 'CN_before': int (coordination number before transition)
        - 'CN_after': int (coordination number after transition)
        - 'entering_neighbor': np.ndarray
    """
    M1 = np.asarray(M1, dtype=np.float64)
    M2 = np.asarray(M2, dtype=np.float64)
    M3 = np.asarray(M3, dtype=np.float64)
    neighbors = np.asarray(neighbors, dtype=np.float64)

    if neighbors.ndim == 1:
        neighbors = neighbors.reshape(1, -1)

    transitions = []

    # For each potential 4th neighbor, find critical R
    for i, M4 in enumerate(neighbors):
        # Skip if M4 is one of the triangle vertices
        if (np.allclose(M4, M1) or np.allclose(M4, M2) or np.allclose(M4, M3)):
            continue

        result = solve_critical_radius(M1, M2, M3, M4, gap_threshold=gap_threshold)

        if result is not None:
            R_crit = result['R_critical']
            if R_range[0] <= R_crit <= R_range[1]:
                transitions.append({
                    'R_critical': R_crit,
                    'position': result['position'],
                    'CN_before': 3,
                    'CN_after': 4,
                    'entering_neighbor': M4,
                    'gap_to_4th': result['gap_to_4th']
                })

    # Sort by R_critical
    transitions.sort(key=lambda x: x['R_critical'])

    return transitions
