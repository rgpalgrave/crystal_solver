"""
CN-6 Site Finder for Octahedral Coordination.

This module finds all octahedral (CN-6) coordination sites analytically
using a simplified approach based on the key insight that CN-6 octahedral
centers are still Voronoi vertices - determined by any 4 of the 6 neighbors,
with the remaining 2 serving as validation constraints.

Mathematical Foundation:
    - Octahedral center is equidistant from all 6 vertices
    - Pick best 4 of 6 vertices to define a tetrahedron
    - Solve for circumcenter using 4-sphere intersection
    - Validate: remaining 2 vertices must also be at distance R

This approach is superior to least-squares because:
    1. No numerical optimization required
    2. Purely analytical (linear algebra only)
    3. Faster computation
    4. Clearer geometric interpretation

Units:
    All distances are in Angstroms (Å) unless otherwise specified.
    Coordinates are Cartesian unless labeled as 'fractional'.

Author: Crystal Structure Solver Project
"""

import numpy as np
from typing import Tuple, Optional, List, Iterator, Dict, Any, TYPE_CHECKING
from itertools import combinations

if TYPE_CHECKING:
    from bravais import BravaisLattice


# Import from local modules - adjust path as needed for your project structure
try:
    from sphere_intersection import (
        find_circumcenter_of_tetrahedron,
        circumradius_of_tetrahedron,
        is_collinear,
    )
except ImportError:
    # Fallback: inline implementations if modules not available
    pass


def is_coplanar(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray, 
                p4: np.ndarray, tolerance: float = 1e-10) -> bool:
    """
    Check if four points are coplanar (lie on the same plane).
    
    Uses the scalar triple product: if (p2-p1) · [(p3-p1) × (p4-p1)] ≈ 0,
    the points are coplanar.
    
    Parameters
    ----------
    p1, p2, p3, p4 : np.ndarray
        Four points to check, each shape (3,).
    tolerance : float, optional
        Relative tolerance for coplanarity test.
        
    Returns
    -------
    bool
        True if points are coplanar within tolerance.
    """
    v1 = np.asarray(p2, dtype=np.float64) - np.asarray(p1, dtype=np.float64)
    v2 = np.asarray(p3, dtype=np.float64) - np.asarray(p1, dtype=np.float64)
    v3 = np.asarray(p4, dtype=np.float64) - np.asarray(p1, dtype=np.float64)
    
    # Scalar triple product
    triple = np.dot(v1, np.cross(v2, v3))
    
    # Normalize by characteristic length scale for scale invariance
    scale = np.linalg.norm(v1) * np.linalg.norm(v2) * np.linalg.norm(v3)
    
    if scale < 1e-30:
        return True  # Degenerate
        
    return np.abs(triple) < tolerance * scale


def find_tetrahedron_circumcenter(M1: np.ndarray, M2: np.ndarray,
                                   M3: np.ndarray, M4: np.ndarray) -> Optional[np.ndarray]:
    """
    Find circumcenter of tetrahedron (equidistant from all 4 vertices).
    
    Solves the linear system from the equal-distance conditions:
        ||x - M1|| = ||x - M2|| = ||x - M3|| = ||x - M4||
    
    Parameters
    ----------
    M1, M2, M3, M4 : np.ndarray
        Tetrahedron vertices, each shape (3,).
        
    Returns
    -------
    np.ndarray or None
        Circumcenter coordinates, shape (3,).
        None if tetrahedron is degenerate (coplanar vertices).
    """
    M1 = np.asarray(M1, dtype=np.float64)
    M2 = np.asarray(M2, dtype=np.float64)
    M3 = np.asarray(M3, dtype=np.float64)
    M4 = np.asarray(M4, dtype=np.float64)
    
    # Check for coplanarity (degenerate tetrahedron)
    if is_coplanar(M1, M2, M3, M4):
        return None
    
    # Build the linear system Ax = b
    # Row i: 2(M_{i+1} - M1) · x = ||M_{i+1}||² - ||M1||²
    A = np.zeros((3, 3))
    b = np.zeros(3)
    
    M1_sq = np.dot(M1, M1)
    
    for i, M in enumerate([M2, M3, M4]):
        A[i, :] = 2 * (M - M1)
        b[i] = np.dot(M, M) - M1_sq
    
    # Check determinant
    det = np.linalg.det(A)
    if np.abs(det) < 1e-10:
        return None
    
    try:
        return np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        return None


def calculate_tetrahedron_regularity(vertices: np.ndarray) -> float:
    """
    Calculate regularity metric for a tetrahedron.
    
    A regular tetrahedron has all 6 edges of equal length.
    The regularity metric is the coefficient of variation (std/mean)
    of edge lengths - lower is more regular.
    
    Parameters
    ----------
    vertices : np.ndarray
        4×3 array of tetrahedron vertices.
        
    Returns
    -------
    float
        Coefficient of variation of edge lengths.
        0.0 = perfect regular tetrahedron.
        Higher values = more irregular.
    """
    # Calculate all 6 edge lengths: C(4,2) = 6
    edges = []
    for i, j in combinations(range(4), 2):
        edge_length = np.linalg.norm(vertices[i] - vertices[j])
        edges.append(edge_length)
    
    edges = np.array(edges)
    mean_edge = np.mean(edges)
    
    if mean_edge < 1e-10:
        return float('inf')  # Degenerate
    
    return np.std(edges) / mean_edge


class CN6Finder:
    """
    Find all CN-6 (octahedral) coordination sites analytically.
    
    SIMPLIFIED APPROACH (no least-squares needed):
        1. Enumerate potential octahedra (6 atom groups)
        2. Select best tetrahedral subset (4 of 6 vertices)
        3. Solve as CN-4 site (standard Voronoi vertex)
        4. Validate: all 6 vertices at distance R
    
    This leverages the key insight that octahedral centers are determined
    by ANY 4 of the 6 neighbors - the other 2 serve as validation.
    
    Parameters
    ----------
    distance_tolerance : float, optional
        Relative tolerance for distance matching (default 0.08 = 8%).
    regularity_threshold : float, optional
        Maximum regularity coefficient for tetrahedral subset (default 0.5).
    
    Attributes
    ----------
    distance_tolerance : float
        Tolerance for validating equidistant vertices.
    regularity_threshold : float
        Threshold for filtering irregular tetrahedra.
    
    Examples
    --------
    >>> lattice = BravaisLattice('cP', {'a': 1.0})
    >>> metals = lattice.generate_atoms(n_cells=1)
    >>> finder = CN6Finder()
    >>> sites = finder.find_all_sites(metals, R_range=(0.4, 0.8))
    """
    
    def __init__(self, distance_tolerance: float = 0.08,
                 regularity_threshold: float = 0.5):
        """Initialize CN6 finder with tolerances."""
        self.distance_tolerance = distance_tolerance
        self.regularity_threshold = regularity_threshold
    
    def find_all_sites(self, metals: np.ndarray, 
                       R_range: Tuple[float, float],
                       lattice: Optional[Any] = None) -> List[Dict[str, Any]]:
        """
        Find all CN-6 sites using CN-4 solver + validation.
        
        Algorithm:
            1. Enumerate potential octahedra (6 vertices)
            2. For each octahedron:
               a. Select best tetrahedral subset (4 most regular)
               b. Solve for center using circumcenter calculation
               c. Validate: check all 6 vertices at distance R
            3. Remove duplicates
        
        Parameters
        ----------
        metals : np.ndarray
            Metal atom positions, shape (N, 3).
        R_range : tuple of float
            (R_min, R_max) - range of valid coordination radii.
        lattice : BravaisLattice, optional
            If provided, fractional coordinates will be calculated.
            
        Returns
        -------
        list of dict
            List of CN-6 sites, each containing:
            - 'position': np.ndarray - Cartesian coordinates
            - 'fractional': np.ndarray - Fractional coordinates (if lattice provided)
            - 'R_critical': float - Coordination radius
            - 'coordination_number': int - Always 6
            - 'nearest_neighbors': list - Indices of 6 coordinating atoms
            - 'neighbor_distances': list - Distances to each neighbor
            - 'distance_variation': float - Std/mean of neighbor distances
            - 'gap_to_next_shell': float - Gap to 7th neighbor
        """
        metals = np.asarray(metals, dtype=np.float64)
        R_min, R_max = R_range
        
        sites = []
        
        # Enumerate all potential octahedra
        for oct_indices in self.enumerate_octahedra(metals, R_range):
            vertices = metals[list(oct_indices)]
            
            # Try to find valid octahedral center
            result = self._solve_octahedral_site(vertices, oct_indices, R_range)
            
            if result is not None:
                # Validate the site
                R = result['R_critical']
                if R_min <= R <= R_max:
                    # Calculate gap to 7th neighbor
                    gap = self._calculate_gap_to_7th(
                        result['position'], metals, oct_indices, R
                    )
                    result['gap_to_next_shell'] = gap
                    
                    # Add fractional coordinates if lattice provided
                    if lattice is not None:
                        try:
                            result['fractional'] = lattice.get_fractional_coords(
                                result['position']
                            )
                        except Exception:
                            result['fractional'] = None
                    else:
                        result['fractional'] = None
                    
                    sites.append(result)
        
        # Remove duplicate sites
        sites = self._remove_duplicates(sites)
        
        return sites
    
    def enumerate_octahedra(self, metals: np.ndarray,
                            R_range: Tuple[float, float]) -> Iterator[Tuple[int, ...]]:
        """
        Yield all potential octahedra (groups of 6 atoms).
        
        Strategy:
            1. For each pair of atoms that could be opposite vertices (distance ~2R)
            2. Find 4 atoms at distance ~R from their midpoint
            3. Yield as potential octahedron
        
        Parameters
        ----------
        metals : np.ndarray
            Metal atom positions, shape (N, 3).
        R_range : tuple of float
            Expected coordination radius range.
            
        Yields
        ------
        tuple of int
            Indices of 6 atoms forming a potential octahedron.
        """
        n_atoms = len(metals)
        R_min, R_max = R_range
        
        if n_atoms < 6:
            return
        
        seen_configurations = set()
        
        # Strategy: Opposite vertices in octahedron are at distance 2R
        # So look for pairs with distance in [2*R_min, 2*R_max]
        for i in range(n_atoms):
            for j in range(i + 1, n_atoms):
                d_ij = np.linalg.norm(metals[i] - metals[j])
                
                # Check if this could be an opposite pair
                if d_ij < 2 * R_min * 0.75 or d_ij > 2 * R_max * 1.25:
                    continue
                
                R_estimate = d_ij / 2
                midpoint = (metals[i] + metals[j]) / 2
                
                # Find 4 atoms at approximately R from midpoint
                distances_to_mid = np.linalg.norm(metals - midpoint, axis=1)
                
                # Candidates: close to R_estimate but not i or j
                tol = R_estimate * 0.3
                candidate_mask = (
                    (distances_to_mid > R_estimate - tol) &
                    (distances_to_mid < R_estimate + tol) &
                    (np.arange(n_atoms) != i) &
                    (np.arange(n_atoms) != j)
                )
                
                candidates = np.where(candidate_mask)[0]
                
                if len(candidates) < 4:
                    continue
                
                # Try combinations of 4 from candidates (limit to avoid explosion)
                max_combos = min(50, len(list(combinations(candidates, 4))))
                for combo in list(combinations(candidates, 4))[:max_combos]:
                    oct_indices = (i, j) + combo
                    canonical = tuple(sorted(oct_indices))
                    
                    if canonical in seen_configurations:
                        continue
                    seen_configurations.add(canonical)
                    
                    if self._is_plausible_octahedron(metals, oct_indices, R_range):
                        yield oct_indices
    
    def _compute_distance_matrix(self, metals: np.ndarray) -> np.ndarray:
        """Compute pairwise distance matrix."""
        n = len(metals)
        diff = metals[:, np.newaxis, :] - metals[np.newaxis, :, :]
        return np.sqrt(np.sum(diff**2, axis=2))
    
    def _is_plausible_octahedron(self, metals: np.ndarray,
                                  indices: Tuple[int, ...],
                                  R_range: Tuple[float, float]) -> bool:
        """
        Quick check if 6 points could form an octahedron.
        
        An ideal octahedron has:
            - 12 edges of length R*sqrt(2)
            - 3 "long" diagonals of length 2R
        
        We check for roughly this structure with generous tolerances.
        """
        vertices = metals[list(indices)]
        
        # Compute all 15 pairwise distances: C(6,2) = 15
        distances = []
        for i, j in combinations(range(6), 2):
            d = np.linalg.norm(vertices[i] - vertices[j])
            distances.append(d)
        
        distances = np.array(sorted(distances))
        
        # Basic sanity check
        if len(distances) < 15:
            return False
        
        # All distances should be positive
        if distances[0] < 1e-6:
            return False
        
        # In ideal octahedron: 12 short edges, 3 long diagonals
        # The ratio of long/short = sqrt(2) ≈ 1.414
        
        # For non-ideal octahedra, we're more lenient
        # Just check that distances cluster into roughly two groups
        
        # Mean distance for reference
        mean_dist = np.mean(distances)
        
        # Check overall variation isn't too extreme
        # Allow up to 80% variation for now (will be filtered later)
        overall_variation = (distances[-1] - distances[0]) / mean_dist
        if overall_variation > 2.0:  # Very lenient
            return False
        
        return True
    
    def select_best_tetrahedron(self, vertices: np.ndarray) -> Tuple[int, int, int, int]:
        """
        From 6 vertices, select the 4 that form the most regular tetrahedron.
        
        Strategy:
            - Try all C(6,4) = 15 possible tetrahedral subsets
            - Rank by regularity (standard deviation of edge lengths)
            - Return indices of best subset
        
        Parameters
        ----------
        vertices : np.ndarray
            6×3 array of octahedral vertices.
            
        Returns
        -------
        tuple of int
            (i, j, k, l): Indices (0-5) of 4 vertices forming best tetrahedron.
        """
        best_regularity = float('inf')
        best_tetrahedron = (0, 1, 2, 3)  # Default
        
        # Try all C(6,4) = 15 combinations
        for indices in combinations(range(6), 4):
            tet_vertices = vertices[list(indices)]
            
            # Skip if coplanar (invalid tetrahedron)
            if is_coplanar(tet_vertices[0], tet_vertices[1],
                          tet_vertices[2], tet_vertices[3]):
                continue
            
            regularity = calculate_tetrahedron_regularity(tet_vertices)
            
            if regularity < best_regularity:
                best_regularity = regularity
                best_tetrahedron = indices
        
        return best_tetrahedron
    
    def solve_octahedral_center(self, vertices: np.ndarray, 
                                 R: Optional[float] = None) -> Optional[np.ndarray]:
        """
        Solve for octahedral center using simplified method.
        
        Algorithm:
            1. Select best tetrahedral subset (4 of 6)
            2. Solve for circumcenter of these 4 atoms
            3. Validate: check if all 6 vertices are at distance R
        
        Parameters
        ----------
        vertices : np.ndarray
            6×3 array of octahedral vertices (M1...M6).
        R : float, optional
            Expected coordination radius. If None, uses the circumradius
            of the selected tetrahedron.
            
        Returns
        -------
        np.ndarray or None
            Position of octahedral center.
            None if validation fails (not a true CN-6 site).
        """
        vertices = np.asarray(vertices, dtype=np.float64)
        
        if len(vertices) != 6:
            return None
        
        # Select best 4 vertices
        i, j, k, l = self.select_best_tetrahedron(vertices)
        M1, M2, M3, M4 = vertices[[i, j, k, l]]
        
        # Solve for circumcenter of tetrahedron
        center = find_tetrahedron_circumcenter(M1, M2, M3, M4)
        
        if center is None:
            return None
        
        # Calculate distances to all 6 vertices
        distances = np.linalg.norm(vertices - center, axis=1)
        
        # Determine R if not provided
        if R is None:
            R = np.mean(distances)
        
        # Validate: all 6 vertices should be at distance R
        if np.allclose(distances, R, rtol=self.distance_tolerance):
            return center
        else:
            return None
    
    def _solve_octahedral_site(self, vertices: np.ndarray,
                                indices: Tuple[int, ...],
                                R_range: Tuple[float, float]) -> Optional[Dict[str, Any]]:
        """
        Solve for octahedral site and return full result dict.
        """
        vertices = np.asarray(vertices, dtype=np.float64)
        R_min, R_max = R_range
        
        # Select best tetrahedron
        tet_indices = self.select_best_tetrahedron(vertices)
        tet_vertices = vertices[list(tet_indices)]
        
        # Check tetrahedron regularity
        regularity = calculate_tetrahedron_regularity(tet_vertices)
        if regularity > self.regularity_threshold:
            return None
        
        # Find circumcenter
        center = find_tetrahedron_circumcenter(
            tet_vertices[0], tet_vertices[1], 
            tet_vertices[2], tet_vertices[3]
        )
        
        if center is None:
            return None
        
        # Calculate distances from center to all 6 vertices
        distances = np.linalg.norm(vertices - center, axis=1)
        R_mean = np.mean(distances)
        
        # Check R is in valid range
        if R_mean < R_min or R_mean > R_max:
            return None
        
        # Validate all 6 are equidistant
        distance_variation = np.std(distances) / R_mean if R_mean > 0 else float('inf')
        
        if distance_variation > self.distance_tolerance:
            return None
        
        # Build result dictionary
        return {
            'position': center,
            'R_critical': R_mean,
            'coordination_number': 6,
            'nearest_neighbors': list(indices),
            'neighbor_distances': distances.tolist(),
            'distance_variation': distance_variation,
            'gap_to_next_shell': None,  # Filled in later
            'stability_score': 1.0 - distance_variation,  # Higher is better
            'tetrahedron_indices': [indices[i] for i in tet_indices],
            'tetrahedron_regularity': regularity
        }
    
    def _calculate_gap_to_7th(self, position: np.ndarray, metals: np.ndarray,
                               neighbor_indices: Tuple[int, ...],
                               R: float) -> float:
        """
        Calculate gap between coordination shell and 7th neighbor.
        
        Parameters
        ----------
        position : np.ndarray
            Site position.
        metals : np.ndarray
            All metal positions.
        neighbor_indices : tuple
            Indices of the 6 coordinating atoms.
        R : float
            Coordination radius.
            
        Returns
        -------
        float
            Relative gap: (d_7th - R) / R
        """
        # Calculate distances to all atoms
        distances = np.linalg.norm(metals - position, axis=1)
        
        # Mask out the 6 neighbors
        mask = np.ones(len(metals), dtype=bool)
        for idx in neighbor_indices:
            mask[idx] = False
        
        other_distances = distances[mask]
        
        if len(other_distances) == 0:
            return float('inf')
        
        d_7th = np.min(other_distances)
        return (d_7th - R) / R
    
    def _remove_duplicates(self, sites: List[Dict[str, Any]],
                           tolerance: float = 0.05) -> List[Dict[str, Any]]:
        """
        Remove duplicate sites based on position proximity.
        
        Parameters
        ----------
        sites : list
            List of site dictionaries.
        tolerance : float
            Distance tolerance for considering sites as duplicates.
            
        Returns
        -------
        list
            Deduplicated list of sites.
        """
        if not sites:
            return sites
        
        unique_sites = []
        
        for site in sites:
            is_duplicate = False
            duplicate_idx = None
            
            for idx, existing in enumerate(unique_sites):
                dist = np.linalg.norm(
                    np.asarray(site['position']) - np.asarray(existing['position'])
                )
                if dist < tolerance:
                    is_duplicate = True
                    # Keep the one with smaller distance variation
                    if site['distance_variation'] < existing['distance_variation']:
                        duplicate_idx = idx
                    break
            
            if is_duplicate and duplicate_idx is not None:
                # Replace with better site
                unique_sites[duplicate_idx] = site
            elif not is_duplicate:
                unique_sites.append(site)
        
        return unique_sites
    
    def validate_cn6_site(self, position: np.ndarray, metals: np.ndarray,
                          R: float) -> bool:
        """
        Validate CN-6 coordination.
        
        Criteria:
            1. Exactly 6 atoms at distance R (within tolerance)
            2. All 6 at similar distance (< 8% variation)
            3. Clear gap to 7th neighbor (> 12%)
        
        Parameters
        ----------
        position : np.ndarray
            Position to validate, shape (3,).
        metals : np.ndarray
            All metal positions, shape (N, 3).
        R : float
            Expected coordination radius.
            
        Returns
        -------
        bool
            True if valid CN-6 site, False otherwise.
        """
        position = np.asarray(position, dtype=np.float64)
        metals = np.asarray(metals, dtype=np.float64)
        
        # Calculate all distances
        distances = np.linalg.norm(metals - position, axis=1)
        sorted_distances = np.sort(distances)
        
        # Check that exactly 6 atoms are at approximately R
        R_min = R * (1 - self.distance_tolerance)
        R_max = R * (1 + self.distance_tolerance)
        
        cn_mask = (distances >= R_min) & (distances <= R_max)
        n_coordinating = np.sum(cn_mask)
        
        if n_coordinating != 6:
            return False
        
        # Check distance variation
        coord_distances = distances[cn_mask]
        mean_dist = np.mean(coord_distances)
        variation = np.std(coord_distances) / mean_dist if mean_dist > 0 else float('inf')
        
        if variation > 0.05:  # Stricter for validation
            return False
        
        # Check gap to 7th neighbor
        if len(sorted_distances) > 6:
            d_6th = sorted_distances[5]
            d_7th = sorted_distances[6]
            gap = (d_7th - d_6th) / d_6th
            
            if gap < 0.12:
                return False
        
        return True


# Convenience function for simple usage
def find_cn6_sites(metals: np.ndarray, 
                   R_range: Tuple[float, float] = (0.3, 1.0),
                   **kwargs) -> List[Dict[str, Any]]:
    """
    Find all CN-6 octahedral coordination sites.
    
    Convenience wrapper around CN6Finder.
    
    Parameters
    ----------
    metals : np.ndarray
        Metal atom positions, shape (N, 3).
    R_range : tuple of float, optional
        Valid radius range. Default is (0.3, 1.0).
    **kwargs
        Additional arguments passed to CN6Finder constructor.
        
    Returns
    -------
    list of dict
        List of CN-6 sites.
        
    Examples
    --------
    >>> metals = generate_fcc_lattice(a=1.0, n_cells=1)
    >>> sites = find_cn6_sites(metals, R_range=(0.4, 0.6))
    """
    finder = CN6Finder(**kwargs)
    return finder.find_all_sites(metals, R_range)
