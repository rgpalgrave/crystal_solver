"""
CN-3 (Trigonal Planar) Site Finder - Analytical Implementation.

This module finds all trigonal (CN-3) coordination sites analytically
by solving constraint equations directly - no parameter scanning.

The core algorithm:
1. Enumerate all valid triangles of metal atoms
2. For each triangle, identify potential 4th neighbors
3. Solve for critical R where the 4th neighbor enters coordination shell
4. Validate solutions for true CN-3 coordination
5. Remove symmetry duplicates and rank by stability

Mathematical Foundation:
    A CN-3 site occurs at the intersection of 3 coordination spheres
    (one centered on each coordinating metal). For a triangle of metals,
    there are two such intersection points: above and below the triangle
    plane. The "critical radius" R_crit is the value where the 4th nearest
    metal is exactly at the edge of the coordination shell (with gap).

Units:
    Distances in Angstroms (Å)
    Fractional coordinates dimensionless [0, 1)

Author: Crystal Structure Solver Project
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Iterator, Optional, Dict, Any
from itertools import combinations

# Import from package structure with fallback for standalone use
try:
    from ..geometry.sphere_intersection import (
        is_collinear,
        solve_3_sphere_intersection,
        circumcenter_3d,
        unit_normal_to_plane,
    )
    from ..geometry.critical_solvers import (
        solve_critical_radius,
        verify_coordination_shell,
    )
    from ..lattices.bravais import BravaisLattice
except ImportError:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from geometry.sphere_intersection import (
        is_collinear,
        solve_3_sphere_intersection,
        circumcenter_3d,
        unit_normal_to_plane,
    )
    from geometry.critical_solvers import (
        solve_critical_radius,
        verify_coordination_shell,
    )
    from lattices.bravais import BravaisLattice


class CN3Finder:
    """
    Find all CN-3 (trigonal planar) coordination sites analytically.
    
    Uses 3-sphere intersection to find candidate positions, then
    validates coordination using critical radius solver.
    
    Attributes
    ----------
    gap_threshold : float
        Minimum relative gap (default 0.15 = 15%) between coordinating
        atoms and the 4th neighbor for a valid CN-3 site.
    min_stability_gap : float
        Minimum gap to 5th neighbor for a "stable" site (default 0.12).
    aspect_ratio_max : float
        Maximum aspect ratio for triangles (filters very flat triangles).
    
    Examples
    --------
    >>> from ..lattices.bravais import BravaisLattice
    >>> lattice = BravaisLattice('tI', {'a': 1.0, 'c': 0.644})
    >>> metals = lattice.generate_atoms(n_cells=1)
    >>> finder = CN3Finder()
    >>> sites = finder.find_all_sites(metals, R_range=(0.35, 0.50))
    >>> print(f"Found {len(sites)} CN-3 sites")
    """
    
    def __init__(self, gap_threshold: float = 0.15, 
                 min_stability_gap: float = 0.12,
                 aspect_ratio_max: float = 10.0):
        """
        Initialize CN-3 site finder.
        
        Parameters
        ----------
        gap_threshold : float, optional
            Minimum relative gap to 4th neighbor. Default 0.15.
        min_stability_gap : float, optional
            Gap threshold for marking sites as "stable". Default 0.12.
        aspect_ratio_max : float, optional
            Maximum triangle aspect ratio. Default 10.0.
        """
        self.gap_threshold = gap_threshold
        self.min_stability_gap = min_stability_gap
        self.aspect_ratio_max = aspect_ratio_max
        self._lattice = None  # Set when using with BravaisLattice
    
    def find_all_sites(self, metals: np.ndarray, 
                       R_range: Tuple[float, float],
                       lattice: Optional[BravaisLattice] = None) -> List[dict]:
        """
        Find all CN-3 sites analytically by solving for critical radii.
        
        Algorithm:
        1. Enumerate all triangles of metals
        2. For each triangle:
           a. Compute circumcenter (potential CN-3 site)
           b. Check if circumradius is in valid R_range
           c. Find nearby atoms (potential 4th neighbors)
           d. For each potential M4:
              - Solve for critical R using solve_critical_radius()
              - If R in valid range, validate as CN-3 coordination
           e. Also check direct circumcenter if it gives valid CN-3
        3. Remove duplicates
        4. Sort by stability (gap_to_5th)
        
        Parameters
        ----------
        metals : np.ndarray
            Metal atom positions (N×3 array) in Cartesian coordinates.
        R_range : tuple of float
            (R_min, R_max) physical bounds on coordination radius.
        lattice : BravaisLattice, optional
            Lattice object for coordinate conversion. If provided,
            enables fractional coordinate output.
        
        Returns
        -------
        list of dict
            List of dicts with structure:
            {
                'position': np.ndarray,           # Cartesian
                'fractional': np.ndarray,         # Unit cell coords (if lattice provided)
                'R_critical': float,              # Optimal radius
                'coordination_number': 3,
                'coordinating_indices': List[int], # Which metals (indices)
                'gap_to_4th': float,             # Separation to 4th neighbor
                'gap_to_5th': float,             # Separation to 5th neighbor
                'stability_metric': float,        # gap_to_5th / gap_to_4th
                'triangle_edges': List[float],    # Edge lengths
                'is_stable': bool                # gap_to_5th > min_stability_gap
            }
        """
        metals = np.asarray(metals, dtype=np.float64)
        R_min, R_max = R_range
        self._lattice = lattice
        
        if metals.ndim == 1:
            metals = metals.reshape(-1, 3)
        
        n_metals = len(metals)
        if n_metals < 3:
            return []
        
        # Pre-compute distance matrix for efficiency
        self._dist_matrix = np.zeros((n_metals, n_metals))
        for i in range(n_metals):
            diffs = metals - metals[i]
            self._dist_matrix[i] = np.linalg.norm(diffs, axis=1)
        
        # Estimate max edge length from R_max - use tighter bound
        # For CN-3, the circumradius ≈ edge/√3 for equilateral triangle
        # So for R_max = 0.5, max_edge ≈ 0.5 * √3 ≈ 0.87
        # Use 2x for safety margin on non-equilateral triangles
        max_edge = 2.0 * R_max
        
        sites = []
        processed_positions = set()  # Track positions to avoid duplicates
        
        # Enumerate all valid triangles
        for i, j, k in self.enumerate_triangles(metals, max_edge):
            M1, M2, M3 = metals[i], metals[j], metals[k]
            
            # First approach: Check both 3-sphere intersection points directly
            circumcenter = circumcenter_3d(M1, M2, M3)
            if circumcenter is None:
                continue
            
            circumradius = np.linalg.norm(M1 - circumcenter)
            
            # Check both intersection points at the circumradius
            if R_min <= circumradius <= R_max:
                sol1, sol2 = solve_3_sphere_intersection(M1, M2, M3, circumradius)
                
                for sol in [sol1, sol2]:
                    if sol is None:
                        continue
                    
                    # Quick position hash to avoid re-checking
                    pos_key = tuple(np.round(sol, 3))
                    if pos_key in processed_positions:
                        continue
                    
                    # Validate as CN-3 site
                    validation = self._validate_cn3_extended(sol, metals, (i, j, k))
                    if validation is not None:
                        processed_positions.add(pos_key)
                        site = self._build_site_dict_from_validation(
                            position=sol,
                            triangle_indices=(i, j, k),
                            metals=metals,
                            validation=validation
                        )
                        sites.append(site)
            
            # Second approach: Use critical radius solver with 4th neighbors
            # Only check a limited number of nearby atoms for efficiency
            triangle_indices = {i, j, k}
            nearby = self._find_nearby_atoms_fast(i, j, k, metals, 
                                                  exclude_indices=triangle_indices,
                                                  search_radius=2.5 * R_max,
                                                  max_neighbors=5)
            
            for m4_idx in nearby:
                M4 = metals[m4_idx]
                
                # Solve for critical R
                result = solve_critical_radius(M1, M2, M3, M4, 
                                              gap_threshold=self.gap_threshold)
                
                if result is None:
                    continue
                
                R_crit = result['R_critical']
                
                # Check if R is in physical range
                if not (R_min < R_crit < R_max):
                    continue
                
                # Get both intersection solutions at this R
                sol1, sol2 = solve_3_sphere_intersection(M1, M2, M3, R_crit)
                
                for sol in [sol1, sol2]:
                    if sol is None:
                        continue
                    
                    # Quick position hash
                    pos_key = tuple(np.round(sol, 3))
                    if pos_key in processed_positions:
                        continue
                    
                    # Validate this is truly a CN-3 site
                    validation = self._validate_cn3_extended(sol, metals, (i, j, k))
                    if validation is not None:
                        processed_positions.add(pos_key)
                        site = self._build_site_dict_from_validation(
                            position=sol,
                            triangle_indices=(i, j, k),
                            metals=metals,
                            validation=validation
                        )
                        sites.append(site)
        
        # Remove duplicates (using fractional coordinates if available)
        unique_sites = self.remove_symmetry_duplicates(sites)
        
        # Sort by stability metric (descending)
        unique_sites.sort(key=lambda s: s.get('stability_metric', 0), reverse=True)
        
        return unique_sites
    
    def _find_nearby_atoms_fast(self, i: int, j: int, k: int,
                                metals: np.ndarray, exclude_indices: set,
                                search_radius: float, max_neighbors: int = 10) -> List[int]:
        """Find indices of metals near the triangle using cached distances."""
        # Use pre-computed distance matrix
        # Compute centroid distance from each vertex
        centroid = (metals[i] + metals[j] + metals[k]) / 3.0
        
        nearby = []
        for idx in range(len(metals)):
            if idx in exclude_indices:
                continue
            
            # Use pre-computed distances to triangle vertices as proxy
            min_dist_to_tri = min(self._dist_matrix[i, idx], 
                                 self._dist_matrix[j, idx],
                                 self._dist_matrix[k, idx])
            
            if min_dist_to_tri < search_radius:
                # Compute actual distance to centroid
                d = np.linalg.norm(metals[idx] - centroid)
                if d < search_radius:
                    nearby.append((d, idx))
        
        # Sort by distance and take closest
        nearby.sort(key=lambda x: x[0])
        return [idx for _, idx in nearby[:max_neighbors]]
    
    def _validate_cn3_extended(self, position: np.ndarray, metals: np.ndarray,
                                triangle_indices: Tuple[int, int, int],
                                cn_tolerance: float = 0.15,
                                min_gap: float = 0.12) -> Optional[dict]:
        """
        Extended validation for CN-3 site with detailed metrics.
        
        Parameters
        ----------
        position : np.ndarray
            Candidate site position.
        metals : np.ndarray
            All metal positions.
        triangle_indices : tuple of int
            Indices of the coordinating triangle.
        cn_tolerance : float
            Tolerance for distance matching between the 3 coordinating atoms.
        min_gap : float
            Minimum gap to 4th neighbor.
        
        Returns
        -------
        dict or None
            Validation metrics if valid, None if invalid.
        """
        # Use pre-computed distances if available
        if hasattr(self, '_position_distances'):
            distances = self._position_distances
        else:
            distances = np.linalg.norm(metals - position, axis=1)
        
        sorted_indices = np.argsort(distances)
        sorted_distances = distances[sorted_indices]
        
        # Check minimum distance (no overlaps)
        if sorted_distances[0] < 0.2:
            return None
        
        # The 3 nearest should be approximately equidistant
        d1, d2, d3 = sorted_distances[0:3]
        R_avg = (d1 + d2 + d3) / 3.0
        
        # Check if all 3 are within tolerance of average
        max_dev = max(abs(d1 - R_avg), abs(d2 - R_avg), abs(d3 - R_avg))
        if max_dev / R_avg > cn_tolerance:
            return None
        
        # Check gap to 4th neighbor
        if len(sorted_distances) >= 4:
            d4 = sorted_distances[3]
            gap_to_4th = (d4 - R_avg) / R_avg
            if gap_to_4th < min_gap:
                return None
        else:
            gap_to_4th = np.inf
        
        # Calculate gap to 5th
        if len(sorted_distances) >= 5:
            d5 = sorted_distances[4]
            gap_to_5th = (d5 - R_avg) / R_avg
        else:
            gap_to_5th = np.inf
        
        # Stability metric
        if gap_to_4th > 0 and gap_to_4th < np.inf:
            stability_metric = gap_to_5th / gap_to_4th if gap_to_5th < np.inf else 10.0
        else:
            stability_metric = gap_to_5th if gap_to_5th < np.inf else 10.0
        
        return {
            'R_critical': R_avg,
            'coordinating_distances': [d1, d2, d3],
            'gap_to_4th': gap_to_4th,
            'gap_to_5th': gap_to_5th,
            'stability_metric': stability_metric,
            'is_stable': gap_to_5th > self.min_stability_gap if gap_to_5th < np.inf else True,
            'nearest_indices': sorted_indices[:3].tolist()
        }
    
    def _build_site_dict_from_validation(self, position: np.ndarray,
                                         triangle_indices: Tuple[int, int, int],
                                         metals: np.ndarray,
                                         validation: dict) -> dict:
        """Build output dictionary from validation results."""
        i, j, k = triangle_indices
        M1, M2, M3 = metals[i], metals[j], metals[k]
        
        # Calculate edge lengths
        e1 = float(np.linalg.norm(M2 - M1))
        e2 = float(np.linalg.norm(M3 - M1))
        e3 = float(np.linalg.norm(M3 - M2))
        
        # Convert to fractional if lattice available
        fractional = None
        if self._lattice is not None:
            fractional = self._lattice.get_fractional_coords(position)
            fractional = fractional % 1.0
        
        return {
            'position': position.copy(),
            'fractional': fractional,
            'R_critical': float(validation['R_critical']),
            'coordination_number': 3,
            'coordinating_indices': list(triangle_indices),
            'gap_to_4th': float(validation['gap_to_4th']) if validation['gap_to_4th'] < np.inf else None,
            'gap_to_5th': float(validation['gap_to_5th']) if validation['gap_to_5th'] < np.inf else None,
            'stability_metric': float(validation['stability_metric']),
            'triangle_edges': [e1, e2, e3],
            'is_stable': validation['is_stable']
        }
    
    def enumerate_triangles(self, metals: np.ndarray, 
                           max_edge: float) -> Iterator[Tuple[int, int, int]]:
        """
        Yield all valid triangles of metal atoms.
        
        Filters out:
        - Collinear atoms (checked via is_collinear)
        - Triangles with any edge > max_edge
        - Very flat triangles (aspect ratio > aspect_ratio_max)
        
        Parameters
        ----------
        metals : np.ndarray
            Metal positions, shape (N, 3).
        max_edge : float
            Maximum triangle edge length.
        
        Yields
        ------
        tuple of (int, int, int)
            Indices of three metals forming valid triangle.
        """
        n = len(metals)
        
        # Pre-compute distance matrix for efficiency
        # Only compute upper triangle
        dist_matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(i+1, n):
                d = np.linalg.norm(metals[j] - metals[i])
                dist_matrix[i, j] = d
                dist_matrix[j, i] = d
        
        # Pre-filter: for each atom, find neighbors within max_edge
        neighbors = {}
        for i in range(n):
            neighbors[i] = [j for j in range(n) if j != i and dist_matrix[i,j] <= max_edge]
        
        # Enumerate triangles using neighbor lists
        seen = set()
        for i in range(n):
            for j in neighbors[i]:
                if j <= i:
                    continue
                for k in neighbors[i]:
                    if k <= j:
                        continue
                    # Check if j-k edge is also valid
                    if dist_matrix[j, k] > max_edge:
                        continue
                    
                    # Create canonical ordering
                    tri = tuple(sorted([i, j, k]))
                    if tri in seen:
                        continue
                    seen.add(tri)
                    
                    M1, M2, M3 = metals[tri[0]], metals[tri[1]], metals[tri[2]]
                    
                    # Get edge lengths
                    e1 = dist_matrix[tri[0], tri[1]]
                    e2 = dist_matrix[tri[0], tri[2]]
                    e3 = dist_matrix[tri[1], tri[2]]
                    
                    max_e = max(e1, e2, e3)
                    min_e = min(e1, e2, e3)
                    
                    # Filter degenerate triangles
                    if min_e < 1e-10:
                        continue
                    
                    # Filter by aspect ratio
                    if max_e / min_e > self.aspect_ratio_max:
                        continue
                    
                    # Filter collinear atoms
                    if is_collinear(M1, M2, M3):
                        continue
                    
                    yield tri
    
    def validate_cn3_site(self, position: np.ndarray, metals: np.ndarray,
                          R: float, tolerance: float = 0.12) -> bool:
        """
        Check if position is valid CN-3 site.
        
        Criteria:
        1. Exactly 3 atoms at distance R (within tolerance)
        2. Clear gap (>12%) to 4th nearest neighbor
        3. No overlaps with other atoms (d > 0.5Å)
        
        Parameters
        ----------
        position : np.ndarray
            Candidate anion position, shape (3,).
        metals : np.ndarray
            All metal positions, shape (N, 3).
        R : float
            Expected coordination radius.
        tolerance : float, optional
            Relative tolerance for distance matching. Default 0.12.
        
        Returns
        -------
        bool
            True if valid CN-3, False otherwise.
        """
        position = np.asarray(position, dtype=np.float64)
        metals = np.asarray(metals, dtype=np.float64)
        
        if metals.ndim == 1:
            metals = metals.reshape(-1, 3)
        
        # Calculate distances to all metals
        distances = np.linalg.norm(metals - position, axis=1)
        sorted_d = np.sort(distances)
        
        # Check for overlaps (too close to any metal)
        if sorted_d[0] < 0.3:
            return False
        
        # Count atoms within tolerance of R
        R_min = R * (1 - tolerance)
        R_max = R * (1 + tolerance)
        in_shell = np.sum((distances >= R_min) & (distances <= R_max))
        
        # Must have exactly 3 coordinating atoms
        if in_shell != 3:
            return False
        
        # Check gap to 4th neighbor
        if len(sorted_d) >= 4:
            d3 = sorted_d[2]
            d4 = sorted_d[3]
            gap = (d4 - d3) / d3
            if gap < self.min_stability_gap:
                return False
        
        return True
    
    def remove_symmetry_duplicates(self, sites: List[dict],
                                   tolerance: float = 0.03) -> List[dict]:
        """
        Remove duplicate sites based on position matching.
        
        Two sites are considered duplicates if their fractional coordinates
        are within tolerance after accounting for periodic boundary conditions.
        
        Note: This does NOT apply crystal symmetry operations - it only
        removes sites that are at the same physical location.
        
        Parameters
        ----------
        sites : list of dict
            List of candidate sites.
        tolerance : float, optional
            Fractional coordinate tolerance. Default 0.03.
        
        Returns
        -------
        list of dict
            Filtered list with positional duplicates removed.
        """
        if not sites:
            return []
        
        unique = []
        
        for site in sites:
            is_duplicate = False
            
            for existing in unique:
                # Compare positions
                if site['fractional'] is not None and existing['fractional'] is not None:
                    # Use fractional coords with periodic wrapping
                    frac1 = np.array(site['fractional']) % 1.0
                    frac2 = np.array(existing['fractional']) % 1.0
                    
                    # Account for periodic boundaries (e.g., 0.01 ≈ 0.99)
                    diff = np.abs(frac1 - frac2)
                    diff = np.minimum(diff, 1.0 - diff)
                    
                    if np.all(diff < tolerance):
                        is_duplicate = True
                        break
                else:
                    # Fall back to Cartesian comparison
                    pos1 = np.array(site['position'])
                    pos2 = np.array(existing['position'])
                    
                    if np.allclose(pos1, pos2, atol=tolerance):
                        is_duplicate = True
                        break
            
            if not is_duplicate:
                unique.append(site)
        
        return unique
    
    def _find_nearby_atoms(self, M1: np.ndarray, M2: np.ndarray, M3: np.ndarray,
                          metals: np.ndarray, exclude_indices: set,
                          search_radius: float) -> List[int]:
        """Find indices of metals near the triangle (potential 4th neighbors)."""
        # Compute triangle centroid
        centroid = (M1 + M2 + M3) / 3.0
        
        nearby = []
        for idx in range(len(metals)):
            if idx in exclude_indices:
                continue
            
            d = np.linalg.norm(metals[idx] - centroid)
            if d < search_radius:
                nearby.append(idx)
        
        return nearby
    
    def _find_5th_neighbor(self, M1: np.ndarray, M2: np.ndarray, M3: np.ndarray,
                          M4: np.ndarray, metals: np.ndarray, 
                          candidate_indices: List[int],
                          R_max: float) -> Optional[np.ndarray]:
        """Find the 5th nearest neighbor for stability calculation."""
        # Compute triangle centroid as reference point
        centroid = (M1 + M2 + M3) / 3.0
        
        # Distance from M4 to centroid gives approximate scale
        d4 = np.linalg.norm(M4 - centroid)
        
        # Find closest atom (other than M1-M4) to centroid
        min_dist = np.inf
        M5 = None
        
        for idx in candidate_indices:
            d = np.linalg.norm(metals[idx] - centroid)
            if d > d4 * 1.05 and d < 3.0 * R_max and d < min_dist:
                min_dist = d
                M5 = metals[idx].copy()
        
        return M5


def find_cn3_analytical(lattice: BravaisLattice, R_min: float, R_max: float,
                        n_cells: int = 1,
                        target_count: Optional[int] = None) -> pd.DataFrame:
    """
    Convenience function returning DataFrame of CN-3 results.
    
    Parameters
    ----------
    lattice : BravaisLattice
        Lattice object to analyze.
    R_min : float
        Minimum coordination radius to consider.
    R_max : float
        Maximum coordination radius to consider.
    n_cells : int, optional
        Number of unit cells to generate. Default 1.
    target_count : int, optional
        If provided, return only the top N sites by stability.
    
    Returns
    -------
    pd.DataFrame
        Columns: position_x, position_y, position_z, frac_x, frac_y, frac_z,
                 R_critical, gap_to_4th, gap_to_5th, stability_metric, is_stable
    """
    metals = lattice.generate_atoms(n_cells=n_cells)
    
    finder = CN3Finder()
    sites = finder.find_all_sites(metals, R_range=(R_min, R_max), lattice=lattice)
    
    # If target_count specified, filter to top N by gap_to_4th
    if target_count is not None and len(sites) > target_count:
        sites.sort(key=lambda s: (s.get('gap_to_4th', 0), s.get('stability_metric', 0)), reverse=True)
        sites = sites[:target_count]
    
    if not sites:
        return pd.DataFrame(columns=[
            'position_x', 'position_y', 'position_z',
            'frac_x', 'frac_y', 'frac_z',
            'R_critical', 'gap_to_4th', 'gap_to_5th', 'stability_metric', 'is_stable'
        ])
    
    rows = []
    for site in sites:
        pos = site['position']
        frac = site['fractional'] if site['fractional'] is not None else [None, None, None]
        
        rows.append({
            'position_x': pos[0],
            'position_y': pos[1],
            'position_z': pos[2],
            'frac_x': frac[0] if frac is not None else None,
            'frac_y': frac[1] if frac is not None else None,
            'frac_z': frac[2] if frac is not None else None,
            'R_critical': site['R_critical'],
            'gap_to_4th': site['gap_to_4th'],
            'gap_to_5th': site['gap_to_5th'],
            'stability_metric': site['stability_metric'],
            'is_stable': site['is_stable']
        })
    
    return pd.DataFrame(rows)


def find_cn3_rutile(a: float = 1.0, c_over_a: float = 0.644,
                    R_range: Tuple[float, float] = (0.35, 0.50),
                    filter_to_best: bool = False,
                    target_count: int = 6,
                    n_cells: int = 1) -> List[dict]:
    """
    Find CN-3 sites in rutile-type structure.
    
    Parameters
    ----------
    a : float, optional
        Lattice parameter a. Default 1.0.
    c_over_a : float, optional
        c/a ratio. Default 0.644 (ideal rutile).
    R_range : tuple, optional
        (R_min, R_max) search range.
    filter_to_best : bool, optional
        If True, return only the best sites (smallest R_critical with highest gaps).
    target_count : int, optional
        Number of sites to return if filter_to_best is True.
    n_cells : int, optional
        Number of unit cells to generate (affects search volume). Default 1.
    
    Returns
    -------
    list of dict
        Found CN-3 sites with all metrics.
    
    Notes
    -----
    Rutile (TiO2) has Ti atoms on a body-centered tetragonal (tI)
    lattice with c/a ≈ 0.644. The oxygen atoms form CN-3
    trigonal planar coordination around specific positions.
    
    For the ideal rutile structure, the 6 most stable CN-3 sites
    are found at R ≈ 0.427a with positions:
    - (x, 1-x, 0) and (1-x, x, 0) where x ≈ 0.305
    - (0.5-x, 0.5+x, 0.5), (0.5+x, 0.5-x, 0.5), and variants
    """
    c = a * c_over_a
    lattice = BravaisLattice('tI', {'a': a, 'c': c})
    metals = lattice.generate_atoms(n_cells=n_cells)
    
    finder = CN3Finder()
    sites = finder.find_all_sites(metals, R_range=R_range, lattice=lattice)
    
    if filter_to_best and len(sites) > target_count:
        # For rutile, the best sites are at the smallest R_critical
        r_groups = {}
        for s in sites:
            r_key = round(s['R_critical'], 2)
            if r_key not in r_groups:
                r_groups[r_key] = []
            r_groups[r_key].append(s)
        
        # Find smallest R with at least target_count sites
        for r_key in sorted(r_groups.keys()):
            group = r_groups[r_key]
            if len(group) >= target_count:
                group.sort(key=lambda s: s.get('gap_to_4th', 0), reverse=True)
                return group[:target_count]
        
        # Fallback
        sites.sort(key=lambda s: s.get('gap_to_4th', 0), reverse=True)
        sites = sites[:target_count]
    
    return sites


def find_cn3_hexagonal(a: float = 1.0, c: float = 1.5,
                       R_range: Tuple[float, float] = (0.4, 0.7),
                       n_cells: int = 1) -> List[dict]:
    """
    Find CN-3 sites in hexagonal lattice.
    
    Parameters
    ----------
    a : float, optional
        Basal plane lattice parameter. Default 1.0.
    c : float, optional
        Axial lattice parameter. Default 1.5.
    R_range : tuple, optional
        (R_min, R_max) search range.
    n_cells : int, optional
        Number of unit cells to generate. Default 1.
    
    Returns
    -------
    list of dict
        Found CN-3 sites with all metrics.
    
    Notes
    -----
    Hexagonal lattices often have CN-3 sites at triangle centers
    in the basal plane.
    """
    lattice = BravaisLattice('hP', {'a': a, 'c': c})
    metals = lattice.generate_atoms(n_cells=n_cells)
    
    finder = CN3Finder()
    return finder.find_all_sites(metals, R_range=R_range, lattice=lattice)
