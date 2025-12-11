"""
CN-4 (Tetrahedral) Coordination Site Finder.

This module provides analytical methods for finding all tetrahedral
coordination sites in metal lattices using 4-sphere intersection
(Voronoi vertex) calculations.

Mathematical Foundation:
    CN-4 sites are the circumcenters of metal tetrahedra - the unique
    point equidistant from all four vertices. Unlike CN-3 sites (which
    have two solutions per triangle), CN-4 gives exactly one point per
    tetrahedron.

    Key advantage: No R scanning required. The circumradius directly
    gives the coordination sphere radius for each tetrahedral site.

Units:
    All distances are in Angstroms (Å) unless otherwise specified.
    Coordinates are Cartesian unless labeled as 'fractional'.

Author: Crystal Structure Solver Project
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Iterator, Any
from itertools import combinations
from scipy.spatial import Delaunay


def is_coplanar(M1: np.ndarray, M2: np.ndarray, M3: np.ndarray, M4: np.ndarray,
                tolerance: float = 1e-8) -> bool:
    """
    Check if four points are coplanar (lie in the same plane).

    Uses the scalar triple product: if (M2-M1)·((M3-M1)×(M4-M1)) ≈ 0,
    the points are coplanar.

    Parameters
    ----------
    M1, M2, M3, M4 : np.ndarray
        Four points to test, each shape (3,).
    tolerance : float, optional
        Relative tolerance for coplanarity. Default is 1e-8.

    Returns
    -------
    bool
        True if points are coplanar within tolerance.

    Examples
    --------
    >>> # Four points in the xy-plane
    >>> M1 = np.array([0, 0, 0])
    >>> M2 = np.array([1, 0, 0])
    >>> M3 = np.array([0, 1, 0])
    >>> M4 = np.array([1, 1, 0])
    >>> is_coplanar(M1, M2, M3, M4)
    True
    """
    v1 = np.asarray(M2, dtype=np.float64) - np.asarray(M1, dtype=np.float64)
    v2 = np.asarray(M3, dtype=np.float64) - np.asarray(M1, dtype=np.float64)
    v3 = np.asarray(M4, dtype=np.float64) - np.asarray(M1, dtype=np.float64)

    # Scalar triple product: v1 · (v2 × v3)
    triple_product = np.abs(np.dot(v1, np.cross(v2, v3)))

    # Scale tolerance by characteristic length (product of edge lengths)
    scale = np.linalg.norm(v1) * np.linalg.norm(v2) * np.linalg.norm(v3)
    if scale < 1e-30:
        return True  # Degenerate case

    return triple_product < tolerance * scale


def find_circumcenter_of_tetrahedron(M1: np.ndarray, M2: np.ndarray,
                                      M3: np.ndarray, M4: np.ndarray) -> Optional[np.ndarray]:
    """
    Find the circumcenter of a tetrahedron (equidistant from all 4 vertices).

    Solves the linear system from radical plane equations:
        2(Mᵢ - M₁)·x = ||Mᵢ||² - ||M₁||²  for i = 2, 3, 4

    Parameters
    ----------
    M1, M2, M3, M4 : np.ndarray
        Vertices of the tetrahedron, each shape (3,).

    Returns
    -------
    np.ndarray or None
        Circumcenter coordinates, shape (3,).
        None if the tetrahedron is degenerate (coplanar vertices).

    Examples
    --------
    >>> # Regular tetrahedron centered near origin
    >>> M1 = np.array([1, 1, 1])
    >>> M2 = np.array([1, -1, -1])
    >>> M3 = np.array([-1, 1, -1])
    >>> M4 = np.array([-1, -1, 1])
    >>> center = find_circumcenter_of_tetrahedron(M1, M2, M3, M4)
    >>> np.allclose(center, [0, 0, 0])
    True
    """
    M1 = np.asarray(M1, dtype=np.float64)
    M2 = np.asarray(M2, dtype=np.float64)
    M3 = np.asarray(M3, dtype=np.float64)
    M4 = np.asarray(M4, dtype=np.float64)

    # Build the linear system Ax = b
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
        return np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        return None


def circumradius_of_tetrahedron(M1: np.ndarray, M2: np.ndarray,
                                 M3: np.ndarray, M4: np.ndarray) -> Optional[float]:
    """
    Calculate the circumradius of a tetrahedron.

    Parameters
    ----------
    M1, M2, M3, M4 : np.ndarray
        Vertices of the tetrahedron, each shape (3,).

    Returns
    -------
    float or None
        Circumradius (distance from circumcenter to any vertex).
        None if the tetrahedron is degenerate.
    """
    center = find_circumcenter_of_tetrahedron(M1, M2, M3, M4)
    if center is None:
        return None
    return np.linalg.norm(center - np.asarray(M1, dtype=np.float64))


def tetrahedron_regularity(M1: np.ndarray, M2: np.ndarray,
                           M3: np.ndarray, M4: np.ndarray) -> float:
    """
    Calculate regularity metric for a tetrahedron (0 to 1).

    A regular tetrahedron has regularity = 1.0. Distorted tetrahedra
    have lower values. This metric is the ratio of the minimum edge
    length to the maximum edge length.

    Parameters
    ----------
    M1, M2, M3, M4 : np.ndarray
        Vertices of the tetrahedron, each shape (3,).

    Returns
    -------
    float
        Regularity metric in range [0, 1].
        1.0 = perfectly regular, 0.0 = maximally distorted.
    """
    vertices = [
        np.asarray(M1, dtype=np.float64),
        np.asarray(M2, dtype=np.float64),
        np.asarray(M3, dtype=np.float64),
        np.asarray(M4, dtype=np.float64)
    ]

    # Calculate all 6 edge lengths
    edges = []
    for i in range(4):
        for j in range(i + 1, 4):
            edges.append(np.linalg.norm(vertices[i] - vertices[j]))

    if max(edges) < 1e-10:
        return 0.0

    return min(edges) / max(edges)


def tetrahedron_volume(M1: np.ndarray, M2: np.ndarray,
                       M3: np.ndarray, M4: np.ndarray) -> float:
    """
    Calculate volume of a tetrahedron.

    Volume = |det([M2-M1, M3-M1, M4-M1])| / 6

    Parameters
    ----------
    M1, M2, M3, M4 : np.ndarray
        Vertices of the tetrahedron, each shape (3,).

    Returns
    -------
    float
        Volume of the tetrahedron (in cubic Å).
    """
    v1 = np.asarray(M2, dtype=np.float64) - np.asarray(M1, dtype=np.float64)
    v2 = np.asarray(M3, dtype=np.float64) - np.asarray(M1, dtype=np.float64)
    v3 = np.asarray(M4, dtype=np.float64) - np.asarray(M1, dtype=np.float64)

    return np.abs(np.dot(v1, np.cross(v2, v3))) / 6.0


class CN4Finder:
    """
    Find all CN-4 (tetrahedral) coordination sites analytically.

    CN-4 sites are Voronoi vertices - the unique intersection point
    of 4 spheres. This is the "standard" case in computational geometry
    where four spheres of equal radius intersect at their circumcenter.

    Key Innovation:
        Instead of scanning R values, we directly compute the circumradius
        of each metal tetrahedron. This circumradius IS the coordination
        sphere radius that places an anion at the tetrahedral site.

    Attributes
    ----------
    gap_threshold : float
        Minimum relative gap between 4th and 5th neighbor for valid CN-4.
    distance_tolerance : float
        Tolerance for identifying atoms at the same distance.
    min_regularity : float
        Minimum tetrahedron regularity (0-1) to consider.

    Examples
    --------
    >>> from src.lattices.bravais import BravaisLattice
    >>> lattice = BravaisLattice('cF', {'a': 4.0})
    >>> metals = lattice.generate_atoms(n_cells=1)
    >>> finder = CN4Finder()
    >>> sites = finder.find_all_sites(metals, R_range=(1.0, 3.0))
    >>> print(f"Found {len(sites)} tetrahedral sites")
    """

    def __init__(self, gap_threshold: float = 0.12, distance_tolerance: float = 0.15,
                 min_regularity: float = 0.3):
        """
        Initialize CN4Finder.

        Parameters
        ----------
        gap_threshold : float, optional
            Minimum relative gap to 5th neighbor (default 0.12 = 12%).
        distance_tolerance : float, optional
            Tolerance for matching coordination distances (default 0.15).
        min_regularity : float, optional
            Minimum tetrahedron regularity to consider (default 0.3).
        """
        self.gap_threshold = gap_threshold
        self.distance_tolerance = distance_tolerance
        self.min_regularity = min_regularity

    def find_all_sites(self, metals: np.ndarray, R_range: Tuple[float, float],
                       lattice: Optional[Any] = None,
                       use_delaunay: bool = True) -> List[Dict[str, Any]]:
        """
        Find all CN-4 (tetrahedral) coordination sites analytically.

        Algorithm:
        1. Enumerate all tetrahedra of metals (using Delaunay or brute force)
        2. For each tetrahedron (M1, M2, M3, M4):
           - Check if coplanar (skip if so)
           - Compute circumcenter (the CN-4 site position)
           - Compute circumradius (the natural R value)
           - Validate that this is a genuine CN-4 site
        3. Remove duplicates (sites at same position)
        4. Rank by regularity and stability

        Parameters
        ----------
        metals : np.ndarray
            Metal atom positions, shape (N, 3) in Cartesian coordinates.
        R_range : tuple of (float, float)
            Range of R values to consider: (R_min, R_max).
            Sites with circumradius outside this range are filtered.
        lattice : BravaisLattice, optional
            Lattice object for converting to fractional coordinates.
        use_delaunay : bool, optional
            Use Delaunay triangulation to find tetrahedra (faster).
            Set False for brute-force enumeration. Default True.

        Returns
        -------
        list of dict
            List of valid CN-4 sites, each containing:
            - 'position': np.ndarray - Cartesian coordinates
            - 'fractional': np.ndarray - Fractional coordinates (if lattice provided)
            - 'R_critical': float - Circumradius (coordination sphere radius)
            - 'coordination_number': int - Always 4
            - 'nearest_neighbors': list - Indices of 4 coordinating metals
            - 'gap_to_next_shell': float - Relative gap to 5th neighbor
            - 'stability_score': float - Ranking metric (regularity × gap)
            - 'regularity': float - Tetrahedron regularity (0-1)

        Notes
        -----
        Unlike CN-3 (which requires solving quadratic equations), CN-4 is
        simpler because the circumcenter is uniquely determined by 4 points,
        and the circumradius directly gives the coordination sphere radius.
        """
        metals = np.asarray(metals, dtype=np.float64)
        if metals.ndim == 1:
            metals = metals.reshape(-1, 3)

        R_min, R_max = R_range
        sites = []

        # Generate tetrahedra
        if use_delaunay and len(metals) >= 4:
            tetrahedra = self._enumerate_delaunay_tetrahedra(metals, R_max)
        else:
            tetrahedra = self._enumerate_tetrahedra_brute(metals, R_max)

        # Process each tetrahedron
        for i1, i2, i3, i4 in tetrahedra:
            M1, M2, M3, M4 = metals[i1], metals[i2], metals[i3], metals[i4]

            # Skip coplanar configurations
            if is_coplanar(M1, M2, M3, M4):
                continue

            # Check regularity (skip very distorted tetrahedra)
            regularity = tetrahedron_regularity(M1, M2, M3, M4)
            if regularity < self.min_regularity:
                continue

            # Find circumcenter and circumradius
            center = find_circumcenter_of_tetrahedron(M1, M2, M3, M4)
            if center is None:
                continue

            R = circumradius_of_tetrahedron(M1, M2, M3, M4)
            if R is None:
                continue

            # Check if R is in the target range
            if not (R_min <= R <= R_max):
                continue

            # Validate this as a genuine CN-4 site
            validation = self.validate_cn4_site(center, metals, R)
            if not validation['is_valid']:
                continue

            # Build site dictionary
            site = {
                'position': center.copy(),
                'fractional': None,
                'R_critical': float(R),
                'coordination_number': 4,
                'nearest_neighbors': [int(i1), int(i2), int(i3), int(i4)],
                'gap_to_next_shell': validation['gap_to_next'],
                'stability_score': regularity * validation['gap_to_next'],
                'regularity': regularity,
                'coordination_distances': validation['coordination_distances'],
            }

            # Convert to fractional if lattice provided
            if lattice is not None:
                site['fractional'] = lattice.get_fractional_coords(center)

            sites.append(site)

        # Remove duplicates
        sites = self._remove_duplicates(sites)

        # Sort by stability score (higher is better)
        sites.sort(key=lambda s: s['stability_score'], reverse=True)

        return sites

    def _enumerate_delaunay_tetrahedra(self, metals: np.ndarray,
                                        max_edge: float) -> Iterator[Tuple[int, int, int, int]]:
        """
        Enumerate tetrahedra using Delaunay triangulation.

        This is much faster than brute force for large point sets.

        Parameters
        ----------
        metals : np.ndarray
            Metal positions, shape (N, 3).
        max_edge : float
            Maximum edge length to consider.

        Yields
        ------
        tuple of (int, int, int, int)
            Indices of four metals forming a tetrahedron.
        """
        try:
            tri = Delaunay(metals)
        except Exception:
            # Fall back to brute force if Delaunay fails
            yield from self._enumerate_tetrahedra_brute(metals, max_edge)
            return

        for simplex in tri.simplices:
            i1, i2, i3, i4 = simplex

            # Check max edge constraint
            vertices = metals[simplex]
            max_found_edge = 0.0
            for i in range(4):
                for j in range(i + 1, 4):
                    edge = np.linalg.norm(vertices[i] - vertices[j])
                    max_found_edge = max(max_found_edge, edge)

            if max_found_edge <= max_edge * 2.5:  # Allow some margin
                yield (i1, i2, i3, i4)

    def _enumerate_tetrahedra_brute(self, metals: np.ndarray,
                                     max_edge: float) -> Iterator[Tuple[int, int, int, int]]:
        """
        Enumerate tetrahedra by brute force (all 4-combinations).

        Slower but guaranteed to find all tetrahedra.

        Parameters
        ----------
        metals : np.ndarray
            Metal positions, shape (N, 3).
        max_edge : float
            Maximum edge length to consider.

        Yields
        ------
        tuple of (int, int, int, int)
            Indices of four metals forming a tetrahedron.
        """
        n = len(metals)

        for i1, i2, i3, i4 in combinations(range(n), 4):
            # Check all edges
            vertices = [metals[i1], metals[i2], metals[i3], metals[i4]]
            skip = False

            for i in range(4):
                if skip:
                    break
                for j in range(i + 1, 4):
                    edge = np.linalg.norm(vertices[i] - vertices[j])
                    if edge > max_edge * 2.5:
                        skip = True
                        break

            if not skip:
                yield (i1, i2, i3, i4)

    def enumerate_tetrahedra(self, metals: np.ndarray, max_edge: float) -> Iterator[Tuple[int, int, int, int]]:
        """
        Yield all valid tetrahedra.

        Filters:
        - Coplanar atoms (4 atoms too flat)
        - Tetrahedra with edges > max_edge
        - Very distorted tetrahedra (regularity < min_regularity)

        Parameters
        ----------
        metals : np.ndarray
            Metal positions, shape (N, 3).
        max_edge : float
            Maximum edge length to consider.

        Yields
        ------
        tuple of (int, int, int, int)
            Indices of four metals forming a valid tetrahedron.
        """
        metals = np.asarray(metals, dtype=np.float64)

        for i1, i2, i3, i4 in self._enumerate_delaunay_tetrahedra(metals, max_edge):
            M1, M2, M3, M4 = metals[i1], metals[i2], metals[i3], metals[i4]

            # Skip coplanar
            if is_coplanar(M1, M2, M3, M4):
                continue

            # Skip low regularity
            if tetrahedron_regularity(M1, M2, M3, M4) < self.min_regularity:
                continue

            yield (i1, i2, i3, i4)

    def validate_cn4_site(self, position: np.ndarray, metals: np.ndarray,
                          R: float) -> Dict[str, Any]:
        """
        Validate that a position is a genuine CN-4 coordination site.

        Checks:
        1. Exactly 4 atoms at distance ≈ R (within tolerance)
        2. Clear gap (> gap_threshold) to 5th neighbor
        3. The 4 coordinating atoms form a valid tetrahedron

        Parameters
        ----------
        position : np.ndarray
            Candidate site position, shape (3,).
        metals : np.ndarray
            All metal positions, shape (N, 3).
        R : float
            Expected coordination distance.

        Returns
        -------
        dict
            Validation results:
            - 'is_valid': bool - True if genuine CN-4
            - 'actual_CN': int - Number of atoms at distance ≈ R
            - 'gap_to_next': float - Relative gap to 5th neighbor
            - 'is_well_separated': bool - True if gap > threshold
            - 'coordination_distances': list - Distances to coordinating atoms
            - 'nearest_distances': list - First 6 distances
        """
        position = np.asarray(position, dtype=np.float64)
        metals = np.asarray(metals, dtype=np.float64)

        # Calculate all distances
        distances = np.linalg.norm(metals - position, axis=1)
        sorted_distances = np.sort(distances)

        # Identify coordinating atoms (within tolerance of R)
        R_min = R * (1 - self.distance_tolerance)
        R_max = R * (1 + self.distance_tolerance)
        coord_mask = (distances >= R_min) & (distances <= R_max)
        actual_CN = int(np.sum(coord_mask))
        coord_distances = sorted(distances[coord_mask].tolist())

        # Calculate gap to 5th neighbor
        if actual_CN >= 4 and len(sorted_distances) > 4:
            # Average distance of closest 4
            d_coord = np.mean(sorted_distances[:4])
            d_5th = sorted_distances[4]
            gap_to_next = (d_5th - d_coord) / d_coord
        elif actual_CN >= 4:
            gap_to_next = float('inf')  # No 5th neighbor
        else:
            gap_to_next = 0.0

        # Determine validity
        is_valid = (actual_CN == 4) and (gap_to_next > self.gap_threshold)

        return {
            'is_valid': is_valid,
            'actual_CN': actual_CN,
            'gap_to_next': float(gap_to_next) if gap_to_next != float('inf') else 1.0,
            'is_well_separated': gap_to_next > self.gap_threshold,
            'coordination_distances': coord_distances,
            'nearest_distances': sorted_distances[:6].tolist() if len(sorted_distances) >= 6 else sorted_distances.tolist()
        }

    def _remove_duplicates(self, sites: List[Dict], tolerance: float = 0.05) -> List[Dict]:
        """
        Remove duplicate sites (same position within tolerance).

        When duplicates are found, keeps the one with highest stability_score.

        Parameters
        ----------
        sites : list of dict
            List of site dictionaries.
        tolerance : float, optional
            Position tolerance for identifying duplicates.

        Returns
        -------
        list of dict
            De-duplicated list of sites.
        """
        if not sites:
            return []

        # Sort by stability score (descending) to keep best when removing duplicates
        sites_sorted = sorted(sites, key=lambda s: s['stability_score'], reverse=True)
        unique = []

        for site in sites_sorted:
            pos = site['position']
            is_duplicate = False

            for existing in unique:
                if np.linalg.norm(pos - existing['position']) < tolerance:
                    is_duplicate = True
                    break

            if not is_duplicate:
                unique.append(site)

        return unique

    def find_sites_at_radius(self, metals: np.ndarray, R_target: float,
                             tolerance: float = 0.1) -> List[Dict[str, Any]]:
        """
        Find CN-4 sites with circumradius close to a target value.

        This is useful when you know the expected R (e.g., from ionic radii).

        Parameters
        ----------
        metals : np.ndarray
            Metal atom positions, shape (N, 3).
        R_target : float
            Target coordination sphere radius.
        tolerance : float, optional
            Fractional tolerance around R_target. Default 0.1 (10%).

        Returns
        -------
        list of dict
            Sites with circumradius in range [R_target*(1-tol), R_target*(1+tol)].
        """
        R_min = R_target * (1 - tolerance)
        R_max = R_target * (1 + tolerance)
        return self.find_all_sites(metals, R_range=(R_min, R_max))

    def find_fractional_sites(self, metals: np.ndarray, R_range: Tuple[float, float],
                               lattice: Any) -> List[Dict[str, Any]]:
        """
        Find CN-4 sites and return with fractional coordinates.

        Convenience wrapper that ensures fractional coordinates are computed
        and wraps them to the unit cell [0, 1).

        Parameters
        ----------
        metals : np.ndarray
            Metal atom positions, shape (N, 3) in Cartesian coordinates.
        R_range : tuple of (float, float)
            Range of R values to consider.
        lattice : BravaisLattice
            Lattice object for coordinate conversion.

        Returns
        -------
        list of dict
            Sites with 'fractional' coordinates wrapped to [0, 1).
        """
        sites = self.find_all_sites(metals, R_range, lattice=lattice)

        for site in sites:
            if site['fractional'] is not None:
                # Wrap to unit cell [0, 1)
                frac = site['fractional'] % 1.0
                # Handle numerical edge case where modulo gives exactly 1.0
                frac = np.where(np.isclose(frac, 1.0), 0.0, frac)
                site['fractional'] = frac

        return sites


# =============================================================================
# Convenience functions
# =============================================================================

def find_tetrahedral_sites(metals: np.ndarray, R_range: Tuple[float, float],
                           **kwargs) -> List[Dict[str, Any]]:
    """
    Convenience function to find all tetrahedral (CN-4) sites.

    Parameters
    ----------
    metals : np.ndarray
        Metal atom positions, shape (N, 3).
    R_range : tuple of (float, float)
        Range of coordination sphere radii to consider.
    **kwargs
        Additional arguments passed to CN4Finder constructor.

    Returns
    -------
    list of dict
        List of CN-4 site dictionaries.
    """
    finder = CN4Finder(**kwargs)
    return finder.find_all_sites(metals, R_range)


def count_tetrahedral_sites(metals: np.ndarray, R_range: Tuple[float, float],
                            **kwargs) -> int:
    """
    Count the number of tetrahedral sites in a metal lattice.

    Parameters
    ----------
    metals : np.ndarray
        Metal atom positions, shape (N, 3).
    R_range : tuple of (float, float)
        Range of coordination sphere radii.
    **kwargs
        Additional arguments passed to CN4Finder.

    Returns
    -------
    int
        Number of CN-4 sites found.
    """
    sites = find_tetrahedral_sites(metals, R_range, **kwargs)
    return len(sites)
