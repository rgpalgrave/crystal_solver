"""
Bravais Lattice Generator for Crystal Structure Prediction.

This module generates metal atom positions for all 14 Bravais lattice types,
which form the foundation for predicting anion positions using sphere
intersection methods.

The 14 Bravais Lattices by Crystal System:
    - Cubic (3): cP (simple), cI (body-centered), cF (face-centered)
    - Tetragonal (2): tP (simple), tI (body-centered)
    - Orthorhombic (4): oP (simple), oI (body), oF (face), oC (base)
    - Hexagonal (1): hP
    - Rhombohedral (1): hR
    - Monoclinic (2): mP (simple), mC (base-centered)
    - Triclinic (1): aP

Units:
    All distances are in Angstroms (Å) unless otherwise specified.
    Angles are in degrees unless specified as radians.

Author: Crystal Structure Solver Project
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Union


class BravaisLattice:
    """
    Bravais lattice generator for all 14 lattice types.

    This class generates atomic positions for any of the 14 Bravais lattice
    types, with support for arbitrary lattice parameters and unit cell
    repetitions.

    Lattice type codes:
        - cP: Primitive cubic (simple cubic)
        - cI: Body-centered cubic (BCC)
        - cF: Face-centered cubic (FCC)
        - tP: Primitive tetragonal
        - tI: Body-centered tetragonal (e.g., rutile Ti sublattice)
        - oP: Primitive orthorhombic
        - oI: Body-centered orthorhombic
        - oF: Face-centered orthorhombic
        - oC: Base-centered orthorhombic (C-centered)
        - hP: Hexagonal
        - hR: Rhombohedral (trigonal)
        - mP: Primitive monoclinic
        - mC: Base-centered monoclinic
        - aP: Triclinic (no symmetry)

    Attributes
    ----------
    lattice_type : str
        Two-letter Bravais lattice code.
    params : dict
        Lattice parameters (a, b, c, alpha, beta, gamma).
    basis_vectors : np.ndarray
        3x3 matrix of basis vectors (rows are vectors).
    motif : np.ndarray
        Fractional coordinates of atoms in the basis.

    Examples
    --------
    >>> lattice = BravaisLattice('cF', {'a': 4.05})
    >>> atoms = lattice.generate_atoms(n_cells=2)
    >>> print(f"Generated {len(atoms)} atoms")

    >>> # Body-centered tetragonal for rutile
    >>> lattice = BravaisLattice('tI', {'a': 4.59, 'c': 2.96})
    >>> atoms = lattice.generate_atoms(n_cells=1)
    """

    # Class-level constants for lattice properties
    LATTICE_TYPES = {
        'cP', 'cI', 'cF',  # Cubic
        'tP', 'tI',        # Tetragonal
        'oP', 'oI', 'oF', 'oC',  # Orthorhombic
        'hP', 'hR',        # Hexagonal/Rhombohedral
        'mP', 'mC',        # Monoclinic
        'aP'               # Triclinic
    }

    # Coordination numbers for each lattice type
    COORDINATION_NUMBERS = {
        'cP': 6, 'cI': 8, 'cF': 12,
        'tP': 6, 'tI': 8,
        'oP': 6, 'oI': 8, 'oF': 12, 'oC': 6,
        'hP': 12, 'hR': 12,
        'mP': 6, 'mC': 6,
        'aP': 6
    }

    def __init__(self, lattice_type: str, params: Dict[str, float]):
        """
        Initialize Bravais lattice.

        Parameters
        ----------
        lattice_type : str
            Two-letter code (e.g., 'cP', 'tI', 'cF').
        params : dict
            Lattice parameters. Required keys depend on lattice type:
            - All types need 'a'
            - Tetragonal, hexagonal need 'c' (or defaults to 'a')
            - Rhombohedral needs 'a' and 'alpha' (or defaults)
            - Orthorhombic needs 'a', 'b', 'c'
            - Monoclinic needs 'a', 'b', 'c', 'beta'
            - Triclinic needs 'a', 'b', 'c', 'alpha', 'beta', 'gamma'

        Raises
        ------
        ValueError
            If lattice_type is not recognized or required parameters missing.
        """
        if lattice_type not in self.LATTICE_TYPES:
            raise ValueError(f"Unknown lattice type: {lattice_type}. "
                           f"Valid types: {sorted(self.LATTICE_TYPES)}")

        self.lattice_type = lattice_type
        self.params = self._validate_and_fill_params(params)
        self.basis_vectors = self._compute_basis_vectors()
        self.motif = self._get_motif()
        self._inverse_basis = None  # Cached for coordinate conversion

    def _validate_and_fill_params(self, params: Dict[str, float]) -> Dict[str, float]:
        """Validate parameters and fill defaults based on lattice type."""
        p = params.copy()

        # Ensure 'a' is always present
        if 'a' not in p:
            raise ValueError("Parameter 'a' is required for all lattice types")

        a = p['a']

        # Set defaults based on crystal system
        system = self.lattice_type[0]  # First letter indicates system

        if system == 'c':  # Cubic
            p.setdefault('b', a)
            p.setdefault('c', a)
            p.setdefault('alpha', 90.0)
            p.setdefault('beta', 90.0)
            p.setdefault('gamma', 90.0)

        elif system == 't':  # Tetragonal
            p.setdefault('b', a)
            p.setdefault('c', a)  # Will often be overridden
            p.setdefault('alpha', 90.0)
            p.setdefault('beta', 90.0)
            p.setdefault('gamma', 90.0)

        elif system == 'o':  # Orthorhombic
            p.setdefault('b', a)
            p.setdefault('c', a)
            p.setdefault('alpha', 90.0)
            p.setdefault('beta', 90.0)
            p.setdefault('gamma', 90.0)

        elif system == 'h':  # Hexagonal or Rhombohedral
            if self.lattice_type == 'hP':
                p.setdefault('b', a)
                p.setdefault('c', a * np.sqrt(8/3))  # Ideal c/a
                p.setdefault('alpha', 90.0)
                p.setdefault('beta', 90.0)
                p.setdefault('gamma', 120.0)
            else:  # hR (Rhombohedral)
                p.setdefault('b', a)
                p.setdefault('c', a)
                p.setdefault('alpha', 60.0)  # Rhombohedral angle
                p.setdefault('beta', 60.0)
                p.setdefault('gamma', 60.0)

        elif system == 'm':  # Monoclinic
            p.setdefault('b', a)
            p.setdefault('c', a)
            p.setdefault('alpha', 90.0)
            p.setdefault('beta', 90.0)  # Typically != 90
            p.setdefault('gamma', 90.0)

        elif system == 'a':  # Triclinic
            p.setdefault('b', a)
            p.setdefault('c', a)
            p.setdefault('alpha', 90.0)
            p.setdefault('beta', 90.0)
            p.setdefault('gamma', 90.0)

        return p

    def _compute_basis_vectors(self) -> np.ndarray:
        """
        Compute the basis vectors for the lattice.

        Returns 3x3 matrix where each row is a basis vector.
        Uses the convention where:
        - a₁ is along x-axis
        - a₂ is in the xy-plane
        - a₃ has components in all three directions (for non-orthogonal systems)
        """
        a = self.params['a']
        b = self.params['b']
        c = self.params['c']
        alpha = np.radians(self.params['alpha'])  # Angle between b and c
        beta = np.radians(self.params['beta'])    # Angle between a and c
        gamma = np.radians(self.params['gamma'])  # Angle between a and b

        # Special case: hexagonal
        if self.lattice_type == 'hP':
            # Hexagonal basis: a₁ along x, a₂ at 120° in xy-plane
            a1 = np.array([a, 0, 0])
            a2 = np.array([a * np.cos(np.radians(120)), a * np.sin(np.radians(120)), 0])
            a3 = np.array([0, 0, c])
            return np.array([a1, a2, a3])

        # Special case: rhombohedral
        if self.lattice_type == 'hR':
            # Rhombohedral in terms of primitive vectors
            cos_alpha = np.cos(alpha)
            sin_alpha = np.sin(alpha)
            
            # Use standard rhombohedral convention
            a1 = np.array([a, 0, 0])
            a2 = np.array([a * cos_alpha, a * sin_alpha, 0])
            
            # a3 makes angle alpha with both a1 and a2
            a3x = a * cos_alpha
            a3y = a * (cos_alpha - cos_alpha * cos_alpha) / sin_alpha
            a3z = a * np.sqrt(1 - a3x**2/a**2 - a3y**2/a**2)
            a3 = np.array([a3x, a3y, a3z])
            
            return np.array([a1, a2, a3])

        # General case using crystallographic convention
        # a₁ along x-axis
        a1 = np.array([a, 0, 0])

        # a₂ in xy-plane
        a2 = np.array([b * np.cos(gamma), b * np.sin(gamma), 0])

        # a₃ general direction
        a3x = c * np.cos(beta)
        a3y = c * (np.cos(alpha) - np.cos(beta) * np.cos(gamma)) / np.sin(gamma)
        a3z_sq = c**2 - a3x**2 - a3y**2
        a3z = np.sqrt(max(a3z_sq, 0))  # Protect against numerical issues
        a3 = np.array([a3x, a3y, a3z])

        return np.array([a1, a2, a3])

    def _get_motif(self) -> np.ndarray:
        """
        Get fractional coordinates of atoms in the unit cell motif.

        Returns array of shape (n_atoms, 3) with fractional coordinates.
        """
        ltype = self.lattice_type

        # Primitive lattices: single atom at origin
        if ltype in ['cP', 'tP', 'oP', 'hP', 'mP', 'aP']:
            return np.array([[0.0, 0.0, 0.0]])

        # Body-centered: origin + body center
        if ltype in ['cI', 'tI', 'oI']:
            return np.array([
                [0.0, 0.0, 0.0],
                [0.5, 0.5, 0.5]
            ])

        # Face-centered: origin + 3 face centers
        if ltype in ['cF', 'oF']:
            return np.array([
                [0.0, 0.0, 0.0],
                [0.5, 0.5, 0.0],
                [0.5, 0.0, 0.5],
                [0.0, 0.5, 0.5]
            ])

        # Base-centered (C-centered): origin + center of ab face
        if ltype in ['oC', 'mC']:
            return np.array([
                [0.0, 0.0, 0.0],
                [0.5, 0.5, 0.0]
            ])

        # Rhombohedral: single atom in primitive cell
        if ltype == 'hR':
            return np.array([[0.0, 0.0, 0.0]])

        # Default fallback
        return np.array([[0.0, 0.0, 0.0]])

    def generate_atoms(self, n_cells: int = 1) -> np.ndarray:
        """
        Generate metal atom positions in Cartesian coordinates.

        Creates a supercell by repeating the unit cell in all three
        directions. The supercell spans from -n_cells to +n_cells
        in each direction (for a total of (2*n_cells+1)³ unit cells).

        Parameters
        ----------
        n_cells : int, optional
            Number of unit cells in each direction from origin.
            Default is 1, which generates a 3×3×3 supercell.

        Returns
        -------
        np.ndarray
            Array of shape (N, 3) with Cartesian coordinates of all atoms.
        """
        atoms = []

        # Generate all unit cell translations
        for i in range(-n_cells, n_cells + 1):
            for j in range(-n_cells, n_cells + 1):
                for k in range(-n_cells, n_cells + 1):
                    # Translation vector in fractional coordinates
                    translation = np.array([i, j, k], dtype=np.float64)

                    # Add each atom in motif
                    for atom_frac in self.motif:
                        frac_pos = atom_frac + translation
                        # Convert to Cartesian
                        cart_pos = frac_pos @ self.basis_vectors
                        atoms.append(cart_pos)

        return np.array(atoms)

    def get_cartesian_coords(self, fractional: np.ndarray) -> np.ndarray:
        """
        Convert fractional to Cartesian coordinates.

        Parameters
        ----------
        fractional : np.ndarray
            Fractional coordinates, shape (N, 3) or (3,).

        Returns
        -------
        np.ndarray
            Cartesian coordinates, same shape as input.
        """
        fractional = np.atleast_2d(fractional)
        # Cartesian = fractional @ basis_vectors
        cartesian = fractional @ self.basis_vectors
        return cartesian.squeeze() if cartesian.shape[0] == 1 else cartesian

    def get_fractional_coords(self, cartesian: np.ndarray) -> np.ndarray:
        """
        Convert Cartesian to fractional coordinates.

        Parameters
        ----------
        cartesian : np.ndarray
            Cartesian coordinates, shape (N, 3) or (3,).

        Returns
        -------
        np.ndarray
            Fractional coordinates, same shape as input.
        """
        cartesian = np.atleast_2d(cartesian)

        # Cache inverse basis matrix
        if self._inverse_basis is None:
            self._inverse_basis = np.linalg.inv(self.basis_vectors)

        # Fractional = cartesian @ inverse_basis
        fractional = cartesian @ self._inverse_basis
        return fractional.squeeze() if fractional.shape[0] == 1 else fractional

    def get_basis_vectors(self) -> np.ndarray:
        """
        Return 3x3 matrix of basis vectors.

        Returns
        -------
        np.ndarray
            Basis vectors as rows: [[a1x, a1y, a1z],
                                    [a2x, a2y, a2z],
                                    [a3x, a3y, a3z]]
        """
        return self.basis_vectors.copy()

    def get_coordination_number(self) -> int:
        """
        Return typical coordination number for this lattice type.

        Note: This is the coordination for the ideal lattice.
        Actual coordination may vary depending on structure.

        Returns
        -------
        int
            Coordination number (6, 8, or 12 for common lattices).
        """
        return self.COORDINATION_NUMBERS.get(self.lattice_type, 6)

    def get_nearest_neighbor_distance(self) -> float:
        """
        Calculate nearest neighbor distance for this lattice.

        Returns
        -------
        float
            Nearest neighbor distance in Angstroms.
        """
        a = self.params['a']
        c = self.params.get('c', a)
        ltype = self.lattice_type

        if ltype == 'cP':
            return a
        elif ltype == 'cI':
            return a * np.sqrt(3) / 2
        elif ltype == 'cF':
            return a / np.sqrt(2)
        elif ltype == 'tP':
            return min(a, c)
        elif ltype == 'tI':
            # Body diagonal / 2
            return np.sqrt(a**2/2 + c**2/4)
        elif ltype == 'hP':
            return a  # In-plane NN distance
        elif ltype == 'hR':
            return a  # Edge length
        else:
            # For other types, compute from generated atoms
            atoms = self.generate_atoms(n_cells=1)
            if len(atoms) < 2:
                return a
            # Find minimum non-zero distance
            distances = []
            for i, atom in enumerate(atoms):
                for j in range(i + 1, len(atoms)):
                    d = np.linalg.norm(atom - atoms[j])
                    if d > 1e-6:
                        distances.append(d)
            return min(distances) if distances else a

    def get_unit_cell_volume(self) -> float:
        """
        Calculate the unit cell volume.

        Returns
        -------
        float
            Volume in Å³.
        """
        return np.abs(np.linalg.det(self.basis_vectors))

    def get_reciprocal_vectors(self) -> np.ndarray:
        """
        Calculate reciprocal lattice vectors.

        Returns
        -------
        np.ndarray
            3x3 matrix of reciprocal vectors (rows).
        """
        V = self.get_unit_cell_volume()
        a1, a2, a3 = self.basis_vectors

        b1 = 2 * np.pi * np.cross(a2, a3) / V
        b2 = 2 * np.pi * np.cross(a3, a1) / V
        b3 = 2 * np.pi * np.cross(a1, a2) / V

        return np.array([b1, b2, b3])

    def wrap_to_unit_cell(self, fractional: np.ndarray) -> np.ndarray:
        """
        Wrap fractional coordinates to [0, 1) range.

        Parameters
        ----------
        fractional : np.ndarray
            Fractional coordinates, any shape with last dim = 3.

        Returns
        -------
        np.ndarray
            Wrapped coordinates in [0, 1).
        """
        return fractional % 1.0

    def find_equivalent_positions(self, position: np.ndarray,
                                   tolerance: float = 0.01) -> np.ndarray:
        """
        Find all symmetry-equivalent positions in the unit cell.

        Parameters
        ----------
        position : np.ndarray
            Fractional coordinates of a position, shape (3,).
        tolerance : float, optional
            Tolerance for identifying equivalent positions.

        Returns
        -------
        np.ndarray
            Array of unique equivalent positions, shape (N, 3).
        """
        # This is a simplified version - full implementation would use
        # space group operations
        positions = [self.wrap_to_unit_cell(position)]

        # For now, just return the input position wrapped to unit cell
        # A full implementation would apply all space group symmetry operations
        return np.array(positions)

    def __repr__(self) -> str:
        """String representation of lattice."""
        return (f"BravaisLattice(type='{self.lattice_type}', "
                f"a={self.params['a']:.4f}, "
                f"atoms_per_cell={len(self.motif)})")


# =============================================================================
# Convenience functions for common lattice types
# =============================================================================

def simple_cubic(a: float, n_cells: int = 1) -> np.ndarray:
    """
    Generate simple cubic lattice (cP).

    Parameters
    ----------
    a : float
        Lattice parameter (Å).
    n_cells : int, optional
        Number of unit cells in each direction.

    Returns
    -------
    np.ndarray
        Atom positions, shape (N, 3).

    Notes
    -----
    Structure:
        - Atoms at corners of cube
        - Coordination number: 6
        - Nearest neighbor distance: a
    """
    lattice = BravaisLattice('cP', {'a': a})
    return lattice.generate_atoms(n_cells)


def body_centered_cubic(a: float, n_cells: int = 1) -> np.ndarray:
    """
    Generate body-centered cubic lattice (cI/BCC).

    Parameters
    ----------
    a : float
        Lattice parameter (Å).
    n_cells : int, optional
        Number of unit cells in each direction.

    Returns
    -------
    np.ndarray
        Atom positions, shape (N, 3).

    Notes
    -----
    Structure:
        - Atoms at corners + body center
        - Coordination number: 8
        - Nearest neighbor distance: a√3/2 ≈ 0.866a

    Examples: Fe, Cr, W, Mo at room temperature
    """
    lattice = BravaisLattice('cI', {'a': a})
    return lattice.generate_atoms(n_cells)


def face_centered_cubic(a: float, n_cells: int = 1) -> np.ndarray:
    """
    Generate face-centered cubic lattice (cF/FCC).

    Parameters
    ----------
    a : float
        Lattice parameter (Å).
    n_cells : int, optional
        Number of unit cells in each direction.

    Returns
    -------
    np.ndarray
        Atom positions, shape (N, 3).

    Notes
    -----
    Structure:
        - Atoms at corners + face centers
        - Coordination number: 12
        - Nearest neighbor distance: a/√2 ≈ 0.707a

    Examples: Cu, Ag, Au, Al, Ni
    """
    lattice = BravaisLattice('cF', {'a': a})
    return lattice.generate_atoms(n_cells)


def body_centered_tetragonal(a: float, c: float, n_cells: int = 1) -> np.ndarray:
    """
    Generate body-centered tetragonal lattice (tI).

    Parameters
    ----------
    a : float
        Basal plane lattice parameter (Å).
    c : float
        Axial lattice parameter (Å).
    n_cells : int, optional
        Number of unit cells in each direction.

    Returns
    -------
    np.ndarray
        Atom positions, shape (N, 3).

    Notes
    -----
    Structure:
        - Like BCC but with c ≠ a
        - Atoms at corners + body center
        - Coordination number: 8 (can vary with c/a)

    The rutile (TiO₂) structure has Ti atoms on this lattice
    with c/a ≈ 0.644.
    """
    lattice = BravaisLattice('tI', {'a': a, 'c': c})
    return lattice.generate_atoms(n_cells)


def primitive_tetragonal(a: float, c: float, n_cells: int = 1) -> np.ndarray:
    """
    Generate primitive tetragonal lattice (tP).

    Parameters
    ----------
    a : float
        Basal plane lattice parameter (Å).
    c : float
        Axial lattice parameter (Å).
    n_cells : int, optional
        Number of unit cells in each direction.

    Returns
    -------
    np.ndarray
        Atom positions, shape (N, 3).
    """
    lattice = BravaisLattice('tP', {'a': a, 'c': c})
    return lattice.generate_atoms(n_cells)


def hexagonal(a: float, c: float, n_cells: int = 1) -> np.ndarray:
    """
    Generate hexagonal lattice (hP).

    Parameters
    ----------
    a : float
        Basal plane lattice parameter (Å).
    c : float
        Axial lattice parameter (Å).
    n_cells : int, optional
        Number of unit cells in each direction.

    Returns
    -------
    np.ndarray
        Atom positions, shape (N, 3).

    Notes
    -----
    Structure:
        - Hexagonal basis in ab-plane
        - Coordination number: 12 (for ideal c/a = √(8/3) ≈ 1.633)
        - Nearest neighbor distance: a (in-plane)

    Examples: Zn, Mg, Ti (α-phase)
    """
    lattice = BravaisLattice('hP', {'a': a, 'c': c})
    return lattice.generate_atoms(n_cells)


def rhombohedral(a: float, alpha: float = 60.0, n_cells: int = 1) -> np.ndarray:
    """
    Generate rhombohedral lattice (hR).

    Parameters
    ----------
    a : float
        Edge length of rhombohedron (Å).
    alpha : float, optional
        Rhombohedral angle in degrees. Default is 60°.
    n_cells : int, optional
        Number of unit cells in each direction.

    Returns
    -------
    np.ndarray
        Atom positions, shape (N, 3).

    Notes
    -----
    The rhombohedral lattice is like a cube stretched along
    the body diagonal. At α = 60°, it's equivalent to FCC;
    at α = 90°, it's simple cubic.
    """
    lattice = BravaisLattice('hR', {'a': a, 'alpha': alpha, 'beta': alpha, 'gamma': alpha})
    return lattice.generate_atoms(n_cells)


def primitive_orthorhombic(a: float, b: float, c: float, n_cells: int = 1) -> np.ndarray:
    """
    Generate primitive orthorhombic lattice (oP).

    Parameters
    ----------
    a, b, c : float
        Lattice parameters (Å).
    n_cells : int, optional
        Number of unit cells in each direction.

    Returns
    -------
    np.ndarray
        Atom positions, shape (N, 3).
    """
    lattice = BravaisLattice('oP', {'a': a, 'b': b, 'c': c})
    return lattice.generate_atoms(n_cells)


def body_centered_orthorhombic(a: float, b: float, c: float, n_cells: int = 1) -> np.ndarray:
    """
    Generate body-centered orthorhombic lattice (oI).

    Parameters
    ----------
    a, b, c : float
        Lattice parameters (Å).
    n_cells : int, optional
        Number of unit cells in each direction.

    Returns
    -------
    np.ndarray
        Atom positions, shape (N, 3).
    """
    lattice = BravaisLattice('oI', {'a': a, 'b': b, 'c': c})
    return lattice.generate_atoms(n_cells)


def face_centered_orthorhombic(a: float, b: float, c: float, n_cells: int = 1) -> np.ndarray:
    """
    Generate face-centered orthorhombic lattice (oF).

    Parameters
    ----------
    a, b, c : float
        Lattice parameters (Å).
    n_cells : int, optional
        Number of unit cells in each direction.

    Returns
    -------
    np.ndarray
        Atom positions, shape (N, 3).
    """
    lattice = BravaisLattice('oF', {'a': a, 'b': b, 'c': c})
    return lattice.generate_atoms(n_cells)


def base_centered_orthorhombic(a: float, b: float, c: float, n_cells: int = 1) -> np.ndarray:
    """
    Generate base-centered orthorhombic lattice (oC).

    Parameters
    ----------
    a, b, c : float
        Lattice parameters (Å).
    n_cells : int, optional
        Number of unit cells in each direction.

    Returns
    -------
    np.ndarray
        Atom positions, shape (N, 3).
    """
    lattice = BravaisLattice('oC', {'a': a, 'b': b, 'c': c})
    return lattice.generate_atoms(n_cells)


def primitive_monoclinic(a: float, b: float, c: float, beta: float = 90.0,
                         n_cells: int = 1) -> np.ndarray:
    """
    Generate primitive monoclinic lattice (mP).

    Parameters
    ----------
    a, b, c : float
        Lattice parameters (Å).
    beta : float, optional
        Monoclinic angle in degrees. Default is 90°.
    n_cells : int, optional
        Number of unit cells in each direction.

    Returns
    -------
    np.ndarray
        Atom positions, shape (N, 3).
    """
    lattice = BravaisLattice('mP', {'a': a, 'b': b, 'c': c, 'beta': beta})
    return lattice.generate_atoms(n_cells)


def base_centered_monoclinic(a: float, b: float, c: float, beta: float = 90.0,
                              n_cells: int = 1) -> np.ndarray:
    """
    Generate base-centered monoclinic lattice (mC).

    Parameters
    ----------
    a, b, c : float
        Lattice parameters (Å).
    beta : float, optional
        Monoclinic angle in degrees. Default is 90°.
    n_cells : int, optional
        Number of unit cells in each direction.

    Returns
    -------
    np.ndarray
        Atom positions, shape (N, 3).
    """
    lattice = BravaisLattice('mC', {'a': a, 'b': b, 'c': c, 'beta': beta})
    return lattice.generate_atoms(n_cells)


def triclinic(a: float, b: float, c: float,
              alpha: float = 90.0, beta: float = 90.0, gamma: float = 90.0,
              n_cells: int = 1) -> np.ndarray:
    """
    Generate triclinic lattice (aP).

    Parameters
    ----------
    a, b, c : float
        Lattice parameters (Å).
    alpha, beta, gamma : float, optional
        Angles in degrees. Default is 90° for all.
    n_cells : int, optional
        Number of unit cells in each direction.

    Returns
    -------
    np.ndarray
        Atom positions, shape (N, 3).

    Notes
    -----
    The triclinic system has no symmetry constraints.
    All lattice parameters and angles can be different.
    """
    lattice = BravaisLattice('aP', {'a': a, 'b': b, 'c': c,
                                     'alpha': alpha, 'beta': beta, 'gamma': gamma})
    return lattice.generate_atoms(n_cells)
