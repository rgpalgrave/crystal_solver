"""
Unified Structure Analyzer for Crystal Coordination Site Prediction.

This module combines CN-3, CN-4, and CN-6 coordination finders with
stability ranking, stoichiometry deduction, and reference structure
comparison capabilities.

The StructureAnalyzer provides a complete analytical workflow:
1. Find all coordination sites (CN-3, CN-4, CN-6)
2. Calculate stability metrics
3. Deduce stoichiometry from site occupancies
4. Rank structures by combined stability score
5. Compare predictions to known reference structures

Units:
    Distances in Angstroms (Å)
    Fractional coordinates dimensionless [0, 1)

Author: Crystal Structure Solver Project
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from fractions import Fraction
from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass, field

# Import coordination finders - these use the package structure
try:
    from ..coordination.cn3_finder import CN3Finder
    from ..coordination.cn4_finder import CN4Finder
    from ..coordination.cn6_finder import CN6Finder
    from ..lattices.bravais import BravaisLattice
except ImportError:
    # Fallback for standalone usage
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from coordination.cn3_finder import CN3Finder
    from coordination.cn4_finder import CN4Finder
    from coordination.cn6_finder import CN6Finder
    from lattices.bravais import BravaisLattice


# Pauling's radius ratio rules for coordination numbers
PAULING_RATIO_RANGES = {
    3: (0.155, 0.225),   # Trigonal planar
    4: (0.225, 0.414),   # Tetrahedral
    6: (0.414, 0.732),   # Octahedral
    8: (0.732, 1.000),   # Cubic
    12: (1.000, np.inf)  # Cuboctahedral
}


@dataclass
class StabilityWeights:
    """Configurable weights for stability scoring."""
    distance_ratios: float = 0.30
    pauling_rules: float = 0.25
    packing_efficiency: float = 0.15
    coordination_gaps: float = 0.20
    madelung_energy: float = 0.10
    
    def normalize(self, include_madelung: bool = True) -> 'StabilityWeights':
        """Return normalized weights summing to 1.0."""
        if include_madelung:
            total = (self.distance_ratios + self.pauling_rules + 
                    self.packing_efficiency + self.coordination_gaps + 
                    self.madelung_energy)
        else:
            total = (self.distance_ratios + self.pauling_rules + 
                    self.packing_efficiency + self.coordination_gaps)
        
        if total == 0:
            total = 1.0
            
        return StabilityWeights(
            distance_ratios=self.distance_ratios / total,
            pauling_rules=self.pauling_rules / total,
            packing_efficiency=self.packing_efficiency / total,
            coordination_gaps=self.coordination_gaps / total,
            madelung_energy=self.madelung_energy / total if include_madelung else 0.0
        )


class StructureAnalyzer:
    """
    Unified interface for complete crystal structure analysis.
    
    Combines CN-3, CN-4, and CN-6 finders with stability metrics
    and stoichiometry deduction.
    
    Parameters
    ----------
    madelung_calculator : callable, optional
        Function that calculates Madelung energy for a structure.
        Signature: madelung_calc(cation_positions, anion_positions, charges) -> float
    stability_weights : StabilityWeights, optional
        Custom weights for stability scoring.
    
    Attributes
    ----------
    cn3_finder : CN3Finder
        Finder for trigonal planar (CN-3) sites.
    cn4_finder : CN4Finder
        Finder for tetrahedral (CN-4) sites.
    cn6_finder : CN6Finder
        Finder for octahedral (CN-6) sites.
    madelung_calc : callable or None
        Optional Madelung energy calculator.
    weights : StabilityWeights
        Weights for combining stability metrics.
    
    Examples
    --------
    >>> from src.lattices.bravais import BravaisLattice
    >>> lattice = BravaisLattice('tI', {'a': 1.0, 'c': 0.644})
    >>> analyzer = StructureAnalyzer()
    >>> results = analyzer.analyze_framework(lattice, R_cation=0.6, R_anion=1.4)
    >>> print(f"Found stoichiometry: {results['stoichiometry']}")
    """
    
    def __init__(self, madelung_calculator: Optional[callable] = None,
                 stability_weights: Optional[StabilityWeights] = None):
        """
        Initialize analyzer with optional Madelung energy calculator.
        
        Args:
            madelung_calculator: Optional integration with existing
                                Madelung energy calculation code
            stability_weights: Custom weights for stability scoring
        """
        self.cn3_finder = CN3Finder()
        self.cn4_finder = CN4Finder()
        self.cn6_finder = CN6Finder()
        self.madelung_calc = madelung_calculator
        self.weights = stability_weights or StabilityWeights()
    
    def analyze_framework(self, lattice: BravaisLattice, 
                          R_cation: float, R_anion: float,
                          n_cells: int = 1,
                          R_tolerance: float = 0.2) -> Dict[str, Any]:
        """
        Complete analysis of crystal structure.
        
        Workflow:
        1. Find all CN-3, CN-4, CN-6 sites
        2. Calculate stability metrics for each
        3. Deduce stoichiometry
        4. Rank structures
        
        Parameters
        ----------
        lattice : BravaisLattice
            Bravais lattice with metal positions.
        R_cation : float
            Cation radius (Å).
        R_anion : float
            Anion radius (Å).
        n_cells : int, optional
            Number of unit cells to generate. Default is 1.
        R_tolerance : float, optional
            Fractional tolerance on R search range. Default is 0.2.
        
        Returns
        -------
        dict
            Complete analysis results:
            - 'cn3_sites': List[dict] - All CN-3 sites found
            - 'cn4_sites': List[dict] - All CN-4 sites found  
            - 'cn6_sites': List[dict] - All CN-6 sites found
            - 'all_sites': List[dict] - Combined list of all sites
            - 'stoichiometry': str - e.g., "MX2"
            - 'stability_metrics': dict - Combined metrics
            - 'ranked_structures': pd.DataFrame - Ranked by stability
            - 'lattice_info': dict - Lattice parameters and type
        """
        # Calculate coordination sphere radius
        R = R_cation + R_anion
        R_min = R * (1 - R_tolerance)
        R_max = R * (1 + R_tolerance)
        R_range = (R_min, R_max)
        
        # Generate metal positions
        metals = lattice.generate_atoms(n_cells=n_cells)
        n_cations = len(lattice.motif)  # Cations per unit cell
        
        # Find all coordination sites
        cn3_sites = self.cn3_finder.find_all_sites(metals, R_range, lattice=lattice)
        cn4_sites = self.cn4_finder.find_all_sites(metals, R_range, lattice=lattice)
        cn6_sites = self.cn6_finder.find_all_sites(metals, R_range, lattice=lattice)
        
        # Add coordination number to each site for unified handling
        for site in cn3_sites:
            site['coordination_number'] = 3
        for site in cn4_sites:
            site['coordination_number'] = 4
        for site in cn6_sites:
            site['coordination_number'] = 6
        
        # Combine all sites
        all_sites = cn3_sites + cn4_sites + cn6_sites
        
        # Filter to unique sites within unit cell
        unique_sites = self._filter_to_unit_cell(all_sites, lattice)
        
        # Calculate stability metrics
        stability_metrics = self.calculate_stability_metrics(
            unique_sites, lattice, R_cation, R_anion, metals
        )
        
        # Deduce stoichiometry
        stoichiometry = self.deduce_stoichiometry(unique_sites, n_cations)
        
        # Rank structures
        ranked_df = self.rank_structures(unique_sites, stability_metrics)
        
        return {
            'cn3_sites': cn3_sites,
            'cn4_sites': cn4_sites,
            'cn6_sites': cn6_sites,
            'all_sites': unique_sites,
            'stoichiometry': stoichiometry,
            'stability_metrics': stability_metrics,
            'ranked_structures': ranked_df,
            'lattice_info': {
                'type': lattice.lattice_type,
                'params': lattice.params.copy(),
                'n_cations_per_cell': n_cations,
                'unit_cell_volume': lattice.get_unit_cell_volume()
            }
        }
    
    def _filter_to_unit_cell(self, sites: List[Dict], 
                              lattice: BravaisLattice,
                              tolerance: float = 0.05) -> List[Dict]:
        """
        Filter sites to those within the unit cell and remove duplicates.
        
        Parameters
        ----------
        sites : list of dict
            All sites found (may include duplicates from periodic images).
        lattice : BravaisLattice
            Lattice for coordinate conversion.
        tolerance : float
            Tolerance for duplicate detection.
            
        Returns
        -------
        list of dict
            Unique sites with fractional coordinates in [0, 1).
        """
        if not sites:
            return []
        
        unique_sites = []
        
        for site in sites:
            # Get fractional coordinates
            if site.get('fractional') is not None:
                frac = np.asarray(site['fractional'])
            else:
                frac = lattice.get_fractional_coords(site['position'])
            
            # Wrap to unit cell [0, 1)
            frac_wrapped = frac % 1.0
            # Handle numerical edge case
            frac_wrapped = np.where(np.isclose(frac_wrapped, 1.0, atol=1e-10), 
                                    0.0, frac_wrapped)
            
            # Check if this is a duplicate
            is_duplicate = False
            duplicate_idx = None
            
            for idx, existing in enumerate(unique_sites):
                existing_frac = np.asarray(existing.get('fractional_wrapped', 
                                                        existing.get('fractional', [0, 0, 0])))
                
                # Check distance in fractional space (with periodic boundary)
                diff = np.abs(frac_wrapped - existing_frac)
                diff = np.minimum(diff, 1.0 - diff)  # Periodic
                
                if np.all(diff < tolerance):
                    # Found a duplicate
                    is_duplicate = True
                    # Keep the one with better stability score
                    if site.get('stability_score', 0) > existing.get('stability_score', 0):
                        duplicate_idx = idx  # Mark for replacement
                    break
            
            if is_duplicate and duplicate_idx is not None:
                # Replace the existing site with the better one
                site_copy = site.copy()
                site_copy['fractional_wrapped'] = frac_wrapped
                unique_sites[duplicate_idx] = site_copy
            elif not is_duplicate:
                site_copy = site.copy()
                site_copy['fractional_wrapped'] = frac_wrapped
                unique_sites.append(site_copy)
        
        return unique_sites
    
    def deduce_stoichiometry(self, all_sites: List[Dict], 
                              n_cations: int) -> str:
        """
        Deduce stoichiometry from site occupancies.
        
        Considers:
        - Number of each type of site
        - Multiplicity (how many equivalent sites by symmetry)
        - Partial occupancy possibilities
        
        Parameters
        ----------
        all_sites : list of dict
            Combined list of all CN sites.
        n_cations : int
            Number of cation positions in unit cell.
        
        Returns
        -------
        str
            Stoichiometry string: "MX", "MX2", "M2X3", etc.
        """
        if not all_sites or n_cations == 0:
            return "Unknown"
        
        # Count anion sites (assuming each site = 1 anion)
        n_anions = len(all_sites)
        
        if n_anions == 0:
            return "M"  # No anions found
        
        # Simplify ratio using fractions
        try:
            ratio = Fraction(n_anions, n_cations).limit_denominator(12)
            
            # Format the stoichiometry string
            m_count = ratio.denominator
            x_count = ratio.numerator
            
            if m_count == 1 and x_count == 1:
                return "MX"
            elif m_count == 1:
                return f"MX{x_count}"
            elif x_count == 1:
                return f"M{m_count}X"
            else:
                return f"M{m_count}X{x_count}"
                
        except (ValueError, ZeroDivisionError):
            return "Unknown"
    
    def calculate_stability_metrics(self, sites: List[Dict], 
                                      lattice: BravaisLattice,
                                      R_cation: float,
                                      R_anion: float,
                                      metals: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Calculate multiple stability indicators.
        
        Metrics:
        1. Distance ratios:
           - d_min(anion-cation) vs d_min(anion-anion)
           - d_min(anion-cation) vs d_min(cation-cation)
        
        2. Pauling's ratio check:
           - For each CN, verify R_cation/R_anion in valid range
        
        3. Packing efficiency:
           - Volume occupied / Total volume
        
        4. Madelung energy (if calculator provided):
           - Electrostatic energy of structure
        
        5. Coordination shell gaps:
           - Average gap_to_next for all sites
        
        Parameters
        ----------
        sites : list of dict
            All coordination sites found.
        lattice : BravaisLattice
            Lattice object for volume calculation.
        R_cation : float
            Cation radius (Å).
        R_anion : float
            Anion radius (Å).
        metals : np.ndarray, optional
            Metal atom positions.
            
        Returns
        -------
        dict
            Stability metrics.
        """
        if not sites:
            return self._empty_metrics()
        
        # Calculate radius ratio
        if R_anion > 0:
            radius_ratio = R_cation / R_anion
        else:
            radius_ratio = 0.0
        
        # Group sites by coordination number
        cn_groups = {}
        for site in sites:
            cn = site.get('coordination_number', 0)
            if cn not in cn_groups:
                cn_groups[cn] = []
            cn_groups[cn].append(site)
        
        # 1. Pauling's ratio check for each CN
        pauling_ratios = {}
        for cn, cn_sites in cn_groups.items():
            if cn in PAULING_RATIO_RANGES:
                r_min, r_max = PAULING_RATIO_RANGES[cn]
                is_valid = r_min <= radius_ratio <= r_max
            else:
                is_valid = True  # No rule defined
            
            pauling_ratios[f'cn{cn}'] = {
                'ratio': radius_ratio,
                'valid': is_valid,
                'n_sites': len(cn_sites)
            }
        
        # 2. Distance ratios
        distance_ratios = self._calculate_distance_ratios(sites, lattice, metals)
        
        # 3. Packing efficiency
        packing_efficiency = self._calculate_packing_efficiency(
            sites, lattice, R_cation, R_anion
        )
        
        # 4. Coordination shell gaps
        gaps = []
        for site in sites:
            gap = site.get('gap_to_next_shell') or site.get('gap_to_4th') or site.get('gap_to_next', 0)
            if gap and gap != float('inf'):
                gaps.append(gap)
        
        avg_coordination_gap = np.mean(gaps) if gaps else 0.0
        
        # 5. Madelung energy (if calculator provided)
        madelung_energy = None
        if self.madelung_calc is not None and sites:
            try:
                anion_positions = np.array([s['position'] for s in sites])
                cation_positions = metals if metals is not None else lattice.generate_atoms(n_cells=1)
                madelung_energy = self.madelung_calc(cation_positions, anion_positions)
            except Exception:
                madelung_energy = None
        
        # Calculate overall stability score
        overall_score = self._calculate_overall_score(
            distance_ratios, pauling_ratios, packing_efficiency,
            avg_coordination_gap, madelung_energy
        )
        
        return {
            'distance_ratios': distance_ratios,
            'pauling_ratios': pauling_ratios,
            'packing_efficiency': packing_efficiency,
            'madelung_energy': madelung_energy,
            'avg_coordination_gap': avg_coordination_gap,
            'overall_stability_score': overall_score,
            'cn_distribution': {cn: len(sites) for cn, sites in cn_groups.items()}
        }
    
    def _empty_metrics(self) -> Dict[str, Any]:
        """Return empty metrics structure."""
        return {
            'distance_ratios': {'AC_vs_AA': None, 'AC_vs_CC': None},
            'pauling_ratios': {},
            'packing_efficiency': 0.0,
            'madelung_energy': None,
            'avg_coordination_gap': 0.0,
            'overall_stability_score': 0.0,
            'cn_distribution': {}
        }
    
    def _calculate_distance_ratios(self, sites: List[Dict],
                                    lattice: BravaisLattice,
                                    metals: Optional[np.ndarray]) -> Dict[str, Optional[float]]:
        """
        Calculate distance ratios for stability assessment.
        
        Returns
        -------
        dict
            - AC_vs_AA: Anion-cation distance vs anion-anion distance
            - AC_vs_CC: Anion-cation distance vs cation-cation distance
        """
        if not sites or metals is None or len(metals) < 2:
            return {'AC_vs_AA': None, 'AC_vs_CC': None}
        
        # Get anion positions
        anion_positions = np.array([s['position'] for s in sites])
        
        # Calculate minimum cation-cation distance
        cc_distances = []
        for i in range(len(metals)):
            for j in range(i + 1, len(metals)):
                cc_distances.append(np.linalg.norm(metals[i] - metals[j]))
        d_CC = min(cc_distances) if cc_distances else 1.0
        
        # Calculate minimum anion-cation distance
        ac_distances = []
        for anion in anion_positions:
            for cation in metals:
                d = np.linalg.norm(anion - cation)
                if d > 0.1:  # Avoid zero distances
                    ac_distances.append(d)
        d_AC = min(ac_distances) if ac_distances else 1.0
        
        # Calculate minimum anion-anion distance
        if len(anion_positions) >= 2:
            aa_distances = []
            for i in range(len(anion_positions)):
                for j in range(i + 1, len(anion_positions)):
                    d = np.linalg.norm(anion_positions[i] - anion_positions[j])
                    if d > 0.1:
                        aa_distances.append(d)
            d_AA = min(aa_distances) if aa_distances else d_AC * 2
        else:
            d_AA = d_AC * 2  # Default ratio
        
        return {
            'AC_vs_AA': d_AC / d_AA if d_AA > 0 else None,
            'AC_vs_CC': d_AC / d_CC if d_CC > 0 else None
        }
    
    def _calculate_packing_efficiency(self, sites: List[Dict],
                                       lattice: BravaisLattice,
                                       R_cation: float,
                                       R_anion: float) -> float:
        """
        Calculate packing efficiency (volume fraction occupied).
        
        Returns
        -------
        float
            Packing efficiency in range [0, 1].
        """
        V_cell = lattice.get_unit_cell_volume()
        if V_cell <= 0:
            return 0.0
        
        # Volume of spheres
        n_cations = len(lattice.motif)
        n_anions = len(sites)
        
        V_cation = (4/3) * np.pi * R_cation**3
        V_anion = (4/3) * np.pi * R_anion**3
        
        V_occupied = n_cations * V_cation + n_anions * V_anion
        
        # Clamp to valid range (can exceed 1.0 if spheres overlap)
        efficiency = min(V_occupied / V_cell, 1.0)
        
        return efficiency
    
    def _calculate_overall_score(self, distance_ratios: Dict,
                                  pauling_ratios: Dict,
                                  packing_efficiency: float,
                                  avg_gap: float,
                                  madelung_energy: Optional[float]) -> float:
        """
        Calculate weighted overall stability score.
        
        Returns
        -------
        float
            Combined stability score in range [0, 1].
        """
        # Normalize weights
        include_madelung = madelung_energy is not None
        weights = self.weights.normalize(include_madelung)
        
        score = 0.0
        
        # Distance ratio score (prefer higher AC_vs_AA and AC_vs_CC)
        if distance_ratios.get('AC_vs_AA') is not None:
            # Ideal: AC should be smaller than AA (ratio < 1)
            ratio_score = min(1.0, distance_ratios['AC_vs_AA'])
            score += weights.distance_ratios * (1.0 - ratio_score)
        
        # Pauling's rules score (fraction of CNs with valid ratios)
        if pauling_ratios:
            valid_count = sum(1 for v in pauling_ratios.values() if v.get('valid', False))
            pauling_score = valid_count / len(pauling_ratios)
            score += weights.pauling_rules * pauling_score
        
        # Packing efficiency (higher is better, but cap at 0.74 for close-packed)
        score += weights.packing_efficiency * min(packing_efficiency / 0.74, 1.0)
        
        # Coordination gap score (larger gaps = more stable)
        # Normalize: gap of 0.15 = 50%, gap of 0.30 = 100%
        gap_score = min(avg_gap / 0.30, 1.0) if avg_gap > 0 else 0.0
        score += weights.coordination_gaps * gap_score
        
        # Madelung energy (if available) - more negative = more stable
        if include_madelung and madelung_energy is not None:
            # Normalize: typical range is -30 to 0 for common structures
            madelung_score = max(0.0, min(1.0, -madelung_energy / 30.0))
            score += weights.madelung_energy * madelung_score
        
        return score
    
    def rank_structures(self, all_sites: List[Dict], 
                         metrics: Dict[str, Any]) -> pd.DataFrame:
        """
        Rank all possible structures by combined stability score.
        
        Parameters
        ----------
        all_sites : list of dict
            All coordination sites found.
        metrics : dict
            Stability metrics from calculate_stability_metrics().
        
        Returns
        -------
        pd.DataFrame
            DataFrame with columns:
            ['site_id', 'coordination_number', 'R_critical', 'position',
             'fractional', 'gap_to_next', 'stability_score', 'pauling_valid']
            Sorted by stability_score (descending).
        """
        if not all_sites:
            return pd.DataFrame(columns=[
                'site_id', 'coordination_number', 'R_critical', 
                'position_x', 'position_y', 'position_z',
                'frac_x', 'frac_y', 'frac_z',
                'gap_to_next', 'stability_score', 'pauling_valid'
            ])
        
        rows = []
        pauling_ratios = metrics.get('pauling_ratios', {})
        
        for i, site in enumerate(all_sites):
            cn = site.get('coordination_number', 0)
            pos = site.get('position', np.zeros(3))
            frac = site.get('fractional_wrapped', site.get('fractional', np.zeros(3)))
            
            if frac is None:
                frac = np.zeros(3)
            
            # Get gap value
            gap = (site.get('gap_to_next_shell') or 
                   site.get('gap_to_4th') or 
                   site.get('gap_to_next', 0))
            if gap == float('inf'):
                gap = 1.0
            
            # Check Pauling validity for this CN
            cn_key = f'cn{cn}'
            pauling_valid = pauling_ratios.get(cn_key, {}).get('valid', True)
            
            # Site-level stability score
            site_score = site.get('stability_score', site.get('stability_metric', 0.5))
            
            rows.append({
                'site_id': i,
                'coordination_number': cn,
                'R_critical': site.get('R_critical', 0),
                'position_x': pos[0] if hasattr(pos, '__getitem__') else 0,
                'position_y': pos[1] if hasattr(pos, '__getitem__') else 0,
                'position_z': pos[2] if hasattr(pos, '__getitem__') else 0,
                'frac_x': frac[0] if hasattr(frac, '__getitem__') else 0,
                'frac_y': frac[1] if hasattr(frac, '__getitem__') else 0,
                'frac_z': frac[2] if hasattr(frac, '__getitem__') else 0,
                'gap_to_next': gap,
                'stability_score': site_score,
                'pauling_valid': pauling_valid
            })
        
        df = pd.DataFrame(rows)
        df = df.sort_values('stability_score', ascending=False).reset_index(drop=True)
        
        return df
    
    def compare_to_reference(self, predicted_sites: List[Dict], 
                              reference_structure: Dict,
                              lattice: Optional[BravaisLattice] = None) -> Dict[str, Any]:
        """
        Compare prediction to known structure from database.
        
        Metrics:
        1. Position RMSD: How close are predicted vs reference positions?
        2. CN match: Do coordination numbers agree?
        3. Stoichiometry match: Same formula?
        4. Site matching: Which predicted sites correspond to which reference?
        
        Parameters
        ----------
        predicted_sites : list of dict
            Sites found by analyzer.
        reference_structure : dict
            Loaded from reference database with keys:
            - 'anion_positions': list of fractional positions
            - 'stoichiometry': string
            - 'coordination_numbers': list of CNs
        lattice : BravaisLattice, optional
            Lattice for coordinate conversion.
        
        Returns
        -------
        dict
            Comparison results:
            - 'position_rmsd': float (Å or fractional)
            - 'cn_match': bool
            - 'stoichiometry_match': bool
            - 'matched_sites': List[Tuple[int, int]] - (pred_idx, ref_idx) pairs
            - 'unmatched_predicted': List[int]
            - 'unmatched_reference': List[int]
            - 'fraction_matched': float
        """
        if not predicted_sites or not reference_structure:
            return {
                'position_rmsd': None,
                'cn_match': False,
                'stoichiometry_match': False,
                'matched_sites': [],
                'unmatched_predicted': list(range(len(predicted_sites))),
                'unmatched_reference': [],
                'fraction_matched': 0.0
            }
        
        # Get reference anion positions
        ref_positions = reference_structure.get('anion_positions', [])
        if not ref_positions:
            ref_positions = reference_structure.get('anion_fractional', [])
        
        ref_positions = [np.asarray(p) for p in ref_positions]
        
        # Get predicted positions (use fractional if available)
        pred_positions = []
        for site in predicted_sites:
            frac = site.get('fractional_wrapped', site.get('fractional'))
            if frac is not None:
                pred_positions.append(np.asarray(frac))
            elif lattice is not None:
                cart = site.get('position', np.zeros(3))
                frac = lattice.get_fractional_coords(cart)
                pred_positions.append(frac % 1.0)
            else:
                pred_positions.append(site.get('position', np.zeros(3)))
        
        # Match sites using Hungarian algorithm (greedy approximation)
        matched_sites = []
        used_pred = set()
        used_ref = set()
        position_errors = []
        match_threshold = 0.1  # Fractional coordinate threshold
        
        # Greedy matching: for each reference, find closest predicted
        for ref_idx, ref_pos in enumerate(ref_positions):
            best_pred_idx = None
            best_dist = float('inf')
            
            for pred_idx, pred_pos in enumerate(pred_positions):
                if pred_idx in used_pred:
                    continue
                
                # Calculate distance with periodic boundary conditions
                diff = np.abs(pred_pos - ref_pos)
                diff = np.minimum(diff, 1.0 - diff)  # Periodic
                dist = np.linalg.norm(diff)
                
                if dist < best_dist:
                    best_dist = dist
                    best_pred_idx = pred_idx
            
            if best_pred_idx is not None and best_dist < match_threshold:
                matched_sites.append((best_pred_idx, ref_idx))
                used_pred.add(best_pred_idx)
                used_ref.add(ref_idx)
                position_errors.append(best_dist)
        
        # Calculate RMSD
        if position_errors:
            position_rmsd = np.sqrt(np.mean(np.array(position_errors)**2))
        else:
            position_rmsd = None
        
        # Unmatched sites
        unmatched_pred = [i for i in range(len(predicted_sites)) if i not in used_pred]
        unmatched_ref = [i for i in range(len(ref_positions)) if i not in used_ref]
        
        # CN match check
        ref_cns = reference_structure.get('coordination_numbers', [])
        pred_cns = [s.get('coordination_number', 0) for s in predicted_sites]
        cn_match = (sorted(ref_cns) == sorted(pred_cns)) if ref_cns else True
        
        # Stoichiometry match
        ref_stoich = reference_structure.get('stoichiometry', '')
        n_cations = reference_structure.get('n_cations', 1)
        pred_stoich = self.deduce_stoichiometry(predicted_sites, n_cations)
        stoichiometry_match = (ref_stoich.upper() == pred_stoich.upper())
        
        # Fraction matched
        total_sites = max(len(ref_positions), len(predicted_sites))
        fraction_matched = len(matched_sites) / total_sites if total_sites > 0 else 0.0
        
        return {
            'position_rmsd': position_rmsd,
            'cn_match': cn_match,
            'stoichiometry_match': stoichiometry_match,
            'matched_sites': matched_sites,
            'unmatched_predicted': unmatched_pred,
            'unmatched_reference': unmatched_ref,
            'fraction_matched': fraction_matched
        }
    
    def find_optimal_structure(self, lattice: BravaisLattice,
                                R_cation: float, R_anion: float,
                                target_stoichiometry: Optional[str] = None,
                                n_cells: int = 1) -> Dict[str, Any]:
        """
        Find the optimal structure matching constraints.
        
        Parameters
        ----------
        lattice : BravaisLattice
            Metal framework lattice.
        R_cation, R_anion : float
            Ionic radii.
        target_stoichiometry : str, optional
            If provided, filter to structures matching this stoichiometry.
        n_cells : int, optional
            Number of unit cells.
            
        Returns
        -------
        dict
            Best structure with highest stability score matching constraints.
        """
        # Run full analysis
        results = self.analyze_framework(lattice, R_cation, R_anion, n_cells)
        
        # Filter by stoichiometry if specified
        if target_stoichiometry is not None:
            if results['stoichiometry'].upper() != target_stoichiometry.upper():
                # Try different combinations of sites
                # This is a simplified approach - could be expanded
                pass
        
        return results


def load_reference_structure(name: str, 
                              data_dir: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
    """
    Load structure from data/reference_structures.json.
    
    Parameters
    ----------
    name : str
        Structure name (e.g., 'TiO2_rutile', 'NaCl_rock_salt').
    data_dir : str or Path, optional
        Directory containing reference_structures.json.
        Defaults to package data directory.
    
    Returns
    -------
    dict
        Reference structure with:
        - lattice_type, params
        - cation_positions, anion_positions (fractional)
        - coordination_numbers
        - stoichiometry
        - source/reference info
    
    Raises
    ------
    FileNotFoundError
        If reference_structures.json not found.
    KeyError
        If structure name not in database.
    """
    if data_dir is None:
        # Look in package data directory
        data_dir = Path(__file__).parent.parent.parent / 'data'
    else:
        data_dir = Path(data_dir)
    
    json_path = data_dir / 'reference_structures.json'
    
    if not json_path.exists():
        raise FileNotFoundError(f"Reference structures file not found: {json_path}")
    
    with open(json_path, 'r') as f:
        all_structures = json.load(f)
    
    # Case-insensitive lookup
    name_lower = name.lower()
    for key, structure in all_structures.items():
        if key.lower() == name_lower:
            return structure
    
    raise KeyError(f"Structure '{name}' not found. Available: {list(all_structures.keys())}")


def list_reference_structures(data_dir: Optional[Union[str, Path]] = None) -> List[str]:
    """
    List all available reference structures.
    
    Returns
    -------
    list of str
        Names of all structures in the reference database.
    """
    if data_dir is None:
        data_dir = Path(__file__).parent.parent.parent / 'data'
    else:
        data_dir = Path(data_dir)
    
    json_path = data_dir / 'reference_structures.json'
    
    if not json_path.exists():
        return []
    
    with open(json_path, 'r') as f:
        all_structures = json.load(f)
    
    return list(all_structures.keys())


# =============================================================================
# Convenience Functions
# =============================================================================

def analyze_structure(lattice_type: str, params: Dict[str, float],
                      R_cation: float, R_anion: float,
                      n_cells: int = 1) -> Dict[str, Any]:
    """
    One-line structure analysis.
    
    Parameters
    ----------
    lattice_type : str
        Bravais lattice code (e.g., 'tI', 'cF').
    params : dict
        Lattice parameters.
    R_cation, R_anion : float
        Ionic radii.
    n_cells : int, optional
        Number of unit cells.
        
    Returns
    -------
    dict
        Complete analysis results.
        
    Examples
    --------
    >>> results = analyze_structure('tI', {'a': 1.0, 'c': 0.644}, 0.6, 1.4)
    >>> print(results['stoichiometry'])
    """
    lattice = BravaisLattice(lattice_type, params)
    analyzer = StructureAnalyzer()
    return analyzer.analyze_framework(lattice, R_cation, R_anion, n_cells)


def validate_against_reference(lattice_type: str, params: Dict[str, float],
                                R_cation: float, R_anion: float,
                                reference_name: str) -> Dict[str, Any]:
    """
    Analyze structure and compare to reference.
    
    Parameters
    ----------
    lattice_type : str
        Bravais lattice code.
    params : dict
        Lattice parameters.
    R_cation, R_anion : float
        Ionic radii.
    reference_name : str
        Name of reference structure to compare.
        
    Returns
    -------
    dict
        Analysis results with comparison metrics.
    """
    lattice = BravaisLattice(lattice_type, params)
    analyzer = StructureAnalyzer()
    
    # Analyze
    results = analyzer.analyze_framework(lattice, R_cation, R_anion)
    
    # Load reference and compare
    try:
        reference = load_reference_structure(reference_name)
        comparison = analyzer.compare_to_reference(
            results['all_sites'], reference, lattice
        )
        results['reference_comparison'] = comparison
    except (FileNotFoundError, KeyError) as e:
        results['reference_comparison'] = {'error': str(e)}
    
    return results
