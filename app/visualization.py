"""
Visualization utilities for crystal structure analysis.

This module provides Plotly-based 3D visualization functions for
metal lattices, coordination sites, and crystal structures.

Author: Crystal Structure Solver Project
"""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import List, Dict, Any, Optional, Tuple, Union


# Color schemes for different elements/coordination numbers
COLORS = {
    'metal': '#1f77b4',      # Blue for metals/cations
    'cn3': '#ff7f0e',        # Orange for CN-3 sites
    'cn4': '#2ca02c',        # Green for CN-4 sites
    'cn6': '#d62728',        # Red for CN-6 sites
    'anion': '#9467bd',      # Purple for anions
    'bond': '#888888',       # Gray for bonds
    'unit_cell': '#444444',  # Dark gray for unit cell edges
}

SIZES = {
    'metal': 12,
    'cn3': 10,
    'cn4': 10,
    'cn6': 10,
    'anion': 8,
}


def create_3d_plot(metals: np.ndarray, 
                   sites: Optional[List[Dict]] = None,
                   bonds: Optional[List[Tuple[int, int]]] = None,
                   title: str = "Crystal Structure",
                   show_unit_cell: bool = True,
                   lattice_vectors: Optional[np.ndarray] = None,
                   show_axes: bool = True) -> go.Figure:
    """
    Create Plotly 3D scatter plot of crystal structure.
    
    Parameters
    ----------
    metals : np.ndarray
        Metal atom positions, shape (N, 3).
    sites : list of dict, optional
        Coordination sites with 'position' and 'coordination_number' keys.
    bonds : list of tuple, optional
        List of (i, j) pairs indicating bonds between atoms.
    title : str, optional
        Plot title.
    show_unit_cell : bool, optional
        Whether to draw unit cell edges.
    lattice_vectors : np.ndarray, optional
        3x3 matrix of lattice vectors for drawing unit cell.
    show_axes : bool, optional
        Whether to show coordinate axes.
        
    Returns
    -------
    go.Figure
        Plotly figure object.
    """
    fig = go.Figure()
    
    # Plot metal atoms
    fig.add_trace(go.Scatter3d(
        x=metals[:, 0],
        y=metals[:, 1],
        z=metals[:, 2],
        mode='markers',
        marker=dict(
            size=SIZES['metal'],
            color=COLORS['metal'],
            symbol='circle',
            line=dict(color='white', width=1)
        ),
        name='Metal atoms',
        hovertemplate='Metal<br>x: %{x:.3f}<br>y: %{y:.3f}<br>z: %{z:.3f}<extra></extra>'
    ))
    
    # Plot coordination sites by CN
    if sites:
        for cn in [3, 4, 6]:
            cn_sites = [s for s in sites if s.get('coordination_number', 0) == cn]
            if cn_sites:
                positions = np.array([s['position'] for s in cn_sites])
                fig.add_trace(go.Scatter3d(
                    x=positions[:, 0],
                    y=positions[:, 1],
                    z=positions[:, 2],
                    mode='markers',
                    marker=dict(
                        size=SIZES[f'cn{cn}'],
                        color=COLORS[f'cn{cn}'],
                        symbol='diamond',
                        line=dict(color='white', width=1)
                    ),
                    name=f'CN-{cn} sites ({len(cn_sites)})',
                    hovertemplate=f'CN-{cn}<br>x: %{{x:.3f}}<br>y: %{{y:.3f}}<br>z: %{{z:.3f}}<extra></extra>'
                ))
    
    # Plot bonds
    if bonds:
        for i, j in bonds:
            fig.add_trace(go.Scatter3d(
                x=[metals[i, 0], metals[j, 0]],
                y=[metals[i, 1], metals[j, 1]],
                z=[metals[i, 2], metals[j, 2]],
                mode='lines',
                line=dict(color=COLORS['bond'], width=2),
                showlegend=False,
                hoverinfo='skip'
            ))
    
    # Draw unit cell
    if show_unit_cell and lattice_vectors is not None:
        add_unit_cell_edges(fig, lattice_vectors)
    
    # Layout
    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title='x (Å)',
            yaxis_title='y (Å)',
            zaxis_title='z (Å)',
            aspectmode='data',
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.5)
            )
        ),
        margin=dict(l=0, r=0, t=40, b=0),
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        )
    )
    
    return fig


def add_unit_cell_edges(fig: go.Figure, lattice_vectors: np.ndarray,
                        color: str = COLORS['unit_cell'], 
                        width: int = 2) -> None:
    """
    Add unit cell edges to a 3D plot.
    
    Parameters
    ----------
    fig : go.Figure
        Plotly figure to add edges to.
    lattice_vectors : np.ndarray
        3x3 matrix of lattice vectors.
    color : str, optional
        Edge color.
    width : int, optional
        Edge line width.
    """
    a1, a2, a3 = lattice_vectors
    origin = np.array([0, 0, 0])
    
    # Define the 12 edges of the unit cell
    edges = [
        (origin, a1),
        (origin, a2),
        (origin, a3),
        (a1, a1 + a2),
        (a1, a1 + a3),
        (a2, a2 + a1),
        (a2, a2 + a3),
        (a3, a3 + a1),
        (a3, a3 + a2),
        (a1 + a2, a1 + a2 + a3),
        (a1 + a3, a1 + a2 + a3),
        (a2 + a3, a1 + a2 + a3),
    ]
    
    for start, end in edges:
        fig.add_trace(go.Scatter3d(
            x=[start[0], end[0]],
            y=[start[1], end[1]],
            z=[start[2], end[2]],
            mode='lines',
            line=dict(color=color, width=width, dash='dash'),
            showlegend=False,
            hoverinfo='skip'
        ))


def create_structure_viewer(metals: np.ndarray, 
                            sites: List[Dict],
                            lattice_vectors: Optional[np.ndarray] = None,
                            title: str = "Interactive Structure Viewer") -> go.Figure:
    """
    Create an interactive structure viewer with rotation controls.
    
    Parameters
    ----------
    metals : np.ndarray
        Metal atom positions.
    sites : list of dict
        Coordination sites.
    lattice_vectors : np.ndarray, optional
        Lattice vectors for unit cell display.
    title : str, optional
        Plot title.
        
    Returns
    -------
    go.Figure
        Interactive Plotly figure.
    """
    fig = create_3d_plot(
        metals, sites, 
        title=title,
        show_unit_cell=True,
        lattice_vectors=lattice_vectors
    )
    
    # Add animation controls
    fig.update_layout(
        updatemenus=[
            dict(
                type="buttons",
                showactive=False,
                y=0.8,
                x=0.1,
                xanchor="left",
                buttons=[
                    dict(
                        label="Reset View",
                        method="relayout",
                        args=[{"scene.camera": dict(eye=dict(x=1.5, y=1.5, z=1.5))}]
                    ),
                    dict(
                        label="Top View",
                        method="relayout",
                        args=[{"scene.camera": dict(eye=dict(x=0, y=0, z=2.5))}]
                    ),
                    dict(
                        label="Side View",
                        method="relayout",
                        args=[{"scene.camera": dict(eye=dict(x=2.5, y=0, z=0))}]
                    ),
                ]
            )
        ]
    )
    
    return fig


def plot_sites_3d(metals: np.ndarray, 
                  sites: List[Dict],
                  cn: Optional[int] = None,
                  lattice_vectors: Optional[np.ndarray] = None) -> go.Figure:
    """
    Plot coordination sites with metal framework.
    
    Parameters
    ----------
    metals : np.ndarray
        Metal positions.
    sites : list of dict
        Sites to plot.
    cn : int, optional
        Filter to specific coordination number.
    lattice_vectors : np.ndarray, optional
        For unit cell display.
        
    Returns
    -------
    go.Figure
        Plotly figure.
    """
    if cn is not None:
        sites = [s for s in sites if s.get('coordination_number') == cn]
    
    return create_3d_plot(
        metals, sites,
        title=f"CN-{cn} Sites" if cn else "Coordination Sites",
        lattice_vectors=lattice_vectors
    )


def plot_structure(structure: Dict[str, Any],
                   show_bonds: bool = True) -> go.Figure:
    """
    Plot a complete structure from analysis results.
    
    Parameters
    ----------
    structure : dict
        Structure dictionary with 'all_sites', 'lattice_info' keys.
    show_bonds : bool, optional
        Whether to show coordination bonds.
        
    Returns
    -------
    go.Figure
        Plotly figure.
    """
    # Extract data
    sites = structure.get('all_sites', [])
    lattice_info = structure.get('lattice_info', {})
    
    # Get lattice vectors if available
    lattice_vectors = None
    if 'basis_vectors' in lattice_info:
        lattice_vectors = np.array(lattice_info['basis_vectors'])
    
    # Get metals from lattice (simplified - just show sites)
    metals = np.array([[0, 0, 0]])  # Placeholder
    if 'metals' in structure:
        metals = np.array(structure['metals'])
    
    title = f"Structure: {lattice_info.get('type', 'Unknown')}"
    if 'stoichiometry' in structure:
        title += f" - {structure['stoichiometry']}"
    
    return create_3d_plot(
        metals, sites,
        title=title,
        lattice_vectors=lattice_vectors
    )


def plot_comparison(predicted: Dict, reference: Dict,
                    title: str = "Prediction vs Reference") -> go.Figure:
    """
    Create side-by-side comparison plot.
    
    Parameters
    ----------
    predicted : dict
        Predicted structure.
    reference : dict
        Reference structure.
    title : str, optional
        Plot title.
        
    Returns
    -------
    go.Figure
        Plotly figure with subplots.
    """
    fig = make_subplots(
        rows=1, cols=2,
        specs=[[{'type': 'scatter3d'}, {'type': 'scatter3d'}]],
        subplot_titles=('Prediction', 'Reference')
    )
    
    # This is a simplified version - full implementation would
    # extract and plot both structures properly
    
    return fig


def plot_ca_optimization(results: List[Dict[str, Any]], 
                         target_cn: int,
                         lattice_type: str) -> go.Figure:
    """
    Plot c/a ratio optimization results.
    
    Parameters
    ----------
    results : list of dict
        List with 'c_a' and 'n_sites' keys.
    target_cn : int
        Target coordination number.
    lattice_type : str
        Lattice type code.
        
    Returns
    -------
    go.Figure
        Line plot of sites vs c/a ratio.
    """
    import plotly.express as px
    import pandas as pd
    
    df = pd.DataFrame(results)
    
    fig = px.line(
        df, x='c_a', y='n_sites',
        title=f'CN-{target_cn} Sites vs c/a Ratio ({lattice_type})',
        labels={'c_a': 'c/a Ratio', 'n_sites': 'Number of Sites'}
    )
    
    fig.update_traces(mode='lines+markers')
    
    # Highlight maximum
    max_idx = df['n_sites'].idxmax()
    max_row = df.loc[max_idx]
    
    fig.add_annotation(
        x=max_row['c_a'],
        y=max_row['n_sites'],
        text=f"Optimal: c/a = {max_row['c_a']:.3f}",
        showarrow=True,
        arrowhead=2
    )
    
    return fig


def export_to_cif(lattice, sites: List[Dict], filename: str,
                  cation_symbol: str = "M", anion_symbol: str = "X") -> str:
    """
    Export structure in CIF format.
    
    Parameters
    ----------
    lattice : BravaisLattice
        Lattice object.
    sites : list of dict
        Coordination sites (anion positions).
    filename : str
        Output filename.
    cation_symbol : str, optional
        Symbol for cation.
    anion_symbol : str, optional
        Symbol for anion.
        
    Returns
    -------
    str
        CIF file content as string.
    """
    params = lattice.params
    
    cif_content = f"""data_{filename.replace('.cif', '')}
_cell_length_a    {params['a']:.6f}
_cell_length_b    {params['b']:.6f}
_cell_length_c    {params['c']:.6f}
_cell_angle_alpha {params['alpha']:.3f}
_cell_angle_beta  {params['beta']:.3f}
_cell_angle_gamma {params['gamma']:.3f}
_symmetry_space_group_name_H-M 'P1'
_symmetry_Int_Tables_number    1

loop_
_atom_site_label
_atom_site_type_symbol
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
_atom_site_occupancy
"""
    
    # Add cations (from lattice motif)
    for i, frac in enumerate(lattice.motif):
        cif_content += f"{cation_symbol}{i+1} {cation_symbol} {frac[0]:.6f} {frac[1]:.6f} {frac[2]:.6f} 1.0\n"
    
    # Add anions (from sites)
    for i, site in enumerate(sites):
        frac = site.get('fractional', site.get('position', [0, 0, 0]))
        if 'fractional' not in site and hasattr(lattice, 'get_fractional_coords'):
            frac = lattice.get_fractional_coords(np.array(site['position']))
        # Wrap to unit cell
        frac = np.array(frac) % 1.0
        cif_content += f"{anion_symbol}{i+1} {anion_symbol} {frac[0]:.6f} {frac[1]:.6f} {frac[2]:.6f} 1.0\n"
    
    return cif_content


def results_to_dataframe(sites: List[Dict]) -> 'pd.DataFrame':
    """
    Convert site list to pandas DataFrame for display.
    
    Parameters
    ----------
    sites : list of dict
        List of coordination sites.
        
    Returns
    -------
    pd.DataFrame
        DataFrame with site information.
    """
    import pandas as pd
    
    if not sites:
        return pd.DataFrame()
    
    data = []
    for i, site in enumerate(sites):
        row = {
            'Site': i + 1,
            'CN': site.get('coordination_number', '?'),
            'x': site['position'][0] if 'position' in site else 0,
            'y': site['position'][1] if 'position' in site else 0,
            'z': site['position'][2] if 'position' in site else 0,
        }
        
        if 'fractional' in site:
            frac = site['fractional']
            row['frac_x'] = frac[0]
            row['frac_y'] = frac[1]
            row['frac_z'] = frac[2]
        
        if 'R_critical' in site:
            row['R_crit (Å)'] = f"{site['R_critical']:.4f}"
        
        if 'gap_to_4th' in site:
            row['Gap to 4th'] = f"{site['gap_to_4th']:.3f}"
        
        if 'stability_metric' in site:
            row['Stability'] = f"{site['stability_metric']:.3f}"
        
        data.append(row)
    
    return pd.DataFrame(data)
