"""
Crystal Structure Solver - Streamlit Application

Interactive web application for analytical crystal structure prediction
using sphere intersection mathematics.

Features:
1. Framework Explorer - Visualize Bravais lattices
2. Coordination Site Finder - Find CN-3, CN-4, CN-6 sites
3. Structure Predictor - Predict complete crystal structures
4. Comparison Tool - Compare predictions with known structures
5. Parameter Optimizer - Find optimal c/a ratios

Author: Crystal Structure Solver Project
"""

import streamlit as st
import numpy as np
import pandas as pd
import json
from pathlib import Path
import sys

# Add source directory to path
SRC_DIR = Path(__file__).parent.parent / "src"
DATA_DIR = Path(__file__).parent.parent / "data"
sys.path.insert(0, str(SRC_DIR))

# Import modules
from lattices.bravais import BravaisLattice
from analysis.structure_analyzer import (
    StructureAnalyzer,
    load_reference_structure,
    list_reference_structures,
    analyze_structure
)
from visualization import (
    create_3d_plot,
    create_structure_viewer,
    plot_sites_3d,
    results_to_dataframe,
    export_to_cif,
    plot_ca_optimization,
    COLORS
)

# Page configuration
st.set_page_config(
    page_title="Crystal Structure Solver",
    page_icon="💎",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        margin-bottom: 1rem;
    }
    .metric-container {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding-left: 20px;
        padding-right: 20px;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'analyzer' not in st.session_state:
    st.session_state.analyzer = StructureAnalyzer()

if 'last_results' not in st.session_state:
    st.session_state.last_results = None


# =============================================================================
# Helper Functions
# =============================================================================

@st.cache_data
def get_lattice_info():
    """Load lattice parameter information."""
    json_path = DATA_DIR / "lattice_parameters.json"
    if json_path.exists():
        with open(json_path, 'r') as f:
            return json.load(f)
    return {"lattice_types": {}}


@st.cache_data
def get_reference_structures():
    """Load reference structures."""
    json_path = DATA_DIR / "reference_structures.json"
    if json_path.exists():
        with open(json_path, 'r') as f:
            return json.load(f)
    return {}


def get_lattice_from_sidebar(key_prefix: str = ""):
    """Create lattice from sidebar inputs."""
    lattice_info = get_lattice_info()
    lattice_types = list(lattice_info.get('lattice_types', {}).keys())
    
    if not lattice_types:
        lattice_types = ['cP', 'cI', 'cF', 'tP', 'tI', 'oP', 'oI', 'oF', 'oC', 'hP', 'hR', 'mP', 'mC', 'aP']
    
    lattice_type = st.sidebar.selectbox(
        "Lattice Type",
        lattice_types,
        key=f"{key_prefix}lattice_type",
        help="Select a Bravais lattice type"
    )
    
    # Get info about selected lattice
    info = lattice_info.get('lattice_types', {}).get(lattice_type, {})
    
    if info:
        st.sidebar.caption(f"**{info.get('name', lattice_type)}**")
        st.sidebar.caption(f"CN: {info.get('coordination_number', '?')}")
    
    # Parameter inputs based on crystal system
    params = {'a': st.sidebar.number_input("a (Å)", 1.0, 20.0, 4.0, 0.1, key=f"{key_prefix}a")}
    
    # Non-cubic systems need c
    if lattice_type[0] in ['t', 'h', 'o', 'm', 'a']:
        if lattice_type == 'hP':
            c_default = params['a'] * 1.633  # Ideal c/a for hcp
        elif lattice_type == 'tI':
            c_default = params['a'] * 0.644  # Rutile-like
        else:
            c_default = params['a']
        
        params['c'] = st.sidebar.number_input("c (Å)", 0.5, 20.0, c_default, 0.1, key=f"{key_prefix}c")
        
        # Show c/a ratio
        st.sidebar.caption(f"c/a = {params['c']/params['a']:.4f}")
    
    # Orthorhombic needs b
    if lattice_type[0] in ['o', 'm', 'a']:
        params['b'] = st.sidebar.number_input("b (Å)", 0.5, 20.0, params['a'], 0.1, key=f"{key_prefix}b")
    
    # Monoclinic/triclinic need angles
    if lattice_type[0] == 'm':
        params['beta'] = st.sidebar.number_input("β (°)", 60.0, 120.0, 90.0, 1.0, key=f"{key_prefix}beta")
    
    if lattice_type == 'aP':
        params['alpha'] = st.sidebar.number_input("α (°)", 60.0, 120.0, 90.0, 1.0, key=f"{key_prefix}alpha")
        params['beta'] = st.sidebar.number_input("β (°)", 60.0, 120.0, 90.0, 1.0, key=f"{key_prefix}beta")
        params['gamma'] = st.sidebar.number_input("γ (°)", 60.0, 120.0, 90.0, 1.0, key=f"{key_prefix}gamma")
    
    if lattice_type == 'hR':
        params['alpha'] = st.sidebar.number_input("α (°)", 30.0, 90.0, 60.0, 1.0, key=f"{key_prefix}alpha")
    
    return lattice_type, params


# =============================================================================
# Page Functions
# =============================================================================

def page_framework_explorer():
    """Page 1: Framework Explorer - Visualize Bravais lattices."""
    st.markdown('<h1 class="main-header">🔬 Framework Explorer</h1>', unsafe_allow_html=True)
    st.markdown("Explore the 14 Bravais lattice types that form metal frameworks in ionic crystals.")
    
    # Sidebar inputs
    st.sidebar.header("Lattice Parameters")
    lattice_type, params = get_lattice_from_sidebar("fw_")
    
    n_cells = st.sidebar.slider("Supercell size", 1, 3, 1, key="fw_ncells",
                                help="Number of unit cells in each direction")
    
    # Generate lattice
    try:
        lattice = BravaisLattice(lattice_type, params)
        metals = lattice.generate_atoms(n_cells)
        
        # Main display
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("3D Visualization")
            fig = create_3d_plot(
                metals, 
                title=f"{lattice_type} Lattice ({len(metals)} atoms)",
                show_unit_cell=True,
                lattice_vectors=lattice.get_basis_vectors()
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("Lattice Information")
            
            # Display metrics
            st.metric("Lattice Type", lattice_type)
            st.metric("Coordination Number", lattice.get_coordination_number())
            st.metric("Atoms per Unit Cell", len(lattice.motif))
            st.metric("Total Atoms (Supercell)", len(metals))
            
            nn_dist = lattice.get_nearest_neighbor_distance()
            st.metric("Nearest Neighbor Distance", f"{nn_dist:.4f} Å")
            
            # Basis vectors
            st.subheader("Basis Vectors")
            basis = lattice.get_basis_vectors()
            basis_df = pd.DataFrame(
                basis,
                columns=['x', 'y', 'z'],
                index=['a₁', 'a₂', 'a₃']
            )
            st.dataframe(basis_df.style.format("{:.4f}"))
            
            # Unit cell volume
            volume = np.abs(np.linalg.det(basis))
            st.metric("Unit Cell Volume", f"{volume:.4f} Å³")
        
        # Show lattice info from database
        lattice_info = get_lattice_info()
        info = lattice_info.get('lattice_types', {}).get(lattice_type, {})
        
        if info:
            with st.expander("📚 Lattice Type Details"):
                st.write(f"**Full Name:** {info.get('name', 'Unknown')}")
                st.write(f"**Crystal System:** {info.get('crystal_system', 'Unknown')}")
                st.write(f"**Centering:** {info.get('centering', 'Unknown')}")
                if 'examples' in info and info['examples']:
                    st.write(f"**Example Elements:** {', '.join(info['examples'])}")
                if 'nearest_neighbor_formula' in info:
                    st.write(f"**NN Distance Formula:** {info['nearest_neighbor_formula']}")
    
    except Exception as e:
        st.error(f"Error generating lattice: {str(e)}")


def page_coordination_finder():
    """Page 2: Coordination Site Finder."""
    st.markdown('<h1 class="main-header">🎯 Coordination Site Finder</h1>', unsafe_allow_html=True)
    st.markdown("Find CN-3, CN-4, and CN-6 coordination sites analytically using sphere intersection mathematics.")
    
    # Sidebar inputs
    st.sidebar.header("Structure Parameters")
    lattice_type, params = get_lattice_from_sidebar("cf_")
    
    st.sidebar.header("Ionic Radii")
    R_cation = st.sidebar.number_input("Cation radius (Å)", 0.3, 2.0, 0.6, 0.05, key="cf_rcation")
    R_anion = st.sidebar.number_input("Anion radius (Å)", 0.5, 3.0, 1.4, 0.05, key="cf_ranion")
    
    st.sidebar.caption(f"**Radius ratio:** r⁺/r⁻ = {R_cation/R_anion:.3f}")
    st.sidebar.caption(f"**Coordination sphere:** R = {R_cation + R_anion:.3f} Å")
    
    cn_types = st.sidebar.multiselect(
        "Find CN types",
        [3, 4, 6],
        default=[3, 4, 6],
        key="cf_cntypes"
    )
    
    n_cells = st.sidebar.slider("Analysis cell size", 1, 2, 1, key="cf_ncells")
    
    # Analysis button
    if st.sidebar.button("🔍 Find Sites", type="primary", key="cf_analyze"):
        try:
            with st.spinner("Analyzing structure..."):
                lattice = BravaisLattice(lattice_type, params)
                analyzer = st.session_state.analyzer
                results = analyzer.analyze_framework(lattice, R_cation, R_anion, n_cells=n_cells)
                st.session_state.last_results = results
                st.session_state.last_lattice = lattice
        except Exception as e:
            st.error(f"Analysis failed: {str(e)}")
            import traceback
            st.code(traceback.format_exc())
            return
    
    # Display results
    if st.session_state.last_results is not None:
        results = st.session_state.last_results
        lattice = st.session_state.last_lattice
        metals = lattice.generate_atoms(n_cells)
        
        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("CN-3 Sites", len(results['cn3_sites']))
        col2.metric("CN-4 Sites", len(results['cn4_sites']))
        col3.metric("CN-6 Sites", len(results['cn6_sites']))
        col4.metric("Stoichiometry", results['stoichiometry'])
        
        # Tabs for different views
        tab1, tab2, tab3 = st.tabs(["📊 Site Tables", "🎨 3D Visualization", "📥 Export"])
        
        with tab1:
            for cn in cn_types:
                sites_key = f'cn{cn}_sites'
                sites = results.get(sites_key, [])
                
                st.subheader(f"CN-{cn} Sites ({len(sites)} found)")
                
                if sites:
                    df = results_to_dataframe(sites)
                    st.dataframe(df, use_container_width=True)
                else:
                    st.info(f"No CN-{cn} sites found in the specified radius range.")
        
        with tab2:
            # Combined visualization
            all_sites = []
            for cn in cn_types:
                all_sites.extend(results.get(f'cn{cn}_sites', []))
            
            fig = create_3d_plot(
                metals,
                sites=all_sites,
                title=f"Coordination Sites in {lattice_type}",
                show_unit_cell=True,
                lattice_vectors=lattice.get_basis_vectors()
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Individual CN visualizations
            st.subheader("Individual CN Types")
            cols = st.columns(len(cn_types))
            for i, cn in enumerate(cn_types):
                with cols[i]:
                    sites = results.get(f'cn{cn}_sites', [])
                    if sites:
                        fig = plot_sites_3d(
                            metals, sites, cn=cn,
                            lattice_vectors=lattice.get_basis_vectors()
                        )
                        fig.update_layout(height=400)
                        st.plotly_chart(fig, use_container_width=True)
        
        with tab3:
            st.subheader("Export Results")
            
            # CSV export
            all_sites = results.get('all_sites', [])
            if all_sites:
                df = results_to_dataframe(all_sites)
                csv = df.to_csv(index=False)
                st.download_button(
                    "📥 Download Sites (CSV)",
                    data=csv,
                    file_name="coordination_sites.csv",
                    mime="text/csv"
                )
            
            # CIF export
            if all_sites:
                cif_content = export_to_cif(lattice, all_sites, "structure.cif")
                st.download_button(
                    "📥 Download Structure (CIF)",
                    data=cif_content,
                    file_name="predicted_structure.cif",
                    mime="chemical/x-cif"
                )
            
            # JSON export
            json_data = {
                'lattice_type': lattice_type,
                'params': params,
                'R_cation': R_cation,
                'R_anion': R_anion,
                'stoichiometry': results['stoichiometry'],
                'n_cn3': len(results['cn3_sites']),
                'n_cn4': len(results['cn4_sites']),
                'n_cn6': len(results['cn6_sites']),
            }
            st.download_button(
                "📥 Download Analysis (JSON)",
                data=json.dumps(json_data, indent=2),
                file_name="analysis_results.json",
                mime="application/json"
            )
    else:
        st.info("👆 Configure parameters in the sidebar and click 'Find Sites' to begin analysis.")


def page_structure_predictor():
    """Page 3: Structure Predictor."""
    st.markdown('<h1 class="main-header">🔮 Structure Predictor</h1>', unsafe_allow_html=True)
    st.markdown("Predict crystal structures for given ionic radii by searching across lattice types.")
    
    # Input columns
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Cation")
        cation_name = st.text_input("Cation", "Ti4+", key="sp_cation")
        R_cation = st.number_input("Cation radius (Å)", 0.3, 2.0, 0.605, 0.01, key="sp_rcation")
    
    with col2:
        st.subheader("Anion")
        anion_name = st.text_input("Anion", "O2-", key="sp_anion")
        R_anion = st.number_input("Anion radius (Å)", 0.5, 3.0, 1.40, 0.01, key="sp_ranion")
    
    # Radius ratio analysis
    ratio = R_cation / R_anion
    st.info(f"**Radius ratio:** r⁺/r⁻ = {ratio:.4f}")
    
    # Pauling's rules prediction
    if ratio < 0.155:
        expected_cn = "2 (linear)"
    elif ratio < 0.225:
        expected_cn = "3 (trigonal planar)"
    elif ratio < 0.414:
        expected_cn = "4 (tetrahedral)"
    elif ratio < 0.732:
        expected_cn = "6 (octahedral)"
    elif ratio < 1.0:
        expected_cn = "8 (cubic)"
    else:
        expected_cn = "12 (cuboctahedral)"
    
    st.info(f"**Pauling's rules prediction:** CN = {expected_cn}")
    
    # Lattice type selection
    lattice_options = st.multiselect(
        "Search lattice types",
        ["cP", "cI", "cF", "tP", "tI", "hP", "oP", "oI"],
        default=["cF", "tI", "hP"],
        key="sp_lattices"
    )
    
    # Advanced options
    with st.expander("⚙️ Advanced Options"):
        c_a_min = st.slider("Min c/a ratio", 0.3, 1.0, 0.5, 0.05)
        c_a_max = st.slider("Max c/a ratio", 1.0, 3.0, 2.0, 0.05)
        c_a_steps = st.slider("c/a scan steps", 5, 50, 20)
    
    # Predict button
    if st.button("🔮 Predict Structures", type="primary", key="sp_predict"):
        predictions = []
        
        progress = st.progress(0)
        status = st.empty()
        
        total_steps = len(lattice_options)
        
        for i, lt in enumerate(lattice_options):
            status.text(f"Analyzing {lt}...")
            
            try:
                # For cubic lattices, just analyze once
                if lt[0] == 'c':
                    lattice = BravaisLattice(lt, {'a': 1.0})
                    results = analyze_structure(lt, {'a': 1.0}, R_cation, R_anion)
                    
                    if results['all_sites']:
                        predictions.append({
                            'lattice_type': lt,
                            'c_a': 1.0,
                            'stoichiometry': results['stoichiometry'],
                            'n_cn3': len(results['cn3_sites']),
                            'n_cn4': len(results['cn4_sites']),
                            'n_cn6': len(results['cn6_sites']),
                            'stability_score': results['stability_metrics'].get('combined_score', 0)
                        })
                
                else:
                    # Scan c/a ratio
                    for c_a in np.linspace(c_a_min, c_a_max, c_a_steps):
                        params = {'a': 1.0, 'c': c_a}
                        results = analyze_structure(lt, params, R_cation, R_anion)
                        
                        if results['all_sites']:
                            predictions.append({
                                'lattice_type': lt,
                                'c_a': c_a,
                                'stoichiometry': results['stoichiometry'],
                                'n_cn3': len(results['cn3_sites']),
                                'n_cn4': len(results['cn4_sites']),
                                'n_cn6': len(results['cn6_sites']),
                                'stability_score': results['stability_metrics'].get('combined_score', 0)
                            })
            
            except Exception as e:
                st.warning(f"Error analyzing {lt}: {str(e)}")
            
            progress.progress((i + 1) / total_steps)
        
        status.text("Analysis complete!")
        
        if predictions:
            # Sort by stability
            predictions.sort(key=lambda x: x['stability_score'], reverse=True)
            st.session_state.predictions = predictions
        else:
            st.warning("No valid structures found.")
    
    # Display predictions
    if 'predictions' in st.session_state and st.session_state.predictions:
        predictions = st.session_state.predictions
        
        st.subheader(f"Top {min(10, len(predictions))} Predicted Structures")
        
        for i, pred in enumerate(predictions[:10]):
            cn_dist = f"CN3:{pred['n_cn3']} CN4:{pred['n_cn4']} CN6:{pred['n_cn6']}"
            
            with st.expander(
                f"#{i+1}: {pred['stoichiometry']} ({pred['lattice_type']}) - "
                f"Score: {pred['stability_score']:.3f}"
            ):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write(f"**Lattice Type:** {pred['lattice_type']}")
                    st.write(f"**c/a Ratio:** {pred['c_a']:.4f}")
                    st.write(f"**Stoichiometry:** {pred['stoichiometry']}")
                
                with col2:
                    st.write(f"**CN Distribution:**")
                    st.write(f"- CN-3: {pred['n_cn3']} sites")
                    st.write(f"- CN-4: {pred['n_cn4']} sites")
                    st.write(f"- CN-6: {pred['n_cn6']} sites")
                
                # Visualize this structure
                if st.button(f"🔍 Visualize", key=f"vis_{i}"):
                    params = {'a': 1.0}
                    if pred['c_a'] != 1.0:
                        params['c'] = pred['c_a']
                    
                    lattice = BravaisLattice(pred['lattice_type'], params)
                    results = analyze_structure(pred['lattice_type'], params, R_cation, R_anion)
                    metals = lattice.generate_atoms(n_cells=1)
                    
                    fig = create_3d_plot(
                        metals,
                        sites=results['all_sites'],
                        title=f"{pred['stoichiometry']} ({pred['lattice_type']})",
                        lattice_vectors=lattice.get_basis_vectors()
                    )
                    st.plotly_chart(fig, use_container_width=True)


def page_comparison_tool():
    """Page 4: Comparison Tool."""
    st.markdown('<h1 class="main-header">⚖️ Comparison Tool</h1>', unsafe_allow_html=True)
    st.markdown("Compare predictions against known reference structures.")
    
    # Get reference structures
    ref_structures = get_reference_structures()
    ref_names = list(ref_structures.keys())
    
    if not ref_names:
        st.warning("No reference structures found. Please check data/reference_structures.json")
        return
    
    # Reference selection
    ref_name = st.selectbox(
        "Select Reference Structure",
        ref_names,
        format_func=lambda x: f"{x} - {ref_structures[x].get('name', x)}",
        key="ct_refname"
    )
    
    reference = ref_structures[ref_name]
    
    # Display reference info
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📚 Reference Structure")
        st.write(f"**Name:** {reference.get('name', ref_name)}")
        st.write(f"**Formula:** {reference.get('formula', 'Unknown')}")
        st.write(f"**Stoichiometry:** {reference.get('stoichiometry', 'Unknown')}")
        st.write(f"**Lattice Type:** {reference.get('lattice_type', 'Unknown')}")
        st.write(f"**Space Group:** {reference.get('space_group', 'Unknown')}")
        
        params = reference.get('params', {})
        if params:
            st.write(f"**Lattice Parameters:** a={params.get('a', '?')} Å")
            if 'c' in params:
                st.write(f"  c={params['c']} Å (c/a = {reference.get('c_over_a', params['c']/params['a']):.4f})")
    
    with col2:
        # Show reference positions
        st.subheader("📍 Reference Positions")
        
        cation_pos = reference.get('cation_positions', [])
        st.write(f"**Cation positions:** {len(cation_pos)}")
        if cation_pos:
            df_cat = pd.DataFrame(cation_pos, columns=['x', 'y', 'z'])
            st.dataframe(df_cat.style.format("{:.4f}"), height=150)
        
        anion_pos = reference.get('anion_positions', [])
        st.write(f"**Anion positions:** {len(anion_pos)}")
        if anion_pos:
            df_an = pd.DataFrame(anion_pos, columns=['x', 'y', 'z'])
            st.dataframe(df_an.style.format("{:.4f}"), height=150)
    
    st.divider()
    
    # Prediction and comparison
    st.subheader("🔮 Predict and Compare")
    
    col1, col2 = st.columns(2)
    with col1:
        R_cation = st.number_input("Cation radius (Å)", 0.3, 2.0, 0.6, 0.05, key="ct_rcation")
    with col2:
        R_anion = st.number_input("Anion radius (Å)", 0.5, 3.0, 1.4, 0.05, key="ct_ranion")
    
    if st.button("⚖️ Predict and Compare", type="primary", key="ct_compare"):
        try:
            with st.spinner("Running analysis..."):
                # Create lattice from reference
                lattice_type = reference['lattice_type']
                params = reference.get('params', {'a': 1.0})
                
                lattice = BravaisLattice(lattice_type, params)
                analyzer = st.session_state.analyzer
                
                # Run prediction
                results = analyzer.analyze_framework(lattice, R_cation, R_anion)
                
                # Compare to reference
                comparison = analyzer.compare_to_reference(
                    results['all_sites'],
                    reference,
                    lattice
                )
            
            # Display comparison results
            st.subheader("📊 Comparison Results")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                rmsd = comparison.get('position_rmsd')
                if rmsd is not None:
                    st.metric("Position RMSD", f"{rmsd:.4f} Å")
                else:
                    st.metric("Position RMSD", "N/A")
            
            with col2:
                cn_match = comparison.get('cn_match', False)
                st.metric("CN Match", "✅ Yes" if cn_match else "❌ No")
            
            with col3:
                stoich_match = comparison.get('stoichiometry_match', False)
                st.metric("Stoichiometry Match", "✅ Yes" if stoich_match else "❌ No")
            
            # Detailed results
            col1, col2 = st.columns(2)
            
            with col1:
                st.write(f"**Matched sites:** {len(comparison.get('matched_sites', []))}")
                st.write(f"**Unmatched prediction:** {len(comparison.get('unmatched_predicted', []))}")
                st.write(f"**Unmatched reference:** {len(comparison.get('unmatched_reference', []))}")
                st.write(f"**Match fraction:** {comparison.get('fraction_matched', 0):.1%}")
            
            with col2:
                st.write(f"**Predicted stoichiometry:** {results['stoichiometry']}")
                st.write(f"**Reference stoichiometry:** {reference.get('stoichiometry', 'Unknown')}")
            
            # Side-by-side visualization
            st.subheader("📊 Visual Comparison")
            
            metals = lattice.generate_atoms(n_cells=1)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Prediction**")
                fig_pred = create_3d_plot(
                    metals,
                    sites=results['all_sites'],
                    title="Predicted Sites",
                    lattice_vectors=lattice.get_basis_vectors()
                )
                fig_pred.update_layout(height=400)
                st.plotly_chart(fig_pred, use_container_width=True)
            
            with col2:
                st.write("**Reference**")
                # Create reference sites from positions
                ref_sites = []
                for pos in reference.get('anion_positions', []):
                    cart_pos = lattice.get_cartesian_coords(np.array(pos))
                    ref_sites.append({
                        'position': cart_pos,
                        'fractional': pos,
                        'coordination_number': 4  # Default
                    })
                
                fig_ref = create_3d_plot(
                    metals,
                    sites=ref_sites,
                    title="Reference Sites",
                    lattice_vectors=lattice.get_basis_vectors()
                )
                fig_ref.update_layout(height=400)
                st.plotly_chart(fig_ref, use_container_width=True)
        
        except Exception as e:
            st.error(f"Comparison failed: {str(e)}")
            import traceback
            st.code(traceback.format_exc())


def page_parameter_optimizer():
    """Page 5: Parameter Optimizer."""
    st.markdown('<h1 class="main-header">⚙️ Parameter Optimizer</h1>', unsafe_allow_html=True)
    st.markdown("Find optimal c/a ratios for maximum coordination sites.")
    
    # Inputs
    col1, col2 = st.columns(2)
    
    with col1:
        lattice_type = st.selectbox(
            "Lattice Type",
            ["tI", "tP", "hP", "oI", "oP"],
            key="po_lattice",
            help="Select a non-cubic lattice type"
        )
        
        target_CN = st.selectbox(
            "Target Anion CN",
            [3, 4, 6],
            key="po_cn",
            help="Coordination number to optimize for"
        )
    
    with col2:
        R_cation = st.number_input("Cation radius (Å)", 0.3, 2.0, 0.6, 0.05, key="po_rcation")
        R_anion = st.number_input("Anion radius (Å)", 0.5, 3.0, 1.4, 0.05, key="po_ranion")
    
    # c/a range
    col1, col2, col3 = st.columns(3)
    with col1:
        c_a_min = st.number_input("Min c/a", 0.3, 1.5, 0.5, 0.05, key="po_camin")
    with col2:
        c_a_max = st.number_input("Max c/a", 1.0, 3.0, 2.0, 0.05, key="po_camax")
    with col3:
        n_points = st.slider("Scan points", 10, 100, 50, key="po_npoints")
    
    # Optimize button
    if st.button("🔍 Find Optimal c/a", type="primary", key="po_optimize"):
        c_a_range = np.linspace(c_a_min, c_a_max, n_points)
        results = []
        
        progress = st.progress(0)
        status = st.empty()
        
        for i, c_a in enumerate(c_a_range):
            status.text(f"Analyzing c/a = {c_a:.3f}...")
            
            try:
                params = {'a': 1.0, 'c': c_a}
                lattice = BravaisLattice(lattice_type, params)
                metals = lattice.generate_atoms(n_cells=1)
                
                analyzer = st.session_state.analyzer
                analysis = analyzer.analyze_framework(lattice, R_cation, R_anion)
                
                n_sites = len(analysis[f'cn{target_CN}_sites'])
                results.append({
                    'c_a': c_a,
                    'n_sites': n_sites,
                    'stoichiometry': analysis['stoichiometry']
                })
            
            except Exception as e:
                results.append({
                    'c_a': c_a,
                    'n_sites': 0,
                    'stoichiometry': 'Error'
                })
            
            progress.progress((i + 1) / len(c_a_range))
        
        status.text("Optimization complete!")
        st.session_state.optimization_results = results
    
    # Display results
    if 'optimization_results' in st.session_state:
        results = st.session_state.optimization_results
        df = pd.DataFrame(results)
        
        # Plot
        fig = plot_ca_optimization(results, target_CN, lattice_type)
        st.plotly_chart(fig, use_container_width=True)
        
        # Find optimal
        if df['n_sites'].max() > 0:
            optimal_idx = df['n_sites'].idxmax()
            optimal = df.loc[optimal_idx]
            
            st.success(
                f"**Optimal c/a = {optimal['c_a']:.4f}** yields "
                f"**{int(optimal['n_sites'])} CN-{target_CN} sites** "
                f"(stoichiometry: {optimal['stoichiometry']})"
            )
            
            # Show visualization at optimal
            if st.button("🔍 Visualize Optimal Structure", key="po_vis"):
                params = {'a': 1.0, 'c': optimal['c_a']}
                lattice = BravaisLattice(lattice_type, params)
                metals = lattice.generate_atoms(n_cells=1)
                
                analysis = analyze_structure(lattice_type, params, R_cation, R_anion)
                
                fig = create_3d_plot(
                    metals,
                    sites=analysis['all_sites'],
                    title=f"Optimal Structure (c/a = {optimal['c_a']:.4f})",
                    lattice_vectors=lattice.get_basis_vectors()
                )
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No sites found in the specified range. Try adjusting the ionic radii or c/a range.")
        
        # Data table
        with st.expander("📊 Full Scan Results"):
            st.dataframe(df)
            
            csv = df.to_csv(index=False)
            st.download_button(
                "📥 Download Results (CSV)",
                data=csv,
                file_name="ca_optimization.csv",
                mime="text/csv"
            )


# =============================================================================
# Main Application
# =============================================================================

def main():
    """Main application entry point."""
    
    # Sidebar navigation
    st.sidebar.title("💎 Crystal Solver")
    st.sidebar.markdown("---")
    
    pages = {
        "🔬 Framework Explorer": page_framework_explorer,
        "🎯 Coordination Finder": page_coordination_finder,
        "🔮 Structure Predictor": page_structure_predictor,
        "⚖️ Comparison Tool": page_comparison_tool,
        "⚙️ Parameter Optimizer": page_parameter_optimizer,
    }
    
    selection = st.sidebar.radio("Navigate", list(pages.keys()), label_visibility="collapsed")
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### About")
    st.sidebar.markdown(
        "This tool predicts ionic crystal structures analytically "
        "using sphere intersection mathematics."
    )
    st.sidebar.markdown(
        "[📖 Documentation](https://github.com/crystal-solver/docs) | "
        "[🐛 Report Issue](https://github.com/crystal-solver/issues)"
    )
    
    # Run selected page
    pages[selection]()


if __name__ == "__main__":
    main()
