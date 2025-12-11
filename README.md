# Crystal Structure Solver - Streamlit Application

Interactive web application for analytical crystal structure prediction using sphere intersection mathematics.

## Features

### 1. 🔬 Framework Explorer
Visualize the 14 Bravais lattice types that form metal frameworks in ionic crystals.
- Interactive 3D visualization with Plotly
- Adjustable lattice parameters (a, b, c, angles)
- Supercell generation
- Unit cell display with basis vectors

### 2. 🎯 Coordination Site Finder
Find CN-3, CN-4, and CN-6 coordination sites analytically.
- Input cation/anion radii
- Automatic radius ratio analysis
- Site tables with fractional coordinates
- 3D visualization of all site types
- Export to CSV, CIF, and JSON formats

### 3. 🔮 Structure Predictor
Predict crystal structures for given ionic radii by searching across lattice types.
- Pauling's rules prediction
- Multi-lattice scanning
- c/a ratio optimization
- Stability scoring and ranking

### 4. ⚖️ Comparison Tool
Compare predictions against known reference structures.
- Built-in reference database (rutile, rock salt, fluorite, etc.)
- Position RMSD calculation
- CN and stoichiometry matching
- Side-by-side visualization

### 5. ⚙️ Parameter Optimizer
Find optimal c/a ratios for maximum coordination sites.
- Systematic c/a scanning
- Interactive optimization plots
- Export scan results

## Installation

```bash
# Install dependencies
pip install streamlit plotly pandas scipy numpy

# Run the application
cd crystal_solver
streamlit run app/streamlit_app.py
```

## Project Structure

```
crystal_solver/
├── app/
│   ├── streamlit_app.py    # Main Streamlit application
│   └── visualization.py    # Plotly visualization utilities
├── src/
│   ├── geometry/           # Sphere intersection math
│   │   ├── sphere_intersection.py
│   │   └── critical_solvers.py
│   ├── lattices/           # Bravais lattice generators
│   │   └── bravais.py
│   ├── coordination/       # CN site finders
│   │   ├── cn3_finder.py
│   │   ├── cn4_finder.py
│   │   └── cn6_finder.py
│   └── analysis/           # Structure analysis
│       └── structure_analyzer.py
├── data/
│   ├── reference_structures.json
│   └── lattice_parameters.json
└── README.md
```

## Usage Examples

### Quick Start
```bash
streamlit run app/streamlit_app.py
```
Then open http://localhost:8501 in your browser.

### Analyze Rutile (TiO₂)
1. Go to "Coordination Finder"
2. Select lattice type: tI
3. Set a = 4.594 Å, c = 2.958 Å
4. Set cation radius = 0.605 Å (Ti⁴⁺)
5. Set anion radius = 1.40 Å (O²⁻)
6. Click "Find Sites"
7. Expected: CN-3 sites matching rutile oxygen positions

### Find Optimal c/a for HCP
1. Go to "Parameter Optimizer"
2. Select lattice type: hP
3. Select target CN: 4
4. Set radii appropriate for your system
5. Click "Find Optimal c/a"
6. View optimization curve

## API Reference

### BravaisLattice
```python
from src.lattices.bravais import BravaisLattice

# Create FCC lattice
lattice = BravaisLattice('cF', {'a': 4.05})
metals = lattice.generate_atoms(n_cells=2)

# Get lattice properties
cn = lattice.get_coordination_number()
nn_dist = lattice.get_nearest_neighbor_distance()
basis = lattice.get_basis_vectors()
```

### StructureAnalyzer
```python
from src.analysis.structure_analyzer import StructureAnalyzer, analyze_structure

# One-line analysis
results = analyze_structure('tI', {'a': 1.0, 'c': 0.644}, R_cation=0.6, R_anion=1.4)

# Full analyzer
analyzer = StructureAnalyzer()
results = analyzer.analyze_framework(lattice, R_cation=0.6, R_anion=1.4)
```

## Reference Structures

The following structures are included for validation:
- **TiO2_rutile** - Rutile titanium dioxide (tI, c/a=0.644)
- **NaCl_rock_salt** - Rock salt sodium chloride (cF)
- **ZnS_zinc_blende** - Zinc blende sphalerite (cF)
- **CaF2_fluorite** - Fluorite calcium fluoride (cF)
- **CsCl_cesium_chloride** - Cesium chloride (cP)
- **ZnO_wurtzite** - Wurtzite zinc oxide (hP)
- **SrTiO3_perovskite** - Perovskite strontium titanate (cP)
- And more...

## Performance

- Computation: <1s for typical lattice analysis
- Memory: Handles lattices up to ~10,000 atoms
- Accuracy: Matches known structures to <0.05 fractional coordinate

## Dependencies

- Python 3.8+
- streamlit >= 1.0
- plotly >= 5.0
- pandas >= 1.0
- numpy >= 1.20
- scipy >= 1.7

## License

MIT License - See LICENSE file for details.

## Contributing

Contributions welcome! Please read CONTRIBUTING.md for guidelines.
