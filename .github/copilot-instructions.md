# Copilot Instructions for DESI DR1 Stellar Stream Tutorial

## Project Overview
- This project analyzes Milky Way stellar streams using DESI DR1 data, focusing on mixture modeling and MCMC to identify stream members.
- Main workflow is in `streamTutorial.ipynb`, with supporting scripts for data processing, modeling, and plotting.
- Data files are in `data/`, with subfolders for models and stream tables. Notebooks for analysis and visualization are in the project root.

## Key Components
- **Notebooks**: Main entry is `streamTutorial.ipynb`. Other notebooks (e.g., `post_mcmc.ipynb`, `plot_multi_streams.ipynb`) provide specialized analyses.
- **Python Modules**:
  - `stream_functions.py`: Core stream modeling, orbit integration, and utility functions.
  - `streamTutorial.py`/`streamTutorial_main.py`: High-level workflow, plotting, and presentation utilities.
  - `gallery_functions.py`: Functions for cross-matching, kinematic cuts, and stream membership.
  - `post_mcmc.py`: Post-processing and visualization of MCMC results.
- **Data**: All required small data files are in `data/`. Large files (e.g., DESI DR1 catalogue) must be downloaded separately.

## Developer Workflows
- **Environment**: Use `conda env create -f env.yml` to set up the required Python environment.
- **Running Analyses**: Execute notebooks in order, starting with `streamTutorial.ipynb`. Scripts can be imported for custom workflows.
- **Plotting**: Use the `set_presentation()` and `presentation()` utilities for consistent figure styling. See `*_presentation_rc` in main scripts.
- **MCMC**: MCMC fitting is performed using `emcee`. Diagnostic plots (chains, corner plots) are generated automatically.
- **Data Cuts**: Use functions like `kin_cut`, `betw`, and `ra_dec_dist_cut` for filtering and preparing data.

## Project Conventions
- **Imports**: All scripts use `import ... as ...` for major libraries (numpy, pandas, astropy, etc.).
- **Suppressions**: Astropy deprecation warnings are suppressed for cleaner output.
- **Plotting**: All plots should use the provided style utilities for consistency.
- **Data Handling**: Use `astropy.table.Table` and pandas DataFrames for tabular data. FITS files are read with astropy.
- **Naming**: Functions and variables use descriptive, lower_snake_case names. Stream-related functions are prefixed with `stream_` or `kin_`.

## Integration Points
- **External Libraries**: Relies on `emcee`, `corner`, `astropy`, `galpy`, `healpy`, `polars`, and others. All dependencies are listed in `env.yml`.
- **Data**: Large data files are not versioned; see README for download instructions.
- **Server Integration**: For large-scale runs, see Eridanus server notes in README.

## Examples
- To plot kinematics with background and stream overlays:
  ```python
  plt_kin.plot_params['sf_in_desi']['alpha'] = 0.5
  plt_kin.plot_params['background']['alpha'] = 0.2
  plt_kin.plot_params['background']['s'] = 1
  ```
- To apply a kinematic cut:
  ```python
  filtered = kin_cut(...)
  ```

## References
- See `README.md` for scientific background, data sources, and further documentation.

---
For any unclear or incomplete sections, please provide feedback to improve these instructions.
