import numpy as np
import matplotlib.pyplot as plt
from matplotlib.path import Path
from matplotlib.lines import Line2D
import pandas as pd
import healpy as hp
import sys
from pathlib import Path
import scipy as sp
import scipy.stats as stats
from astropy.io import fits
from astropy import table
import astropy.coordinates as coord
from astropy.coordinates.matrix_utilities import rotation_matrix
import astropy.units as u
import matplotlib
import importlib
import stream_functions as stream_funcs
import emcee
import corner
from astropy.table import Table, join
from collections import OrderedDict
import time
from scipy.interpolate import interp1d, splrep, splev
import os
import re
#import feh_correct
import warnings
from astropy.utils.exceptions import AstropyDeprecationWarning
import copy
import multiprocessing
from tqdm.auto import tqdm
# Ensure scripts directory is on sys.path when running from repo root
SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))
# Suppress specific Astropy deprecation warnings
warnings.filterwarnings("ignore", category=AstropyDeprecationWarning, module='gala.dynamics.core')
import polars as pl

# -----------------------------
# Presentation style utilities
# -----------------------------
from contextlib import contextmanager
import matplotlib as mpl

_PRESENTATION_RC = {
    'figure.dpi': 200,
    'savefig.dpi': 300,
    'axes.titlesize': 18,
    'axes.labelsize': 18,
    'axes.linewidth': 2.2,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'xtick.major.size': 6,
    'ytick.major.size': 6,
    'xtick.major.width': 1.8,
    'ytick.major.width': 1.8,
    'legend.fontsize': 14,
    'lines.linewidth': 2.2,
    'lines.markersize': 6.5,
    'grid.alpha': 0.25,
}

_pres_old = None

def set_presentation(on=True, restore='previous', **overrides):
    """Enable or disable presentation plotting style.

    Parameters
    - on: bool
        True applies presentation rcParams; False reverts style.
    - restore: {'previous', 'defaults'}
        When disabling, 'previous' restores values saved at first enable;
        'defaults' resets to Matplotlib defaults (rcdefaults()).
    - overrides: dict
        Extra rcParams to layer on top when enabling.

    Usage
        set_presentation(True)
        ... make plots ...
        set_presentation(False)              # restore previous (default)
        set_presentation(False, 'defaults')  # restore mpl defaults
    """
    global _pres_old
    if on:
        if _pres_old is None:
            _pres_old = {k: mpl.rcParams.get(k) for k in _PRESENTATION_RC}
        rc = {**_PRESENTATION_RC, **overrides}
        mpl.rcParams.update(rc)
        return

    # Turning OFF
    if restore == 'defaults':
        mpl.rcdefaults()
        _pres_old = None
        return

    if _pres_old is not None:
        mpl.rcParams.update(_pres_old)
        _pres_old = None
    else:
        # If we have nothing saved, fall back to defaults
        mpl.rcdefaults()

@contextmanager
def presentation(on=True, **overrides):
    """Context manager version; resets style on exit.

    with presentation(True):
        ... make plots ...
    """
    if not on:
        yield
        return
    old = {k: mpl.rcParams.get(k) for k in _PRESENTATION_RC}
    mpl.rcParams.update({**_PRESENTATION_RC, **overrides})
    try:
        yield
    finally:
        mpl.rcParams.update(old)

def corner_presentation_kwargs(scale=1.0):
    """Handy kwargs for corner.corner() that match current rcParams."""
    fs = int(mpl.rcParams.get('axes.labelsize', 12) * scale)
    return {
        'label_kwargs': {'fontsize': fs},
        'title_kwargs': {'fontsize': fs},
        'max_n_ticks': 4,
    }

# -----------------------------
# Interactive spline utilities
# -----------------------------

# Default path for persisting draggable control points
_CTRL_FILE_DEFAULT = Path(__file__).resolve().with_name("drag_spline_ctrls.npz")


class DraggablePoints:
    """Minimal draggable point handler for Matplotlib interactive backends."""

    def __init__(self, ax, x, y, update_cb=None, tol=7, fixed_x=None):
        self.ax = ax
        self.fixed_x = np.asarray(fixed_x, dtype=float) if fixed_x is not None else None
        self.line = Line2D(x, y, marker='o', color='k', ls='', picker=tol)
        ax.add_line(self.line)
        self.update_cb = update_cb
        self._ind = None
        canvas = self.line.figure.canvas
        self.cid_press = canvas.mpl_connect('button_press_event', self.on_press)
        self.cid_release = canvas.mpl_connect('button_release_event', self.on_release)
        self.cid_motion = canvas.mpl_connect('motion_notify_event', self.on_motion)

    def on_press(self, event):
        if event.inaxes != self.ax:
            return
        contains, info = self.line.contains(event)
        if not contains:
            return
        self._ind = info['ind'][0]

    def on_motion(self, event):
        if self._ind is None or event.inaxes != self.ax:
            return
        x, y = self.line.get_xdata(), self.line.get_ydata()
        new_x = x[self._ind]
        if self.fixed_x is not None and len(self.fixed_x) > self._ind:
            new_x = self.fixed_x[self._ind]
        x[self._ind], y[self._ind] = new_x, event.ydata
        self.line.set_data(x, y)
        if self.update_cb:
            self.update_cb(x, y)
        self.line.figure.canvas.draw_idle()

    def on_release(self, event):
        self._ind = None


def _default_ctrl(phi1, y, n=5):
    """Build a simple set of control points along phi1 for initialization."""
    x = np.linspace(np.nanmin(phi1), np.nanmax(phi1), n)
    order = np.argsort(phi1)
    y_interp = np.interp(x, np.array(phi1)[order], np.array(y)[order])
    return x, y_interp


def _save_controls(ctrl_dict, path=_CTRL_FILE_DEFAULT):
    """Persist draggable control points to disk."""
    np.savez(
        path,
        vgsr_x=ctrl_dict["vgsr"]["x"],
        vgsr_y=ctrl_dict["vgsr"]["y"],
        pmra_x=ctrl_dict["pmra"]["x"],
        pmra_y=ctrl_dict["pmra"]["y"],
        pmdec_x=ctrl_dict["pmdec"]["x"],
        pmdec_y=ctrl_dict["pmdec"]["y"],
    )
    print(f"Saved control points to {path}")
    return ctrl_dict


def drag_spline(
    phi1,
    vgsr,
    pmra,
    pmdec,
    ctrl_path=_CTRL_FILE_DEFAULT,
    tol=7,
    sf_in_desi=None,
    sf_not_desi=None,
    n_ctrl=None,
    phi1_fixed=None,
):
    """Interactive spline controls for vgsr/pmra/pmdec vs phi1.

    Returns a dict with fig/axes plus a ``save()`` callable that writes
    control points to ``ctrl_path`` and returns them as a dict.
    """
    phi1 = np.asarray(phi1)
    phi1_fixed = np.asarray(phi1_fixed, dtype=float) if phi1_fixed is not None else None
    datasets = {
        "vgsr": (np.asarray(vgsr), "V_GSR [km/s]"),
        "pmra": (np.asarray(pmra), "pmra [mas/yr]"),
        "pmdec": (np.asarray(pmdec), "pmdec [mas/yr]"),
    }

    def _align_to_fixed_x(xc, yc):
        if phi1_fixed is None:
            return np.asarray(xc, dtype=float), np.asarray(yc, dtype=float)
        xc = np.asarray(xc, dtype=float)
        yc = np.asarray(yc, dtype=float)
        if len(xc) == len(phi1_fixed):
            return phi1_fixed, yc
        # interpolate y onto fixed grid
        order = np.argsort(xc)
        return phi1_fixed, np.interp(phi1_fixed, xc[order], yc[order])

    if Path(ctrl_path).exists():
        data = np.load(ctrl_path)
        ctrl_init = {
            "vgsr": _align_to_fixed_x(data["vgsr_x"], data["vgsr_y"]),
            "pmra": _align_to_fixed_x(data["pmra_x"], data["pmra_y"]),
            "pmdec": _align_to_fixed_x(data["pmdec_x"], data["pmdec_y"]),
        }
        print(f"Loaded control points from {ctrl_path}")
        if phi1_fixed is not None:
            print("Locked φ1 to provided phi1_fixed grid.")
    else:
        n_default = len(phi1_fixed) if phi1_fixed is not None else (n_ctrl if n_ctrl is not None else 5)
        if phi1_fixed is not None:
            def interp_y(yvals):
                order = np.argsort(phi1)
                return np.interp(np.sort(phi1_fixed), np.array(phi1)[order], np.array(yvals)[order])
            ctrl_init = {
                k: (np.sort(phi1_fixed), interp_y(v))
                for k, (v, _) in datasets.items()
            }
        else:
            ctrl_init = {k: _default_ctrl(phi1, v, n=n_default) for k, (v, _) in datasets.items()}
        print(f"Using default control points (n={n_default}); drag to adjust then run the save cell.")

    fig, axes = plt.subplots(3, 1, sharex=True, figsize=(9, 8))
    lines = {}
    draggables = {}

    overlay_styles = {
        "sf_in": {"marker": "D", "facecolors": "forestgreen", "edgecolors": "k", "s": 55, "alpha": 0.9, "label": "SF in DESI"},
        "sf_not": {"marker": "D", "facecolors": "none", "edgecolors": "k", "s": 55, "alpha": 0.9, "label": "SF (not in DESI)"},
    }

    def _pick(data, candidates):
        for cand in candidates:
            if isinstance(data, dict) and cand in data:
                return np.asarray(data[cand])
            if hasattr(data, "columns") and cand in data.columns:
                return np.asarray(data[cand])
        return None

    def _maybe_plot_overlay(ax, key, overlay_key, overlay_data):
        if overlay_data is None:
            return 0
        name_map = {
            "vgsr": ["vgsr", "VGSR"],
            "pmra": ["pmra", "PMRA", "pmRA"],
            "pmdec": ["pmdec", "PMDEC", "pmDE"],
            "phi1": ["phi1", "PHI1", "phi_1"],
        }
        x_overlay = _pick(overlay_data, name_map["phi1"])
        y_overlay = _pick(overlay_data, name_map[key])
        if x_overlay is None or y_overlay is None:
            return 0
        style = overlay_styles[overlay_key].copy()
        ax.scatter(x_overlay, y_overlay, **style)
        return len(x_overlay)

    def redraw(key, xc, yc):
        order = np.argsort(xc)
        xc_s = np.asarray(xc)[order]
        yc_s = np.asarray(yc)[order]
        k = min(3, len(xc_s) - 1)
        if k < 1:
            return
        tck = splrep(xc_s, yc_s, s=0, k=k)
        xx = np.linspace(xc_s.min(), xc_s.max(), 400)
        yy = splev(xx, tck)
        lines[key].set_data(xx, yy)

    sf_in_count = 0
    sf_not_count = 0

    for ax, key in zip(axes, ["vgsr", "pmra", "pmdec"]):
        ydata, ylabel = datasets[key]
        ax.scatter(phi1, ydata, s=6, alpha=0.25, label="data")
        sf_in_count += _maybe_plot_overlay(ax, key, "sf_in", sf_in_desi)
        sf_not_count += _maybe_plot_overlay(ax, key, "sf_not", sf_not_desi)
        cx, cy = ctrl_init[key]
        line, = ax.plot([], [], "k-", lw=2.5, label="spline")
        lines[key] = line
        draggables[key] = DraggablePoints(
            ax,
            np.array(cx, dtype=float),
            np.array(cy, dtype=float),
            update_cb=lambda xc, yc, k=key: redraw(k, xc, yc),
            tol=tol,
            fixed_x=phi1_fixed if phi1_fixed is not None else np.array(cx, dtype=float),
        )
        redraw(key, cx, cy)
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.25)
    axes[-1].set_xlabel(r"$\phi_1$")
    plt.tight_layout()
    if sf_in_count or sf_not_count:
        print(f"Overlayed SF stars: in DESI = {sf_in_count}, not in DESI = {sf_not_count}")

    # Show the interactive figure - plt.show(block=False) works better in VS Code
    plt.show(block=False)

    def gather():
        return {
            key: {
                "x": draggables[key].line.get_xdata(),
                "y": draggables[key].line.get_ydata(),
            }
            for key in draggables
        }

    return {
        "fig": fig,
        "axes": axes,
        "controls": gather,
        "save": lambda path=ctrl_path: _save_controls(gather(), path),
    }

class Data:
    def __init__(self, desi_path, sf_path='/raid/catalogs/streamfinder_gaiadr3.fits', cleaned_data=False, dist_path=''):
        self.desi_path = desi_path
        self.sf_path = sf_path
        self.cleaned_data = cleaned_data
        if not cleaned_data:
            decals_path = '/raid/DESI/catalogs/loa/rv_output/241119/legacyphot-loa-241126.fits'
            self.decals_data = stream_funcs.load_fits_columns(decals_path, ['EBV', 'FLUX_G', 'FLUX_R', 'FLUX_Z'])
            # These are the columns from the DESI data that we want to import
            desired_columns = [
            'VRAD', 'VRAD_ERR', 'RVS_WARN', 'TEFF', 'LOGG', ## TEFF and LOGG needed for FeH correction
            'RR_SPECTYPE', 
            'TARGET_RA', 'TARGET_DEC', 'FEH', 'FEH_ERR',
            'TARGETID', 'PRIMARY',
            'SOURCE_ID', 'PMRA', 'PMRA_ERROR', 'PMDEC', 'PMDEC_ERROR', 'PARALLAX', 'PARALLAX_ERROR', 'PMRA_PMDEC_CORR'
            ]
            desi_hdu_indices = [1,3,4]
            self.desi_vrad_data = stream_funcs.load_fits_columns(desi_path, desired_columns, desi_hdu_indices)
            self.desi_vrad_data.label='DESI'
            self.desi_data = table.hstack([self.desi_vrad_data, self.decals_data]) 
            # Drop the rows with NaN values in all columns
            print(f"Length of DESI Data before Cuts: {len(self.desi_data)}")
            self.desi_data = stream_funcs.dropna_Table(self.desi_data, columns = desired_columns)
            self.desi_data = self.desi_data[(self.desi_data['RVS_WARN'] == 0) & (self.desi_data['RR_SPECTYPE'] == 'STAR') & (self.desi_data['PRIMARY']) &\
            (self.desi_data['VRAD_ERR'] < 10) & (self.desi_data['FEH_ERR'] < 0.5)] 
            self.desi_data.remove_columns(['RVS_WARN', 'RR_SPECTYPE'])
            self.desi_data = self.desi_data.to_pandas()
            # remove stars with FLUX_G and FLUX_R as NaN
            self.desi_data = self.desi_data[~self.desi_data['FLUX_G'].isna() | ~self.desi_data['FLUX_R'].isna() |  ~self.desi_data['EBV'].isna()]
            
            print(f"Length after NaN cut: {len(self.desi_data)}")
        

            # Applying additional errors in quadrature
            self.desi_data['VRAD_ERR'] = np.sqrt(self.desi_data['VRAD_ERR']**2 + 0.9**2) ### Turn into its own column
            self.desi_data['PMRA_ERROR'] = np.sqrt(self.desi_data['PMRA_ERROR']**2 + (np.sqrt(550)*0.001)**2) ### Turn into its own column
            self.desi_data['PMDEC_ERROR'] = np.sqrt(self.desi_data['PMDEC_ERROR']**2 + (np.sqrt(550)*0.001)**2) ### Turn into its own column
            self.desi_data['FEH_ERR'] = np.sqrt(self.desi_data['FEH_ERR']**2 + 0.01**2) ### Turn into its own column


            # # Apply Metallicity correction to the DR1 Data (See Section 4.2 https://arxiv.org/pdf/2505.14787)
            # print("Adding empirical FEH calibration (can find uncalibrated data in column['FEH_uncalib])")
            # self.desi_data['FEH_uncalib'] = self.desi_data['FEH']
            # self.desi_data['FEH'] = feh_correct.calibrate(self.desi_data['FEH'], self.desi_data['TEFF'], self.desi_data['LOGG'])
            
            # Switch to VGSR instead of VRAD
            self.desi_data['VGSR'] = np.array(
                stream_funcs.vhel_to_vgsr(
                    np.array(self.desi_data['TARGET_RA']) * u.deg,
                    np.array(self.desi_data['TARGET_DEC']) * u.deg,
                    np.array(self.desi_data['VRAD']) * (u.km / u.s),
                ).value
            )
        else:
            desi_data_tbl = table.Table.read(self.desi_path, format='fits')
            # Convert Astropy Table to Pandas DataFrame properly
            self.desi_data = desi_data_tbl.to_pandas()


        # Lets load the STREAMFINDER data for Gaia DR3
        if sf_path:
            sf_data = table.Table.read(self.sf_path)
            self.sf_data = sf_data.to_pandas()
            self.sf_data['VGSR'] = np.array(
                stream_funcs.vhel_to_vgsr(
                    np.array(self.sf_data['RAdeg']) * u.deg,
                    np.array(self.sf_data['DEdeg']) * u.deg,
                    np.array(self.sf_data['VHel']) * (u.km / u.s),
                ).value
            )
        else:
            print('No STREAMFINDER path given.')

        if dist_path:
            distance_data = stream_funcs.load_fits_columns(dist_path, ['TARGETID', 'dist_mod', 'dist_mod_err'])
            distance_data = pl.from_pandas(distance_data.to_pandas())
            desi_data_pl = pl.from_pandas(self.desi_data)

            # Ensure same dtype
            distance_data = distance_data.with_columns(
                distance_data['TARGETID'].cast(desi_data_pl['TARGETID'].dtype)
            )

            # Fast hash join
            desi_data_pl = desi_data_pl.join(distance_data, on='TARGETID', how='left')

            # Back to pandas if needed downstream
            self.desi_data = desi_data_pl.to_pandas()

            print("RVS distances added")

        

    def select(self, mask_or_func):
        """
        Applies a mask and returns a new, filtered Data object.

        Args:
            mask_or_func (function or pd.Series): 
                - If a function, it must take a DataFrame and return a boolean Series.
                - If a Series, it must be a boolean mask with an index matching desi_data.

        Returns:
            Data: A new Data object containing only the filtered data.
        """
        new_data_object = copy.copy(self)
        original_df = new_data_object.desi_data

        if callable(mask_or_func):
            # It's a function, so call it to get the mask
            mask = mask_or_func(original_df)
        else:
            # Assume it's a pre-computed boolean series
            mask = mask_or_func
        
        new_data_object.desi_data = original_df[mask].copy()

        # Copy over stream-related attributes if they exist (for when this is called on stream data)
        stream_attrs = ['SoI_streamfinder', 'frame', 'SoI_galstream']
        for attr in stream_attrs:
            if hasattr(self, attr):
                setattr(new_data_object, attr, getattr(self, attr))
        
        return new_data_object

    def sfTable(self):
        ''''
        Not working, ask Joseph how he got his table
        
        '''
        if hasattr(self, 'sf_data') and hasattr(self, 'desi_data'):
            # Convert to pandas if needed
            sf_df = self.sf_data.to_pandas() if not isinstance(self.sf_data, pd.DataFrame) else self.sf_data
            desi_df = self.desi_data.to_pandas() if not isinstance(self.desi_data, pd.DataFrame) else self.desi_data

            # Inner join on Gaia == TARGETID to find common entries
            merged = pd.merge(sf_df, desi_df, left_on='Gaia', right_on='TARGETID', how='inner')

            # Count how many matched entries per Stream
            stream_counts = merged['Stream'].value_counts().reset_index()
            stream_counts.columns = ['Stream', 'Matched_Count']

            # Also count how many total entries per Stream in sf_data
            total_counts = sf_df['Stream'].value_counts().reset_index()
            total_counts.columns = ['Stream', 'SF_Count']

            # Merge both counts into one table
            stream_counts = pd.merge(total_counts, stream_counts, on='Stream', how='left')
            stream_counts['Matched_Count'] = stream_counts['Matched_Count'].fillna(0).astype(int)

            # Display the result
            print(stream_counts.to_string(index=False))

    def sfCrossMatch(self, isin=True):
        if not hasattr(self, 'SoI_streamfinder') or self.SoI_streamfinder is None or len(self.SoI_streamfinder) == 0:
            # Ensure downstream attributes exist even when SF3 is absent
            if isin:
                if not hasattr(self, 'confirmed_sf_and_desi'):
                    # Create empty DataFrame with correct dtypes for numeric columns
                    self.confirmed_sf_and_desi = pd.DataFrame({
                        'TARGET_RA': pd.Series(dtype='float64'),
                        'TARGET_DEC': pd.Series(dtype='float64'),
                        'phi1': pd.Series(dtype='float64'),
                        'phi2': pd.Series(dtype='float64'),
                        'VGSR': pd.Series(dtype='float64'),
                        'PMRA': pd.Series(dtype='float64'),
                        'PMDEC': pd.Series(dtype='float64'),
                        'VRAD': pd.Series(dtype='float64'),
                        'VRAD_ERR': pd.Series(dtype='float64'),
                        'PMRA_ERROR': pd.Series(dtype='float64'),
                        'PMDEC_ERROR': pd.Series(dtype='float64'),
                        'FEH': pd.Series(dtype='float64'),
                        'FEH_ERR': pd.Series(dtype='float64')
                    })
                if not hasattr(self, 'cut_confirmed_sf_and_desi'):
                    self.cut_confirmed_sf_and_desi = self.confirmed_sf_and_desi.copy()
            else:
                if not hasattr(self, 'confirmed_sf_not_desi'):
                    self.confirmed_sf_not_desi = pd.DataFrame({
                        'RAdeg': pd.Series(dtype='float64'),
                        'DEdeg': pd.Series(dtype='float64'),
                        'phi1': pd.Series(dtype='float64'),
                        'phi2': pd.Series(dtype='float64'),
                        'VGSR': pd.Series(dtype='float64'),
                        'pmRA': pd.Series(dtype='float64'),
                        'pmDE': pd.Series(dtype='float64'),
                        'VHel': pd.Series(dtype='float64')
                    })
            print('No STREAMFINDER members available; skipping cross-match.')
            return

        gaia_source_ids = self.SoI_streamfinder.columns[0]
        
        # Determine the attribute name to use
        if isin:
            # Store the original data for comparison if it exists
            # Priority: existing confirmed_sf_and_desi > confirmed_sf_and_desi_full > original_confirmed_sf_and_desi
            if hasattr(self, 'confirmed_sf_and_desi') and len(self.confirmed_sf_and_desi) > 0:
                self.old_base = self.confirmed_sf_and_desi.copy()
            elif hasattr(self, 'confirmed_sf_and_desi_full'):
                self.old_base = self.confirmed_sf_and_desi_full.copy()
            elif hasattr(self, 'original_confirmed_sf_and_desi'):
                self.old_base = self.original_confirmed_sf_and_desi.copy()
            
            base_name = 'confirmed_sf_and_desi'
            attr_name = base_name
            if hasattr(self, base_name):
                import string
                suffix = 'b'
                while hasattr(self ,f'{base_name}_{suffix}'):
                    suffix = chr(ord(suffix) + 1)
                attr_name = f'{base_name}_{suffix}'
        
            # Perform the merge and assign to the chosen attribute
            merged = pd.merge(
                self.SoI_streamfinder.drop_duplicates(subset=[gaia_source_ids]),
                self.desi_data.drop_duplicates(subset=['SOURCE_ID']),
                left_on=gaia_source_ids,
                right_on='SOURCE_ID',
                how='inner',
                suffixes=('_sf', '_desi')
            )
            merged.dropna(inplace=True)
            
            # Calculate phi1 and phi2 coordinates from TARGET_RA and TARGET_DEC (DESI coordinates)
            if len(merged) > 0:
                merged['phi1'], merged['phi2'] = stream_funcs.ra_dec_to_phi1_phi2(self.frame,np.array(merged['TARGET_RA'])*u.deg, np.array(merged['TARGET_DEC'])*u.deg)
            else:
                # If no matches, create empty phi1 and phi2 columns
                merged['phi1'] = pd.Series(dtype='float64')
                merged['phi2'] = pd.Series(dtype='float64')
            
            if 'VRAD' in merged.columns and len(merged) > 0 and merged['VRAD'].notnull().any():
                merged['VGSR'] = np.array(
                    stream_funcs.vhel_to_vgsr(
                        np.array(merged['TARGET_RA']) * u.deg,
                        np.array(merged['TARGET_DEC']) * u.deg,
                        np.array(merged['VRAD']) * u.km/u.s
                    ).value
                )
            else:
                # nans length of the dataframe or empty series if no data
                if len(merged) > 0:
                    merged['VGSR'] = np.nan * np.ones(len(merged))
                    print("No valid VRAD values found in 'merged'; skipping VGSR computation.")
                else:
                    merged['VGSR'] = pd.Series(dtype='float64')
            setattr(self, base_name, merged) # NOTE change base_name to attr_name if I want to not overwrite past confirmed_sf_and_desi
            if hasattr(self, 'old_base') and len(self.old_base) > 0:
                # Find stars that were in old_base but not in the new merged data
                # Use SOURCE_ID for comparison as it's the unique identifier
                old_source_ids = set(self.old_base['SOURCE_ID']) if 'SOURCE_ID' in self.old_base.columns else set()
                new_source_ids = set(merged['SOURCE_ID']) if 'SOURCE_ID' in merged.columns else set()
                cut_source_ids = old_source_ids - new_source_ids
                
                if cut_source_ids:
                    not_in_merged = self.old_base[self.old_base['SOURCE_ID'].isin(cut_source_ids)]
                    self.cut_confirmed_sf_and_desi = not_in_merged
                    print(f"Created cut_confirmed_sf_and_desi with {len(not_in_merged)} stars that were filtered out")
                else:
                    # If no stars were cut, create an empty DataFrame with same structure
                    self.cut_confirmed_sf_and_desi = self.old_base.iloc[0:0].copy()
                    print("No stars were cut - cut_confirmed_sf_and_desi is empty")
            else:
                # No old_base available - create empty DataFrame
                if hasattr(self, 'confirmed_sf_and_desi') and len(self.confirmed_sf_and_desi) > 0:
                    self.cut_confirmed_sf_and_desi = self.confirmed_sf_and_desi.iloc[0:0].copy()
                else:
                    self.cut_confirmed_sf_and_desi = pd.DataFrame()
                print("No original data available for comparison - cut_confirmed_sf_and_desi is empty")


            print(f"Number of stars in SF: {len(self.SoI_streamfinder)}, Number of DESI and SF stars: {len(merged)}")
            print(f"Saved merged DataFrame as self.data.{attr_name}")
        else:
            base_name = 'confirmed_sf_not_desi'
            attr_name = base_name
            if hasattr(self, base_name):
                import string
                suffix = 'b'
                while hasattr(self, f'{base_name}_{suffix}'):
                    suffix = chr(ord(suffix)+1)
                attr_name = f'{base_name}_{suffix}'
            # Use a memory-friendly left-anti join instead of a full outer merge
            sf_unique = self.SoI_streamfinder.drop_duplicates(subset=[gaia_source_ids])
            desi_unique_ids = self.desi_data.drop_duplicates(subset=['SOURCE_ID'])['SOURCE_ID']
            only_in_SoI = sf_unique.loc[~sf_unique[gaia_source_ids].isin(desi_unique_ids)].copy()
            
            if len(only_in_SoI) > 0:
                only_in_SoI['phi1'], only_in_SoI['phi2'] = stream_funcs.ra_dec_to_phi1_phi2(self.frame,np.array(only_in_SoI['RAdeg'])*u.deg, np.array(only_in_SoI['DEdeg'])*u.deg)
                only_in_SoI['VGSR'] = np.array(stream_funcs.vhel_to_vgsr(np.array(only_in_SoI['RAdeg'])*u.deg, np.array(only_in_SoI['DEdeg'])*u.deg, np.array(only_in_SoI['VHel'])*u.km/u.s).value)
            else:
                # If no SF-only stars, create empty phi1, phi2, and VGSR columns
                only_in_SoI['phi1'] = pd.Series(dtype='float64')
                only_in_SoI['phi2'] = pd.Series(dtype='float64')
                only_in_SoI['VGSR'] = pd.Series(dtype='float64')
            
            setattr(self, base_name, only_in_SoI)
            print(f'Stars only in SF3: {len(only_in_SoI)}')

class Selection:
    """
    A class to manage and apply multiple selection criteria (masks) to a DataFrame.
    
    This class allows for the programmatic building of a complex filter by adding
    individual masks, which are then combined with a logical AND.
    """
    def __init__(self, data_frame):
        """
        Initializes the Selection object.

        Args:
            data_frame (pd.DataFrame): The pandas DataFrame to which the selections 
                                       will be applied.
        """
        if not isinstance(data_frame, pd.DataFrame):
            raise TypeError("Input 'data_frame' must be a pandas DataFrame.")
        
        self.df = data_frame
        self.masks = {} # A dictionary to store named mask functions
        print(f"Selection object created for DataFrame with {len(self.df)} rows.")

    def add_mask(self, name, mask_func):
        """
        Adds a new filtering criterion to the selection.

        Args:
            name (str): A descriptive name for the mask (e.g., 'metal_poor_cut').
            mask_func (function): A function that takes a DataFrame and returns a 
                                  boolean Series (the mask).
        """
        self.masks[name] = mask_func
        print(f"Mask added: '{name}'")

    def remove_mask(self, name):
        """Removes a mask by its name."""
        if name in self.masks:
            del self.masks[name]
            print(f"Mask removed: '{name}'")
        else:
            print(f"Warning: Mask '{name}' not found.")
            
    def list_masks(self):
        """Prints the names of all currently active masks."""
        if not self.masks:
            print("No masks are currently active.")
        else:
            print("Active masks:")
            for name in self.masks:
                print(f"- {name}")

    def get_final_mask(self):
        """
        Computes the final combined boolean mask.

        All individual masks are combined using a logical AND.

        Returns:
            pd.Series: A boolean Series representing the final combined mask.
        """
        if not self.masks:
            print("No masks to apply, returning an all-True mask.")
            return pd.Series([True] * len(self.df), index=self.df.index)

        # Start with a mask that is True for all entries
        final_mask = pd.Series(True, index=self.df.index)
        
        print("Combining masks...")
        for name, mask_func in self.masks.items():
            individual_mask = mask_func(self.df)
            final_mask &= individual_mask # Combine with logical AND
            print(f"...'{name}' selected {individual_mask.sum()} stars")

        print(f"Selection: {final_mask.sum()} / {len(self.df)} stars.")
        return final_mask
    
    def get_masks(self, mask_names):
        """
        Computes the final mask for a specific list of mask names.

        All individual masks are combined using a logical AND.

        Returns:
            pd.Series: A boolean Series representing the final combined mask for the specified names.
        """
        if not mask_names:
            print("No mask names provided, returning an all-True mask.")
            return pd.Series([True] * len(self.df), index=self.df.index)

        combined_mask = pd.Series(True, index=self.df.index)
        for name in mask_names:
            if name in self.masks:
                individual_mask = self.masks[name](self.df)
                combined_mask &= individual_mask  # Combine with logical AND
                print(f"...'{name}' selected {individual_mask.sum()} stars")
            else:
                print(f"Warning: Mask '{name}' not found. Skipping.")
        print(f"Selection for specified masks: {combined_mask.sum()} / {len(self.df)} stars.")

        return combined_mask


class stream:
    def __init__(self, data_object, streamName='Sylgr-I21', frame=None, add_bhb=True, **kwargs):
        self.streamName = streamName
        self.local = data_object.cleaned_data
        self.frame = frame
        self.add_bhb = add_bhb

        # Status flags for downstream logic
        self.from_sf3 = False
        self.from_galstreams = False
        self.galstreams_only = False
        self.source_catalog = None

        import os
        # file lives in DESI_streamTutorial/libraries; data/ is one level up
        sf3_table_path = Path(__file__).resolve().parent.parent / 'data' / 'sf3_table_dr1.csv'
        sf3_table = pd.read_csv(sf3_table_path)

        # Try to find stream index by matching either 'Stream' or 'Galstream Stream name' columns
        stream_match = sf3_table[
            (sf3_table['Stream'] == streamName) | 
            (sf3_table['Galstream Stream name'] == streamName)
        ]

        if len(stream_match) == 1:
            self.from_sf3 = True
            self.streamNo = int(stream_match['SF3 Stream Index'].iloc[0])
            print(f"Found stream '{streamName}' in SF3 with index {self.streamNo}")
        elif len(stream_match) > 1:
            raise ValueError(f"Multiple matches found for stream '{streamName}'. Please be more specific.")
        else:
            self.streamNo = None
            print(f"Stream '{streamName}' not found in sf3_table_dr1.csv; trying galstreams.")

        # Store a reference to the data object instead of re-running the init
        self.data = data_object

        # Import galstreams and attempt to load the track regardless of SF3 status
        print('Importing galstreams module...')
        import galstreams
        mwsts = galstreams.MWStreams(verbose=False, implement_Off=True)
        self.mwsts = mwsts
        self.data.SoI_galstream = mwsts.get(streamName, None)
        self.from_galstreams = self.data.SoI_galstream is not None

        # Decide which catalog(s) are available
        if self.from_sf3 and self.from_galstreams:
            self.source_catalog = 'sf3+galstreams'
        elif self.from_sf3:
            self.source_catalog = 'sf3'
        elif self.from_galstreams:
            self.source_catalog = 'galstreams_only'
            self.galstreams_only = True
        else:
            raise ValueError(
                f"Stream '{streamName}' not found in sf3_table_dr1.csv or galstreams MWStreams."
            )

        # Attach SF3 members if available; otherwise set empty placeholders
        if self.from_sf3:
            self.data.SoI_streamfinder = self.data.sf_data[self.data.sf_data['Stream'] == self.streamNo]
        else:
            # Empty placeholder with expected columns to avoid downstream KeyErrors
            self.data.SoI_streamfinder = pd.DataFrame(columns=['RAdeg', 'DEdeg', 'VHel', 'Stream'])

        if self.from_galstreams:
            self.min_dist = np.nanmin(self.data.SoI_galstream.track.distance.value)
            self.frame = self.frame or self.data.SoI_galstream.stream_frame
            self.data.frame = self.frame
            self.data.SoI_galstream.gal_phi1 = self.data.SoI_galstream.track.transform_to(self.frame).phi1.deg
            self.data.SoI_galstream.gal_phi2 = self.data.SoI_galstream.track.transform_to(self.frame).phi2.deg
            print(f"Using galstreams track for '{streamName}' (min dist ~ {self.min_dist:.3f} kpc).")
        else:
            self.min_dist = np.nan
            print('No galstream track available for this stream.')

        # Allow user override of the coordinate frame
        custom_frame = kwargs.get('custom_frame', None)
        if custom_frame is not None:
            self.frame = custom_frame
            self.data.frame = custom_frame

        if self.frame is None:
            raise ValueError("No coordinate frame available. Provide custom_frame or ensure galstreams track exists.")

        print('Creating combined DataFrame of SF and DESI')
        # Access desi_data through self.data
        if self.from_sf3:
            self.data.sfCrossMatch()  # saved as confirmed_sf_and_desi
            self.data.sfCrossMatch(isin=False)  # creates DF of stars not in DESI
        else:
            # Empty placeholders so downstream plotting does not fail
            empty_sf_desi_cols = [
                'TARGET_RA', 'TARGET_DEC', 'phi1', 'phi2', 'VGSR', 'PMRA', 'PMDEC',
                'VRAD', 'VRAD_ERR', 'PMRA_ERROR', 'PMDEC_ERROR', 'FEH', 'FEH_ERR'
            ]
            self.data.confirmed_sf_and_desi = pd.DataFrame(columns=empty_sf_desi_cols)
            self.data.cut_confirmed_sf_and_desi = self.data.confirmed_sf_and_desi.copy()
            self.data.confirmed_sf_not_desi = pd.DataFrame(columns=['RAdeg', 'DEdeg', 'phi1', 'phi2', 'VGSR', 'pmRA', 'pmDE', 'VHel'])

        # Compute phi1/phi2 for DESI and SF3 (if present)
        self.data.desi_data['phi1'], self.data.desi_data['phi2'] = stream_funcs.ra_dec_to_phi1_phi2(
            self.frame,
            np.array(self.data.desi_data['TARGET_RA']) * u.deg,
            np.array(self.data.desi_data['TARGET_DEC']) * u.deg,
        )

        if self.from_sf3 and not self.data.SoI_streamfinder.empty:
            self.data.SoI_streamfinder['phi1'], self.data.SoI_streamfinder['phi2'] = stream_funcs.ra_dec_to_phi1_phi2(
                self.frame,
                np.array(self.data.SoI_streamfinder['RAdeg']) * u.deg,
                np.array(self.data.SoI_streamfinder['DEdeg']) * u.deg,
            )

        if self.from_sf3 and not self.data.confirmed_sf_and_desi.empty:
            self.data.confirmed_sf_and_desi['phi1'], self.data.confirmed_sf_and_desi['phi2'] = stream_funcs.ra_dec_to_phi1_phi2(
                self.frame,
                np.array(self.data.confirmed_sf_and_desi['TARGET_RA']) * u.deg,
                np.array(self.data.confirmed_sf_and_desi['TARGET_DEC']) * u.deg,
            )

        if self.from_sf3 and not self.data.confirmed_sf_not_desi.empty:
            self.data.confirmed_sf_not_desi['phi1'], self.data.confirmed_sf_not_desi['phi2'] = stream_funcs.ra_dec_to_phi1_phi2(
                self.frame,
                np.array(self.data.confirmed_sf_not_desi['RAdeg']) * u.deg,
                np.array(self.data.confirmed_sf_not_desi['DEdeg']) * u.deg,
            )
            # convert sf from VHel to VGSR
            self.data.confirmed_sf_not_desi['VGSR'] = np.array(
                stream_funcs.vhel_to_vgsr(
                    np.array(self.data.confirmed_sf_not_desi['RAdeg']) * u.deg,
                    np.array(self.data.confirmed_sf_not_desi['DEdeg']) * u.deg,
                    np.array(self.data.confirmed_sf_not_desi['VHel']) * u.km / u.s,
                ).value
            )

        if self.from_sf3 and not self.data.confirmed_sf_and_desi.empty:
            self.data.confirmed_sf_and_desi['VGSR'] = np.array(
                stream_funcs.vhel_to_vgsr(
                    np.array(self.data.confirmed_sf_and_desi['TARGET_RA']) * u.deg,
                    np.array(self.data.confirmed_sf_and_desi['TARGET_DEC']) * u.deg,
                    np.array(self.data.confirmed_sf_and_desi['VRAD']) * u.km / u.s,
                ).value
            )
    
    def mask_stream(self, mask_or_func):
        """
        Create a new stream object with filtered data and perform all necessary cross-matching.
        This replaces the 4-line pattern:
        - trimmed_desi = SoI.data.select(final_mask)
        - trimmed_stream = copy.copy(SoI)
        - trimmed_stream.data = trimmed_desi
        - trimmed_stream.data.sfCrossMatch(); trimmed_stream.data.sfCrossMatch(False)
        
        Args:
            mask_or_func: The mask or function to apply for filtering
        
        Returns:
            stream: A new stream object with filtered and cross-matched data
        """
        # Step 1: Apply the mask to the data
        trimmed_data = self.data.select(mask_or_func)
        
        # Step 2: Create a copy of the stream object
        trimmed_stream = copy.copy(self)
        
        # Step 3: Assign the filtered data
        trimmed_stream.data = trimmed_data
        
        # Step 4: Perform cross-matching
        trimmed_stream.data.sfCrossMatch()  # Creates confirmed_sf_and_desi
        trimmed_stream.data.sfCrossMatch(False)  # Creates confirmed_sf_not_desi
        
        # Step 5: Automatically compute VGSR for confirmed_sf_not_desi if it exists and has VHel data
        if hasattr(trimmed_stream.data, 'confirmed_sf_not_desi') and len(trimmed_stream.data.confirmed_sf_not_desi) > 0:
            if 'VHel' in trimmed_stream.data.confirmed_sf_not_desi.columns:
                # Copy VHel and set 0 values to np.nan
                vhel = np.array(trimmed_stream.data.confirmed_sf_not_desi['VHel'], dtype=float)
                vhel[vhel == 0] = np.nan
                
                # Only compute VGSR if we have valid VHel values
                if not np.all(np.isnan(vhel)):
                    # Compute VGSR using the stream_functions
                    trimmed_stream.data.confirmed_sf_not_desi['VGSR'] = stream_funcs.vhel_to_vgsr(
                        np.array(trimmed_stream.data.confirmed_sf_not_desi['RAdeg']) * u.deg,
                        np.array(trimmed_stream.data.confirmed_sf_not_desi['DEdeg']) * u.deg,
                        vhel * u.km/u.s
                    ).value
        
        return trimmed_stream

    def isochrone(self, metallicity, age):
        """
        Placeholder for isochrone fitting logic.
        """
        dotter_directory='./data/dotter/' if self.local else '/home/jupyter-nasserm/raid/nasserm/data/dotter/'
        mass_fraction = 0.0181 * 10 ** metallicity
        print(f'Mass Fraction (Z): {mass_fraction}')

        dotter_mass_frac = np.array([
        0.00006, 0.00007, 0.00009, 0.00010, 0.00011, 0.00013, 0.00014, 0.00016,
        0.00017, 0.00019, 0.00021, 0.00024, 0.00028, 0.00032, 0.00037, 0.00042,
        0.00049, 0.00057, 0.00063, 0.00072, 0.00082, 0.00093, 0.00108, 0.00124,
        0.00144, 0.00166, 0.00189, 0.00213, 0.00242, 0.00276, 0.00316, 0.00363,
        0.00417
        ])
        dotter_mass_frac_str = [
        "0.00006", "0.00007", "0.00009", "0.00010", "0.00011", "0.00013", "0.00014", "0.00016",
        "0.00017", "0.00019", "0.00021", "0.00024", "0.00028", "0.00032", "0.00037", "0.00042",
        "0.00049", "0.00057", "0.00063", "0.00072", "0.00082", "0.00093", "0.00108", "0.00124",
        "0.00144", "0.00166", "0.00189", "0.00213", "0.00242", "0.00276", "0.00316", "0.00363",
        "0.00417"
    ]
        print('')
        use_mass_frac = dotter_mass_frac_str[np.argmin(np.abs(dotter_mass_frac - mass_fraction))]


        isochrone_path = dotter_directory + 'iso_a' + str(age) + '_z' + str(use_mass_frac) + '.dat'
        print(f'using {isochrone_path}')
        dotter_mp = np.loadtxt(isochrone_path)
        self.isochrone_path = isochrone_path

        # Obtain the M_g and M_r color band data
        self.dotter_g_mp = dotter_mp[:,6]
        self.dotter_r_mp = dotter_mp[:,7]

        if np.round(self.min_dist,4) != 1:
            # interpolate distance
            interpolate_distances = interp1d(self.data.SoI_galstream.gal_phi1, self.data.SoI_galstream.track.distance.value*1000, kind='linear', fill_value='extrapolate')
            distance_sf = interpolate_distances(self.data.confirmed_sf_and_desi['phi1']) if not self.data.confirmed_sf_and_desi.empty else np.array([])
            distance_desi = interpolate_distances(self.data.desi_data['phi1'])
            distance_cut_sf = interpolate_distances(self.data.cut_confirmed_sf_and_desi['phi1']) if hasattr(self.data, 'cut_confirmed_sf_and_desi') and not self.data.cut_confirmed_sf_and_desi.empty else np.array([])
            print('Using distance gradient')
        elif not self.data.confirmed_sf_and_desi.empty:
            distance_sf = 1/np.nanmean(self.data.confirmed_sf_and_desi['PARALLAX'])*1000
            distance_desi = distance_sf
            distance_cut_sf = 1/np.nanmean(self.data.cut_confirmed_sf_and_desi['PARALLAX'])*1000 if hasattr(self.data, 'cut_confirmed_sf_and_desi') else None
            print(f'set distance to {distance_sf} pc')
        else:
            # Galstreams-only mode: use galstream distance for DESI data
            if hasattr(self.data, 'SoI_galstream') and self.data.SoI_galstream is not None:
                interpolate_distances = interp1d(self.data.SoI_galstream.gal_phi1, self.data.SoI_galstream.track.distance.value*1000, kind='linear', fill_value='extrapolate')
                distance_desi = interpolate_distances(self.data.desi_data['phi1'])
                distance_sf = np.array([])
                distance_cut_sf = np.array([])
                print('Using galstreams distance gradient for DESI data')
            else:
                print('No distance for the stream, go look in literature and set manually with self.min_dist = XX') #kpc)
                return
        self.data.desi_colour_idx, self.data.desi_abs_mag, self.data.desi_r_mag = stream_funcs.get_colour_index_and_abs_mag(self.data.desi_data['EBV'], self.data.desi_data['FLUX_G'], self.data.desi_data['FLUX_R'], distance_desi)
        if not self.data.confirmed_sf_and_desi.empty:
            self.data.sf_colour_idx, self.data.sf_abs_mag, self.data.sf_r_mag = stream_funcs.get_colour_index_and_abs_mag(self.data.confirmed_sf_and_desi['EBV'], self.data.confirmed_sf_and_desi['FLUX_G'], self.data.confirmed_sf_and_desi['FLUX_R'], distance_sf)
        else:
            self.data.sf_colour_idx = np.array([])
            self.data.sf_abs_mag = np.array([])
            self.data.sf_r_mag = np.array([])
        if hasattr(self.data, 'cut_confirmed_sf_and_desi') and not self.data.cut_confirmed_sf_and_desi.empty:
            self.data.cut_sf_colour_idx, self.data.cut_sf_abs_mag, self.data.cut_sf_r_mag = stream_funcs.get_colour_index_and_abs_mag(self.data.cut_confirmed_sf_and_desi['EBV'], self.data.cut_confirmed_sf_and_desi['FLUX_G'], self.data.cut_confirmed_sf_and_desi['FLUX_R'], distance_cut_sf)
        else:
            self.data.cut_sf_colour_idx = np.array([])
            self.data.cut_sf_abs_mag = np.array([])
            self.data.cut_sf_r_mag = np.array([])
        g_r_color_dif = self.dotter_g_mp - self.dotter_r_mp
        sorted_indices = np.argsort(self.dotter_r_mp)
        sorted_dotter_r_mp = self.dotter_r_mp[sorted_indices]
        g_r_color_dif = g_r_color_dif[sorted_indices]

        # Fit for the isochrone line
        self.isochrone_fit = sp.interpolate.UnivariateSpline(sorted_dotter_r_mp, g_r_color_dif, s=0)

#class Orbit: WIP

class StreamPlotter:
    """
    For really clean and easy plotting
    """
    def __init__(self, stream_or_mcmeta_object, save_dir='plots/'):
        """
        Initializes the plotter with a stream object or MCMeta object.
        
        Args:
            stream_or_mcmeta_object: Either a stream instance or MCMeta instance
            save_dir (str): Directory to save plots.
        """
        # Check if it's an MCMeta object or stream object
        if hasattr(stream_or_mcmeta_object, 'initial_params'):
            # It's an MCMeta object
            self.mcmeta = stream_or_mcmeta_object
            self.stream = stream_or_mcmeta_object.stream
            self.data = stream_or_mcmeta_object.stream.data
            # propagate fixed PM sigma if set on MCMeta
            self.lsigpm_ = getattr(stream_or_mcmeta_object, 'lsigpm_', None)
            # Backward-compatible alias expected by existing plotting helpers
            self.meta = self.mcmeta
        else:
            # It's a stream object
            self.stream = stream_or_mcmeta_object
            self.data = stream_or_mcmeta_object.data
            self.mcmeta = None
            # No MCMeta available; default to None
            self.lsigpm_ = None
            self.meta = None
            
        self.save_dir = save_dir
        # Create directory if it doesn't exist
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

        self.plot_params = {
            'sf_in_desi': {
                'marker': 'd',
                's': 40,
                'color': 'green',
                'label': r'SF $\in$ DESI',
                'edgecolor': 'k',
                'zorder': 5
            },
            'sf_not_desi': {
                'marker': 'd',
                's': 30,
                'color': 'none',
                'alpha': 1,
                'edgecolor': 'k',
                'label': 'SF (not in DESI)',
                'zorder': 4
            },
            'sf_in_desi_notsel': {
                'marker': 'd',
                's': 40,
                'color': 'g',
                # 'label': r'SF $\in$ DESI, Cut',
                'edgecolor': 'k',
                'zorder': 5
            },
            'sf_not_desi_notsel': {
                'marker': 'd',
                's': 30,
                'color': 'none',
                'alpha': 1,
                'edgecolor': 'r',
                'label': 'SF (not in DESI), Cut',
                'zorder': 4
            },
            'desi_field': {
                'marker': '.',
                's': 1,
                'color': 'k',
                'alpha': 0.1,
                'label': 'DESI Field Stars',
                'zorder': 1
            },
            'galstream_track': {
                'color': 'y',
                'lw': 2,
                'alpha': 0.5,
                'label': 'galstream',
                'zorder': 2
            },
            'spline_track': {
                'color': 'b',
                'ls': '-.',
                'lw': 1,
                'label': 'Spline',
                'zorder': 3
            },
            'orbit_track': {
                'color': 'r',
                'ls': 'dotted',
                'lw': 1,
                'label': 'Best-fit Orbit',
                'zorder': 3
            },
            'background':{
                'color':'k',
                's':2,
                'alpha': 0.02,
                'zorder':0
            },
            # Centralized styles used in sixD_plot
            'sf_errorbar': {
                'ecolor': 'k',
                'elinewidth': 0.8,
                'capsize': 2,
                'alpha': 0.8,
                'zorder': 4
            },
            'member_errorbar': {
                'ecolor': 'black',
                'elinewidth': 0.8,
                'capsize': 2,
                'alpha': 0.8,
                'zorder': 5
            },
            'initial_spline_line': {
                'color': 'k',
                'lw': 2,
                'alpha': 0.8,
                'label': 'Initial Spline'
            },
            'optimized_spline_line': {
                'color': 'r',
                'lw': 2,
                'alpha': 0.8,
                'label': 'Optimized Spline'
            },
            'initial_feh_line': {
                'color': 'k',
                'lw': 2,
                'alpha': 0.8,
                'label': 'Initial [Fe/H]'
            },
            'optimized_feh_line': {
                'color': 'r',
                'lw': 2,
                'alpha': 0.8,
                'label': 'Optimized [Fe/H]'
            },
            'spline_knot_initial': {
                'marker': 'o',
                'color': 'black',
                's': 50,
                'zorder': 10,
                'alpha': 0.8,
                'edgecolors': 'white',
                'linewidth': 1
            },
            'spline_knot_optimized': {
                'marker': 'o',
                'color': 'red',
                's': 50,
                'zorder': 10,
                'alpha': 0.8,
                'edgecolors': 'white',
                'linewidth': 1
            },
            'membership_scatter': {
                'marker': 'o',
                's': 25,
                'edgecolor': 'black',
                'linewidth': 0.5,
                'alpha': 0.8,
                'zorder': 6
            },
            'sf_high_prob_diamond': {
                'marker': 'D',
                's': 40,
                'edgecolor': 'black',
                'linewidth': 1,
                'alpha': 1.0,
                'zorder': 7
            },
            'sf_low_prob_diamond': {
                'marker': 'D',
                's': 40,
                'color': 'black',
                'edgecolor': 'black',
                'linewidth': 1,
                'alpha': 1.0,
                'zorder': 7
            },
            'membership_colorbar': {
                'cmap': 'viridis',
                'label': 'Membership Probability',
                'pad': 0.02,
                'aspect': 50,
                'shrink': 1.0,
                'location': 'right',
                'labelpad': 15
            },
            'limits': {
                'feh_ylim_default': (-4, -0.5),
                'residual_pad_vgsr': 20,
                'residual_pad_pm': 2,
                'residual_pad_feh': 0.2
            }
        }
    def plot_galstreams(self):
        """
        Plots all the galstream stars within the frame of your SoI
        """
        import astropy.coordinates as ac
        mwsts = self.stream.mwsts
        tnames = mwsts.get_track_names_in_sky_window([np.nanmin(self.data.desi_data['TARGET_RA']), np.nanmax(self.data.desi_data['TARGET_RA'])]*u.deg, [np.nanmin(self.data.desi_data['TARGET_DEC']), np.nanmax(self.data.desi_data['TARGET_DEC'])]*u.deg, frame=ac.ICRS, 
                                           On_only=False, wrap_angle=180.*u.deg)
        fig, ax = plt.subplots(1,1, figsize=(7,3))
        ax.scatter(self.data.desi_data['TARGET_RA'], self.data.desi_data['TARGET_DEC'], color='grey', s=5, label='DESI data', alpha=0.9)
        for st in tnames:
            ax.plot(mwsts[st].track.ra, mwsts[st].track.dec, '-o', ms=2., label=st)
        ax.set_xlim(np.nanmin(self.data.desi_data['TARGET_RA'])-2, np.nanmax(self.data.desi_data['TARGET_RA'])+2)
        ax.set_ylim(np.nanmin(self.data.desi_data['TARGET_DEC'])-2, np.nanmax(self.data.desi_data['TARGET_DEC'])+2)
        ax.set_xlabel('RA [deg]')
        ax.set_ylabel('Dec [deg]')

        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        return fig, ax
    def on_sky(self, stream_frame=True, showStream=True, save=False, galstream=True, orbit=True, background=False):
        """
        Plots the stream on-sky either in RA, DEC or phi1, phi2
        """
        sf_in_desi = getattr(self.data, 'confirmed_sf_and_desi', pd.DataFrame())
        sf_not_desi = getattr(self.data, 'confirmed_sf_not_desi', pd.DataFrame())
        cut_sf = getattr(self.data, 'cut_confirmed_sf_and_desi', pd.DataFrame())
        if stream_frame:
            col_x = 'phi1'
            col_x_ = 'phi1'
            label_x = r'$\phi_1$'
            col_y = 'phi2'
            col_y_ = 'phi2'
            label_y = r'$\phi_2$'
        else:
            col_x = 'TARGET_RA'
            col_x_ = 'RAdeg'
            col_y = 'TARGET_DEC'
            col_y_ = 'DEdeg'
            label_x = 'RA (deg)'
            label_y = 'DEC (deg)'

        fig, ax = plt.subplots(figsize=(10, 3))
        if showStream:
            if not sf_in_desi.empty:
                ax.scatter(
                    sf_in_desi[col_x],
                    sf_in_desi[col_y],
                    **self.plot_params['sf_in_desi']
                )
            if not sf_not_desi.empty:
                ax.scatter(
                    sf_not_desi[col_x_],
                    sf_not_desi[col_y_],
                    **self.plot_params['sf_not_desi']
                )
            if not cut_sf.empty:
                ax.scatter(
                    cut_sf[col_x],
                    cut_sf[col_y],
                    **self.plot_params['sf_in_desi_notsel']
                )
            if galstream and hasattr(self.data, 'SoI_galstream') and (self.data.SoI_galstream is not None):
                if stream_frame:
                    ax.plot(
                        self.data.SoI_galstream.gal_phi1,
                        self.data.SoI_galstream.gal_phi2,
                        **self.plot_params['galstream_track']
                    )
                else:
                    ax.plot(
                        self.data.SoI_galstream.track.ra,
                        self.data.SoI_galstream.track.dec,
                        **self.plot_params['galstream_track']
                    )

        if background:
            ax.scatter(
                self.data.desi_data[col_x],
                self.data.desi_data[col_y],
                **self.plot_params['background']
            )

        # Placeholder for orbit plotting logic
        # if hasattr(self.stream, orbit):
        #     ax.plot(
        #         self.orbit.<x>,
        #         self.orbit.<y>,
        #         **self.plot_params['orbit_track'])

        ax.legend(loc='upper left', ncol=4)
        ax.set_ylabel(label_y)
        ax.set_xlabel(label_x)
        stream_funcs.plot_form(ax)  # Make sure this is defined or imported

    def plx_cut(self, showStream=True, background=True, save=False, galstream=False):
        # Skip this plot if PARALLAX data is missing
        if 'PARALLAX' not in self.data.desi_data.columns:
            print("Warning: PARALLAX column not found in desi_data; skipping plx_cut plot.")
            return None, None
            
        fig, ax = plt.subplots(figsize=(10, 5))
        col_x = 'TARGET_RA'
        col_x_ = 'RAdeg'
        label_x = 'RA (deg)'
        label_y = r'Parallax - 2* Paralalx Error'
        if background:
            ax.scatter(
                self.data.desi_data[col_x],
                self.data.desi_data['PARALLAX']-2*self.data.desi_data['PARALLAX_ERROR'],
                **self.plot_params['background']
            )
        if showStream and not self.data.confirmed_sf_and_desi.empty and 'PARALLAX' in self.data.confirmed_sf_and_desi.columns:
            ax.scatter(
                self.data.confirmed_sf_and_desi[col_x],
                self.data.confirmed_sf_and_desi['PARALLAX']-2* self.data.confirmed_sf_and_desi['PARALLAX_ERROR'],
                **self.plot_params['sf_in_desi']
            )
            # ax.scatter(
            #     self.data.confirmed_sf_not_desi[col_x_],
            #     self.data.confirmed_sf_not_desi['plx']-2*np.nanmean(self.data.desi_data['PARALLAX_ERROR']), # NOTE, error is not given, may want to get from Gaia?
            #     **self.plot_params['sf_not_desi']
            # )

        # WIP, want to show if any stars are cut. Right now its failing for some reason.
        if hasattr(self.data, 'cut_confirmed_sf_and_desi') and not self.data.cut_confirmed_sf_and_desi.empty and 'PARALLAX' in self.data.cut_confirmed_sf_and_desi.columns:
            if showStream:
                ax.scatter(
                    self.data.cut_confirmed_sf_and_desi[col_x],
                    self.data.cut_confirmed_sf_and_desi['PARALLAX']-2* self.data.cut_confirmed_sf_and_desi['PARALLAX_ERROR'],
                    **self.plot_params['sf_in_desi_notsel']
                )
        #         ax.scatter(
        #             self.data.notsel.confirmed_sf_not_desi[col_x_],
        #             self.data.notsel.confirmed_sf_not_desi['PARALLAX']-2* self.data.notsel.confirmed_sf_not_desi['PARALLAX_ERROR'],
        #             **self.plot_params['sf_not_desi_notsel']
        #         )

        # Draw reference line if distance is known
        if np.isfinite(self.stream.min_dist):
            ax.axhline(y=1/self.stream.min_dist, color='r', linestyle='--', label=f'1 / min_dist ({self.stream.min_dist:.2f})')

        # Robust y-limits: prefer SF3 plx if available, else DESI parallaxes
        plx_vals = None
        if hasattr(self.data, 'SoI_streamfinder') and not self.data.SoI_streamfinder.empty and 'plx' in self.data.SoI_streamfinder:
            plx_vals = np.asarray(self.data.SoI_streamfinder['plx'])
        if plx_vals is None or len(plx_vals) == 0 or np.all(~np.isfinite(plx_vals)):
            # fallback to DESI parallax estimates
            plx_vals = np.asarray(self.data.desi_data['PARALLAX']) - 2*np.asarray(self.data.desi_data['PARALLAX_ERROR'])
        finite_plx = plx_vals[np.isfinite(plx_vals)]
        if finite_plx.size:
            pad = 1.0
            ax.set_ylim(finite_plx.min()-pad, finite_plx.max()+pad)
        
        ax.legend(loc='upper left', ncol=4)
        ax.set_ylabel(label_y)
        ax.set_xlabel(label_x)
        stream_funcs.plot_form(ax)  # Make sure this is defined or imported

    def kin_plot(self, showStream=True, show_sf_only=False, background=True, save=False, stream_frame=True,
                 show_hist=False, show_feh=False, hist_kwargs=None):  # , galstream=False):
        """
        Plots the stream kinematics either on-sky or stream_frame.

        Parameters
        ----------
        show_hist : bool, optional
            If True, append a horizontal density histogram of the DESI data to the right of
            the velocity panel.
        show_feh : bool, optional
            If True, add a [Fe/H] panel beneath the kinematic panels. Compatible with show_hist.
        hist_kwargs : dict, optional
            Extra keyword arguments forwarded to each histogram panel.
        """
        if stream_frame:
            col_x = 'phi1'
            col_x_ = 'phi1'
            label_x = r'$\phi_1$'
            col_y = 'VGSR'
            col_y_ = 'VGSR'
            label_y = r'$V_{\mathrm{GSR}}$ (km/s)'
        else:
            col_x = 'TARGET_RA'
            col_x_ = 'RAdeg'
            col_y = 'VGSR'
            col_y_ = 'VGSR'
            label_x = 'RA (deg)'
            label_y = r'$V_{\mathrm{GSR}}$ (km/s)'

        n_rows = 3 + int(show_feh)
        fig_height = 10 + 3 * int(show_feh)
        fig_width = 12 if show_hist else 10

        fig, axes = plt.subplots(n_rows, 1, figsize=(fig_width, fig_height), sharex=True)
        axes = np.atleast_1d(axes)
        fig.subplots_adjust(hspace=0.1)
        hist_axes = None

        if show_hist:
            fig.subplots_adjust(right=0.83)
            fig.canvas.draw()
            hist_axes = []
            for main_ax in axes:
                bbox = main_ax.get_position()
                hist_height = bbox.height
                hist_width = bbox.width / 7.0
                hist_left = bbox.x1
                hist_bottom = bbox.y0
                hist_ax = fig.add_axes([hist_left, hist_bottom, hist_width, hist_height])
                hist_axes.append(hist_ax)
            hist_axes = np.array(hist_axes, dtype=object)

        vel_ax, pmra_ax, pmdec_ax = axes[:3]
        feh_ax = axes[3] if show_feh else None

        has_sf_in = hasattr(self.data, 'confirmed_sf_and_desi') and not self.data.confirmed_sf_and_desi.empty
        has_sf_cut = hasattr(self.data, 'cut_confirmed_sf_and_desi') and not self.data.cut_confirmed_sf_and_desi.empty
        has_sf_not = hasattr(self.data, 'confirmed_sf_not_desi') and not self.data.confirmed_sf_not_desi.empty
        galstream = getattr(self.data, 'SoI_galstream', None)
        use_galstream_track = galstream is not None and (getattr(self.stream, 'galstreams_only', False) or not has_sf_in)

        def _to_val(arr):
            if arr is None:
                return None
            return np.asarray(arr.value if hasattr(arr, 'value') else arr)

        if showStream:
            if has_sf_in:
                vel_ax.scatter(
                    self.data.confirmed_sf_and_desi[col_x],
                    self.data.confirmed_sf_and_desi[col_y],
                    **self.plot_params['sf_in_desi']
                )
                pmra_ax.scatter(
                    self.data.confirmed_sf_and_desi[col_x],
                    self.data.confirmed_sf_and_desi['PMRA'],
                    **self.plot_params['sf_in_desi']
                )
                pmdec_ax.scatter(
                    self.data.confirmed_sf_and_desi[col_x],
                    self.data.confirmed_sf_and_desi['PMDEC'],
                    **self.plot_params['sf_in_desi']
                )
                # WIP, option to show sf not in desi
                if has_sf_cut:
                    vel_ax.scatter(
                        self.data.cut_confirmed_sf_and_desi[col_x],
                        self.data.cut_confirmed_sf_and_desi[col_y_],
                        **self.plot_params['sf_in_desi_notsel']
                    )
                    pmra_ax.scatter(
                        self.data.cut_confirmed_sf_and_desi[col_x],
                        self.data.cut_confirmed_sf_and_desi['PMRA'],
                        **self.plot_params['sf_in_desi_notsel']
                    )
                    pmdec_ax.scatter(
                        self.data.cut_confirmed_sf_and_desi[col_x],
                        self.data.cut_confirmed_sf_and_desi['PMDEC'],
                        **self.plot_params['sf_in_desi_notsel']
                    )
            elif use_galstream_track:
                phi1_g = _to_val(getattr(galstream, 'gal_phi1', None))
                vgsr_g = _to_val(getattr(galstream.track, 'radial_velocity', None)) if hasattr(galstream, 'track') else None
                if vgsr_g is not None and hasattr(galstream, 'track'):
                    try:
                        ra_g = _to_val(getattr(galstream.track, 'ra', None))
                        dec_g = _to_val(getattr(galstream.track, 'dec', None))
                        if ra_g is not None and dec_g is not None:
                            vgsr_g = stream_funcs.vhel_to_vgsr(
                                np.asarray(ra_g) * u.deg,
                                np.asarray(dec_g) * u.deg,
                                np.asarray(vgsr_g) * (u.km / u.s)
                            ).value
                    except Exception:
                        pass
                pmra_g = _to_val(getattr(galstream.track, 'pm_ra_cosdec', None)) if hasattr(galstream, 'track') else None
                pmdec_g = _to_val(getattr(galstream.track, 'pm_dec', None)) if hasattr(galstream, 'track') else None

                if phi1_g is not None and vgsr_g is not None and len(phi1_g) == len(vgsr_g):
                    vel_ax.plot(phi1_g, vgsr_g, **self.plot_params['galstream_track'])
                if phi1_g is not None and pmra_g is not None and len(phi1_g) == len(pmra_g):
                    pmra_ax.plot(phi1_g, pmra_g, **self.plot_params['galstream_track'])
                if phi1_g is not None and pmdec_g is not None and len(phi1_g) == len(pmdec_g):
                    pmdec_ax.plot(phi1_g, pmdec_g, **self.plot_params['galstream_track'])

        if show_sf_only and has_sf_not:
            vel_ax.scatter(
                self.data.confirmed_sf_not_desi[col_x_],
                self.data.confirmed_sf_not_desi[col_y_],
                **self.plot_params['sf_not_desi']
            )    
            pmra_ax.scatter(
                self.data.confirmed_sf_not_desi[col_x_],
                self.data.confirmed_sf_not_desi['pmRA'],
                **self.plot_params['sf_not_desi']
            )  
            pmdec_ax.scatter(
                self.data.confirmed_sf_not_desi[col_x_],
                self.data.confirmed_sf_not_desi['pmDE'],
                **self.plot_params['sf_not_desi']
            )  
        if background:
            vel_ax.scatter(
                self.data.desi_data[col_x],
                self.data.desi_data[col_y],
                **self.plot_params['background']
            )
            pmra_ax.scatter(
                self.data.desi_data[col_x],
                self.data.desi_data['PMRA'],
                **self.plot_params['background']
            )
            pmdec_ax.scatter(
                self.data.desi_data[col_x],
                self.data.desi_data['PMDEC'],
                **self.plot_params['background']
            )

        # Placeholder for orbit plotting logic
        # if hasattr(self.stream, orbit):
        #     ax.plot(
        #         self.orbit.<x>,
        #         self.orbit.<y>,
        #         **self.plot_params['orbit_track'])

        if showStream:
            # Safely set y-limits only when SF3 data exist
            def _safe_concat(columns):
                arrays = []
                for df in (getattr(self.data, 'confirmed_sf_and_desi', pd.DataFrame()),
                           getattr(self.data, 'cut_confirmed_sf_and_desi', pd.DataFrame())):
                    if not df.empty and columns in df:
                        arrays.append(np.asarray(df[columns]))
                return np.concatenate(arrays) if arrays else None

            v_vals = _safe_concat(col_y_)
            if v_vals is not None and len(v_vals) > 0:
                vel_ax.set_ylim(np.nanmin(v_vals) - 100, np.nanmax(v_vals) + 100)

            pmra_vals = _safe_concat('PMRA')
            if pmra_vals is not None and len(pmra_vals) > 0:
                pmra_ax.set_ylim(np.nanmin(pmra_vals) - 7, np.nanmax(pmra_vals) + 7)

            pmdec_vals = _safe_concat('PMDEC')
            if pmdec_vals is not None and len(pmdec_vals) > 0:
                pmdec_ax.set_ylim(np.nanmin(pmdec_vals) - 7, np.nanmax(pmdec_vals) + 7)

        if show_hist:
            background_style = self.plot_params.get('background', {})
            hist_color = background_style.get('color', 'k')
            hist_style = {
                'bins': 60,
                'density': True,
                'histtype': 'stepfilled',
                'alpha': 0.5,
            }
            if hist_kwargs:
                hist_style.update(hist_kwargs)
            hist_style['orientation'] = 'horizontal'
            hist_style.setdefault('color', hist_color)
            hist_style.setdefault('edgecolor', hist_style.get('color', hist_color))

            hist_columns = [col_y, 'PMRA', 'PMDEC']
            if show_feh:
                hist_columns.append('FEH')

            for main_ax, hist_ax, column in zip(axes, hist_axes, hist_columns):
                data_vals = np.asarray(self.data.desi_data[column])
                data_vals = data_vals[np.isfinite(data_vals)]
                if data_vals.size:
                    hist_ax.hist(data_vals, **hist_style)
                hist_ax.set_facecolor('none')
                hist_ax.grid(False)
                hist_ax.tick_params(axis='both', which='both', bottom=False, top=False,
                                    labelbottom=False, left=False, right=False, labelleft=False)
                hist_ax.set_xticks([])
                hist_ax.set_yticks([])
                for spine in hist_ax.spines.values():
                    spine.set_visible(False)
                hist_ax.set_xlim(left=0)
                hist_ax.margins(x=0)

        if show_feh and feh_ax is not None:
            if showStream:
                feh_ax.scatter(
                    self.data.confirmed_sf_and_desi[col_x],
                    self.data.confirmed_sf_and_desi['FEH'],
                    **self.plot_params['sf_in_desi']
                )
                if hasattr(self.data, 'cut_confirmed_sf_and_desi'):
                    feh_ax.scatter(
                        self.data.cut_confirmed_sf_and_desi[col_x],
                        self.data.cut_confirmed_sf_and_desi['FEH'],
                        **self.plot_params['sf_in_desi_notsel']
                    )
            if show_sf_only:
                feh_ax.scatter(
                    self.data.confirmed_sf_not_desi[col_x_],
                    self.data.confirmed_sf_not_desi['FEH'],
                    **self.plot_params['sf_not_desi']
                )
            if background:
                feh_ax.scatter(
                    self.data.desi_data[col_x],
                    self.data.desi_data['FEH'],
                    **self.plot_params['background']
                )
            feh_ax.set_ylim(-4, 0.5)
            feh_ax.set_ylabel(r'[Fe/H]')

        vel_ax.legend(loc='upper left', ncol=4)
        vel_ax.set_ylabel(label_y)
        pmra_ax.set_ylabel(r'$\mu_{\alpha}$ [mas/yr]')
        pmdec_ax.set_ylabel(r'$\mu_{\delta}$ [mas/yr]')
        for main_ax in axes[:-1]:
            main_ax.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
        axes[-1].set_xlabel(label_x)
        stream_funcs.plot_form(vel_ax)
        stream_funcs.plot_form(pmra_ax)
        stream_funcs.plot_form(pmdec_ax)
        if show_feh and feh_ax is not None:
            stream_funcs.plot_form(feh_ax)
        if show_hist:
            for main_ax, hist_ax in zip(axes, hist_axes):
                hist_ax.set_ylim(main_ax.get_ylim())

        return fig, axes

    def feh_plot(self, showStream=True, show_sf_only=False, background=True, save=False, stream_frame=True):
        """
        Plots the stream metallicity either on-sky or stream_frame
        """
        if stream_frame:
            col_x = 'phi1'
            col_x_ = 'phi1'
            label_x = r'$\phi_1$'
            col_y = 'FEH'
            col_y_ = 'FEH'
            label_y = r'[Fe/H]'
        else:
            col_x = 'TARGET_RA'
            col_x_ = 'RAdeg'
            col_y = 'FEH'
            col_y_ = 'FEH'
            label_x = 'RA (deg)'
            label_y = r'[Fe/H]'
        fig, ax = plt.subplots(figsize=(10, 5))
        if showStream:
            ax.scatter(
                self.data.confirmed_sf_and_desi[col_x],
                self.data.confirmed_sf_and_desi[col_y],
                **self.plot_params['sf_in_desi']
            )
            # WIP, option to show sf not in desi
            if hasattr(self.data, 'cut_confirmed_sf_and_desi'):
                if showStream:
                    ax.scatter(
                        self.data.cut_confirmed_sf_and_desi[col_x],
                        self.data.cut_confirmed_sf_and_desi[col_y_],
                        **self.plot_params['sf_in_desi_notsel']
                    )
        if show_sf_only:
            ax.scatter(
                self.data.confirmed_sf_not_desi[col_x_],
                self.data.confirmed_sf_not_desi[col_y_],
                **self.plot_params['sf_not_desi']
            )
        if background:
            ax.scatter(
                self.data.desi_data[col_x],
                self.data.desi_data[col_y],
                **self.plot_params['background']
            )
        
        ax.set_ylim(-4, 0.5)
        ax.legend(loc='upper left', ncol=4)
        ax.set_ylabel(label_y)
        ax.set_xlabel(label_x)
        stream_funcs.plot_form(ax) 

        return fig, ax

    def iso_plot(self, wiggle = 0.18, showStream=True, show_sf_only=False, background=True, save=False, absolute=True, BHB=False, bhb_wiggle=False):
        """
        Plotting the isochrone and stars
        """
        fig, ax = plt.subplots(figsize=(6, 7))
        if showStream and hasattr(self.data, 'sf_colour_idx') and np.size(self.data.sf_colour_idx) > 0:
            if absolute:
                ax.scatter(self.data.sf_colour_idx, self.data.sf_abs_mag,
                            **self.plot_params['sf_in_desi'])
            else:
                ax.scatter(self.data.sf_colour_idx, self.data.sf_r_mag,
                            **self.plot_params['sf_in_desi'])
        if background:
            if absolute:
                ax.scatter(self.data.desi_colour_idx, self.data.desi_abs_mag,
                            **self.plot_params['background'])
            else:
                ax.scatter(self.data.desi_colour_idx, self.data.desi_r_mag,
                            **self.plot_params['background'])
    
        ax.plot(self.stream.isochrone_fit(self.stream.dotter_r_mp), self.stream.dotter_r_mp,
                c='b', ls='-.')
        ax.plot(self.stream.isochrone_fit(self.stream.dotter_r_mp)+wiggle, self.stream.dotter_r_mp,
                c='b', ls='dotted', alpha=0.5, label='Colour wiggle')
        ax.plot(self.stream.isochrone_fit(self.stream.dotter_r_mp)-wiggle, self.stream.dotter_r_mp,
                c='b', ls='dotted', alpha=0.5)
        if hasattr(self.data, 'cut_confirmed_sf_and_desi') and hasattr(self.data, 'cut_sf_colour_idx'):
            if showStream and np.size(self.data.cut_sf_colour_idx) > 0:
                ax.scatter(
                    self.data.cut_sf_colour_idx,
                    self.data.cut_sf_abs_mag if absolute else self.data.cut_sf_r_mag,
                    **self.plot_params['sf_in_desi_notsel']
                )
        # Hard coded
        if BHB:

            # build the BHB using empirical data from M92
            dm_m92_harris = 14.59 #dm of M92
            m92ebv = 0.023
            m92ag = m92ebv * 3.184
            m92ar = m92ebv * 2.130
            m92_hb_r = np.array([17.3, 15.8, 15.38, 15.1, 15.05])
            m92_hb_col = np.array([-0.39, -0.3, -0.2, -0.0, 0.1])
            m92_hb_g = m92_hb_r + m92_hb_col
            des_m92_hb_g = m92_hb_g - 0.104 * (m92_hb_g - m92_hb_r) + 0.01
            des_m92_hb_r = m92_hb_r - 0.102 * (m92_hb_g - m92_hb_r) + 0.02
            des_m92_hb_g = des_m92_hb_g - m92ag - dm_m92_harris
            des_m92_hb_r = des_m92_hb_r - m92ar - dm_m92_harris
            ax.plot(des_m92_hb_g - des_m92_hb_r, des_m92_hb_r, c='b', alpha=1, ls='-.')
            if bhb_wiggle:
                bhb_color_wiggle = 0.4
                bhb_abs_mag_wiggle = 0.1
                ax.plot(des_m92_hb_g - des_m92_hb_r, des_m92_hb_r-bhb_color_wiggle, 'b:', alpha=0.5)
                ax.plot(des_m92_hb_g - des_m92_hb_r, des_m92_hb_r+bhb_color_wiggle, 'b:', alpha=0.5)
                ax.plot(des_m92_hb_g - des_m92_hb_r+bhb_abs_mag_wiggle, des_m92_hb_r, 'b:', alpha=0.5)
                ax.plot(des_m92_hb_g - des_m92_hb_r-bhb_abs_mag_wiggle, des_m92_hb_r, 'b:', alpha=0.5)
        # Hard coded
        
        ax.legend(loc='lower left')
        ax.set_xlabel('g-r',fontsize=15)
        ax.set_ylabel('$M_r$',fontsize=15)
        ax.set_xlim(-0.5, 1.2)
        ax.set_ylim(-1.5, 8)
        ax.invert_yaxis()
        stream_funcs.plot_form(ax)

    def sixD_plot(self, showStream=True, show_sf_only=False, background=True, save=False, stream_frame=True, galstream=False, show_cut=False, 
                  show_initial_splines=False, show_optimized_splines=False, show_mcmc_splines=False, show_sf_errors=True, 
                  show_membership_prob=False, stream_prob=None, min_prob=0.5, show_residuals=False, mcmc_object=None,
                  sigma_pad_frac=0.2, show_ppc=False, ppc_nsamples=200, ppc_phi_grid=100, ppc_alpha=0.25):
        """
        Plots the stream phi1 vs phi2, vgsr, pmra, pmdec, and feh
        
        Parameters:
        -----------
        show_cut : bool, optional
            Whether to show cut stars (red X markers). Default is True.
        show_initial_splines : bool, optional
            Whether to show initial guess splines in black. Default is False.
        show_optimized_splines : bool, optional
            Whether to show optimized splines in red. Default is False.
        show_mcmc_splines : bool, optional
            Whether to show MCMC results splines in blue. Default is False.
        show_sf_errors : bool, optional
            Whether to show error bars on StreamFinder stars. Default is True.
        show_membership_prob : bool, optional
            Whether to plot high membership probability stars with special styling. Default is False.
        stream_prob : array-like, optional
            Array of membership probabilities for DESI stars. Required if show_membership_prob=True.
        min_prob : float, optional
            Minimum membership probability threshold for highlighting stars. Default is 0.5.
        show_residuals : bool, optional
            If True and show_mcmc_splines=True in stream_frame, plot residuals (data - MCMC spline) for
            VGSR/PMRA/PMDEC/FEH so the MCMC spline lies along y=0. Ignored if MCMC results are unavailable.
        sigma_pad_frac : float, optional
            Fraction of sigma used as padding when setting y-limits. In residual mode, PM panels use
            ±(2*sigma + pad). In non-residual mode, if `stream_prob` is provided, PM panels use
            member-based limits computed as mean ± (2*sigma + pad) of the member values.
        """
        if stream_frame:
            col_x = 'phi1'
            col_x_ = 'phi1'
            label_x = r'$\phi_1$'
        else:
            col_x = 'TARGET_RA'
            col_x_ = 'RAdeg'
            label_x = 'RA (deg)'
        if show_membership_prob:
            fig, ax = plt.subplots(5, 1, figsize=(15, 15))
        else:
            fig, ax = plt.subplots(5, 1, figsize=(10, 15))

        # Residuals mode prep (only applicable with MCMC splines in stream frame)
        residual_mode = False
        preds = {'desi': {}, 'sf': {}, 'sf_cut': {}, 'sf_only': {}}
        # Prepare holders so we can later use MCMC spline ranges for y-limit logic ("faded blue regions")
        vgsr_mcmc = pmra_mcmc = pmdec_mcmc = feh_mcmc = None
        sigma_vgsr = sigma_pmra = sigma_pmdec = sigma_feh = None
        if mcmc_object is not None:
                meds = mcmc_object.meds
                ep = mcmc_object.ep
                em = mcmc_object.em
        if show_residuals and show_mcmc_splines and stream_frame and hasattr(self, 'mcmeta') and self.mcmeta is not None and hasattr(self.mcmeta, 'phi1_spline_points'):
            try:

                npts = len(self.mcmeta.phi1_spline_points)
                vgsr_knots = np.array([meds[f'vgsr{i}'] for i in range(1, npts+1)])
                pmra_knots = np.array([meds[f'pmra{i}'] for i in range(1, npts+1)])
                pmdec_knots = np.array([meds[f'pmdec{i}'] for i in range(1, npts+1)])
                feh_const = meds['feh1']

                def eval_spline(phi1_vals, knots):
                    return stream_funcs.apply_spline(phi1_vals, self.mcmeta.phi1_spline_points, knots, k=2)

                # DESI predictions
                phi1_desi = self.data.desi_data[col_x].values
                preds['desi']['vgsr'] = eval_spline(phi1_desi, vgsr_knots)
                preds['desi']['pmra'] = eval_spline(phi1_desi, pmra_knots)
                preds['desi']['pmdec'] = eval_spline(phi1_desi, pmdec_knots)
                preds['desi']['feh'] = np.full_like(phi1_desi, feh_const, dtype=float)

                # SF in DESI predictions
                if hasattr(self.data, 'confirmed_sf_and_desi'):
                    phi1_sf = self.data.confirmed_sf_and_desi[col_x].values
                    preds['sf']['vgsr'] = eval_spline(phi1_sf, vgsr_knots)
                    preds['sf']['pmra'] = eval_spline(phi1_sf, pmra_knots)
                    preds['sf']['pmdec'] = eval_spline(phi1_sf, pmdec_knots)
                    preds['sf']['feh'] = np.full_like(phi1_sf, feh_const, dtype=float)

                # Cut SF predictions
                if hasattr(self.data, 'cut_confirmed_sf_and_desi'):
                    phi1_sf_cut = self.data.cut_confirmed_sf_and_desi[col_x].values
                    preds['sf_cut']['vgsr'] = eval_spline(phi1_sf_cut, vgsr_knots)
                    preds['sf_cut']['pmra'] = eval_spline(phi1_sf_cut, pmra_knots)
                    preds['sf_cut']['pmdec'] = eval_spline(phi1_sf_cut, pmdec_knots)
                    preds['sf_cut']['feh'] = np.full_like(phi1_sf_cut, feh_const, dtype=float)

                # SF only (not in DESI)
                if hasattr(self.data, 'confirmed_sf_not_desi'):
                    phi1_sfo = self.data.confirmed_sf_not_desi[col_x_].values
                    preds['sf_only']['vgsr'] = eval_spline(phi1_sfo, vgsr_knots)
                    preds['sf_only']['pmRA'] = eval_spline(phi1_sfo, pmra_knots)
                    preds['sf_only']['pmDE'] = eval_spline(phi1_sfo, pmdec_knots)

                residual_mode = True
            except Exception as e:
                print(f"Warning: Residuals mode disabled (MCMC meds not available): {e}")
        
        # Plot 1: phi2 vs phi1 (or DEC vs RA)
        if stream_frame:
            col_y0 = 'phi2'
            col_y0_ = 'phi2'
            label_y0 = r'$\phi_2$'
        else:
            col_y0 = 'TARGET_DEC'
            col_y0_ = 'DEdeg'
            label_y0 = 'DEC (deg)'
            
        if showStream:
            if show_sf_errors:
                # Draw error bars first (behind), then overlay member points
                err_params = {'fmt': 'none'}
                err_params.update(self.plot_params.get('sf_errorbar', {}))
                # Only kinematics/abundance panels have measurement errors; no errorbar for phi2
                ax[1].errorbar(
                    self.data.confirmed_sf_and_desi[col_x],
                    (self.data.confirmed_sf_and_desi['VGSR'] - preds['sf'].get('vgsr', 0)) if residual_mode else self.data.confirmed_sf_and_desi['VGSR'],
                    yerr=self.data.confirmed_sf_and_desi['VRAD_ERR'], **err_params
                )
                ax[2].errorbar(
                    self.data.confirmed_sf_and_desi[col_x],
                    (self.data.confirmed_sf_and_desi['PMRA'] - preds['sf'].get('pmra', 0)) if residual_mode else self.data.confirmed_sf_and_desi['PMRA'],
                    yerr=self.data.confirmed_sf_and_desi['PMRA_ERROR'], **err_params
                )
                ax[3].errorbar(
                    self.data.confirmed_sf_and_desi[col_x],
                    (self.data.confirmed_sf_and_desi['PMDEC'] - preds['sf'].get('pmdec', 0)) if residual_mode else self.data.confirmed_sf_and_desi['PMDEC'],
                    yerr=self.data.confirmed_sf_and_desi['PMDEC_ERROR'], **err_params
                )
                ax[4].errorbar(
                    self.data.confirmed_sf_and_desi[col_x],
                    (self.data.confirmed_sf_and_desi['FEH'] - preds['sf'].get('feh', 0)) if residual_mode else self.data.confirmed_sf_and_desi['FEH'],
                    yerr=self.data.confirmed_sf_and_desi['FEH_ERR'], **err_params
                )
                # Now overlay the member points
                ax[0].scatter(
                    self.data.confirmed_sf_and_desi[col_x],
                    self.data.confirmed_sf_and_desi[col_y0],
                    **self.plot_params['sf_in_desi']
                )
                ax[1].scatter(
                    self.data.confirmed_sf_and_desi[col_x],
                    (self.data.confirmed_sf_and_desi['VGSR'] - preds['sf'].get('vgsr', 0)) if residual_mode else self.data.confirmed_sf_and_desi['VGSR'],
                    **self.plot_params['sf_in_desi']
                )
                ax[2].scatter(
                    self.data.confirmed_sf_and_desi[col_x],
                    (self.data.confirmed_sf_and_desi['PMRA'] - preds['sf'].get('pmra', 0)) if residual_mode else self.data.confirmed_sf_and_desi['PMRA'],
                    **self.plot_params['sf_in_desi']
                )
                ax[3].scatter(
                    self.data.confirmed_sf_and_desi[col_x],
                    (self.data.confirmed_sf_and_desi['PMDEC'] - preds['sf'].get('pmdec', 0)) if residual_mode else self.data.confirmed_sf_and_desi['PMDEC'],
                    **self.plot_params['sf_in_desi']
                )
                ax[4].scatter(
                    self.data.confirmed_sf_and_desi[col_x],
                    (self.data.confirmed_sf_and_desi['FEH'] - preds['sf'].get('feh', 0)) if residual_mode else self.data.confirmed_sf_and_desi['FEH'],
                    **self.plot_params['sf_in_desi']
                )
            else:
                # Plot without error bars (original behavior)
                ax[0].scatter(
                    self.data.confirmed_sf_and_desi[col_x],
                    self.data.confirmed_sf_and_desi[col_y0],
                    **self.plot_params['sf_in_desi']
                )
                ax[1].scatter(
                    self.data.confirmed_sf_and_desi[col_x],
                    (self.data.confirmed_sf_and_desi['VGSR'] - preds['sf'].get('vgsr', 0)) if residual_mode else self.data.confirmed_sf_and_desi['VGSR'],
                    **self.plot_params['sf_in_desi']
                )
                ax[2].scatter(
                    self.data.confirmed_sf_and_desi[col_x],
                    (self.data.confirmed_sf_and_desi['PMRA'] - preds['sf'].get('pmra', 0)) if residual_mode else self.data.confirmed_sf_and_desi['PMRA'],
                    **self.plot_params['sf_in_desi']
                )
                ax[3].scatter(
                    self.data.confirmed_sf_and_desi[col_x],
                    (self.data.confirmed_sf_and_desi['PMDEC'] - preds['sf'].get('pmdec', 0)) if residual_mode else self.data.confirmed_sf_and_desi['PMDEC'],
                    **self.plot_params['sf_in_desi']
                )
                ax[4].scatter(
                    self.data.confirmed_sf_and_desi[col_x],
                    (self.data.confirmed_sf_and_desi['FEH'] - preds['sf'].get('feh', 0)) if residual_mode else self.data.confirmed_sf_and_desi['FEH'],
                    **self.plot_params['sf_in_desi']
                )
            
            if hasattr(self.data, 'cut_confirmed_sf_and_desi') and show_cut:
                ax[0].scatter(
                    self.data.cut_confirmed_sf_and_desi[col_x],
                    self.data.cut_confirmed_sf_and_desi[col_y0],
                    **self.plot_params['sf_in_desi_notsel']
                )
                ax[1].scatter(
                    self.data.cut_confirmed_sf_and_desi[col_x],
                    (self.data.cut_confirmed_sf_and_desi['VGSR'] - preds['sf_cut'].get('vgsr', 0)) if residual_mode else self.data.cut_confirmed_sf_and_desi['VGSR'],
                    **self.plot_params['sf_in_desi_notsel']
                )
                ax[2].scatter(
                    self.data.cut_confirmed_sf_and_desi[col_x],
                    (self.data.cut_confirmed_sf_and_desi['PMRA'] - preds['sf_cut'].get('pmra', 0)) if residual_mode else self.data.cut_confirmed_sf_and_desi['PMRA'],
                    **self.plot_params['sf_in_desi_notsel']
                )
                ax[3].scatter(
                    self.data.cut_confirmed_sf_and_desi[col_x],
                    (self.data.cut_confirmed_sf_and_desi['PMDEC'] - preds['sf_cut'].get('pmdec', 0)) if residual_mode else self.data.cut_confirmed_sf_and_desi['PMDEC'],
                    **self.plot_params['sf_in_desi_notsel']
                )
                ax[4].scatter(
                    self.data.cut_confirmed_sf_and_desi[col_x],
                    (self.data.cut_confirmed_sf_and_desi['FEH'] - preds['sf_cut'].get('feh', 0)) if residual_mode else self.data.cut_confirmed_sf_and_desi['FEH'],
                    **self.plot_params['sf_in_desi_notsel']
                )
                
            if stream_frame and galstream and hasattr(self.data, 'SoI_galstream') and self.data.SoI_galstream is not None:
                ax[0].plot(
                    self.data.SoI_galstream.gal_phi1,
                    self.data.SoI_galstream.gal_phi2,
                    **self.plot_params['galstream_track']
                )
            elif not stream_frame and galstream and hasattr(self.data, 'SoI_galstream') and self.data.SoI_galstream is not None:
                ax[0].plot(
                    self.data.SoI_galstream.track.ra,
                    self.data.SoI_galstream.track.dec,
                    **self.plot_params['galstream_track']
                )
                
        if show_sf_only:
            ax[0].scatter(
                self.data.confirmed_sf_not_desi[col_x_],
                self.data.confirmed_sf_not_desi[col_y0_],
                **self.plot_params['sf_not_desi']
            )
            ax[1].scatter(
                self.data.confirmed_sf_not_desi[col_x_],
                (self.data.confirmed_sf_not_desi['VGSR'] - preds['sf_only'].get('vgsr', 0)) if residual_mode else self.data.confirmed_sf_not_desi['VGSR'],
                **self.plot_params['sf_not_desi']
            )
            ax[2].scatter(
                self.data.confirmed_sf_not_desi[col_x_],
                (self.data.confirmed_sf_not_desi['pmRA'] - preds['sf_only'].get('pmRA', 0)) if residual_mode else self.data.confirmed_sf_not_desi['pmRA'],
                **self.plot_params['sf_not_desi']
            )
            ax[3].scatter(
                self.data.confirmed_sf_not_desi[col_x_],
                (self.data.confirmed_sf_not_desi['pmDE'] - preds['sf_only'].get('pmDE', 0)) if residual_mode else self.data.confirmed_sf_not_desi['pmDE'],
                **self.plot_params['sf_not_desi']
            )
            # Note: FEH not available in sf_not_desi, skip this subplot
            
        if background:
            ax[0].scatter(
                self.data.desi_data[col_x],
                self.data.desi_data[col_y0],
                **self.plot_params['background']
            )
            ax[1].scatter(
                self.data.desi_data[col_x],
                (self.data.desi_data['VGSR'] - preds['desi'].get('vgsr', 0)) if residual_mode else self.data.desi_data['VGSR'],
                **self.plot_params['background']
            )
            ax[2].scatter(
                self.data.desi_data[col_x],
                (self.data.desi_data['PMRA'] - preds['desi'].get('pmra', 0)) if residual_mode else self.data.desi_data['PMRA'],
                **self.plot_params['background']
            )
            ax[3].scatter(
                self.data.desi_data[col_x],
                (self.data.desi_data['PMDEC'] - preds['desi'].get('pmdec', 0)) if residual_mode else self.data.desi_data['PMDEC'],
                **self.plot_params['background']
            )
            ax[4].scatter(
                self.data.desi_data[col_x],
                (self.data.desi_data['FEH'] - preds['desi'].get('feh', 0)) if residual_mode else self.data.desi_data['FEH'],
                **self.plot_params['background']
            )
        
        # Plot membership probability stars if requested
        if show_membership_prob and stream_prob is not None:
            import matplotlib.cm as cm
            import matplotlib.colors as colors
            
            # Validate stream_prob length matches DESI data
            if len(stream_prob) != len(self.data.desi_data):
                raise ValueError(f"stream_prob length ({len(stream_prob)}) must match DESI data length ({len(self.data.desi_data)})")
            
            # Get high probability stars
            high_prob_mask = stream_prob.values >= min_prob
            high_prob_indices = np.where(high_prob_mask)[0]
            from astropy.table import Table, hstack

            # Convert to Astropy Table if needed
            if not isinstance(self.data.desi_data, Table):
                desi_tab = Table.from_pandas(self.data.desi_data)
            else:
                desi_tab = self.data.desi_data

            # Select only high-probability rows
            subset = desi_tab[high_prob_indices]
            prob_col = Table([stream_prob.values[high_prob_indices]], names=['stream_prob'])

            # Combine correctly
            self.data.members = hstack([subset, prob_col])
                                    
            if len(high_prob_indices) > 0:
                # Unified colormap/norm for membership probability
                norm = colors.PowerNorm(gamma=5, vmin=min_prob, vmax=1.0)
                cmap = cm.get_cmap('magma_r')
                
                # Plot high probability DESI stars as circles with viridis colormap
                base_member = self.plot_params.get('membership_scatter', {})
                scatter_params = {**base_member,
                                  'c': stream_prob.values[high_prob_indices],
                                  'cmap': cmap,
                                  'norm': norm,
                                  'label': f'High Prob Stars (≥{min_prob:.1f})'}
                
                # Prepare y values (observed or residual) for errorbars and scatter
                x_hp = self.data.desi_data[col_x].iloc[high_prob_indices]
                y1_hp = self.data.desi_data['VGSR'].iloc[high_prob_indices]
                y2_hp = self.data.desi_data['PMRA'].iloc[high_prob_indices]
                y3_hp = self.data.desi_data['PMDEC'].iloc[high_prob_indices]
                y4_hp = self.data.desi_data['FEH'].iloc[high_prob_indices]
                if residual_mode:
                    y1_hp = y1_hp - preds['desi'].get('vgsr', np.zeros(len(self.data.desi_data)))[high_prob_indices]
                    y2_hp = y2_hp - preds['desi'].get('pmra', np.zeros(len(self.data.desi_data)))[high_prob_indices]
                    y3_hp = y3_hp - preds['desi'].get('pmdec', np.zeros(len(self.data.desi_data)))[high_prob_indices]
                    y4_hp = y4_hp - preds['desi'].get('feh', np.zeros(len(self.data.desi_data)))[high_prob_indices]

                # Error bars for high-probability members (kinematics/abundance panels)
                # Ensure error bars are behind the colored member markers
                err_style = {'fmt': 'none'}
                err_style.update(self.plot_params.get('member_errorbar', {}))

                # Plot on each subplot
                ax[0].scatter(x_hp, self.data.desi_data[col_y0].iloc[high_prob_indices], **scatter_params)
                # Add error bars on panels 1-4 using respective measurement errors
                ax[1].errorbar(x_hp, y1_hp, yerr=self.data.desi_data['VRAD_ERR'].iloc[high_prob_indices], **err_style)
                ax[2].errorbar(x_hp, y2_hp, yerr=self.data.desi_data['PMRA_ERROR'].iloc[high_prob_indices], **err_style)
                ax[3].errorbar(x_hp, y3_hp, yerr=self.data.desi_data['PMDEC_ERROR'].iloc[high_prob_indices], **err_style)
                ax[4].errorbar(x_hp, y4_hp, yerr=self.data.desi_data['FEH_ERR'].iloc[high_prob_indices], **err_style)
                # Then scatter the colored member markers
                ax[1].scatter(x_hp, y1_hp, **{k: v for k, v in scatter_params.items() if k != 'label'})
                ax[2].scatter(x_hp, y2_hp, **{k: v for k, v in scatter_params.items() if k != 'label'})
                ax[3].scatter(x_hp, y3_hp, **{k: v for k, v in scatter_params.items() if k != 'label'})
                ax[4].scatter(x_hp, y4_hp, **{k: v for k, v in scatter_params.items() if k != 'label'})

                # Note: error bars for members are already plotted above via err_style
                
                # Add colorbar to the far right of all plots
                # Add colorbar to the overall figure instead of individual subplot
                # If ax is an array of Axes (like from plt.subplots)
                sm = cm.ScalarMappable(norm=norm, cmap=cmap)
                sm.set_array([])

                # Ensure ax is a flat list of axes
                if isinstance(ax, np.ndarray):
                    ax = ax.ravel()

                # Position colorbar next to all subplots
                cbp = self.plot_params.get('membership_colorbar', {})
                cbar = fig.colorbar(sm, ax=ax,
                                    pad=cbp.get('pad', 0.02),
                                    aspect=cbp.get('aspect', 50),
                                    shrink=cbp.get('shrink', 1.0),
                                    location=cbp.get('location', 'right'))
                cbar.set_label(cbp.get('label', 'Membership Probability'), rotation=270, labelpad=cbp.get('labelpad', 15), fontsize=16)
                # Align cbar range and ticks to current min_prob threshold
                try:
                    vmin = float(min_prob)
                except Exception:
                    vmin = 0.5
                ticks = [vmin]
                for t in (0.9, 0.95, 1.0):
                    if t > vmin:
                        ticks.append(t)
                cbar.set_ticks(ticks)
                cbar.set_ticklabels([f"{int(t*100)}%" for t in ticks])
                
                # y-limits for residual mode will be set in a general block below (based on members)
            
            # Modify StreamFinder star styling when membership prob is shown
            if showStream:
                # Calculate membership probabilities for StreamFinder stars if possible
                sf_in_desi_indices = self.data.confirmed_sf_and_desi.index
                desi_indices = self.data.desi_data.index
                
                # Find which DESI indices correspond to SF stars
                sf_desi_mask = desi_indices.isin(sf_in_desi_indices)
                sf_prob_values = stream_prob[sf_desi_mask]
                
                # Determine colors for SF stars based on membership probability
                sf_high_prob_mask = sf_prob_values >= min_prob
                
                # Override existing StreamFinder star plots with new styling
                base_high = self.plot_params.get('sf_high_prob_diamond', {})
                sf_diamond_params_high = {**base_high,
                                          'c': sf_prob_values[sf_high_prob_mask],
                                          'cmap': cmap,
                                          'norm': norm}
                
                sf_diamond_params_low = self.plot_params.get('sf_low_prob_diamond', {}).copy()
                
                # Get high and low probability SF star indices
                sf_indices_high = sf_in_desi_indices[sf_high_prob_mask]
                sf_indices_low = sf_in_desi_indices[~sf_high_prob_mask]
                
                # Plot high probability SF stars with colormap
                if len(sf_indices_high) > 0:
                    sf_data_high = self.data.confirmed_sf_and_desi.loc[sf_indices_high]
                    
                    ax[0].scatter(
                        sf_data_high[col_x],
                        sf_data_high[col_y0],
                        **sf_diamond_params_high
                    )
                    ax[1].scatter(
                        sf_data_high[col_x],
                        (sf_data_high['VGSR'] - preds['sf'].get('vgsr', 0)[:len(sf_data_high)]) if residual_mode else sf_data_high['VGSR'],
                        **{k: v for k, v in sf_diamond_params_high.items() if k != 'label'}
                    )
                    ax[2].scatter(
                        sf_data_high[col_x],
                        (sf_data_high['PMRA'] - preds['sf'].get('pmra', 0)[:len(sf_data_high)]) if residual_mode else sf_data_high['PMRA'],
                        **{k: v for k, v in sf_diamond_params_high.items() if k != 'label'}
                    )
                    ax[3].scatter(
                        sf_data_high[col_x],
                        (sf_data_high['PMDEC'] - preds['sf'].get('pmdec', 0)[:len(sf_data_high)]) if residual_mode else sf_data_high['PMDEC'],
                        **{k: v for k, v in sf_diamond_params_high.items() if k != 'label'}
                    )
                    ax[4].scatter(
                        sf_data_high[col_x],
                        (sf_data_high['FEH'] - preds['sf'].get('feh', 0)[:len(sf_data_high)]) if residual_mode else sf_data_high['FEH'],
                        **{k: v for k, v in sf_diamond_params_high.items() if k != 'label'}
                    )
                
                # Plot low probability SF stars as black diamonds
                if len(sf_indices_low) > 0:
                    sf_data_low = self.data.confirmed_sf_and_desi.loc[sf_indices_low]
                    
                    ax[0].scatter(
                        sf_data_low[col_x],
                        sf_data_low[col_y0],
                        **sf_diamond_params_low,
                        label=f'SF Stars (<{min_prob:.1f})'
                    )
                    ax[1].scatter(
                        sf_data_low[col_x],
                        (sf_data_low['VGSR'] - preds['sf'].get('vgsr', 0)[-len(sf_data_low):]) if residual_mode else sf_data_low['VGSR'],
                        **sf_diamond_params_low
                    )
                    ax[2].scatter(
                        sf_data_low[col_x],
                        (sf_data_low['PMRA'] - preds['sf'].get('pmra', 0)[-len(sf_data_low):]) if residual_mode else sf_data_low['PMRA'],
                        **sf_diamond_params_low
                    )
                    ax[3].scatter(
                        sf_data_low[col_x],
                        (sf_data_low['PMDEC'] - preds['sf'].get('pmdec', 0)[-len(sf_data_low):]) if residual_mode else sf_data_low['PMDEC'],
                        **sf_diamond_params_low
                    )
                    ax[4].scatter(
                        sf_data_low[col_x],
                        (sf_data_low['FEH'] - preds['sf'].get('feh', 0)[-len(sf_data_low):]) if residual_mode else sf_data_low['FEH'],
                        **sf_diamond_params_low
                    )

        # Set consistent x-axis limits for all panels based on the data being plotted
        phi1_values_for_limits = []
        
        # Collect phi1 values from data sources that are actually being plotted
        if background and len(self.data.desi_data) > 0:
            phi1_values_for_limits.extend(self.data.desi_data[col_x].values)
        if showStream and hasattr(self.data, 'confirmed_sf_and_desi') and len(self.data.confirmed_sf_and_desi) > 0:
            phi1_values_for_limits.extend(self.data.confirmed_sf_and_desi[col_x].values)
        if show_sf_only and hasattr(self.data, 'confirmed_sf_not_desi') and len(self.data.confirmed_sf_not_desi) > 0:
            phi1_values_for_limits.extend(self.data.confirmed_sf_not_desi[col_x_].values)
        if hasattr(self.data, 'cut_confirmed_sf_and_desi') and show_cut and len(self.data.cut_confirmed_sf_and_desi) > 0:
            phi1_values_for_limits.extend(self.data.cut_confirmed_sf_and_desi[col_x].values)
        
        # Prefer member stars for x-limits if membership probabilities are available
        if show_membership_prob and stream_prob is not None:
            high_prob_mask = stream_prob >= min_prob
            high_prob_indices = np.where(high_prob_mask)[0]
            if len(high_prob_indices) > 0:
                # Get phi1 range from high probability member stars
                high_prob_phi1 = self.data.desi_data[col_x].iloc[high_prob_indices]
                phi1_min_members = high_prob_phi1.min()
                phi1_max_members = high_prob_phi1.max()
                
                # Extend range to include all spline points if splines are being shown
                if (show_initial_splines or show_optimized_splines or show_mcmc_splines) and hasattr(self, 'mcmeta') and self.mcmeta is not None and hasattr(self.mcmeta, 'phi1_spline_points'):
                    spline_min = np.min(self.mcmeta.phi1_spline_points)
                    spline_max = np.max(self.mcmeta.phi1_spline_points)
                    phi1_min_members = min(phi1_min_members, spline_min)
                    phi1_max_members = max(phi1_max_members, spline_max)
                
                # Add some padding
                phi1_range_members = phi1_max_members - phi1_min_members
                phi1_padding = 0.05 * phi1_range_members if phi1_range_members > 0 else 1.0
                x_limits = (phi1_min_members - phi1_padding, phi1_max_members + phi1_padding)
                
                # Apply consistent x-limits to all panels
                for panel_ax in ax:
                    panel_ax.set_xlim(x_limits)
        elif phi1_values_for_limits:
            # Use all plotted data for x-limits if no membership probabilities
            phi1_min_data = np.min(phi1_values_for_limits)
            phi1_max_data = np.max(phi1_values_for_limits)
            
            # Extend range to include all spline points if splines are being shown
            if (show_initial_splines or show_optimized_splines or show_mcmc_splines) and hasattr(self, 'mcmeta') and self.mcmeta is not None and hasattr(self.mcmeta, 'phi1_spline_points'):
                spline_min = np.min(self.mcmeta.phi1_spline_points)
                spline_max = np.max(self.mcmeta.phi1_spline_points)
                phi1_min_data = min(phi1_min_data, spline_min)
                phi1_max_data = max(phi1_max_data, spline_max)
            
            phi1_range_data = phi1_max_data - phi1_min_data
            phi1_padding = 0.02 * phi1_range_data if phi1_range_data > 0 else 1.0
            x_limits = (phi1_min_data - phi1_padding, phi1_max_data + phi1_padding)
            
            # Apply consistent x-limits to all panels
            for panel_ax in ax:
                panel_ax.set_xlim(x_limits)

            # In residuals mode, prefer y-limits based on high-probability member stars
            # Use member stars for limits whenever probabilities are available (independent of whether they are drawn)
            if residual_mode and stream_prob is not None:
                high_prob_mask = stream_prob >= min_prob
                high_prob_indices = np.where(high_prob_mask)[0]
                if len(high_prob_indices) > 0:
                    pad_v = self.plot_params.get('limits', {}).get('residual_pad_vgsr', 20)
                    pad_pm = self.plot_params.get('limits', {}).get('residual_pad_pm', 2)
                    pad_feh = self.plot_params.get('limits', {}).get('residual_pad_feh', 0.2)
                    # Build residual arrays from high-probability members
                    y1_hp = (self.data.desi_data['VGSR'].iloc[high_prob_indices]
                             - preds['desi'].get('vgsr', np.zeros(len(self.data.desi_data)))[high_prob_indices]).to_numpy()
                    y2_hp = (self.data.desi_data['PMRA'].iloc[high_prob_indices]
                             - preds['desi'].get('pmra', np.zeros(len(self.data.desi_data)))[high_prob_indices]).to_numpy()
                    y3_hp = (self.data.desi_data['PMDEC'].iloc[high_prob_indices]
                             - preds['desi'].get('pmdec', np.zeros(len(self.data.desi_data)))[high_prob_indices]).to_numpy()
                    # Set limits
                    ax[1].set_ylim(np.nanmin(y1_hp) - pad_v, np.nanmax(y1_hp) + pad_v)
                    ax[2].set_ylim(np.nanmin(y2_hp) - pad_pm, np.nanmax(y2_hp) + pad_pm)
                    ax[3].set_ylim(np.nanmin(y3_hp) - pad_pm, np.nanmax(y3_hp) + pad_pm)
                    if 'FEH' in self.data.desi_data.columns and isinstance(preds['desi'].get('feh', None), np.ndarray):
                        y4_hp = (self.data.desi_data['FEH'].iloc[high_prob_indices] - preds['desi']['feh'][high_prob_indices]).to_numpy()
                        ax[4].set_ylim(np.nanmin(y4_hp) - pad_feh, np.nanmax(y4_hp) + pad_feh)
                    else:
                        ax[4].set_ylim(-1, 1)

        # Set y-axis limits (non-residual mode) based on member stars OR faded blue MCMC regions (whichever extends further)
        if not residual_mode:
            pad_settings = self.plot_params.get('limits', {})
            pad_v = pad_settings.get('nonresidual_pad_vgsr', 50)
            pad_pm = pad_settings.get('nonresidual_pad_pm', 5)
            pad_feh = pad_settings.get('nonresidual_pad_feh', 0.5)
            feh_default = pad_settings.get('feh_ylim_default', (-4, -0.5))

            member_idx = None
            if show_membership_prob and stream_prob is not None:
                high_prob_mask = stream_prob >= min_prob
                member_idx = np.where(high_prob_mask)[0]
            elif showStream and hasattr(self.data, 'confirmed_sf_and_desi'):
                member_idx = self.data.confirmed_sf_and_desi.index

            # Helper to compute combined min/max including MCMC fill regions
            def combine_range(member_values, mcmc_curve, sigma):
                vals = []
                if member_values is not None and len(member_values) > 0:
                    vals.append(member_values.values if hasattr(member_values, 'values') else np.array(member_values))
                if (mcmc_curve is not None) and (sigma is not None):
                    # Include +/-2 sigma envelope (faded blue outer region)
                    vals.append(mcmc_curve - 2 * sigma)
                    vals.append(mcmc_curve + 2 * sigma)
                if len(vals) == 0:
                    return None
                allv = np.concatenate([np.array(v, dtype=float).ravel() for v in vals])
                # Remove NaNs
                allv = allv[~np.isnan(allv)]
                if len(allv) == 0:
                    return None
                return float(np.nanmin(allv)), float(np.nanmax(allv))

            # Build member value series
            member_vgsr = member_pmra = member_pmdec = member_feh = None
            if member_idx is not None and len(member_idx) > 0:
                if show_membership_prob and stream_prob is not None:
                    # member_idx are DESI row indices
                    member_vgsr = self.data.desi_data['VGSR'].iloc[member_idx]
                    member_pmra = self.data.desi_data['PMRA'].iloc[member_idx]
                    member_pmdec = self.data.desi_data['PMDEC'].iloc[member_idx]
                    if 'FEH' in self.data.desi_data.columns:
                        member_feh = self.data.desi_data['FEH'].iloc[member_idx]
                else:
                    # StreamFinder confirmed members
                    member_vgsr = self.data.confirmed_sf_and_desi['VGSR']
                    member_pmra = self.data.confirmed_sf_and_desi['PMRA']
                    member_pmdec = self.data.confirmed_sf_and_desi['PMDEC']
                    if 'FEH' in self.data.desi_data.columns:
                        member_feh = self.data.desi_data['FEH'] if 'FEH' in self.data.desi_data else None
                    # Optionally include cut stars to extend range
                    if hasattr(self.data, 'cut_confirmed_sf_and_desi') and show_cut and len(self.data.cut_confirmed_sf_and_desi) > 0:
                        member_vgsr = pd.concat([member_vgsr, self.data.cut_confirmed_sf_and_desi['VGSR']])
                        member_pmra = pd.concat([member_pmra, self.data.cut_confirmed_sf_and_desi['PMRA']])
                        member_pmdec = pd.concat([member_pmdec, self.data.cut_confirmed_sf_and_desi['PMDEC']])

            # Determine combined ranges
            vgsr_range = combine_range(member_vgsr, vgsr_mcmc, sigma_vgsr)
            pmra_range = combine_range(member_pmra, pmra_mcmc, sigma_pmra)
            pmdec_range = combine_range(member_pmdec, pmdec_mcmc, sigma_pmdec)
            feh_range = combine_range(member_feh, feh_mcmc, sigma_feh) if member_feh is not None else None

            # Apply ranges with padding; fall back gracefully if None
            if vgsr_range is not None:
                ax[1].set_ylim(vgsr_range[0] - pad_v, vgsr_range[1] + pad_v)
            if pmra_range is not None:
                ax[2].set_ylim(pmra_range[0] - pad_pm, pmra_range[1] + pad_pm)
            if pmdec_range is not None:
                ax[3].set_ylim(pmdec_range[0] - pad_pm, pmdec_range[1] + pad_pm)
            if feh_range is not None:
                ax[4].set_ylim(feh_range[0] - pad_feh, feh_range[1] + pad_feh)
            else:
                ax[4].set_ylim(*feh_default)
        
        # Plot splines if requested and available
        if (show_initial_splines or show_optimized_splines or show_mcmc_splines) and stream_frame and self.mcmeta is not None:
            # Create phi1 range for spline plotting based on data range, not current axis limits
            phi1_values = []
            
            # Collect phi1 values from all data sources that will be plotted
            if background and len(self.data.desi_data) > 0:
                phi1_values.extend(self.data.desi_data[col_x].values)
            if showStream and hasattr(self.data, 'confirmed_sf_and_desi') and len(self.data.confirmed_sf_and_desi) > 0:
                phi1_values.extend(self.data.confirmed_sf_and_desi[col_x].values)
            if show_sf_only and hasattr(self.data, 'confirmed_sf_not_desi') and len(self.data.confirmed_sf_not_desi) > 0:
                phi1_values.extend(self.data.confirmed_sf_not_desi[col_x_].values)
            if hasattr(self.data, 'cut_confirmed_sf_and_desi') and show_cut and len(self.data.cut_confirmed_sf_and_desi) > 0:
                phi1_values.extend(self.data.cut_confirmed_sf_and_desi[col_x].values)
            
            # Also include membership probability data if it's being shown
            if show_membership_prob and stream_prob is not None:
                # Always include the full DESI dataset range when showing membership probabilities
                phi1_values.extend(self.data.desi_data[col_x].values)
            
            # If we have phi1 values, use them to determine the range, otherwise fall back to axis limits
            if phi1_values:
                phi1_min = np.min(phi1_values)
                phi1_max = np.max(phi1_values)
                # Add some padding to ensure we cover the full range
                phi1_range = phi1_max - phi1_min
                phi1_min -= 0.1 * phi1_range
                phi1_max += 0.1 * phi1_range
            else:
                # Fallback to current axis limits if no data available
                phi1_min = ax[1].get_xlim()[0]
                phi1_max = ax[1].get_xlim()[1]
            
            # Ensure the range extends beyond spline points to cover the full data/plotting range
            if hasattr(self.mcmeta, 'phi1_spline_points'):
                spline_min = np.min(self.mcmeta.phi1_spline_points)
                spline_max = np.max(self.mcmeta.phi1_spline_points)
                # Extend range to at least the spline points, plus some buffer for extrapolation
                spline_range = spline_max - spline_min
                buffer = 0.2 * spline_range  # 20% buffer beyond spline range
                phi1_min = min(phi1_min, spline_min - buffer)
                phi1_max = max(phi1_max, spline_max + buffer)
            
            phi1_spline_plot = np.linspace(phi1_min, phi1_max, 200)  # Increased resolution
            
        # If optimized splines are requested but optimized_params aren't attached,
        # try to source them from common places used in the notebook (silently).
            if show_optimized_splines and (not hasattr(self.mcmeta, 'optimized_params') or self.mcmeta.optimized_params is None):
                try:
                    import inspect
                    sourced = False
                    # 1) Look for a variable named `optimized_for_plotting` in caller frames
                    frame = inspect.currentframe()
                    caller_frame = frame.f_back if frame is not None else None
                    while caller_frame:
                        if 'optimized_for_plotting' in caller_frame.f_globals:
                            ofp = caller_frame.f_globals['optimized_for_plotting']
                            if ofp is not None:
                                self.mcmeta.optimized_params = ofp
                                sourced = True
                                break
                        caller_frame = caller_frame.f_back
                    # 2) Fallback to MCMeta.sp_output (from scipy_optimize)
                    if not sourced and hasattr(self.mcmeta, 'sp_output') and self.mcmeta.sp_output is not None:
                        self.mcmeta.optimized_params = self.mcmeta.sp_output
                        sourced = True
                except Exception as e:
                    pass
            
            # Plot initial guess splines in black
            if show_initial_splines:
                if hasattr(self.mcmeta, 'phi1_spline_points'):
                    try:
                        # VGSR spline
                        vgsr_initial = stream_funcs.apply_spline(
                            phi1_spline_plot, self.mcmeta.phi1_spline_points, 
                            self.mcmeta.initial_params['vgsr_spline_points'], k=2
                        )
                        if 'vgsr_spline_points' in self.mcmeta.initial_params and 'vgsr' in locals():
                            pass
                        # Residual vs MCMC if applicable
                        if 'vgsr_knots' in locals() and 'residual_mode' in locals() and residual_mode:
                            vgsr_mcmc_eval = stream_funcs.apply_spline(
                                phi1_spline_plot, self.mcmeta.phi1_spline_points, vgsr_knots, k=2
                            )
                            vgsr_initial_plot = vgsr_initial - vgsr_mcmc_eval
                        else:
                            vgsr_initial_plot = vgsr_initial
                        ax[1].plot(
                            phi1_spline_plot,
                            vgsr_initial_plot,
                            **self.plot_params.get('initial_spline_line', {})
                        )
                        
                        # Add circle markers at spline points
                        if 'vgsr_knots' in locals() and residual_mode:
                            vgsr_mcmc_knots = stream_funcs.apply_spline(
                                self.mcmeta.phi1_spline_points, self.mcmeta.phi1_spline_points, vgsr_knots, k=2
                            )
                            vgsr_init_knots_scatter = self.mcmeta.initial_params['vgsr_spline_points'] - vgsr_mcmc_knots
                        else:
                            vgsr_init_knots_scatter = self.mcmeta.initial_params['vgsr_spline_points']
                        ax[1].scatter(
                            self.mcmeta.phi1_spline_points,
                            vgsr_init_knots_scatter,
                            **self.plot_params.get('spline_knot_initial', {})
                        )
                        
                        # PMRA spline
                        pmra_initial = stream_funcs.apply_spline(
                            phi1_spline_plot, self.mcmeta.phi1_spline_points, 
                            self.mcmeta.initial_params['pmra_spline_points'], k=2
                        )
                        if 'pmra_knots' in locals() and 'residual_mode' in locals() and residual_mode:
                            pmra_mcmc_eval = stream_funcs.apply_spline(
                                phi1_spline_plot, self.mcmeta.phi1_spline_points, pmra_knots, k=2
                            )
                            pmra_initial_plot = pmra_initial - pmra_mcmc_eval
                        else:
                            pmra_initial_plot = pmra_initial
                        ax[2].plot(
                            phi1_spline_plot,
                            pmra_initial_plot,
                            **self.plot_params.get('initial_spline_line', {})
                        )
                        
                        # Add circle markers at spline points
                        if 'pmra_knots' in locals() and residual_mode:
                            pmra_mcmc_knots = stream_funcs.apply_spline(
                                self.mcmeta.phi1_spline_points, self.mcmeta.phi1_spline_points, pmra_knots, k=2
                            )
                            pmra_init_knots_scatter = self.mcmeta.initial_params['pmra_spline_points'] - pmra_mcmc_knots
                        else:
                            pmra_init_knots_scatter = self.mcmeta.initial_params['pmra_spline_points']
                        ax[2].scatter(
                            self.mcmeta.phi1_spline_points,
                            pmra_init_knots_scatter,
                            **self.plot_params.get('spline_knot_initial', {})
                        )
                        
                        # PMDEC spline
                        pmdec_initial = stream_funcs.apply_spline(
                            phi1_spline_plot, self.mcmeta.phi1_spline_points, 
                            self.mcmeta.initial_params['pmdec_spline_points'], k=2
                        )
                        if 'pmdec_knots' in locals() and 'residual_mode' in locals() and residual_mode:
                            pmdec_mcmc_eval = stream_funcs.apply_spline(
                                phi1_spline_plot, self.mcmeta.phi1_spline_points, pmdec_knots, k=2
                            )
                            pmdec_initial_plot = pmdec_initial - pmdec_mcmc_eval
                        else:
                            pmdec_initial_plot = pmdec_initial
                        ax[3].plot(
                            phi1_spline_plot,
                            pmdec_initial_plot,
                            **self.plot_params.get('initial_spline_line', {})
                        )
                        
                        # Add circle markers at spline points
                        if 'pmdec_knots' in locals() and residual_mode:
                            pmdec_mcmc_knots = stream_funcs.apply_spline(
                                self.mcmeta.phi1_spline_points, self.mcmeta.phi1_spline_points, pmdec_knots, k=2
                            )
                            pmdec_init_knots_scatter = self.mcmeta.initial_params['pmdec_spline_points'] - pmdec_mcmc_knots
                        else:
                            pmdec_init_knots_scatter = self.mcmeta.initial_params['pmdec_spline_points']
                        ax[3].scatter(
                            self.mcmeta.phi1_spline_points,
                            pmdec_init_knots_scatter,
                            **self.plot_params.get('spline_knot_initial', {})
                        )
                        
                        # FEH constant line
                        feh_initial = np.full_like(phi1_spline_plot, self.mcmeta.initial_params['feh1'])
                        if 'feh_const' in locals() and 'residual_mode' in locals() and residual_mode:
                            feh_initial_plot = feh_initial - feh_const
                        else:
                            feh_initial_plot = feh_initial
                        ax[4].plot(
                            phi1_spline_plot,
                            feh_initial_plot,
                            **self.plot_params.get('initial_feh_line', {})
                        )
                        
                        # No FEH spline markers on metallicity panel (line only)
                    except Exception as e:
                        print(f"Warning: Could not plot initial splines: {e}")
            
            # Plot optimized splines in red
            if show_optimized_splines and hasattr(self.mcmeta, 'optimized_params'):
                if hasattr(self.mcmeta, 'phi1_spline_points'):
                    try:
                        # VGSR spline
                        vgsr_optimized = stream_funcs.apply_spline(
                            phi1_spline_plot, self.mcmeta.phi1_spline_points, 
                            self.mcmeta.optimized_params['vgsr_spline_points'], k=2
                        )
                        if 'vgsr_knots' in locals() and 'residual_mode' in locals() and residual_mode:
                            vgsr_mcmc_eval = stream_funcs.apply_spline(
                                phi1_spline_plot, self.mcmeta.phi1_spline_points, vgsr_knots, k=2
                            )
                            vgsr_optimized_plot = vgsr_optimized - vgsr_mcmc_eval
                        else:
                            vgsr_optimized_plot = vgsr_optimized
                        ax[1].plot(
                            phi1_spline_plot,
                            vgsr_optimized_plot,
                            **self.plot_params.get('optimized_spline_line', {})
                        )
                        
                        # Add circle markers at spline points
                        if 'vgsr_knots' in locals() and 'residual_mode' in locals() and residual_mode:
                            vgsr_mcmc_knots = stream_funcs.apply_spline(
                                self.mcmeta.phi1_spline_points, self.mcmeta.phi1_spline_points, vgsr_knots, k=2
                            )
                            vgsr_knots_scatter = self.mcmeta.optimized_params['vgsr_spline_points'] - vgsr_mcmc_knots
                        else:
                            vgsr_knots_scatter = self.mcmeta.optimized_params['vgsr_spline_points']
                        ax[1].scatter(
                            self.mcmeta.phi1_spline_points,
                            vgsr_knots_scatter,
                            **self.plot_params.get('spline_knot_optimized', {})
                        )
                        
                        # PMRA spline
                        pmra_optimized = stream_funcs.apply_spline(
                            phi1_spline_plot, self.mcmeta.phi1_spline_points, 
                            self.mcmeta.optimized_params['pmra_spline_points'], k=2
                        )
                        if 'pmra_knots' in locals() and 'residual_mode' in locals() and residual_mode:
                            pmra_mcmc_eval = stream_funcs.apply_spline(
                                phi1_spline_plot, self.mcmeta.phi1_spline_points, pmra_knots, k=2
                            )
                            pmra_optimized_plot = pmra_optimized - pmra_mcmc_eval
                        else:
                            pmra_optimized_plot = pmra_optimized
                        ax[2].plot(
                            phi1_spline_plot,
                            pmra_optimized_plot,
                            **self.plot_params.get('optimized_spline_line', {})
                        )
                        
                        # Add circle markers at spline points
                        if 'pmra_knots' in locals() and 'residual_mode' in locals() and residual_mode:
                            pmra_mcmc_knots = stream_funcs.apply_spline(
                                self.mcmeta.phi1_spline_points, self.mcmeta.phi1_spline_points, pmra_knots, k=2
                            )
                            pmra_knots_scatter = self.mcmeta.optimized_params['pmra_spline_points'] - pmra_mcmc_knots
                        else:
                            pmra_knots_scatter = self.mcmeta.optimized_params['pmra_spline_points']
                        ax[2].scatter(
                            self.mcmeta.phi1_spline_points,
                            pmra_knots_scatter,
                            **self.plot_params.get('spline_knot_optimized', {})
                        )
                        
                        # PMDEC spline
                        pmdec_optimized = stream_funcs.apply_spline(
                            phi1_spline_plot, self.mcmeta.phi1_spline_points, 
                            self.mcmeta.optimized_params['pmdec_spline_points'], k=2
                        )
                        if 'pmdec_knots' in locals() and 'residual_mode' in locals() and residual_mode:
                            pmdec_mcmc_eval = stream_funcs.apply_spline(
                                phi1_spline_plot, self.mcmeta.phi1_spline_points, pmdec_knots, k=2
                            )
                            pmdec_optimized_plot = pmdec_optimized - pmdec_mcmc_eval
                        else:
                            pmdec_optimized_plot = pmdec_optimized
                        ax[3].plot(
                            phi1_spline_plot,
                            pmdec_optimized_plot,
                            **self.plot_params.get('optimized_spline_line', {})
                        )
                        
                        # Add circle markers at spline points
                        if 'pmdec_knots' in locals() and 'residual_mode' in locals() and residual_mode:
                            pmdec_mcmc_knots = stream_funcs.apply_spline(
                                self.mcmeta.phi1_spline_points, self.mcmeta.phi1_spline_points, pmdec_knots, k=2
                            )
                            pmdec_knots_scatter = self.mcmeta.optimized_params['pmdec_spline_points'] - pmdec_mcmc_knots
                        else:
                            pmdec_knots_scatter = self.mcmeta.optimized_params['pmdec_spline_points']
                        ax[3].scatter(
                            self.mcmeta.phi1_spline_points,
                            pmdec_knots_scatter,
                            **self.plot_params.get('spline_knot_optimized', {})
                        )
                        
                        # FEH constant line
                        feh_optimized = np.full_like(phi1_spline_plot, self.mcmeta.optimized_params['feh1'])
                        if 'feh_const' in locals() and 'residual_mode' in locals() and residual_mode:
                            feh_optimized_plot = feh_optimized - feh_const
                        else:
                            feh_optimized_plot = feh_optimized
                        ax[4].plot(
                            phi1_spline_plot,
                            feh_optimized_plot,
                            **self.plot_params.get('optimized_feh_line', {})
                        )
                        
                        # No FEH spline markers on metallicity panel (line only)
                    except Exception as e:
                        print(f"Warning: Could not plot optimized splines: {e}")
            
            # Plot MCMC splines in blue (requires external meds dictionary with MCMC results)
            if show_mcmc_splines and hasattr(self.mcmeta, 'phi1_spline_points'):
                        
                            
                        # Extract spline points from meds dictionary
                        no_of_spline_points = len(self.mcmeta.phi1_spline_points)
                        
                        # Build the spline arrays from the flattened meds
                        vgsr_mcmc_points = []
                        pmra_mcmc_points = []  
                        pmdec_mcmc_points = []
                        
                        # Extract vgsr spline points
                        for i in range(1, no_of_spline_points + 1):
                            vgsr_mcmc_points.append(meds[f'vgsr{i}'])
                            
                        # Extract pmra spline points  
                        for i in range(1, no_of_spline_points + 1):
                            pmra_mcmc_points.append(meds[f'pmra{i}'])
                            
                        # Extract pmdec spline points
                        for i in range(1, no_of_spline_points + 1):
                            pmdec_mcmc_points.append(meds[f'pmdec{i}'])
                        
                        # Convert to arrays
                        vgsr_mcmc_points = np.array(vgsr_mcmc_points)
                        pmra_mcmc_points = np.array(pmra_mcmc_points)  
                        pmdec_mcmc_points = np.array(pmdec_mcmc_points)
                        
                        # VGSR spline
                        vgsr_mcmc = stream_funcs.apply_spline(
                            phi1_spline_plot, self.mcmeta.phi1_spline_points, 
                            vgsr_mcmc_points, k=2
                        )
                        if residual_mode:
                            ax[1].axhline(0, color='blue', linewidth=2, alpha=0.8, label='MCMC Spline (resid)')
                        else:
                            ax[1].plot(phi1_spline_plot, vgsr_mcmc, 'b-', linewidth=2, 
                                      label='MCMC Spline', alpha=0.8)
                        
                        # Plot 1 sigma and 2 sigma regions covering full spline length
                        expanded_labels = getattr(mcmc_object, 'expanded_param_labels', None)
                        if expanded_labels is None:
                            expanded_labels = getattr(mcmc_object, '_build_expanded_param_labels', lambda: [])()

                        meta_obj = getattr(self, 'meta', None) or getattr(self, 'mcmeta', None)
                        if meta_obj is None and mcmc_object is not None:
                            meta_obj = getattr(mcmc_object, 'meta', None) or getattr(mcmc_object, 'mcmeta', None)

                        lsigv_keys = [lbl for lbl in expanded_labels if lbl.startswith('lsigvgsr')]
                        if lsigv_keys:
                            lsigv_vals = [meds[key] for key in lsigv_keys]
                        else:
                            if meta_obj is None:
                                raise AttributeError('No MCMeta metadata available for lsigv initialization')
                            init_lsigv = meta_obj.initial_params['lsigvgsr']
                            if isinstance(init_lsigv, np.ndarray):
                                lsigv_vals = init_lsigv.tolist()
                            else:
                                lsigv_vals = [init_lsigv]
                        if len(lsigv_vals) == 1:
                            lsigv_meds = lsigv_vals[0]
                        else:
                            lsigv_meds = np.array(lsigv_vals, dtype=float)

                        if np.isscalar(lsigv_meds):
                            sigma_vgsr = np.full_like(phi1_spline_plot, 10**lsigv_meds, dtype=float)
                        else:
                            lsigv_meds = np.asarray(lsigv_meds, dtype=float)
                            if lsigv_meds.size == 1:
                                sigma_vgsr = np.full_like(
                                    phi1_spline_plot, 10**lsigv_meds.ravel()[0], dtype=float
                                )
                            else:
                                if meta_obj is None:
                                    raise AttributeError('No MCMeta metadata available for lsigv spline evaluation')

                                lsigv_phi1_points = getattr(meta_obj, 'lsigv_phi1_spline_points', None)
                                if lsigv_phi1_points is None:
                                    lsigv_phi1_points = getattr(meta_obj, 'phi1_spline_points')

                                lsigv_k = getattr(meta_obj, 'spline_k_lsigv', None)
                                if lsigv_k is None:
                                    lsigv_k = getattr(meta_obj, 'spline_k', 2)

                                sigma_log = stream_funcs.apply_spline(
                                    phi1_spline_plot,
                                    lsigv_phi1_points,
                                    lsigv_meds,
                                    k=lsigv_k
                                )
                                sigma_vgsr = 10**sigma_log

                        if residual_mode:
                            ax[1].fill_between(phi1_spline_plot, -sigma_vgsr, +sigma_vgsr, color='blue', alpha=0.1)
                            ax[1].fill_between(phi1_spline_plot, -2*sigma_vgsr, +2*sigma_vgsr, color='blue', alpha=0.05)
                        else:
                            ax[1].fill_between(phi1_spline_plot, vgsr_mcmc - sigma_vgsr, vgsr_mcmc + sigma_vgsr, color='blue', alpha=0.1)
                            ax[1].fill_between(phi1_spline_plot, vgsr_mcmc - 2*sigma_vgsr, vgsr_mcmc + 2*sigma_vgsr, color='blue', alpha=0.05)

                        vgsr_mcmc_ep = []
                        for i in range(1, no_of_spline_points + 1):
                            vgsr_mcmc_ep.append(ep['vgsr'+str(i)])

                        vgsr_mcmc_em = []
                        for i in range(1, no_of_spline_points + 1):
                            vgsr_mcmc_em.append(np.abs(em['vgsr'+str(i)]))
                        
                        vgsr_mcmc_errors = np.array([vgsr_mcmc_em, vgsr_mcmc_ep])
                        yvals = np.zeros_like(self.mcmeta.phi1_spline_points) if residual_mode else vgsr_mcmc_points
                        ax[1].errorbar(self.mcmeta.phi1_spline_points, yvals, 
                                        yerr=vgsr_mcmc_errors, fmt='o', color='blue', 
                                        markersize=8, zorder=10, alpha=0.8, markeredgecolor='white', 
                                        markeredgewidth=1, capsize=3, capthick=1.5, elinewidth=1.5)

                        # PMRA spline
                        pmra_mcmc = stream_funcs.apply_spline(
                            phi1_spline_plot, self.mcmeta.phi1_spline_points, 
                            pmra_mcmc_points, k=2
                        )
                        if residual_mode:
                            ax[2].axhline(0, color='blue', linewidth=2, alpha=0.8, label='MCMC Spline (resid)')
                        else:
                            ax[2].plot(phi1_spline_plot, pmra_mcmc, 'b-', linewidth=2, 
                                      label='MCMC Spline', alpha=0.8)
                        
                        # Plot 1 sigma and 2 sigma regions for PMRA
                        # Use fixed PM sigma if available; else fall back to MCMC meds; else try mcmc_object.meta
                        _lsigpmra_log = self.lsigpm_ if self.lsigpm_ is not None else meds.get('lsigpmra', None)
                        if _lsigpmra_log is None and mcmc_object is not None:
                            _lsigpmra_log = getattr(getattr(mcmc_object, 'meta', None), 'lsigpm_', None)
                        if _lsigpmra_log is None:
                            raise AttributeError("No lsigpmra in meds and no fixed lsigpm_ available for PMRA band.")
                        sigma_pmra = 10**_lsigpmra_log
                        if residual_mode:
                            ax[2].fill_between(phi1_spline_plot, -sigma_pmra, +sigma_pmra, color='blue', alpha=0.1)
                            ax[2].fill_between(phi1_spline_plot, -2*sigma_pmra, +2*sigma_pmra, color='blue', alpha=0.05)
                        else:
                            ax[2].fill_between(phi1_spline_plot, pmra_mcmc - sigma_pmra, pmra_mcmc + sigma_pmra, color='blue', alpha=0.1)
                            ax[2].fill_between(phi1_spline_plot, pmra_mcmc - 2*sigma_pmra, pmra_mcmc + 2*sigma_pmra, color='blue', alpha=0.05)

                        # Extract error bars for PMRA spline points
                        pmra_mcmc_ep = []
                        for i in range(1, no_of_spline_points + 1):
                            pmra_mcmc_ep.append(ep['pmra'+str(i)])

                        pmra_mcmc_em = []
                        for i in range(1, no_of_spline_points + 1):
                            pmra_mcmc_em.append(np.abs(em['pmra'+str(i)]))
                        pmra_mcmc_errors = np.array([pmra_mcmc_em, pmra_mcmc_ep])

                        yvals = np.zeros_like(self.mcmeta.phi1_spline_points) if residual_mode else pmra_mcmc_points
                        ax[2].errorbar(self.mcmeta.phi1_spline_points, yvals, 
                                        yerr=pmra_mcmc_errors, fmt='o', color='blue', 
                                        markersize=8, zorder=10, alpha=0.8, markeredgecolor='white', 
                                        markeredgewidth=1, capsize=3, capthick=1.5, elinewidth=1.5)

                        # Set y-limits for PMRA only in residual mode (centered around 0)
                        if residual_mode:
                            pad = sigma_pad_frac * sigma_pmra
                            ylo, yhi = -2*sigma_pmra - pad, 2*sigma_pmra + pad
                            ax[2].set_ylim(ylo, yhi)

                        
                        # PMDEC spline
                        pmdec_mcmc = stream_funcs.apply_spline(
                            phi1_spline_plot, self.mcmeta.phi1_spline_points, 
                            pmdec_mcmc_points, k=2
                        )
                        if residual_mode:
                            ax[3].axhline(0, color='blue', linewidth=2, alpha=0.8, label='MCMC Spline (resid)')
                        else:
                            ax[3].plot(phi1_spline_plot, pmdec_mcmc, 'b-', linewidth=2, 
                                      label='MCMC Spline', alpha=0.8)
                        
                        # Plot 1 sigma and 2 sigma regions for PMDEC
                        _lsigpmdec_log = self.lsigpm_ if self.lsigpm_ is not None else meds.get('lsigpmdec', None)
                        if _lsigpmdec_log is None and mcmc_object is not None:
                            _lsigpmdec_log = getattr(getattr(mcmc_object, 'meta', None), 'lsigpm_', None)
                        if _lsigpmdec_log is None:
                            raise AttributeError("No lsigpmdec in meds and no fixed lsigpm_ available for PMDEC band.")
                        sigma_pmdec = 10**_lsigpmdec_log
                        if residual_mode:
                            ax[3].fill_between(phi1_spline_plot, -sigma_pmdec, +sigma_pmdec, color='blue', alpha=0.1)
                            ax[3].fill_between(phi1_spline_plot, -2*sigma_pmdec, +2*sigma_pmdec, color='blue', alpha=0.05)
                        else:
                            ax[3].fill_between(phi1_spline_plot, pmdec_mcmc - sigma_pmdec, pmdec_mcmc + sigma_pmdec, color='blue', alpha=0.1)
                            ax[3].fill_between(phi1_spline_plot, pmdec_mcmc - 2*sigma_pmdec, pmdec_mcmc + 2*sigma_pmdec, color='blue', alpha=0.05)
                        
                        # Extract error bars for PMDEC spline points


                        pmdec_mcmc_ep = []
                        for i in range(1, no_of_spline_points + 1):
                            pmdec_mcmc_ep.append(ep['pmdec'+str(i)])

                        pmdec_mcmc_em = []
                        for i in range(1, no_of_spline_points + 1):
                            pmdec_mcmc_em.append(np.abs(em['pmdec'+str(i)]))
                        pmdec_mcmc_errors = np.array([pmdec_mcmc_em, pmdec_mcmc_ep])

                        yvals = np.zeros_like(self.mcmeta.phi1_spline_points) if residual_mode else pmdec_mcmc_points
                        ax[3].errorbar(self.mcmeta.phi1_spline_points, yvals, 
                                        yerr=pmdec_mcmc_errors, fmt='o', color='blue', 
                                        markersize=8, zorder=10, alpha=0.8, markeredgecolor='white', 
                                        markeredgewidth=1, capsize=3, capthick=1.5, elinewidth=1.5)

                        # Set y-limits for PMDEC only in residual mode (centered around 0)
                        if residual_mode:
                            pad_dec = sigma_pad_frac * sigma_pmdec
                            ylo, yhi = -2*sigma_pmdec - pad_dec, 2*sigma_pmdec + pad_dec
                            ax[3].set_ylim(ylo, yhi)

        # If not plotting residuals, optionally zoom PM panels using members' distribution
        if not residual_mode:
            try:
                # Validate provided stream_prob
                if stream_prob is not None:
                    sp = np.asarray(stream_prob)
                    if sp.shape[0] == len(self.data.desi_data):
                        mem_mask = np.isfinite(sp) & (sp >= float(min_prob))
                        # PMRA member-based limits
                        mem_pmra = np.asarray(self.data.desi_data['PMRA'].values)[mem_mask]
                        if mem_pmra.size >= 3 and np.isfinite(mem_pmra).any():
                            mu, sig = float(np.nanmean(mem_pmra)), float(np.nanstd(mem_pmra))
                            pad = sigma_pad_frac * sig
                            ax[2].set_ylim(mu - 2*sig - pad, mu + 2*sig + pad)
                        # PMDEC member-based limits
                        mem_pmdec = np.asarray(self.data.desi_data['PMDEC'].values)[mem_mask]
                        if mem_pmdec.size >= 3 and np.isfinite(mem_pmdec).any():
                            mu, sig = float(np.nanmean(mem_pmdec)), float(np.nanstd(mem_pmdec))
                            pad = sigma_pad_frac * sig
                            ax[3].set_ylim(mu - 2*sig - pad, mu + 2*sig + pad)
            except Exception:
                # Be permissive: plotting should not fail due to y-limit heuristics
                pass

        # Posterior Predictive Check (PPC): overlay 2D density for stream-only component
        if show_ppc and (mcmc_object is not None) and hasattr(mcmc_object, 'flatchain') and hasattr(mcmc_object, 'expanded_param_labels') and hasattr(self, 'mcmeta') and (self.mcmeta is not None):
            try:
                labels = list(mcmc_object.expanded_param_labels)
                flatchain = np.asarray(mcmc_object.flatchain)
                if flatchain.ndim != 2 or flatchain.shape[1] != len(labels):
                    raise ValueError('MCMC flatchain/labels shape mismatch')

                # helper to find indices safely
                def idx_or_none(name):
                    try:
                        return labels.index(name)
                    except ValueError:
                        return None

                npts = len(self.mcmeta.phi1_spline_points)
                v_idx = [labels.index(f'vgsr{i}') for i in range(1, npts+1)]
                pmra_idx = [labels.index(f'pmra{i}') for i in range(1, npts+1)]
                pmdec_idx = [labels.index(f'pmdec{i}') for i in range(1, npts+1)]

                idx_lsigv = idx_or_none('lsigvgsr')
                idx_feh1 = idx_or_none('feh1')
                idx_lsigfeh = idx_or_none('lsigfeh')
                idx_lsigpmra = idx_or_none('lsigpmra')
                idx_lsigpmdec = idx_or_none('lsigpmdec')

                # choose sample rows
                nsamp = int(min(max(10, ppc_nsamples), flatchain.shape[0]))
                samp_idx = np.linspace(0, flatchain.shape[0]-1, nsamp, dtype=int)

                # phi grid to evaluate
                phi_min = float(np.min(self.mcmeta.phi1_spline_points))
                phi_max = float(np.max(self.mcmeta.phi1_spline_points))
                phi_grid = np.linspace(phi_min, phi_max, int(max(20, ppc_phi_grid)))

                # containers per panel
                X_v, Y_v = [], []
                X_ra, Y_ra = [], []
                X_de, Y_de = [], []
                X_fh, Y_fh = [], []

                # fixed PM sigma if used
                fixed_lsigpm = getattr(getattr(mcmc_object, 'meta', None), 'lsigpm_', None)

                for si in samp_idx:
                    row = flatchain[si]
                    # means
                    v_knots = row[v_idx]
                    pmra_knots = row[pmra_idx]
                    pmdec_knots = row[pmdec_idx]
                    v_mean = stream_funcs.apply_spline(phi_grid, self.mcmeta.phi1_spline_points, v_knots, k=2)
                    ra_mean = stream_funcs.apply_spline(phi_grid, self.mcmeta.phi1_spline_points, pmra_knots, k=2)
                    de_mean = stream_funcs.apply_spline(phi_grid, self.mcmeta.phi1_spline_points, pmdec_knots, k=2)
                    feh_mean = np.full_like(phi_grid, row[idx_feh1]) if idx_feh1 is not None else np.full_like(phi_grid, np.nan)

                    # dispersions
                    sig_v = 10**row[idx_lsigv] if idx_lsigv is not None else 1.0
                    sig_fh = 10**row[idx_lsigfeh] if idx_lsigfeh is not None else 0.1
                    sig_ra = 10**row[idx_lsigpmra] if idx_lsigpmra is not None else (10**fixed_lsigpm if fixed_lsigpm is not None else 0.1)
                    sig_de = 10**row[idx_lsigpmdec] if idx_lsigpmdec is not None else (10**fixed_lsigpm if fixed_lsigpm is not None else 0.1)

                    # draw predictive samples per grid point (1 per grid point for efficiency)
                    if residual_mode:
                        yv = np.random.normal(0.0, sig_v, size=phi_grid.size)
                        yra = np.random.normal(0.0, sig_ra, size=phi_grid.size)
                        yde = np.random.normal(0.0, sig_de, size=phi_grid.size)
                        yfh = np.random.normal(0.0, sig_fh, size=phi_grid.size)
                    else:
                        yv = np.random.normal(v_mean, sig_v)
                        yra = np.random.normal(ra_mean, sig_ra)
                        yde = np.random.normal(de_mean, sig_de)
                        yfh = np.random.normal(feh_mean, sig_fh)

                    X_v.append(phi_grid); Y_v.append(yv)
                    X_ra.append(phi_grid); Y_ra.append(yra)
                    X_de.append(phi_grid); Y_de.append(yde)
                    X_fh.append(phi_grid); Y_fh.append(yfh)

                # stack and histogram
                def _hist2d(axh, x, y, bins_x=60, bins_y=60):
                    x = np.concatenate(x); y = np.concatenate(y)
                    if not np.isfinite(x).any() or not np.isfinite(y).any():
                        return
                    hb = axh.hist2d(x, y, bins=[bins_x, bins_y], cmap='Blues', density=True, alpha=float(ppc_alpha), zorder=0)
                    # reduce edge color visibility
                    if isinstance(hb, tuple) and len(hb) == 4:
                        hb[3].set_linewidth(0)

                _hist2d(ax[1], X_v, Y_v)
                _hist2d(ax[2], X_ra, Y_ra)
                _hist2d(ax[3], X_de, Y_de)
                _hist2d(ax[4], X_fh, Y_fh)
            except Exception as e:
                print(f"Warning: PPC overlay skipped due to error: {e}")

        # FEH constant line and bands (if MCMC meds available)
        try:
            feh_mcmc = np.full_like(phi1_spline_plot, meds['feh1'])
            if residual_mode:
                ax[4].axhline(0, color='blue', linewidth=2, alpha=0.8, label='MCMC [Fe/H] (resid)')
            else:
                ax[4].plot(phi1_spline_plot, feh_mcmc, 'b-', linewidth=2, 
                          label='MCMC [Fe/H]', alpha=0.8)
            # 1σ and 2σ regions
            sigma_feh = 10**meds['lsigfeh']
            if residual_mode:
                ax[4].fill_between(phi1_spline_plot, -sigma_feh, +sigma_feh, color='blue', alpha=0.1)
                ax[4].fill_between(phi1_spline_plot, -2*sigma_feh, +2*sigma_feh, color='blue', alpha=0.05)
            else:
                ax[4].fill_between(phi1_spline_plot, feh_mcmc - sigma_feh, feh_mcmc + sigma_feh, color='blue', alpha=0.1)
                ax[4].fill_between(phi1_spline_plot, feh_mcmc - 2*sigma_feh, feh_mcmc + 2*sigma_feh, color='blue', alpha=0.05)
        except Exception:
            pass

        
        # Labels and formatting
        if show_initial_splines or show_optimized_splines or show_mcmc_splines:
            # Add legends to kinematic plots if splines are shown
            for i in [1]:  # Show legends on all kinematic plots
                if ax[i].get_lines():  # Only add legend if there are lines to show
                    ax[i].legend(loc='best', fontsize='small')
        # Place top panel legend (phi2 vs phi1) at bottom-left
        ax[0].legend(loc='lower left', ncol=4)
        ax[0].set_ylabel(label_y0)
        ax[1].set_ylabel(r'V$_{GSR}$ (km/s)' if not residual_mode else r'V$_{GSR}$ - v$_{\rm MCMC}$ (km/s)')
        ax[2].set_ylabel(r'$\mu_{\alpha}$ [mas/yr]' if not residual_mode else r'$\mu_{\alpha}$ - $\mu_{\alpha,\rm MCMC}$ [mas/yr]')
        ax[3].set_ylabel(r'$\mu_{\delta}$ [mas/yr]' if not residual_mode else r'$\mu_{\delta}$ - $\mu_{\delta,\rm MCMC}$ [mas/yr]')
        ax[4].set_ylabel(r'[Fe/H]' if not residual_mode else r'[Fe/H] - [Fe/H]$_{\rm MCMC}$')
        ax[-1].set_xlabel(label_x)
        
        for a in ax:
            stream_funcs.plot_form(a)
            
        if save:
            plt.tight_layout()
            plt.savefig(f"{self.save_dir}sixD_plot_{self.stream.streamName}.png", dpi=300, bbox_inches='tight')
            
        return fig, ax
    
    def gaussian_mixture_plot(self, showStream=True, background=True, save=False, show_model=True, show_total=True):
        """
        Plots Gaussian mixture model distributions for stream vs background in 4 dimensions:
        VGSR, FEH, PMRA, PMDEC
        
        Uses truncated Gaussians based on the selection cuts applied to the data.
        Requires MCMeta object to be initialized with initial parameters.
        """
        if self.mcmeta is None:
            raise ValueError("MCMeta object required for gaussian_mixture_plot. Initialize StreamPlotter with MCMeta object.")
            
        colors = list(plt.rcParams["axes.prop_cycle"].by_key()["color"])
        from scipy.stats import truncnorm
        
        fig, axes = plt.subplots(2, 2, figsize=(9, 9))
        
        # Get data arrays
        desi_data = self.data.desi_data
        sf_data = self.data.confirmed_sf_and_desi if hasattr(self.data, 'confirmed_sf_and_desi') else pd.DataFrame()
        
        # Define plotting parameters
        alpha_stream = 0.7
        alpha_bg = 0.5
        bins = 50
        
        # Estimate mixture weights
        n_stream = len(sf_data) if len(sf_data) > 0 else 1
        n_total = len(desi_data)
        stream_weight = n_stream / n_total
        bg_weight = 1 - stream_weight
        
        # VGSR plot (top left)
        ax = axes[0, 0]
        if background:
            ax.hist(desi_data['VGSR'], density=True, color='lightgrey', bins=bins, alpha=0.7, label='DESI Data')
        
        if showStream and len(sf_data) > 0:
            ax.hist(sf_data['VGSR'], density=True, color='lightblue', bins=bins, alpha=0.8, label='SF Stars')
            
        # Plot truncated Gaussian components
        if self.mcmeta.truncation_params['vgsr_min'] != -np.inf or self.mcmeta.truncation_params['vgsr_max'] != np.inf:
            vgsr_range = np.linspace(self.mcmeta.truncation_params['vgsr_min'] - 50, 
                                    self.mcmeta.truncation_params['vgsr_max'] + 50, 200)
        else:
            vgsr_range = np.linspace(np.min(desi_data['VGSR']) - 50, 
                                    np.max(desi_data['VGSR']) + 50, 200)
        
        # Stream component (using mean of spline points as approximation)
        stream_vgsr_mean = np.mean(self.mcmeta.initial_params['vgsr_spline_points'])
        lsigv_init = np.atleast_1d(self.mcmeta.initial_params['lsigvgsr'])
        stream_vgsr_std = float(np.mean(10**lsigv_init))
        
        # Background component
        bg_vgsr_mean = self.mcmeta.initial_params['bv']
        bg_vgsr_std = 10**self.mcmeta.initial_params['lsigbv']
        
        # Stream truncated normal
        stream_a = (self.mcmeta.truncation_params['vgsr_min'] - stream_vgsr_mean) / stream_vgsr_std
        stream_b = (self.mcmeta.truncation_params['vgsr_max'] - stream_vgsr_mean) / stream_vgsr_std
        stream_vgsr_pdf = truncnorm.pdf(vgsr_range, stream_a, stream_b, loc=stream_vgsr_mean, scale=stream_vgsr_std)
        
        # Background truncated normal
        bg_a = (self.mcmeta.truncation_params['vgsr_min'] - bg_vgsr_mean) / bg_vgsr_std
        bg_b = (self.mcmeta.truncation_params['vgsr_max'] - bg_vgsr_mean) / bg_vgsr_std
        bg_vgsr_pdf = truncnorm.pdf(vgsr_range, bg_a, bg_b, loc=bg_vgsr_mean, scale=bg_vgsr_std)
        
        if show_model:
            ax.plot(vgsr_range, stream_weight * stream_vgsr_pdf, ':', color=colors[0], label='Stream Component', lw=3, zorder=2)
            ax.plot(vgsr_range, bg_weight * bg_vgsr_pdf, ':', color=colors[1], label='Background Component', lw=3, zorder=2)
        if show_model and show_total:
            ax.plot(vgsr_range, stream_weight * stream_vgsr_pdf + bg_weight * bg_vgsr_pdf, 'k-', label='Total Model', lw=3,zorder=1)
            
        ax.set_xlabel(r'V$_{GSR}$ (km/s)', fontsize=12)
        if self.mcmeta.truncation_params['vgsr_min'] != -np.inf and self.mcmeta.truncation_params['vgsr_max'] != np.inf:
            ax.set_xlim(self.mcmeta.truncation_params['vgsr_min'] - 50, self.mcmeta.truncation_params['vgsr_max'] + 50)
        ax.legend(fontsize='large')
        ax.tick_params(axis='both', labelsize=14)
        stream_funcs.plot_form(ax)
        
        # FEH plot (top right)
        ax = axes[0, 1]
        if background:
            ax.hist(desi_data['FEH'], density=True, color='lightgrey', bins=bins, alpha=0.7)
            
        if showStream and len(sf_data) > 0:
            ax.hist(sf_data['FEH'], density=True, color='lightblue', bins=bins, alpha=0.8)
            
        # Plot truncated Gaussian components
        if self.mcmeta.truncation_params['feh_min'] != -np.inf or self.mcmeta.truncation_params['feh_max'] != np.inf:
            feh_range = np.linspace(self.mcmeta.truncation_params['feh_min'] - 0.5, 
                                self.mcmeta.truncation_params['feh_max'] + 0.5, 200)
        else:
            feh_range = np.linspace(np.min(desi_data['FEH']) - 0.5, 
                                np.max(desi_data['FEH']) + 0.5, 200)

        # Stream component
        stream_feh_mean = self.mcmeta.initial_params['feh1']
        stream_feh_std = 10**self.mcmeta.initial_params['lsigfeh']
        
        # Background component
        bg_feh_mean = self.mcmeta.initial_params['bfeh']
        bg_feh_std = 10**self.mcmeta.initial_params['lsigbfeh']
        
        # Stream truncated normal
        stream_a = (self.mcmeta.truncation_params['feh_min'] - stream_feh_mean) / stream_feh_std
        stream_b = (self.mcmeta.truncation_params['feh_max'] - stream_feh_mean) / stream_feh_std
        stream_feh_pdf = truncnorm.pdf(feh_range, stream_a, stream_b, loc=stream_feh_mean, scale=stream_feh_std)
        
        # Background truncated normal
        bg_a = (self.mcmeta.truncation_params['feh_min'] - bg_feh_mean) / bg_feh_std
        bg_b = (self.mcmeta.truncation_params['feh_max'] - bg_feh_mean) / bg_feh_std
        bg_feh_pdf = truncnorm.pdf(feh_range, bg_a, bg_b, loc=bg_feh_mean, scale=bg_feh_std)
        if show_model:
            ax.plot(feh_range, stream_weight * stream_feh_pdf, ':', color=colors[0], lw=3, zorder=2)
            ax.plot(feh_range, bg_weight * bg_feh_pdf, ':', color=colors[1], lw=3, zorder=2)
        if show_model and show_total:
            ax.plot(feh_range, stream_weight * stream_feh_pdf + bg_weight * bg_feh_pdf, 'k-', lw=3, zorder=1)
            
        ax.set_xlabel('[Fe/H]', fontsize=12)
        if self.mcmeta.truncation_params['feh_min'] != -np.inf and self.mcmeta.truncation_params['feh_max'] != np.inf:
            ax.set_xlim(self.mcmeta.truncation_params['feh_min'] - 0.5, self.mcmeta.truncation_params['feh_max'] + 0.5)
        ax.tick_params(axis='both', labelsize=14)
        stream_funcs.plot_form(ax)
        
        # PMRA plot (bottom left)
        ax = axes[1, 0]
        if background:
            ax.hist(desi_data['PMRA'], density=True, color='lightgrey', bins=bins, alpha=0.7)
            
        if showStream and len(sf_data) > 0:
            ax.hist(sf_data['PMRA'], density=True, color='lightblue', bins=bins, alpha=0.8)
            
        # Plot truncated Gaussian components
        if self.mcmeta.truncation_params['pmra_min'] != -np.inf or self.mcmeta.truncation_params['pmra_max'] != np.inf:
            pmra_range = np.linspace(self.mcmeta.truncation_params['pmra_min'] - 15, 
                                self.mcmeta.truncation_params['pmra_max'] + 15, 200)
        else:
            pmra_range = np.linspace(np.min(desi_data['PMRA']) - 15, 
                                np.max(desi_data['PMRA']) + 15, 200)

        # Stream component (using mean of spline points as approximation)
        stream_pmra_mean = np.mean(self.mcmeta.initial_params['pmra_spline_points'])
        _lsigpmra_log = self.mcmeta.initial_params.get('lsigpmra', getattr(self.mcmeta, 'lsigpm_', None))
        if _lsigpmra_log is None:
            raise KeyError("lsigpmra not found in initial_params and no fixed lsigpm_ set on MCMeta")
        stream_pmra_std = 10**_lsigpmra_log
        
        # Background component
        bg_pmra_mean = self.mcmeta.initial_params['bpmra']
        bg_pmra_std = 10**self.mcmeta.initial_params['lsigbpmra']
        
        # Stream truncated normal
        stream_a = (self.mcmeta.truncation_params['pmra_min'] - stream_pmra_mean) / stream_pmra_std
        stream_b = (self.mcmeta.truncation_params['pmra_max'] - stream_pmra_mean) / stream_pmra_std
        stream_pmra_pdf = truncnorm.pdf(pmra_range, stream_a, stream_b, loc=stream_pmra_mean, scale=stream_pmra_std)
        
        # Background truncated normal
        bg_a = (self.mcmeta.truncation_params['pmra_min'] - bg_pmra_mean) / bg_pmra_std
        bg_b = (self.mcmeta.truncation_params['pmra_max'] - bg_pmra_mean) / bg_pmra_std
        bg_pmra_pdf = truncnorm.pdf(pmra_range, bg_a, bg_b, loc=bg_pmra_mean, scale=bg_pmra_std)
        if show_model:
            ax.plot(pmra_range, stream_weight * stream_pmra_pdf, ':', color=colors[0], lw=3, zorder=2)
            ax.plot(pmra_range, bg_weight * bg_pmra_pdf, ':', color=colors[1], lw=3, zorder=2)
        if show_model and show_total:
            ax.plot(pmra_range, stream_weight * stream_pmra_pdf + bg_weight * bg_pmra_pdf, 'k-', lw=3, zorder=1)
            
 
        ax.set_xlabel(r'$\mu_{RA}$ (mas/yr)', fontsize=12)
        if self.mcmeta.truncation_params['pmra_min'] != -np.inf and self.mcmeta.truncation_params['pmra_max'] != np.inf:
            ax.set_xlim(self.mcmeta.truncation_params['pmra_min'] - 15, self.mcmeta.truncation_params['pmra_max'] + 15)
        ax.tick_params(axis='both', labelsize=14)
        stream_funcs.plot_form(ax)
        
        # PMDEC plot (bottom right)
        ax = axes[1, 1]
        if background:
            ax.hist(desi_data['PMDEC'], density=True, color='lightgrey', bins=bins, alpha=0.7)
            
        if showStream and len(sf_data) > 0:
            ax.hist(sf_data['PMDEC'], density=True, color='lightblue', bins=bins, alpha=0.8)
            
        # Plot truncated Gaussian components
        if self.mcmeta.truncation_params['pmdec_min'] != -np.inf or self.mcmeta.truncation_params['pmdec_max'] != np.inf:
            pmdec_range = np.linspace(self.mcmeta.truncation_params['pmdec_min'] - 15, 
                                 self.mcmeta.truncation_params['pmdec_max'] + 15, 200)
        else:
            pmdec_range = np.linspace(np.min(desi_data['PMDEC']) - 15, 
                                 np.max(desi_data['PMDEC']) + 15, 200)

        # Stream component (using mean of spline points as approximation)
        stream_pmdec_mean = np.mean(self.mcmeta.initial_params['pmdec_spline_points'])
        _lsigpmdec_log = self.mcmeta.initial_params.get('lsigpmdec', getattr(self.mcmeta, 'lsigpm_', None))
        if _lsigpmdec_log is None:
            raise KeyError("lsigpmdec not found in initial_params and no fixed lsigpm_ set on MCMeta")
        stream_pmdec_std = 10**_lsigpmdec_log
        
        # Background component
        bg_pmdec_mean = self.mcmeta.initial_params['bpmdec']
        bg_pmdec_std = 10**self.mcmeta.initial_params['lsigbpmdec']
        
        # Stream truncated normal
        stream_a = (self.mcmeta.truncation_params['pmdec_min'] - stream_pmdec_mean) / stream_pmdec_std
        stream_b = (self.mcmeta.truncation_params['pmdec_max'] - stream_pmdec_mean) / stream_pmdec_std
        stream_pmdec_pdf = truncnorm.pdf(pmdec_range, stream_a, stream_b, loc=stream_pmdec_mean, scale=stream_pmdec_std)
        
        # Background truncated normal
        bg_a = (self.mcmeta.truncation_params['pmdec_min'] - bg_pmdec_mean) / bg_pmdec_std
        bg_b = (self.mcmeta.truncation_params['pmdec_max'] - bg_pmdec_mean) / bg_pmdec_std
        bg_pmdec_pdf = truncnorm.pdf(pmdec_range, bg_a, bg_b, loc=bg_pmdec_mean, scale=bg_pmdec_std)
        if show_model:
            ax.plot(pmdec_range, stream_weight * stream_pmdec_pdf, ':', color=colors[0], lw=3, zorder=2)
            ax.plot(pmdec_range, bg_weight * bg_pmdec_pdf, ':', color=colors[1], lw=3, zorder=2)
        if show_model and show_total:
            ax.plot(pmdec_range, stream_weight * stream_pmdec_pdf + bg_weight * bg_pmdec_pdf, 'k-', lw=3, zorder=1)
        
        ax.set_xlabel(r'$\mu_{DEC}$ (mas/yr)', fontsize=12)
        if self.mcmeta.truncation_params['pmdec_min'] != -np.inf and self.mcmeta.truncation_params['pmdec_max'] != np.inf:
            ax.set_xlim(self.mcmeta.truncation_params['pmdec_min'] - 15, self.mcmeta.truncation_params['pmdec_max'] + 15)
        ax.tick_params(axis='both', labelsize=14)
        stream_funcs.plot_form(ax)
        
        plt.tight_layout()
        
        if save:
            plt.savefig(f"{self.save_dir}gaussian_mixture_{self.stream.streamName}.png", dpi=300, bbox_inches='tight')
            
        return fig, axes


class MCMeta:
    """
    For creating and plotting a spline track of the stream.
    """

    @staticmethod
    def _spline_k_for_points(num_points):
        try:
            npts = int(num_points)
        except Exception:
            npts = 1
        if npts <= 1:
            return 1
        if npts > 3:
            return 3
        return npts - 1

    def __init__(self, no_of_spline_points, stream_object, sf_data, truncation_params=None,
                 phi1_min=None, phi1_max=None, pstream_no_of_spline_points=1,
                 lsigv_no_of_spline_points=1, **kwargs):
        self.stream = stream_object
        self.no_of_spline_points = int(no_of_spline_points)
        self.sf_data = sf_data
        self.pstream_no_of_spline_points = max(1, int(pstream_no_of_spline_points))
        self.lsigv_no_of_spline_points = max(1, int(lsigv_no_of_spline_points))

        self.spline_k = self._spline_k_for_points(self.no_of_spline_points)
        self.spline_k_pstream = self._spline_k_for_points(self.pstream_no_of_spline_points)
        self.spline_k_lsigv = self._spline_k_for_points(self.lsigv_no_of_spline_points)

        # Use provided phi1 range or default to available data (SF3 → galstreams → DESI)
        def _phi1_bounds():
            # SF3 members
            sf_df = getattr(self.stream.data, 'SoI_streamfinder', pd.DataFrame())
            if not sf_df.empty and 'phi1' in sf_df:
                return sf_df['phi1'].min(), sf_df['phi1'].max()
            # galstreams track
            gs = getattr(self.stream.data, 'SoI_galstream', None)
            if gs is not None and hasattr(gs, 'gal_phi1'):
                return np.nanmin(gs.gal_phi1), np.nanmax(gs.gal_phi1)
            # DESI phi1 range
            desi_df = getattr(self.stream.data, 'desi_data', pd.DataFrame())
            if not desi_df.empty and 'phi1' in desi_df:
                return np.nanmin(desi_df['phi1']), np.nanmax(desi_df['phi1'])
            return -50.0, 50.0

        if phi1_min is None or phi1_max is None:
            default_min, default_max = _phi1_bounds()
            phi1_min = default_min if phi1_min is None else phi1_min
            phi1_max = default_max if phi1_max is None else phi1_max
        
        self.phi1_spline_points = kwargs.get('phi1_spline_points', np.linspace(phi1_min, phi1_max, self.no_of_spline_points))
        self.phi1_min = self.phi1_spline_points.min() if phi1_min is None else phi1_min
        self.phi1_max = self.phi1_spline_points.max() if phi1_max is None else phi1_max

        if self.pstream_no_of_spline_points == 1:
            self.pstream_phi1_spline_points = np.array([self.phi1_min], dtype=float)
        else:
            self.pstream_phi1_spline_points = np.linspace(
                self.phi1_min, self.phi1_max, self.pstream_no_of_spline_points
            )

        if self.lsigv_no_of_spline_points == 1:
            self.lsigv_phi1_spline_points = np.array([self.phi1_min], dtype=float)
        else:
            self.lsigv_phi1_spline_points = np.linspace(
                self.phi1_min, self.phi1_max, self.lsigv_no_of_spline_points
            )

        # Store truncation parameters for plotting
        if truncation_params is None:
            # Default truncation values based on data range
            data = self.stream.data.desi_data
            self.truncation_params = {
                'vgsr_min': -np.inf,
                'vgsr_max': np.inf,
                'feh_min': -np.inf,
                'feh_max': np.inf,
                'pmra_min': -np.inf,
                'pmra_max': np.inf,
                'pmdec_min': -np.inf,
                'pmdec_max': np.inf
            }
        else:
            self.truncation_params = truncation_params

        self.lsigpm_ = kwargs.get('lsigpm_set', None)

        # If lsigpm_ is provided (fixed), exclude lsigpmra/lsigpmdec from parameters
        # IMPORTANT: Include 'pstream' first to match parameter vector order
        self.param_labels = [
            "pstream",
            "vgsr_spline_points", "lsigvgsr",
            "feh1", "lsigfeh",
            "pmra_spline_points",
            *(["lsigpmra"] if self.lsigpm_ is None else []),
            "pmdec_spline_points",
            *(["lsigpmdec"] if self.lsigpm_ is None else []),
            "bv", "lsigbv", "bfeh", "lsigbfeh",
            "bpmra", "lsigbpmra", "bpmdec", "lsigbpmdec"
        ]

        # Initialize the initial_params dictionary
        self.initial_params = {}

        print('Making stream initial guess based on galstream and STREAMFINDER...')

        total_desi = max(len(self.stream.data.desi_data), 1)
        base_pstream = np.abs(len(self.sf_data) / total_desi)
        # Keep pstream within (0, 1) to avoid transform issues during sampling
        base_pstream = float(np.clip(base_pstream, 1e-4, 1 - 1e-4))
        if self.pstream_no_of_spline_points == 1:
            self.initial_params['pstream'] = base_pstream
        else:
            self.initial_params['pstream'] = np.full(
                self.pstream_no_of_spline_points, base_pstream, dtype=float
            )

        # Helper to compute a safe log10(sigma) from std and mean error
        def _safe_lsig(std_val, mean_err, default_sigma):
            try:
                var_eff = float(std_val)**2 - float(mean_err)**2
            except Exception:
                var_eff = np.nan
            if not np.isfinite(var_eff) or var_eff <= 0:
                sigma = float(default_sigma)
            else:
                sigma = float(np.sqrt(var_eff))
                if not np.isfinite(sigma) or sigma <= 0:
                    sigma = float(default_sigma)
            # guard against log of non-positive
            sigma = max(sigma, 1e-8)
            return np.log10(sigma)

        if self.sf_data is not None and len(self.sf_data) > 0:
            p = np.polyfit(self.sf_data['phi1'].values, self.sf_data['VGSR'].values, 1)
            self.vgsr_fit = np.poly1d(p)
            lsigv_base = _safe_lsig(
                self.sf_data['VGSR'].values.std(),
                np.mean(self.sf_data['VRAD_ERR'].values),
                default_sigma=1.0
            )
            if self.lsigv_no_of_spline_points == 1:
                self.initial_params['lsigvgsr'] = lsigv_base
            else:
                self.initial_params['lsigvgsr'] = np.full(
                    self.lsigv_no_of_spline_points, lsigv_base, dtype=float
                )
            self.initial_params['vgsr_spline_points'] = self.vgsr_fit(self.phi1_spline_points)
            lsigv_linear = 10**np.atleast_1d(self.initial_params['lsigvgsr'])
            if lsigv_linear.size == 1:
                print(f"Stream VGSR dispersion from trimmed SF: {lsigv_linear[0]:.2f} km/s")
            else:
                print(
                    "Stream VGSR dispersion from trimmed SF (initial knots): "
                    f"{lsigv_linear.min():.2f}–{lsigv_linear.max():.2f} km/s"
                )

            self.initial_params['feh1'] = self.sf_data['FEH'].values.mean()
            self.initial_params['lsigfeh'] = _safe_lsig(self.sf_data['FEH'].values.std(), np.mean(self.sf_data['FEH_ERR'].values), default_sigma=0.05)
            print(f'Stream mean metallicity from trimmed SF: {self.initial_params["feh1"]:.2f} +- {10**self.initial_params["lsigfeh"]:.3f} dex')

            p = np.polyfit(self.sf_data['phi1'].values, self.sf_data['PMRA'].values, 1)
            self.pmra_fit = np.poly1d(p)
            self.initial_params['lsigpmra'] = _safe_lsig(self.sf_data['PMRA'].values.std(), np.mean(self.sf_data['PMRA_ERROR'].values), default_sigma=0.1)
            self.initial_params['pmra_spline_points'] = self.pmra_fit(self.phi1_spline_points)
            print(f"Stream PMRA dispersion from trimmed SF: {10**self.initial_params['lsigpmra']:.2f} mas/yr")

            p = np.polyfit(self.sf_data['phi1'].values, self.sf_data['PMDEC'].values, 1)
            self.pmdec_fit = np.poly1d(p)
            self.initial_params['lsigpmdec'] = _safe_lsig(self.sf_data['PMDEC'].values.std(), np.mean(self.sf_data['PMDEC_ERROR'].values), default_sigma=0.1)
            self.initial_params['pmdec_spline_points'] = self.pmdec_fit(self.phi1_spline_points)
            print(f"Stream PMDEC dispersion from trimmed SF: {10**self.initial_params['lsigpmdec']:.2f} mas/yr")
        else:
            print('No STREAMFINDER members; using DESI data for initial stream guess.')
            desi = self.stream.data.desi_data
            vgsr_med = np.nanmedian(desi['VGSR'].values)
            pmra_med = np.nanmedian(desi['PMRA'].values)
            pmdec_med = np.nanmedian(desi['PMDEC'].values)
            feh_med = np.nanmedian(desi['FEH'].values)

            self.vgsr_fit = np.poly1d([0.0, vgsr_med])
            self.pmra_fit = np.poly1d([0.0, pmra_med])
            self.pmdec_fit = np.poly1d([0.0, pmdec_med])

            lsigv_base = _safe_lsig(
                np.nanstd(desi['VGSR'].values),
                np.nanmean(desi['VRAD_ERR'].values),
                default_sigma=5.0
            )
            if self.lsigv_no_of_spline_points == 1:
                self.initial_params['lsigvgsr'] = lsigv_base
            else:
                self.initial_params['lsigvgsr'] = np.full(
                    self.lsigv_no_of_spline_points, lsigv_base, dtype=float
                )
            self.initial_params['vgsr_spline_points'] = np.full(
                len(self.phi1_spline_points), vgsr_med, dtype=float
            )

            self.initial_params['feh1'] = feh_med
            self.initial_params['lsigfeh'] = _safe_lsig(
                np.nanstd(desi['FEH'].values),
                np.nanmean(desi['FEH_ERR'].values),
                default_sigma=0.2
            )
            print(f"Stream mean metallicity from DESI: {self.initial_params['feh1']:.2f} +- {10**self.initial_params['lsigfeh']:.3f} dex")

            self.initial_params['lsigpmra'] = _safe_lsig(
                np.nanstd(desi['PMRA'].values),
                np.nanmean(desi['PMRA_ERROR'].values),
                default_sigma=1.0
            )
            self.initial_params['pmra_spline_points'] = np.full(
                len(self.phi1_spline_points), pmra_med, dtype=float
            )

            self.initial_params['lsigpmdec'] = _safe_lsig(
                np.nanstd(desi['PMDEC'].values),
                np.nanmean(desi['PMDEC_ERROR'].values),
                default_sigma=1.0
            )
            self.initial_params['pmdec_spline_points'] = np.full(
                len(self.phi1_spline_points), pmdec_med, dtype=float
            )

        print('Making background initial guess...')
        counts, edges = np.histogram(np.array(self.stream.data.desi_data['VGSR']), bins=50)
        self.initial_params['bv'] = 0.5 * (edges[np.argmax(counts)] + edges[np.argmax(counts) + 1])

        self.initial_params['lsigbv'] = _safe_lsig(np.std(np.array(self.stream.data.desi_data['VGSR'])), np.mean(np.array(self.stream.data.desi_data['VRAD_ERR'])), default_sigma=1.0)
        print(f"Background velocity: {self.initial_params['bv']:.2f} +- {10**self.initial_params['lsigbv']:.2f} km/s")


        counts, edges = np.histogram(np.array(self.stream.data.desi_data['FEH']), bins=50)
        self.initial_params['bfeh'] = 0.5 * (edges[np.argmax(counts)] + edges[np.argmax(counts) + 1])
        self.initial_params['lsigbfeh'] = _safe_lsig(np.std(np.array(self.stream.data.desi_data['FEH'])), np.mean(np.array(self.stream.data.desi_data['FEH_ERR'])), default_sigma=0.05)
        print(f"Background metallicity: {self.initial_params['bfeh']:.2f} +- {10**self.initial_params['lsigbfeh']:.3f} dex")

        counts, edges = np.histogram(np.array(self.stream.data.desi_data['PMRA']), bins=50)
        self.initial_params['bpmra'] = 0.5 * (edges[np.argmax(counts)] + edges[np.argmax(counts) + 1])
        self.initial_params['lsigbpmra'] = _safe_lsig(np.std(np.array(self.stream.data.desi_data['PMRA'])), np.mean(np.array(self.stream.data.desi_data['PMRA_ERROR'])), default_sigma=0.1)
        print(f"Background PMRA: {self.initial_params['bpmra']:.2f} +- {10**self.initial_params['lsigbpmra']:.2f} mas/yr")

        counts, edges = np.histogram(np.array(self.stream.data.desi_data['PMDEC']), bins=50)
        self.initial_params['bpmdec'] = 0.5 * (edges[np.argmax(counts)] + edges[np.argmax(counts) + 1])
        self.initial_params['lsigbpmdec'] = _safe_lsig(np.std(np.array(self.stream.data.desi_data['PMDEC'])), np.mean(np.array(self.stream.data.desi_data['PMDEC_ERROR'])), default_sigma=0.1)
        print(f"Background PMDEC: {self.initial_params['bpmdec']:.2f} +- {10**self.initial_params['lsigbpmdec']:.2f} mas/yr")
    
    def priors(self, prior_arr):
        self.p0_guess = [
            self.initial_params['pstream'],                       # pstream (possibly spline points)
            self.initial_params['vgsr_spline_points'],             # VGSR spline points
            self.initial_params['lsigvgsr'],                       # lsigvgsr (possibly spline points)
            self.initial_params['feh1'],                           # mean [Fe/H]
            self.initial_params['lsigfeh'],                        # log(sigma_[Fe/H])
            self.initial_params['pmra_spline_points'],             # PMRA spline points
            *([self.initial_params['lsigpmra']] if self.lsigpm_ is None else []),
            self.initial_params['pmdec_spline_points'],            # PMDEC spline points
            *([self.initial_params['lsigpmdec']] if self.lsigpm_ is None else []),
            self.initial_params['bv'],                             # background VGSR
            self.initial_params['lsigbv'],                         # log(sigma_background_vgsr)
            self.initial_params['bfeh'],                           # background [Fe/H]
            self.initial_params['lsigbfeh'],                       # log(sigma_background_feh)
            self.initial_params['bpmra'],                          # background PMRA
            self.initial_params['lsigbpmra'],                      # log(sigma_background_pmra)
            self.initial_params['bpmdec'],                         # background PMDEC
            self.initial_params['lsigbpmdec']                      # log(sigma_background_pmdec)
        ]

        self.vgsr_trunc = [self.truncation_params['vgsr_min'], self.truncation_params['vgsr_max']]
        self.feh_trunc = [self.truncation_params['feh_min'], self.truncation_params['feh_max']]  
        self.pmra_trunc = [self.truncation_params['pmra_min'], self.truncation_params['pmra_max']]
        self.pmdec_trunc = [self.truncation_params['pmdec_min'], self.truncation_params['pmdec_max']]

        self.array_lengths = [len(x) if isinstance(x, np.ndarray) else 1 for x in self.p0_guess]
        self.flat_p0_guess = np.hstack(self.p0_guess)
        self.prior_arr = self._normalize_prior(prior_arr)

        # Auto-set walkers: ensure at least 2 * ndim to satisfy emcee red-blue move
        ndim = len(self.flat_p0_guess)
        min_walkers = 2 * ndim
        existing = getattr(self, "nwalkers", None)
        if (existing is None) or (existing < min_walkers):
            # Keep the old 70 fallback if it is larger than the strict minimum
            self.nwalkers = max(min_walkers, 70)
            print(f"[MCMeta] Auto-setting nwalkers to {self.nwalkers} (2 * ndim = {min_walkers})")
        else:
            print(f"[MCMeta] Using user-provided nwalkers = {existing} (2 * ndim = {min_walkers})")

        # Sanity check: prior length must match parameter vector length
        if len(self.flat_p0_guess) != len(self.prior_arr):
            msg = (
                f"Prior/parameter length mismatch: prior={len(self.prior_arr)} vs params={len(self.flat_p0_guess)}. "
                f"If you set lsigpm_set (fixed PM sigmas), remove 'lsigpmra' and 'lsigpmdec' bounds from the prior."
            )
            raise ValueError(msg)

    def _normalize_prior(self, prior_arr):
        """Expand prior definitions to match flattened parameter vector length."""
        prior_list = list(prior_arr)
        n_params = len(self.flat_p0_guess)
        if len(prior_list) == n_params:
            expanded = []
            for bounds in prior_list:
                arr = np.asarray(bounds, dtype=float)
                if arr.shape != (2,):
                    raise ValueError(f"Prior bounds {bounds} are not a valid (min, max) pair")
                expanded.append((float(arr[0]), float(arr[1])))
            return expanded

        if len(prior_list) != len(self.p0_guess):
            raise ValueError(
                f"Prior list must have length {n_params} (flattened) or {len(self.p0_guess)} (grouped); got {len(prior_list)}"
            )

        expanded = []
        for group_len, bounds in zip(self.array_lengths, prior_list):
            arr = np.asarray(bounds, dtype=float)
            if arr.ndim == 1:
                if arr.size != 2:
                    raise ValueError(f"Prior bounds {bounds} are not a valid (min, max) pair")
                arr = np.tile(arr, (group_len, 1)) if group_len > 1 else arr.reshape(1, 2)
            elif arr.ndim == 2 and arr.shape[1] == 2:
                if arr.shape[0] == group_len:
                    pass
                elif arr.shape[0] == 1:
                    if group_len > 1:
                        arr = np.repeat(arr, group_len, axis=0)
                elif group_len == 1:
                    arr = arr[:1]
                else:
                    raise ValueError(
                        f"Prior bounds array shape {arr.shape} incompatible with parameter group of length {group_len}"
                    )
            else:
                raise ValueError(f"Prior bounds {bounds} must be convertible to shape (N, 2)")

            for row in np.atleast_2d(arr):
                expanded.append((float(row[0]), float(row[1])))

        return expanded

    def scipy_optimize(self, method='COBYLA'):
        self.optfunc = lambda theta: -stream_funcs.lnprob(
            theta,
            self.prior_arr,
            self.phi1_spline_points,
            self.stream.data.desi_data['VGSR'].values, self.stream.data.desi_data['VRAD_ERR'].values,
            self.stream.data.desi_data['FEH'].values, self.stream.data.desi_data['FEH_ERR'].values,
            self.stream.data.desi_data['PMRA'].values, self.stream.data.desi_data['PMRA_ERROR'].values,
            self.stream.data.desi_data['PMDEC'].values, self.stream.data.desi_data['PMDEC_ERROR'].values,
            self.stream.data.desi_data['phi1'].values,
            trunc_fit=True, assert_prior=False, feh_fit=True,
            k=self.spline_k, reshape_arr_shape=self.array_lengths,
            vgsr_trunc=self.vgsr_trunc, feh_trunc=self.feh_trunc,
            pmra_trunc=self.pmra_trunc, pmdec_trunc=self.pmdec_trunc,
            lsigpm_set=self.lsigpm_
        )
        # Run optimization
        print("Running optimization...")
        self.sp_result = sp.optimize.minimize(self.optfunc, self.flat_p0_guess, method=method, options={'maxiter': 20000})
        print(self.sp_result.message)

        self.reshaped_result = stream_funcs.reshape_arr(self.sp_result.x, self.array_lengths)
        self.sp_output = stream_funcs.get_paramdict(self.reshaped_result, labels=self.param_labels)
        # Make optimized parameters available for plotting/debugging
        self.optimized_params = self.sp_output

        print("\nOptimized Parameters:")
        for label, value in self.sp_output.items():
            if label.startswith('l'):
                if isinstance(value, np.ndarray):
                    print(f"{label[1:]}: {10**value}")
                else:
                    print(f"{label[1:]}: {10**value:.4f}")
            else:
                if isinstance(value, np.ndarray):
                    print(f"{label}: {value}")
                else:
                    print(f"{label}: {value:.4f}" if isinstance(value, (int, float)) else f"{label}: {value}")

    def prior_validation(self):
        self.nparams = len(self.param_labels)
        # Ensure walker count still satisfies the red-blue move requirement
        ndim = len(self.flat_p0_guess)
        min_walkers = 2 * ndim
        current = getattr(self, "nwalkers", 0) or 0
        self.nwalkers = max(current, min_walkers, 70)
        print(f"[MCMeta] prior_validation using nwalkers = {self.nwalkers} (2 * ndim = {min_walkers})")

        self.p0 = self.flat_p0_guess 
        self.ep0 = np.zeros(len(self.p0)) + 0.01

        # Generate walker positions around the starting point
        p0s = np.random.multivariate_normal(self.p0, np.diag(self.ep0)**2, size=self.nwalkers)

        # Clip all walker positions to be within their prior ranges
        for i in range(len(self.prior_arr)):
            min_val, max_val = self.prior_arr[i]
            # Add a small buffer to avoid being exactly on the boundary
            buffer = 1e-7
            p0s[:, i] = np.clip(p0s[:, i], min_val + buffer, max_val - buffer)

        # Ensure all pstream spline points remain within (0, 1)
        pstream_dim = self.array_lengths[0]
        p0s[:, :pstream_dim] = np.clip(p0s[:, :pstream_dim], 1e-10, 1 - 1e-10)

        # Test likelihood for all walkers using the modified function
        print("Testing walker likelihoods...")
        lkhds = []
        failed_walkers = []
        
        for j in range(self.nwalkers):
            try:
                lkhd = stream_funcs.lnprob(
                    p0s[j], self.prior_arr, self.phi1_spline_points, 
                    self.stream.data.desi_data['VGSR'].values, self.stream.data.desi_data['VRAD_ERR'].values,
                    self.stream.data.desi_data['FEH'].values, self.stream.data.desi_data['FEH_ERR'].values,
                    self.stream.data.desi_data['PMRA'].values, self.stream.data.desi_data['PMRA_ERROR'].values,
                    self.stream.data.desi_data['PMDEC'].values, self.stream.data.desi_data['PMDEC_ERROR'].values,
                    self.stream.data.desi_data['phi1'].values, 
                    trunc_fit=True, assert_prior=True, feh_fit=True, k=self.spline_k, 
                    reshape_arr_shape=self.array_lengths,
                    vgsr_trunc=self.vgsr_trunc, feh_trunc=self.feh_trunc, 
                    pmra_trunc=self.pmra_trunc, pmdec_trunc=self.pmdec_trunc,
                    lsigpm_set=self.lsigpm_,
                    pstream_phi1_spline_points=self.pstream_phi1_spline_points,
                    lsigv_phi1_spline_points=self.lsigv_phi1_spline_points,
                    pstream_spline_k=self.spline_k_pstream,
                    lsigv_spline_k=self.spline_k_lsigv
                )
                lkhds.append(lkhd)
                if lkhd <= -9e9:
                    failed_walkers.append((j, lkhd, p0s[j]))
            except Exception as e:
                print(f"Walker {j} failed with error: {e}")
                lkhds.append(-np.inf)
                failed_walkers.append((j, -np.inf, p0s[j]))
        
        # Show details about failed walkers if any
        if failed_walkers:
            print(f"\nFailed walker details:")
            print("Walker | Likelihood | Parameters")
            print("-------|------------|------------")
            for walker_idx, lkhd, params in failed_walkers[:5]:  # Show first 5 failed walkers
                print(f"{walker_idx:6d} | {lkhd:10.2e} | {params[:3]}...")  # Show first 3 params
            
            # Check parameter ranges for the first failed walker
            if failed_walkers:
                print(f"\nDiagnosing parameter issues for walker {failed_walkers[0][0]}:")
                bad_params = failed_walkers[0][2]
                for i, (param_val, (low, high), label) in enumerate(zip(bad_params, self.prior_arr, self.param_labels)):
                    if not (low <= param_val <= high):
                        print(f"  {label}: {param_val:.4f} outside range ({low:.4f}, {high:.4f})")
                    elif abs(param_val - low) < 1e-6 or abs(param_val - high) < 1e-6:
                        print(f"  {label}: {param_val:.4f} too close to boundary ({low:.4f}, {high:.4f})")
            
            # Try testing the prior function directly
            print(f"\nTesting prior function directly on walker {failed_walkers[0][0]}:")
            try:
                prior_result = stream_funcs.lnprior(
                    failed_walkers[0][2], self.prior_arr, self.phi1_spline_points,
                    assert_prior=True, reshape_arr_shape=self.array_lengths
                )
                print(f"Prior result: {prior_result}")
            except Exception as e:
                print(f"Prior function failed: {e}")

        # Check if prior is good - this is the key test from your original code
        if sum(np.array(lkhds) > -9e9) == self.nwalkers:
            print('Your prior is good, you\'ve found something!')
        elif sum(np.array(lkhds) > -9e9) != self.nwalkers:
            print('Your prior is too restrictive, try changing the values listed above!')

        # Assert that all walkers have good likelihoods
        assert np.all(np.array(lkhds) > -9e9), f"Only {sum(np.array(lkhds) > -9e9)}/{self.nwalkers} walkers have valid likelihoods"

        print(f"All {self.nwalkers} walkers initialized successfully!")


    

class MCMC:
    """
    For running MCMC and intial outputs
    """
    def __init__(self, MCMeta_object, output_dir=''):
        
        self.meta = MCMeta_object
        self.lsigpm_ = self.meta.lsigpm_
        self.stream = self.meta.stream
        self.output_dir = output_dir
        self.backend = emcee.backends.HDFBackend(self.output_dir+'/'+self.stream.streamName+str(self.meta.no_of_spline_points)+'.h5')
        self.backend.reset(self.meta.nwalkers,len(self.meta.p0))
        # Book-keeping for runtime metrics
        self.mcmc_start_time = None
        self.mcmc_end_time = None
        self.mcmc_elapsed_seconds = None
        # Try to infer a human note from the output_dir naming convention used in notebooks
        try:
            base = os.path.basename(os.path.abspath(self.output_dir.rstrip('/')))
            # Expect pattern like: <streamName>_<yymmdd or yyyymmdd>_<note>
            m = re.search(r"_\d{6,8}_(.+)$", base)
            if m:
                self.note = m.group(1)
        except Exception:
            pass

    @property
    def lsigpm_(self):
        """Expose the PM dispersion toggle directly from MCMeta."""
        return getattr(self.meta, 'lsigpm_', None)

    @lsigpm_.setter
    def lsigpm_(self, value):
        setattr(self.meta, 'lsigpm_', value)
    #WIP
    def run(self, nproc=32, nsteps=10000, use_optimized_start=True, **kwargs):
        """
        Run a single continuous MCMC chain for `nsteps` iterations. Burn-in can
        be applied later using `apply_burnin(discard, thin)`.

        Backward compatibility: if called with legacy keywords `nburnin` and
        `nstep`, we will ignore the two-phase pattern and instead run for
        `nsteps = nburnin + nstep`.
        """
        from multiprocessing import Pool
        self.nproc = nproc
        self.use_optimized_start = use_optimized_start

        # Back-compat: allow old signature nburnin+nstep
        if 'nburnin' in kwargs or 'nstep' in kwargs:
            nburnin_legacy = int(kwargs.get('nburnin', 0) or 0)
            nstep_legacy = int(kwargs.get('nstep', 0) or 0)
            nsteps = nburnin_legacy + nstep_legacy if (nburnin_legacy + nstep_legacy) > 0 else nsteps
            print(f"[MCMC] Back-compat: using nsteps = nburnin({nburnin_legacy}) + nstep({nstep_legacy}) = {nsteps}")

        self.nsteps = int(nsteps)
        if self.use_optimized_start:
            print("Using optimized parameters as starting positions...")
            start_params = self.meta.sp_result.x
            start_label = "optimized"
        else:
            print("Using initial guess as starting positions...")
            start_params = self.meta.flat_p0_guess
            start_label = "initial_guess"

        with Pool(self.nproc) as pool:
            p0 = start_params
            ep0 = np.zeros(len(p0)) + 0.01
            assert np.all(np.isfinite(start_params)), "start_params contains NaN or inf"
            # Generate walker positions around the starting point
            p0s = np.random.multivariate_normal(p0, np.diag(ep0)**2, size=self.meta.nwalkers)

            print("Clipping all walker positions to be within prior ranges...")
            for i in range(len(self.meta.prior_arr)):
                min_val, max_val = self.meta.prior_arr[i]
                # Add a small buffer to avoid being exactly on the boundary
                buffer = 1e-10
                p0s[:, i] = np.clip(p0s[:, i], min_val + buffer, max_val - buffer)

            # Special clipping for pstream to [0, 1] across its parameter group
            pstream_dim = self.meta.array_lengths[0]
            p0s[:, :pstream_dim] = np.clip(p0s[:, :pstream_dim], 1e-10, 1 - 1e-10)
                
            start = time.time()
            self.mcmc_start_time = start
            print(f"Running a single continuous chain for {self.nsteps} iterations starting from {start_label} parameters...")
            # Use top-level lnprob and pass fixed PM sigma as a positional arg for pickling across processes
            es = emcee.EnsembleSampler(
                self.meta.nwalkers, len(self.meta.flat_p0_guess), stream_funcs.lnprob,
                args=(self.meta.prior_arr, self.meta.phi1_spline_points, 
                    self.meta.stream.data.desi_data['VGSR'].values, self.meta.stream.data.desi_data['VRAD_ERR'].values,
                    self.meta.stream.data.desi_data['FEH'].values, self.meta.stream.data.desi_data['FEH_ERR'].values,
                    self.meta.stream.data.desi_data['PMRA'].values, self.meta.stream.data.desi_data['PMRA_ERROR'].values,
                    self.meta.stream.data.desi_data['PMDEC'].values, self.meta.stream.data.desi_data['PMDEC_ERROR'].values,
                    self.meta.stream.data.desi_data['phi1'].values, 
                    True,  # trunc_fit
                    False,  # assert_prior
                    True, # feh_fit
                    self.meta.spline_k, self.meta.array_lengths,
                    self.meta.vgsr_trunc, self.meta.feh_trunc, self.meta.pmra_trunc, self.meta.pmdec_trunc,
                    self.lsigpm_,  # lsigpm_set as positional arg to lnprob
                    ),
                kwargs={
                    'pstream_phi1_spline_points': self.meta.pstream_phi1_spline_points,
                    'lsigv_phi1_spline_points': self.meta.lsigv_phi1_spline_points,
                    'pstream_spline_k': self.meta.spline_k_pstream,
                    'lsigv_spline_k': self.meta.spline_k_lsigv,
                },
                pool=pool, backend=self.backend)
            
            # Run MCMC with progress bar
            with tqdm(total=self.nsteps, desc="MCMC Progress", unit="step", 
                     bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]') as pbar:
                for sample in es.sample(p0s, iterations=self.nsteps, progress=False):
                    pbar.update(1)
                    # Update progress bar with acceptance fraction every 100 steps
                    if pbar.n % 100 == 0:
                        try:
                            acc_frac = np.mean(es.acceptance_fraction)
                            pbar.set_postfix({'acc_frac': f'{acc_frac:.3f}'})
                        except:
                            pass
            
            self.mcmc_end_time = time.time()
            self.mcmc_elapsed_seconds = float(self.mcmc_end_time - start)
            print(f'Took {self.mcmc_elapsed_seconds:.1f} seconds ({self.mcmc_elapsed_seconds/60:.1f} minutes)')
            # Store chains in the shape expected by show_chains(): (nwalkers, nsteps, ndim)
            try:
                chain_arr = es.get_chain()  # (nsteps, nwalkers, ndim)
                self.chain = np.swapaxes(chain_arr, 0, 1)
            except Exception:
                # Fallback to legacy attributes if available
                self.chain = getattr(es, 'chain', None)

            print('Computing flatchain (no burn-in discarded; use apply_burnin to change)...')
            try:
                self.flatchain = es.get_chain(flat=True)
            except Exception:
                self.flatchain = getattr(es, 'flatchain', None)

            # Track current discard/thin used to compute flatchain
            self.current_discard = 0
            self.current_thin = 1

    def apply_burnin(self, discard=0, thin=1):
        """
        Apply burn-in and thinning post-hoc to set `self.flatchain` from the
        stored backend; updates `current_discard` and `current_thin`.
        """
        try:
            # Access the sampler via the backend
            if hasattr(self, 'backend'):
                # emcee provides a convenience method on the backend
                chain_arr = self.backend.get_chain(discard=discard, thin=thin, flat=True)
            else:
                raise AttributeError('No backend found on MCMC object')
        except Exception:
            # Fallback: if we have the full non-flat chain in self.chain
            chain_local = getattr(self, 'chain', None)
            if chain_local is None:
                raise RuntimeError('No chain available to apply burn-in to.')
            # self.chain is (nwalkers, nsteps, ndim); apply discard/thin
            chain_swapped = np.swapaxes(chain_local, 0, 1)  # (nsteps, nwalkers, ndim)
            chain_cut = chain_swapped[discard::thin]
            chain_arr = chain_cut.reshape(-1, chain_cut.shape[-1])

        self.flatchain = chain_arr
        self.current_discard = int(discard)
        self.current_thin = int(thin)
        print(f"Applied burn-in: discard={self.current_discard}, thin={self.current_thin}. Flatchain shape: {self.flatchain.shape}")

    def _build_expanded_param_labels(self):
        """Return parameter labels that exactly match the flattened chain length."""

        array_lengths = list(getattr(self.meta, 'array_lengths', []))
        labels: list[str] = []

        # Helper to consume the next length from array_lengths with validation
        idx = 0

        def pop_length(expected_label: str | None = None) -> int:
            nonlocal idx
            if idx >= len(array_lengths):
                raise ValueError(
                    "array_lengths is shorter than expected when constructing parameter labels"
                )
            length = int(array_lengths[idx])
            idx += 1
            if length < 1:
                raise ValueError(
                    f"Invalid array length {length} for parameter group"
                    + (f" '{expected_label}'" if expected_label else '')
                )
            return length

        def extend_series(prefix: str, count: int, *, single_label: str | None = None):
            if count == 1:
                label = single_label if single_label is not None else f'{prefix}1'
                labels.append(label)
            else:
                for i in range(1, count + 1):
                    labels.append(f'{prefix}{i}')

        if array_lengths:
            # pstream block
            pstream_len = pop_length('pstream')
            extend_series('pstream', pstream_len, single_label='pstream')

            # VGSR spline points
            vgsr_len = pop_length('vgsr')
            extend_series('vgsr', vgsr_len)

            # lsigv block (log sigma of VGSR)
            lsigv_len = pop_length('lsigvgsr')
            if lsigv_len == 1:
                labels.append('lsigvgsr')
            else:
                labels.append('lsigvgsr')
                for i in range(2, lsigv_len + 1):
                    labels.append(f'lsigvgsr{i}')

            # Fixed-length scalar parameters
            labels.append('feh1')
            if pop_length('feh1') != 1:
                raise ValueError('Expected feh1 to have exactly one element')
            labels.append('lsigfeh')
            if pop_length('lsigfeh') != 1:
                raise ValueError('Expected lsigfeh to have exactly one element')

            # PMRA spline points
            pmra_len = pop_length('pmra')
            extend_series('pmra', pmra_len)

            if self.lsigpm_ is None:
                lsigpmra_len = pop_length('lsigpmra')
                if lsigpmra_len != 1:
                    raise ValueError('Expected lsigpmra to have exactly one element')
                labels.append('lsigpmra')

            # PMDEC spline points
            pmdec_len = pop_length('pmdec')
            extend_series('pmdec', pmdec_len)

            if self.lsigpm_ is None:
                lsigpmdec_len = pop_length('lsigpmdec')
                if lsigpmdec_len != 1:
                    raise ValueError('Expected lsigpmdec to have exactly one element')
                labels.append('lsigpmdec')

            # Background scalars (order matches MCMeta.p0_guess)
            bg_labels = ['bv', 'lsigbv', 'bfeh', 'lsigbfeh', 'bpmra', 'lsigbpmra', 'bpmdec', 'lsigbpmdec']
            for name in bg_labels:
                if pop_length(name) != 1:
                    raise ValueError(f"Expected {name} to have exactly one element")
                labels.append(name)

            if idx != len(array_lengths):
                raise ValueError(
                    f"array_lengths has {len(array_lengths)} entries but only {idx} were consumed while "
                    "building labels"
                )

            return labels

        # Fallback: if array_lengths is unavailable, use metadata counts
        fallback = []

        if getattr(self.meta, 'pstream_no_of_spline_points', 1) <= 1:
            fallback.append('pstream')
        else:
            fallback.extend(
                f'pstream{i}' for i in range(1, self.meta.pstream_no_of_spline_points + 1)
            )

        fallback.extend(f'vgsr{i}' for i in range(1, self.meta.no_of_spline_points + 1))

        if getattr(self.meta, 'lsigv_no_of_spline_points', 1) <= 1:
            fallback.append('lsigvgsr')
        else:
            fallback.append('lsigvgsr')
            fallback.extend(
                f'lsigvgsr{i}' for i in range(2, self.meta.lsigv_no_of_spline_points + 1)
            )

        fallback.extend(['feh1', 'lsigfeh'])
        fallback.extend(f'pmra{i}' for i in range(1, self.meta.no_of_spline_points + 1))
        if self.lsigpm_ is None:
            fallback.append('lsigpmra')
        fallback.extend(f'pmdec{i}' for i in range(1, self.meta.no_of_spline_points + 1))
        if self.lsigpm_ is None:
            fallback.append('lsigpmdec')
        fallback.extend(['bv', 'lsigbv', 'bfeh', 'lsigbfeh', 'bpmra', 'lsigbpmra', 'bpmdec', 'lsigbpmdec'])

        return fallback

    def show_chains(self, trimmed=False):
        # Build a label list that matches the flattened parameter order exactly
        self.expanded_param_labels = self._build_expanded_param_labels()

        # Fetch raw or trimmed chain from backend/in-memory
        if trimmed:
            discard = int(getattr(self, 'current_discard', 0))
            thin = int(getattr(self, 'current_thin', 1))
            chain = None
            try:
                # (nsteps_trim, nwalkers, ndim)
                chain_backend = self.backend.get_chain(discard=discard, thin=thin, flat=False)
                chain = np.swapaxes(chain_backend, 0, 1)  # -> (nwalkers, nsteps_trim, ndim)
            except Exception:
                pass
            if chain is None:
                chain_local = getattr(self, 'chain', None)
                if chain_local is None:
                    raise RuntimeError('No MCMC chain available. Run mcmc.run() first.')
                chain_swapped = np.swapaxes(chain_local, 0, 1)  # (nsteps, nwalkers, ndim)
                chain_cut = chain_swapped[discard::thin]
                chain = np.swapaxes(chain_cut, 0, 1)  # back to (nwalkers, nsteps_trim, ndim)
        else:
            chain = None
            try:
                # (nsteps, nwalkers, ndim)
                chain_backend = self.backend.get_chain(flat=False)
                chain = np.swapaxes(chain_backend, 0, 1)  # -> (nwalkers, nsteps, ndim)
            except Exception:
                pass
            if chain is None:
                chain = getattr(self, 'chain', None)
            if chain is None:
                raise RuntimeError('No MCMC chain available. Run mcmc.run() first.')

        chain = np.asarray(chain)
        Nrow = chain.shape[2]
        fig, axes = plt.subplots(Nrow, figsize=(6, 2 * Nrow))
        if Nrow == 1:
            axes = [axes]

        for iparam, ax in enumerate(axes):
            for j in range(self.meta.nwalkers):
                ax.plot(chain[j, :, iparam], lw=.5, alpha=.2)
            ax.set_ylabel(self.expanded_param_labels[iparam])

        fig.tight_layout()

    def show_corner(self):
        # Prefer any user-filtered flatchain (from apply_burnin), else fallback to backend
        flatchain = getattr(self, 'flatchain', None)
        if flatchain is None:
            try:
                flatchain = self.backend.get_chain(flat=True)
            except Exception:
                pass
        if flatchain is None:
            raise RuntimeError('No flatchain available. Run mcmc.run() first.')
        _ = flatchain.shape  # ensure it's an ndarray
        fig = corner.corner(flatchain, labels=self.expanded_param_labels, quantiles=[0.16,0.50,0.84], show_titles=True)
        
    def show_stream_corner(self, include_pstream=True, transform_logs=True, corner_kwargs=None):
        """
        Corner plot for stream-only parameters.

        - include_pstream: include 'pstream' as the first parameter (default True)
        - transform_logs: convert any 'lsig*' parameters to linear space via 10** (default True)
        - corner_kwargs: optional dict passed to corner.corner()
        """
        # Acquire flatchain
        flatchain = getattr(self, 'flatchain', None)
        if flatchain is None:
            try:
                flatchain = self.backend.get_chain(flat=True)
            except Exception:
                pass
        if flatchain is None:
            raise RuntimeError('No flatchain available. Run mcmc.run() first.')

        # Build expanded labels to match flattened parameter order
        self.expanded_param_labels = self._build_expanded_param_labels()

        # Select stream-only labels (and indices)
        labels = list(self.expanded_param_labels)
        def _take(names):
            return [labels.index(n) for n in names]

        n = self.meta.no_of_spline_points
        stream_labels = []
        if include_pstream:
            stream_labels.extend([lbl for lbl in labels if lbl.startswith('pstream')])
        stream_labels += [f'vgsr{i}' for i in range(1, n+1)]
        stream_labels += [lbl for lbl in labels if lbl.startswith('lsigvgsr')]
        stream_labels += ['feh1', 'lsigfeh']
        stream_labels += [f'pmra{i}' for i in range(1, n+1)]
        if self.lsigpm_ is None:
            stream_labels += ['lsigpmra']
        stream_labels += [f'pmdec{i}' for i in range(1, n+1)]
        if self.lsigpm_ is None:
            stream_labels += ['lsigpmdec']

        # Defensive: keep only labels that actually exist
        stream_labels = [lbl for lbl in stream_labels if lbl in labels]
        idxs = _take(stream_labels)

        # Slice the chain
        chain_sel = flatchain[:, idxs]

        # Optionally transform log-dispersion parameters to linear space for readability
        if transform_logs:
            for j, lbl in enumerate(stream_labels):
                if lbl.startswith('lsig'):
                    chain_sel[:, j] = 10 ** chain_sel[:, j]

        # Prepare defaults and invoke corner
        if corner_kwargs is None:
            corner_kwargs = {}
        corner_kwargs.setdefault('labels', stream_labels)
        corner_kwargs.setdefault('quantiles', [0.16, 0.50, 0.84])
        corner_kwargs.setdefault('show_titles', True)

        fig = corner.corner(chain_sel, **corner_kwargs)
        return fig


    def print_result(self):
        # Ensure labels exist (in case user didn't call show_chains first)
        if not hasattr(self, 'expanded_param_labels'):
            self.expanded_param_labels = self._build_expanded_param_labels()
        # Ensure flatchain is available; prefer backend with current burn-in settings if any
        if getattr(self, 'flatchain', None) is None:
            try:
                self.flatchain = self.backend.get_chain(flat=True)
            except Exception:
                raise RuntimeError('No flatchain available. Run mcmc.run() first.')

        result = stream_funcs.process_chain(self.flatchain, labels = self.expanded_param_labels)
        self.result = result
        if len(result) == 2:
            self.meds, self.errs = result
        else:
            self.meds, self.errs, _ = result
        print(len(self.meds))
        print(self.meds)
        
        exp_flatchain = np.copy(self.flatchain)
        for i, label in enumerate(self.meds.keys()):
            if label[0] == 'l':
                exp_flatchain[:,i]= 10 ** exp_flatchain[:,i]
        result = stream_funcs.process_chain(exp_flatchain, labels = self.expanded_param_labels)
        if len(result) == 2:
            self.exp_meds, self.exp_errs = result
        else:
            self.exp_meds, self.exp_errs, _ = result
            
        result = stream_funcs.process_chain(self.flatchain, avg_error=False, labels = self.expanded_param_labels)
        if len(result) == 2:
            _, self.ep = result
            self.em = None
        else:
            _, self.ep, self.em = result
            
        exp_flatchain = np.copy(self.flatchain)
        for i, label in enumerate(self.meds.keys()):
            if label[0] == 'l':
                exp_flatchain[:,i]= 10 ** exp_flatchain[:,i]
        result = stream_funcs.process_chain(exp_flatchain, avg_error=False, labels = self.expanded_param_labels)
        if len(result) == 2:
            _, self.exp_ep = result
            self.exp_em = None
        else:
            _, self.exp_ep, self.exp_em = result

        i = 0
        # print("{:<10} {:>10} {:>10} {:>10} {:>10}".format('param','med','err','exp(med)','exp(err)'))
        print("{:<10} {:>10} {:>10} {:>10} {:>10} {:>10} {:>10}".format('param','med', 'em','ep','exp(med)', 'exp(em)','exp(ep)'))
        print('--------------------------------------------------------------------------------------')
        em_dict = self.em if isinstance(self.em, dict) else {}
        ep_dict = self.ep if isinstance(self.ep, dict) else {}
        exp_em_dict = self.exp_em if isinstance(self.exp_em, dict) else {}
        exp_ep_dict = self.exp_ep if isinstance(self.exp_ep, dict) else {}
        for label,v in self.meds.items():
            # if label[:8] == 'lpstream':
            #     print("{:<10} {:>10.3f} {:>10.3f} {:>10.5f} {:>10.5f}".format(label,v,errs[label], np.e**v, np.log(10)*(np.e**v)*errs[label]))
            if label[0] == 'l':
                # print("{:<10} {:>10.3f} {:>10.3f} {:>10.3f} {:>10.3f} ".format(label,v,errs[label], exp_meds[label], exp_errs[label]))
                print("{:<10} {:>10.3f} {:>10.3f} {:>10.3f} {:>10.3f} {:>10.3f} {:>10.3f}".format(
                    label,
                    v,
                    em_dict.get(label, np.nan),
                    ep_dict.get(label, np.nan),
                    self.exp_meds[label],
                    exp_em_dict.get(label, np.nan),
                    exp_ep_dict.get(label, np.nan)
                ))
            else:
                print("{:<10} {:>10.3f} {:>10.3f} {:>10.3f}".format(label, v, em_dict.get(label, np.nan), ep_dict.get(label, np.nan)))
            i += 1

    def memprob(self, N_samples = 0):
        #Calculate membership probabilities using the new memprob function
        # Get the data from the stream object that was optimized
        data = self.stream.data.desi_data

        # Extract the relevant parameters from the MCMC results
        if N_samples == 0:
            print('not sampling posterior, using median parameters')
            theta_final = list(self.meds.values())
                    # Calculate membership probabilities
            stream_prob = stream_funcs.memprob(
                theta=theta_final,
                prior=self.meta.prior_arr,
                spline_x_points=self.meta.phi1_spline_points,
                vgsr=data['VGSR'].values,
                vgsr_err=data['VRAD_ERR'].values,
                feh=data['FEH'].values,
                feh_err=data['FEH_ERR'].values,
                pmra=data['PMRA'].values,
                pmra_err=data['PMRA_ERROR'].values,
                pmdec=data['PMDEC'].values,
                pmdec_err=data['PMDEC_ERROR'].values,
                phi1=data['phi1'].values,
                trunc_fit=True,  # Use truncated fitting as in your setup
                assert_prior=False,
                feh_fit=True,
                k=self.meta.spline_k,
                reshape_arr_shape=self.meta.array_lengths,
                vgsr_trunc=self.meta.vgsr_trunc,
                feh_trunc=self.meta.feh_trunc,
                pmra_trunc=self.meta.pmra_trunc,
                pmdec_trunc=self.meta.pmdec_trunc,
                lsigpm_set=self.lsigpm_
            )
            stream_prob_em = None
            stream_prob_ep = None
        else:  # Use the median parameters from MCMC
            print(f'sampling the posterior with {N_samples} ')
            n_total = self.flatchain.shape[0]
            sample_indices = np.random.choice(n_total, size=min(N_samples, n_total), replace=False)
            data = self.stream.data.desi_data

            prob_samples = np.zeros((len(sample_indices), len(data)))
    
            # Compute membership probability for each sampled parameter set
            for i, idx in enumerate(sample_indices):
                theta_sample = self.flatchain[idx]
                
                prob_samples[i] = stream_funcs.memprob(
                    theta=theta_sample,
                    prior=self.meta.prior_arr,
                    spline_x_points=self.meta.phi1_spline_points,
                    vgsr=data['VGSR'].values,
                    vgsr_err=data['VRAD_ERR'].values,
                    feh=data['FEH'].values,
                    feh_err=data['FEH_ERR'].values,
                    pmra=data['PMRA'].values,
                    pmra_err=data['PMRA_ERROR'].values,
                    pmdec=data['PMDEC'].values,
                    pmdec_err=data['PMDEC_ERROR'].values,
                    phi1=data['phi1'].values,
                    trunc_fit=True,
                    assert_prior=False,
                    feh_fit=True,
                    k=self.meta.spline_k,
                    reshape_arr_shape=self.meta.array_lengths,
                    vgsr_trunc=self.meta.vgsr_trunc,
                    feh_trunc=self.meta.feh_trunc,
                    pmra_trunc=self.meta.pmra_trunc,
                    pmdec_trunc=self.meta.pmdec_trunc,
                    lsigpm_set=self.lsigpm_
                )
            
            # Take median across samples for each star
            stream_prob = np.median(prob_samples, axis=0)
            stream_prob_em = np.percentile(prob_samples, 16, axis=0)   # Lower 95% CI
            stream_prob_ep = np.percentile(prob_samples, 84, axis=0) 

        print(f"Calculated membership probabilities for {len(stream_prob)} stars")
        print(f"Membership probabilities range from {np.min(stream_prob):.3f} to {np.max(stream_prob):.3f}")
        print(f"Mean membership probability: {np.mean(stream_prob):.3f}")
        print(f"Stars with >50% probability: {len(stream_prob[stream_prob > 0.5])}")
        print(f"Stars with >70% probability: {len(stream_prob[stream_prob > 0.7])}")
        print(f"Stars with >90% probability: {len(stream_prob[stream_prob > 0.9])}")
        
        self.stream_prob = stream_prob
        self.stream_prob_em = stream_prob_em
        self.stream_prob_ep = stream_prob_ep

        # Create a copy of the original data and add the new columns
        result_data = self.stream.data.desi_data.copy()
        result_data['stream_prob'] = stream_prob
        result_data['stream_prob_em'] = stream_prob_em if stream_prob_em is not None else np.nan
        result_data['stream_prob_ep'] = stream_prob_ep if stream_prob_ep is not None else np.nan

        return result_data
    
    def save_run(self, note=None):
        """
        Save MCMC results and membership probabilities to files.
        This method saves various outputs from the MCMC run including chains,
        parameters, and high-probability stream members.

        If `note` is not provided, attempts to use `self.note` or `self.meta.note`.
        If those are unset, tries to parse a trailing note from the output directory
        name following the notebook convention: runs/<stream>_<date>_<note>.
        """
        # Ensure output directory ends with a slash for consistency
        outdir = self.output_dir.rstrip('/') 

        with open(outdir + '/isochrone_path.txt', 'w') as f:
            f.write(getattr(self.stream, 'isochrone_path', ''))

        mcmc_dict = {
            "flatchain": self.flatchain,
            "extended_param_labels": self.expanded_param_labels,
        }

        importlib.reload(stream_funcs)
        theta_final = []
        exp_theta_final = []
        errs_list = []
        exp_errs_list = []
        ep_list = []
        em_list = []
        exp_ep_list = []
        exp_em_list = []

        for label, i in self.meds.items():
            theta_final.append(i)
            exp_theta_final.append(self.exp_meds[label])
            errs_list.append(self.errs[label])
            exp_errs_list.append(self.exp_errs[label])
            # Use safe dict access in case certain diagnostics aren't present
            ep_list.append((self.ep if isinstance(self.ep, dict) else {}).get(label, np.nan))
            em_list.append(abs((self.em if isinstance(self.em, dict) else {}).get(label, np.nan)))
            exp_ep_list.append((self.exp_ep if isinstance(self.exp_ep, dict) else {}).get(label, np.nan))
            exp_em_list.append(abs((self.exp_em if isinstance(self.exp_em, dict) else {}).get(label, np.nan)))

        nested_list_meds = stream_funcs.reshape_arr(theta_final, self.meta.array_lengths)
        nested_list_exp_meds = stream_funcs.reshape_arr(exp_theta_final, self.meta.array_lengths)
        nested_list_errs = stream_funcs.reshape_arr(errs_list, self.meta.array_lengths)
        nested_list_exp_errs = stream_funcs.reshape_arr(exp_errs_list, self.meta.array_lengths)
        nested_list_ep = stream_funcs.reshape_arr(ep_list, self.meta.array_lengths)
        nested_list_em = stream_funcs.reshape_arr(em_list, self.meta.array_lengths)
        nested_list_exp_ep = stream_funcs.reshape_arr(exp_ep_list, self.meta.array_lengths)
        nested_list_exp_em = stream_funcs.reshape_arr(exp_em_list, self.meta.array_lengths)

        nested_dict = {
            "meds": nested_list_meds,
            "exp_meds": nested_list_exp_meds,
            "errs": nested_list_errs,
            "exp_errs": nested_list_exp_errs,
            "ep": nested_list_ep,
            "em": nested_list_em,
            "exp_ep": nested_list_exp_ep,
            "exp_em": nested_list_exp_em,
            "array_lengths": self.meta.array_lengths,
            "param_labels": self.meta.param_labels,
            "expanded_param_labels": self.expanded_param_labels,
            # Save fixed PM dispersion if used during MCMC (log10 units); else None
            "lsig_pm_fixed_log10": (self.lsigpm_ if getattr(self, 'lsigpm_', None) is not None else None),
        }

        np.save(f'{outdir}/mcmc_dict.npy', mcmc_dict)
        np.save(f'{outdir}/nested_dict.npy', nested_dict)
        np.savetxt(f'{outdir}/{self.stream.streamName}_{getattr(self, "phi2_wiggle", "default")}.txt', np.array(theta_final))

        # Save spline points dict (user requested)
        try:
            phi1_spline_points = getattr(self.meta, 'phi1_spline_points', None)
            # Prefer explicitly provided variants, else fall back to phi1_spline_points
            pstream_phi1_spline_points = getattr(self.meta, 'pstream_phi1_spline_points', None)
            if pstream_phi1_spline_points is None:
                pstream_phi1_spline_points = phi1_spline_points

            lsigv_phi1_spline_points = getattr(self.meta, 'lsigv_phi1_spline_points', None)
            if lsigv_phi1_spline_points is None:
                lsigv_phi1_spline_points = phi1_spline_points

            spline_k = getattr(self.meta, 'spline_k', None)
            spline_k_lsigv = getattr(self.meta, 'spline_k_lsigv', spline_k)
            spline_k_pstream = getattr(self.meta, 'spline_k_pstream', spline_k)

            spline_points_dict = {
                'phi1_spline_points': phi1_spline_points,
                'pstream_phi1_spline_points': pstream_phi1_spline_points,
                'lsigv_phi1_spline_points': lsigv_phi1_spline_points,
                'spline_k': spline_k,
                'spline_k_lsigv': spline_k_lsigv,
                'spline_k_pstream': spline_k_pstream
            }

            np.save(f'{outdir}/spline_points_dict.npy', spline_points_dict)
        except Exception as e:
            print(f"Warning: could not save spline_points_dict: {e}")

        # Calculate membership probabilities if not already done
        if not hasattr(self, 'stream_prob'):
            self.stream_prob = self.memprob()

        dataframe = self.stream.data.desi_data.copy()
        dataframe['stream_prob'] = self.stream_prob
        dataframe['stream_prob_em'] = self.stream_prob_em if self.stream_prob_em is not None else np.nan
        dataframe['stream_prob_ep'] = self.stream_prob_ep if self.stream_prob_ep is not None else np.nan

        # Default minimum probability threshold
        min_prob = getattr(self, 'min_prob', 0.5)

        # Save high-probability members (above min_prob threshold)
        high_prob_mask = self.stream_prob >= min_prob
        high_prob_dataframe = dataframe[high_prob_mask]
        high_prob_table = Table.from_pandas(high_prob_dataframe)
        output_path = f'{outdir}/{self.stream.streamName}_phi2_spline_{int(min_prob*100)}%_mem.fits'
        high_prob_table.write(output_path, format='fits', overwrite=True)
        print(f"Saved {len(high_prob_dataframe)} high-probability members to: {output_path}")

        # Save all stars with membership probabilities
        all_table = Table.from_pandas(dataframe)
        output_path = f'{outdir}/{self.stream.streamName}_phi2_spline_all%_mem.fits'
        all_table.write(output_path, format='fits', overwrite=True)
        print(f"Saved {len(dataframe)} total stars to: {output_path}")

        # ------------------
        # Write metrics.txt
        # ------------------
        try:
            # Count members at different thresholds
            n_members_05 = int(np.sum(self.stream_prob >= 0.5))
            n_members_07 = int(np.sum(self.stream_prob >= 0.7))
            n_members_09 = int(np.sum(self.stream_prob >= 0.9))
            n_total_modeled = int(len(dataframe))

            # Time spent on MCMC
            elapsed_sec = getattr(self, 'mcmc_elapsed_seconds', None)
            elapsed_min = (elapsed_sec / 60.0) if elapsed_sec is not None else None

            # Output dir (absolute) and isochrone path/name
            outdir_abs = os.path.abspath(outdir)
            iso_path = getattr(self.stream, 'isochrone_path', '') or ''
            iso_name = os.path.basename(iso_path) if iso_path else ''

            # Spline points used and their locations
            phi1_spline_points = getattr(self.meta, 'phi1_spline_points', None)
            pstream_phi1_spline_points = getattr(self.meta, 'pstream_phi1_spline_points', None)
            lsigv_phi1_spline_points = getattr(self.meta, 'lsigv_phi1_spline_points', None)

            def fmt_array(arr):
                if arr is None:
                    return '[]'
                arr = np.asarray(arr, dtype=float)
                return np.array2string(arr, precision=4, separator=', ')

            # Count SF stars above 0.5 memprob (SF∩DESI subset)
            n_sf_above_05 = 0
            if hasattr(self.stream.data, 'confirmed_sf_and_desi') and len(self.stream.data.confirmed_sf_and_desi) > 0:
                sf_ids = set(self.stream.data.confirmed_sf_and_desi['SOURCE_ID']) if 'SOURCE_ID' in self.stream.data.confirmed_sf_and_desi.columns else set()
                if 'SOURCE_ID' in dataframe.columns:
                    mask_sf = dataframe['SOURCE_ID'].isin(sf_ids)
                    n_sf_above_05 = int(np.sum((dataframe.loc[mask_sf, 'stream_prob'] >= 0.5)))

            # Optional note: prefer explicit arg, then attributes, then parse from outdir
            note_val = note or getattr(self, 'note', None) or getattr(self.meta, 'note', None)
            if not note_val:
                try:
                    base = os.path.basename(outdir_abs)
                    m = re.search(r"_\d{6,8}_(.+)$", base)
                    if m:
                        note_val = m.group(1)
                except Exception:
                    note_val = ''

            metrics_lines = [
                f"note: {note_val}",
                f"members>=0.5: {n_members_05}",
                f"members>=0.7: {n_members_07}",
                f"members>=0.9: {n_members_09}",
                f"total_modeled: {n_total_modeled}",
                f"mcmc_elapsed_sec: {elapsed_sec if elapsed_sec is not None else 'NA'}",
                f"mcmc_elapsed_min: {elapsed_min if elapsed_min is not None else 'NA'}",
                f"output_dir: {outdir_abs}",
                f"isochrone_name: {iso_name}",
                f"isochrone_path: {iso_path}",
                f"vgsr_phi1_spline_points_count: {len(phi1_spline_points) if phi1_spline_points is not None else 0}",
                f"vgsr_phi1_spline_points: {fmt_array(phi1_spline_points)}",
                f"pstream_phi1_spline_points_count: {len(pstream_phi1_spline_points) if pstream_phi1_spline_points is not None else 0}",
                f"pstream_phi1_spline_points: {fmt_array(pstream_phi1_spline_points)}",
                f"lsigv_phi1_spline_points_count: {len(lsigv_phi1_spline_points) if lsigv_phi1_spline_points is not None else 0}",
                f"lsigv_phi1_spline_points: {fmt_array(lsigv_phi1_spline_points)}",
                f"sf_members_above_0.5: {n_sf_above_05}",
            ]

            with open(f"{outdir}/metrics.txt", 'w') as f:
                f.write("\n".join(metrics_lines) + "\n")
            print(f"Saved run metrics to: {outdir}/metrics.txt")
        except Exception as e:
            print(f"Warning: failed to write metrics.txt due to: {e}")
