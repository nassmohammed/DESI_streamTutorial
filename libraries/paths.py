"""
Central path resolver for DESI_streamTutorial.

Repo-bundled data (sf3_only_table.csv, streamfinder_gaiadr3.fits, dotter/)
is resolved automatically from this file's location — no user action required.

External data (DESI catalogues, DECaLS, RVS distances) is read from
config.yaml at the project root. To override for a local machine, create
config.local.yaml (copy config.yaml and edit it). That file is gitignored.

Usage:
    from libraries.paths import PATHS
    Data = st.Data(PATHS.desi_path, sf_path=PATHS.sf_path, ...)
    # Override a single path without editing config files:
    # PATHS.desi_path = Path('/your/local/copy.fits')
"""

from pathlib import Path
import yaml

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_CONFIG_FILE  = _PROJECT_ROOT / "config.yaml"
_LOCAL_CONFIG = _PROJECT_ROOT / "config.local.yaml"


def _load_external():
    with open(_CONFIG_FILE) as f:
        cfg = yaml.safe_load(f)
    ext = cfg.get("external_data", {})
    if _LOCAL_CONFIG.exists():
        with open(_LOCAL_CONFIG) as f:
            local = yaml.safe_load(f) or {}
        ext.update(local.get("external_data", {}))
    return ext


class _Paths:
    def __init__(self):
        ext = _load_external()

        # ── Repo-bundled data (always works after git clone, no config needed) ──
        self.sf3_table  = _PROJECT_ROOT / "data" / "sf3_only_table.csv"
        self.dotter_dir = _PROJECT_ROOT / "data" / "dotter"
        self.sf_bundled = _PROJECT_ROOT / "data" / "streamfinder_gaiadr3.fits"

        # ── External data (set in config.yaml / config.local.yaml) ──────────────
        self.desi_path   = Path(ext["desi_path"])
        self.sf_path     = Path(ext["sf_path"])
        self.dist_path   = Path(ext["dist_path"])
        self.decals_path = Path(ext["decals_path"])

        # Optional LOA-specific catalogues (empty string in config → None)
        self.bhb_path = Path(ext["bhb_path"]) if ext.get("bhb_path") else None
        self.rrl_path = Path(ext["rrl_path"]) if ext.get("rrl_path") else None


PATHS = _Paths()
