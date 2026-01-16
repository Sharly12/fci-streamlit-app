# utils/data_loader.py
import os
from pathlib import Path
import tempfile
import requests
import rasterio
from rasterio.warp import reproject, Resampling as RioResampling
from pysheds.grid import Grid
import geopandas as gpd
import streamlit as st

# --------------------------------------------------------------------
# Dropbox fallbacks (only used if local files not found)
# --------------------------------------------------------------------
DEM_URL = (
    "https://www.dropbox.com/scl/fi/xc8lovzfem79wx1i6z8pi/mosaic.tif?rlkey=0fj002ryj0bpsjx6ezit04a3q&st=kkgvet4b&dl=1"
    
)
CN_URL = (
    "https://www.dropbox.com/scl/fi/xfseghib9vg31loxan294/"
    "CN.tif?rlkey=6o75z9l36l8viuiivxmgiame7&st=e7jfq1vi&dl=1"
)
PARCELS_URL = (
    "https://www.dropbox.com/scl/fi/dv1br78ds9mz7zdtwc1sr/"
    "grid-network.geojson?rlkey=rbq4kyi8u9nl4byzz7wu6rkq4&st=qiyefkny&dl=1"
)
LU_URL = (
    "https://www.dropbox.com/scl/fi/0529835pzec0jbrz7fndn/"
    "LU_Demo.geojson?rlkey=ulr4g1i2rkeywrwd2ryri4943&st=v8wnjr14&dl=1"
)

# --------------------------------------------------------------------
# Shared download helper (used for DEM, CN, parcels, LU)
# --------------------------------------------------------------------
def _download_from_dropbox(url: str, dest: Path, label: str) -> None:
    """
    Download a file from Dropbox to `dest` if it doesn't already exist.
    """
    if dest.exists():
        return

    st.info(f"Downloading {label} from Dropbox ...")
    dest.parent.mkdir(parents=True, exist_ok=True)

    with requests.get(url, stream=True, allow_redirects=True) as r:
        r.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)

# --------------------------------------------------------------------
# DEM / CN / Parcels loader (used by FCI and can be reused by others)
# --------------------------------------------------------------------
@st.cache_resource(show_spinner="🔍 Locating input data...")
def get_data_paths():
    """
    Returns (DEM_PATH, PARCELS_PATH, CN_PATH).

    1. First tries local files in `data/raw`.
    2. If missing, downloads from Dropbox to a temporary folder.
    """
    # Try project-local data folder first
    base_dir = Path(__file__).resolve().parents[1]
    raw_dir = base_dir / "data" / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    dem_local = raw_dir / "Dem-demo.tif"
    cn_local = raw_dir / "CN.tif"
    parcels_local = raw_dir / "grid-network.geojson"

    if dem_local.exists() and cn_local.exists() and parcels_local.exists():
        return str(dem_local), str(parcels_local), str(cn_local)

    # Fallback: download to temp
    data_dir = Path(tempfile.gettempdir()) / "hydro_app_data"
    data_dir.mkdir(parents=True, exist_ok=True)

    dem_path = data_dir / "Dem-demo.tif"
    cn_path = data_dir / "CN.tif"
    parcels_path = data_dir / "grid-network.geojson"

    _download_from_dropbox(DEM_URL, dem_path, "DEM")
    _download_from_dropbox(CN_URL, cn_path, "Curve Number raster")
    _download_from_dropbox(PARCELS_URL, parcels_path, "Parcels GeoJSON")

    return str(dem_path), str(parcels_path), str(cn_path)

# --------------------------------------------------------------------
# LU_Demo.geojson loader (for SEI and any LU-based model)
# --------------------------------------------------------------------
@st.cache_resource(show_spinner="🔍 Locating land-use layer (LU_Demo.geojson)...")
def get_lu_path() -> str:
    """
    Returns a local path to LU_Demo.geojson.

    1. First tries `data/raw/LU_Demo.geojson` in the repo.
    2. If missing, downloads from Dropbox (LU_URL) to a temp folder.
    """
    base_dir = Path(__file__).resolve().parents[1]
    raw_dir = base_dir / "data" / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    lu_local = raw_dir / "LU_Demo.geojson"
    if lu_local.exists():
        return str(lu_local)

    data_dir = Path(tempfile.gettempdir()) / "hydro_app_data"
    data_dir.mkdir(parents=True, exist_ok=True)

    lu_path = data_dir / "LU_Demo.geojson"
    _download_from_dropbox(LU_URL, lu_path, "Land-use (LU_Demo.geojson)")

    return str(lu_path)

# --------------------------------------------------------------------
# Shared DEM/CN/parcels preparation (used by FCI and can be reused)
# --------------------------------------------------------------------
@st.cache_resource(show_spinner="🧮 Preparing DEM, CN, flow directions & parcels...")
def load_base_data(DEM_PATH_IN, PARCELS_PATH, CN_RASTER_PATH):
    """Precompute DEM, flow directions, aligned CN and parcels (shared by all models)."""
    # --- DEM metadata ---
    with rasterio.open(DEM_PATH_IN) as dem_src:
        dem_profile = dem_src.profile.copy()
        dem_crs = dem_src.crs
        dem_transform = dem_src.transform
        dem_nodata = dem_src.nodata
        height, width = dem_src.height, dem_src.width

    # --- DEM + flow directions ---
    grid = Grid.from_raster(DEM_PATH_IN)
    dem = grid.read_raster(DEM_PATH_IN).astype("float32")

    if dem_nodata is not None:
        valid_mask = (dem != dem_nodata) & (dem == dem)
    else:
        valid_mask = (dem == dem)

    dem_filled = grid.fill_depressions(dem)
    dem_conditioned = grid.resolve_flats(dem_filled)
    flow_directions = grid.flowdir(dem_conditioned)

    # --- CN alignment ---
    with rasterio.open(CN_RASTER_PATH) as cn_src:
        cn_data = cn_src.read(1).astype("float32")
        cn_nodata = cn_src.nodata
        cn_transform = cn_src.transform
        cn_crs = cn_src.crs

    cn_aligned = dem.copy()
    cn_aligned[:] = float("nan")

    reproject(
        source=cn_data,
        destination=cn_aligned,
        src_transform=cn_transform,
        src_crs=cn_crs,
        dst_transform=dem_transform,
        dst_crs=dem_crs,
        resampling=RioResampling.nearest,
        src_nodata=cn_nodata,
        dst_nodata=float("nan"),
    )

    # --- Parcels ---
    parcels = gpd.read_file(PARCELS_PATH)
    if parcels.crs is None:
        raise ValueError("Parcels file has no CRS defined.")
    if parcels.crs.to_string() != dem_crs.to_string():
        parcels = parcels.to_crs(dem_crs)
    if "area_m2" not in parcels.columns:
        parcels["area_m2"] = parcels.geometry.area

    return {
        "dem": dem,
        "valid_mask": valid_mask,
        "flow_directions": flow_directions,
        "cn_aligned": cn_aligned,
        "dem_profile": dem_profile,
        "dem_transform": dem_transform,
        "dem_crs": dem_crs,
        "height": height,
        "width": width,
        "parcels": parcels,
    }




