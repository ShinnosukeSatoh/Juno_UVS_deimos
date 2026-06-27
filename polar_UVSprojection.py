from UniversalColor import UniversalColor
from PIL import Image
from matplotlib.colors import LogNorm, Normalize, LinearSegmentedColormap
import matplotlib.pyplot as plt
from datetime import datetime
from scipy.io import readsav
from astropy.io import fits
import spiceypy as spice
import numpy as np
import re
import glob
import os
import Leadangle_wave as Wave

from RAW_trace_2 import load_best_fit               # noqa: E402
from Leadangle_fit_JunoUVS import moonS3wlon_arr    # noqa: E402
from Leadangle_fit_JunoUVS import spice_moonS3      # noqa: E402

UC = UniversalColor()
UC.set_palette()

Shin = True


# ------------------------------------------------------------
# Hardcoded Juno Perijove UTC times
# ------------------------------------------------------------
JUNO_PJ_TIMES = {
    0: "2016-07-05 02:47:31.851",
    1: "2016-08-27 12:50:44.060",
    2: "2016-10-19 18:10:53.669",
    3: "2016-12-11 17:03:40.665",
    4: "2017-02-02 12:57:08.935",
    5: "2017-03-27 08:51:51.552",
    6: "2017-05-19 06:00:46.945",
    7: "2017-07-11 01:54:42.322",
    8: "2017-09-01 21:48:50.499",
    9: "2017-10-24 17:42:31.438",
    10: "2017-12-16 17:56:58.997",
    11: "2018-02-07 13:51:29.722",
    12: "2018-04-01 09:45:42.502",
    13: "2018-05-24 05:39:50.502",
    14: "2018-07-16 05:17:21.832",
    15: "2018-09-07 01:11:40.519",
    16: "2018-10-29 21:05:59.956",
    17: "2018-12-21 16:59:48.319",
    18: "2019-02-12 17:34:30.940",
    19: "2019-04-06 12:14:22.473",
    20: "2019-05-29 08:08:18.282",
    21: "2019-07-21 04:02:43.348",
    22: "2019-09-12 03:40:44.422",
    23: "2019-11-03 22:18:13.850",
    24: "2019-12-26 17:36:12.571",
    25: "2020-02-17 17:51:55.133",
    26: "2020-04-10 13:47:40.171",
    27: "2020-06-02 10:20:02.882",
    28: "2020-07-25 06:15:27.223",
    29: "2020-09-16 02:10:52.328",
    30: "2020-11-08 01:49:42.104",
    31: "2020-12-30 21:45:44.567",
    32: "2021-02-21 17:40:34.245",
}


# ------------------------------------------------------------
# Constants
# ------------------------------------------------------------
MU0 = 1.26E-6            # 真空中の透磁率
AMU2KG = 1.66E-27        # 原子質量をkgに変換するファクタ [kg amu^-1]
RJ = 71492E+3            # JUPITER RADIUS [m]
MJ = 1.90E+27            # JUPITER MASS [kg]
C = 2.99792E+8           # LIGHT SPEED [m/s]
G = 6.67E-11             # 万有引力定数  [m^3 kg^-1 s^-2]
e = 1.60218E-19          # 素電荷 [J]
me = 9.10E-31            # 電子質量 [kg]


# ------------------------------------------------------------
# Colormaps and caches
# ------------------------------------------------------------
IDL_BLUE_WHITE_CMAP = LinearSegmentedColormap.from_list(
    "idl_blue_white",
    [
        (0.00, (0.00, 0.00, 0.00)),
        (0.10, (0.00, 0.00, 0.22)),
        (0.25, (0.02, 0.02, 0.48)),
        (0.45, (0.08, 0.22, 0.68)),
        (0.65, (0.08, 0.55, 0.88)),
        (0.82, (0.45, 0.70, 0.88)),
        (0.92, (0.75, 0.84, 0.90)),
        (1.00, (1.00, 1.00, 1.00)),
    ],
)
NO_DATA_COLOR = (51 / 255.0, 51 / 255.0, 51 / 255.0)
if Shin:
    NO_DATA_COLOR = (1 / 255.0, 1 / 255.0, 1 / 255.0)
IDL_BLUE_WHITE_CMAP.set_bad(NO_DATA_COLOR)

OVAL_CACHE = {}
SATELLITE_FP_CACHE = {}
OVERLAY_CACHE = {}


# ------------------------------------------------------------
# FITS corruption check
# ------------------------------------------------------------
def check_fits_file(filepath):
    try:
        with fits.open(filepath, memmap=True) as hdul:
            for hdu in hdul:
                _ = hdu.header
                if hdu.data is not None:
                    _ = np.asarray(hdu.data).shape
        return True, None
    except Exception as e:
        return False, str(e)


def filter_valid_fits_records(selected_records):
    valid_records = []

    for record in selected_records:
        filepath = record["filepath"]
        is_valid, error_msg = check_fits_file(filepath)

        if not is_valid:
            print(f"ERROR: corrupted FITS file: {filepath}")
            print(f"       reason: {error_msg}")
            continue

        valid_records.append(record)

    return valid_records


# ------------------------------------------------------------
# Output helpers
# ------------------------------------------------------------
def sanitize_time_string_for_filename(time_string):
    return (
        time_string.replace(":", "")
        .replace("-", "")
        .replace("T", "_")
        .replace(" ", "_")
    )


def build_output_png_path(pj_number, tstart, tend, zoomed=False):
    out_dir = os.path.join("./Output", f"PJ{pj_number}")
    os.makedirs(out_dir, exist_ok=True)

    start_str = sanitize_time_string_for_filename(tstart)
    end_str = sanitize_time_string_for_filename(tend)

    base = f"{start_str}_{end_str}.png"
    if zoomed:
        base = f"Zoomed_{base}"

    return os.path.join(out_dir, base)


# ------------------------------------------------------------
# GIF helpers
# ------------------------------------------------------------
def build_gif_from_pngs(gif_path, png_paths, duration=0.8):
    if len(png_paths) == 0:
        return None

    frames = []
    target_size = None

    for png_path in png_paths:
        pil_img = Image.open(png_path).convert("RGB")

        if target_size is None:
            target_size = pil_img.size

        if pil_img.size != target_size:
            try:
                resample_method = Image.Resampling.LANCZOS
            except AttributeError:
                resample_method = Image.LANCZOS
            pil_img = pil_img.resize(target_size, resample_method)

        frames.append(pil_img.copy())

    if len(frames) == 1:
        frames[0].save(
            gif_path,
            format="GIF",
            save_all=True,
            append_images=[],
            duration=int(duration * 1000),
            loop=0,
        )
    else:
        frames[0].save(
            gif_path,
            format="GIF",
            save_all=True,
            append_images=frames[1:],
            duration=int(duration * 1000),
            loop=0,
        )

    print(f"Saved GIF: {gif_path}")
    return gif_path


def build_hemisphere_gifs(pj_number, hemisphere_name, avg_png_paths, zoom_png_paths, duration=0.8):
    out_dir = os.path.join(".", "Output", f"PJ{pj_number}")
    os.makedirs(out_dir, exist_ok=True)

    avg_gif_path = None
    zoom_gif_path = None

    if len(avg_png_paths) > 0:
        avg_gif_path = os.path.join(
            out_dir, f"PJ{pj_number}_{hemisphere_name}_Average.gif")
        build_gif_from_pngs(avg_gif_path, avg_png_paths, duration=duration)

    if len(zoom_png_paths) > 0:
        zoom_gif_path = os.path.join(
            out_dir, f"PJ{pj_number}_{hemisphere_name}_Zoomed.gif")
        build_gif_from_pngs(zoom_gif_path, zoom_png_paths, duration=duration)

    return avg_gif_path, zoom_gif_path


# ------------------------------------------------------------
# Get PJ UTC time
# ------------------------------------------------------------
def get_pj_time(pj_number):
    return JUNO_PJ_TIMES[pj_number]


# ------------------------------------------------------------
# Load / unload SPICE kernels
# ------------------------------------------------------------
def load_spice_kernels(meta_kernel):
    spice.furnsh(meta_kernel)


def unload_spice_kernels():
    spice.kclear()


# ------------------------------------------------------------
# Time helpers
# ------------------------------------------------------------
def utc_to_et(utc_string):
    return spice.str2et(utc_string)


def et_to_utc(et):
    return spice.et2utc(et, "ISOC", 0)


def doy_utc_to_et(year, doy, hh, mm, ss):
    dt = datetime.strptime(
        f"{year} {doy:03d} {hh:02d}:{mm:02d}:{ss:02d}", "%Y %j %H:%M:%S")
    return spice.str2et(dt.strftime("%Y-%m-%dT%H:%M:%S"))


# ------------------------------------------------------------
# Build a regular ET grid
# ------------------------------------------------------------
def build_time_grid(t0, t1, dt):
    if t1 <= t0:
        return np.array([], dtype=np.float64)

    time_et = np.arange(t0, t1 + dt, dt, dtype=np.float64)
    time_et = time_et[time_et <= t1]

    if time_et.size == 0 or time_et[-1] < t1:
        time_et = np.append(time_et, t1)

    return time_et


# ------------------------------------------------------------
# Generate time windows for both hemispheres
# Pre-PJ => NORTH
# Post-PJ => SOUTH
# ------------------------------------------------------------
def generate_time_windows(
    pjtime_et,
    dt,
    use_north=True,
    use_south=True,
    south_start_override_et=None,
):
    windows = []

    if use_north:
        # t0_n = pjtime_et - 4.0 * 3600.0       # default: 4 hours before pj
        # t1_n = pjtime_et + 15.0 * 60.0       # default: 15 minutes after pj

        time_et_n = build_time_grid(t0_n, t1_n, dt)
        windows.append(("NORTH", time_et_n))

    if use_south:
        # t0_s = pjtime_et + 15.0 * 60.0       # default: 15 minutes before pj
        # t1_s = pjtime_et + 4.0 * 3600.0      # default: 4 hours after pj

        if south_start_override_et is not None:
            t0_s = max(t0_s, south_start_override_et)

        time_et_s = build_time_grid(t0_s, t1_s, dt)
        windows.append(("SOUTH", time_et_s))

    return windows


# ------------------------------------------------------------
# Parse filename time from:
# UVS_SSI_YYYYDOYHHMMSS_P14_V01.FIT
# ------------------------------------------------------------
def parse_uvs_filename_time(filename):
    base = os.path.basename(filename)
    pattern = r"^UVS_SSI_(\d{4})(\d{3})(\d{2})(\d{2})(\d{2})_P(\d+)_.*\.FIT$"
    match = re.match(pattern, base, re.IGNORECASE)

    if match is None:
        return None, None

    year = int(match.group(1))
    doy = int(match.group(2))
    hour = int(match.group(3))
    minute = int(match.group(4))
    second = int(match.group(5))
    pj_number = int(match.group(6))

    dt = datetime.strptime(
        f"{year} {doy:03d} {hour:02d}:{minute:02d}:{second:02d}",
        "%Y %j %H:%M:%S",
    )

    return dt, pj_number


# ------------------------------------------------------------
# List all FITS files for a given PJ
# ------------------------------------------------------------
def list_pj_files(data_root, pj_number):
    pj_dir = os.path.join(data_root, f"PJ{pj_number:02d}")
    pattern = os.path.join(pj_dir, f"UVS_SSI_*_P{pj_number:02d}_*.FIT")
    return sorted(glob.glob(pattern))


# ------------------------------------------------------------
# Convert file list to timed records
# ------------------------------------------------------------
def build_file_time_records(file_list):
    records = []

    for filepath in file_list:
        file_time_dt, file_pj = parse_uvs_filename_time(filepath)

        if file_time_dt is None:
            continue

        records.append(
            {
                "filepath": filepath,
                "filename": os.path.basename(filepath),
                "datetime": file_time_dt,
                "utc": file_time_dt.strftime("%Y-%m-%dT%H:%M:%S"),
                "pj": file_pj,
            }
        )

    return records


# ------------------------------------------------------------
# Find files contained within a UTC time range
# ------------------------------------------------------------
def find_files_in_time_range(file_records, tstart_utc, tend_utc):
    tstart_dt = datetime.strptime(tstart_utc, "%Y-%m-%dT%H:%M:%S")
    tend_dt = datetime.strptime(tend_utc, "%Y-%m-%dT%H:%M:%S")

    selected = []

    for record in file_records:
        if tstart_dt <= record["datetime"] <= tend_dt:
            selected.append(record)

    return selected


# ------------------------------------------------------------
# Orthographic projection
# ------------------------------------------------------------
def ortho_latlon_to_xy(lat, lon, lat_0, lon_0):
    dtor = np.pi / 180.0

    x = np.cos(lat * dtor) * np.sin((lon - lon_0) * dtor)

    y = (
        np.cos(lat_0 * dtor) * np.sin(lat * dtor)
        - np.sin(lat_0 * dtor) * np.cos(lat * dtor) *
        np.cos((lon - lon_0) * dtor)
    )

    return x, y


def ortho_visible_mask(lat, lon, lat_0, lon_0):
    dtor = np.pi / 180.0

    cosc = (
        np.sin(lat_0 * dtor) * np.sin(lat * dtor)
        + np.cos(lat_0 * dtor) * np.cos(lat * dtor) *
        np.cos((lon - lon_0) * dtor)
    )

    return cosc >= 0.0


# ------------------------------------------------------------
# Great-circle distance helpers
# ------------------------------------------------------------
def great_circle_distance_deg(lat1_deg, lon1_deg, lat2_deg, lon2_deg):
    lat1 = np.deg2rad(lat1_deg)
    lon1 = np.deg2rad(lon1_deg)
    lat2 = np.deg2rad(lat2_deg)
    lon2 = np.deg2rad(lon2_deg)

    cos_sigma = (
        np.sin(lat1) * np.sin(lat2)
        + np.cos(lat1) * np.cos(lat2) * np.cos(lon2 - lon1)
    )
    cos_sigma = np.clip(cos_sigma, -1.0, 1.0)

    return np.rad2deg(np.arccos(cos_sigma))


def great_circle_distance_km(lat1_deg, lon1_deg, lat2_deg, lon2_deg, radius_km):
    return np.deg2rad(great_circle_distance_deg(lat1_deg, lon1_deg, lat2_deg, lon2_deg)) * radius_km


# ------------------------------------------------------------
# Orthographic grid
# ------------------------------------------------------------
def grid_ortho(dlon, dlat, proj_lat, proj_lon):
    nlon = int(360.0 / dlon) + 1
    nlat = int(180.0 / dlat) + 1

    latgrid_lon = np.zeros((181, nlon), dtype=np.float64)
    for ilon in range(nlon):
        latgrid_lon[:, ilon] = ilon * dlon

    latgrid_lat = np.repeat(
        np.linspace(-90.0, 90.0, 181)[:, None], nlon, axis=1)

    xlat, ylat = ortho_latlon_to_xy(
        latgrid_lat, latgrid_lon, proj_lat, proj_lon)
    vis_lat = ortho_visible_mask(latgrid_lat, latgrid_lon, proj_lat, proj_lon)
    xlat = np.where(vis_lat, xlat, np.nan)
    ylat = np.where(vis_lat, ylat, np.nan)

    longrid_lat = np.zeros((nlat, 361), dtype=np.float64)
    for ilat in range(nlat):
        longrid_lat[ilat, :] = ilat * dlat - 90.0

    longrid_lon = np.repeat(np.linspace(0.0, 360.0, 361)[
                            None, :], nlat, axis=0)

    xlon, ylon = ortho_latlon_to_xy(
        longrid_lat, longrid_lon, proj_lat, proj_lon)
    vis_lon = ortho_visible_mask(longrid_lat, longrid_lon, proj_lat, proj_lon)
    xlon = np.where(vis_lon, xlon, np.nan)
    ylon = np.where(vis_lon, ylon, np.nan)

    nlon_grid = nlon
    nlat_grid = nlat

    return (
        xlat,
        ylat,
        xlon,
        ylon,
        nlon_grid,
        nlat_grid,
        latgrid_lon,
        latgrid_lat,
        longrid_lon,
        longrid_lat,
    )


# ------------------------------------------------------------
# Retrieve N_TIME from FITS header
# ------------------------------------------------------------
def get_nadir_time_from_fits(filepath):
    is_valid, error_msg = check_fits_file(filepath)
    if not is_valid:
        raise IOError(f"Corrupted FITS file: {filepath} | {error_msg}")

    with fits.open(filepath) as hdul:
        for hdu in hdul:
            if "N_TIME" in hdu.header:
                return hdu.header["N_TIME"]

    raise KeyError(f"N_TIME not found in any header of {filepath}")


def get_nadir_times_utc(selected_records):
    nadir_times_utc = []

    for record in selected_records:
        nadir_time_utc = get_nadir_time_from_fits(record["filepath"])
        nadir_times_utc.append(nadir_time_utc)

    return np.array(nadir_times_utc)


# ------------------------------------------------------------
# SPICE geometry helpers
# ------------------------------------------------------------
def get_jupiter_radii():
    radii = spice.bodvrd("JUPITER", "RADII", 3)[1]
    req = float(radii[0])
    rpol = float(radii[2])
    return req, rpol


def compute_subspacecraft_latitude(et, target="JUPITER", observer="JUNO", frame="IAU_JUPITER", abcorr="NONE"):
    spoint, _, _ = spice.subpnt(
        "Near Point: Ellipsoid",
        target,
        et,
        frame,
        abcorr,
        observer,
    )
    _, _, lat = spice.reclat(spoint)
    return np.degrees(lat)


def compute_nadir_altitude(et, target="JUPITER", observer="JUNO", frame="IAU_JUPITER", abcorr="NONE"):
    _, _, srfvec = spice.subpnt(
        "Near Point: Ellipsoid",
        target,
        et,
        frame,
        abcorr,
        observer,
    )
    return float(spice.vnorm(srfvec))


def convert_s3_lon_vec(lon_deg):
    lon_deg = np.asarray(lon_deg, dtype=np.float64).copy()
    ind_l0 = lon_deg < 0.0
    ind_g0 = ~ind_l0
    lon_deg[ind_l0] = -lon_deg[ind_l0]
    lon_deg[ind_g0] = 360.0 - lon_deg[ind_g0]
    lon_deg = np.mod(lon_deg, 360.0)
    return lon_deg


def compute_io_system3_w_longitude(et):
    moon_state, _ = spice.spkezr("IO", et, "IAU_JUPITER", "CN+S", "JUPITER")
    _, lon_rad, lat_rad = spice.reclat(moon_state[:3])
    submoon_jupiter_lon = np.degrees(lon_rad)
    submoon_jupiter_lat = np.degrees(lat_rad)
    submoon_jupiter_lon_w = convert_s3_lon_vec(submoon_jupiter_lon)
    return float(submoon_jupiter_lon_w), float(submoon_jupiter_lat)


def angular_separation_deg(lon1, lon2):
    return np.abs((lon1 - lon2 + 180.0) % 360.0 - 180.0)


def compute_spin_ifp_from_io_lon(hemi_spin, time_et_array,
                                 satellite_fp_path="/Users/shin/Documents/Research/Juno/UVS/Code/PolarProjection/Input/Satellite_FP_JRM33_LA.sav"):
    satellite_fp = load_satellite_fp_for_hemisphere(
        hemi_spin, sav_path=satellite_fp_path)

    io_sat_lon = np.asarray(satellite_fp["io_sat_lon"], dtype=np.float64)
    ifp_lon = np.asarray(satellite_fp["ifp_lon"], dtype=np.float64)
    ifp_lat = np.asarray(satellite_fp["ifp_lat"], dtype=np.float64)

    io_s3_w_lon = []
    ifp_lon_spin = []
    ifp_lat_spin = []
    ifp_match_index = []

    for et in np.asarray(time_et_array, dtype=np.float64):
        io_lon_w, _ = compute_io_system3_w_longitude(et)
        idx = int(np.argmin(angular_separation_deg(io_sat_lon, io_lon_w)))

        io_s3_w_lon.append(io_lon_w)
        ifp_lon_spin.append(ifp_lon[idx])
        ifp_lat_spin.append(ifp_lat[idx])
        ifp_match_index.append(idx)

    return {
        "io_s3_w_lon": np.array(io_s3_w_lon, dtype=np.float64),
        "ifp_lon_spin": np.array(ifp_lon_spin, dtype=np.float64),
        "ifp_lat_spin": np.array(ifp_lat_spin, dtype=np.float64),
        "ifp_match_index": np.array(ifp_match_index, dtype=np.int64),
    }


def compute_midpoint_ifp_position(
    hemi_spin,
    tstart,
    tend,
    proj_lat,
    proj_lon,
    satellite_fp_path="/Users/shin/Documents/Research/Juno/UVS/Code/PolarProjection/Input/Satellite_FP_JRM33_LA.sav",
):
    tstart_et = utc_to_et(tstart)
    tend_et = utc_to_et(tend)
    tmid_et = 0.5 * (tstart_et + tend_et)

    mid_ifp = compute_spin_ifp_from_io_lon(
        hemi_spin=hemi_spin,
        time_et_array=np.array([tmid_et], dtype=np.float64),
        satellite_fp_path=satellite_fp_path,
    )

    ifp_lon_mid = float(mid_ifp["ifp_lon_spin"][0])
    ifp_lat_mid = float(mid_ifp["ifp_lat_spin"][0])

    avg_x_ifp_spin, avg_y_ifp_spin = ortho_latlon_to_xy(
        ifp_lat_mid,
        ifp_lon_mid,
        proj_lat,
        proj_lon,
    )

    is_visible = ortho_visible_mask(
        ifp_lat_mid, ifp_lon_mid, proj_lat, proj_lon)
    if not bool(is_visible):
        avg_x_ifp_spin = np.nan
        avg_y_ifp_spin = np.nan

    return {
        "tmid_et": float(tmid_et),
        "ifp_lon_mid": ifp_lon_mid,
        "ifp_lat_mid": ifp_lat_mid,
        "avg_x_ifp_spin": float(avg_x_ifp_spin),
        "avg_y_ifp_spin": float(avg_y_ifp_spin),
    }


# ------------------------------------------------------------
# Projection setup
# Use the requested hemisphere, not sub-spacecraft latitude.
# ------------------------------------------------------------
def get_projection_setup_from_hemisphere(hemisphere_name):
    hemi_name = hemisphere_name.upper()

    if hemi_name == "NORTH":
        hemi_spin = "N"
        minx = -0.5
        maxx = 0.5
        miny = -0.3
        maxy = 0.7
        proj_lat = 90.0
        proj_lon = 0.0
    elif hemi_name == "SOUTH":
        hemi_spin = "S"
        minx = -0.6
        maxx = 0.6
        miny = -0.6
        maxy = 0.6
        proj_lat = -90.0
        proj_lon = 0.0
    else:
        raise ValueError(f"Unknown hemisphere_name: {hemisphere_name}")

    return hemi_spin, minx, maxx, miny, maxy, proj_lat, proj_lon


def get_plot_limits(hemi_spin):
    if hemi_spin == "N":
        return {
            "xmin": -0.5,
            "xmax": 0.5,
            "ymin": -0.3,
            "ymax": 0.7,
        }
    else:
        return {
            "xmin": -0.4,
            "xmax": 0.6,
            "ymin": -0.45,
            "ymax": 0.57,
        }


def get_zoom_limits(center_x, center_y, hemi_spin, half_size=0.2):
    plot_limits = get_plot_limits(hemi_spin)

    if not np.isfinite(center_x) or not np.isfinite(center_y):
        return (
            (plot_limits["xmin"], plot_limits["xmax"]),
            (plot_limits["ymin"], plot_limits["ymax"]),
        )

    return (
        (center_x - half_size, center_x + half_size),
        (center_y - half_size, center_y + half_size),
    )


# ------------------------------------------------------------
# Build map bins
# ------------------------------------------------------------
def build_map_bins(minx, maxx, miny, maxy, delta_xy):
    xbin = np.arange(minx, maxx + delta_xy, delta_xy)
    ybin = np.arange(miny, maxy + delta_xy, delta_xy)
    nxmap = len(xbin)
    nymap = len(ybin)
    return xbin, ybin, nxmap, nymap


# ------------------------------------------------------------
# Load needed FITS products
# ------------------------------------------------------------
def load_fits_products(filepath):
    is_valid, error_msg = check_fits_file(filepath)
    if not is_valid:
        raise IOError(f"Corrupted FITS file: {filepath} | {error_msg}")

    with fits.open(filepath) as hdul:
        phot_map = hdul[0].data.astype(np.float64)
        num_det_map = hdul["DETECTION MAP"].data.astype(np.float64)
        int_time_map = hdul["INTEGRATION TIME MAP"].data.astype(np.float64)
        lat = hdul["LATITUDE"].data.astype(np.float64)
        lon = hdul["LONGITUDE"].data.astype(np.float64)
        slit_mask_map = hdul["SLIT MASK MAP"].data.astype(np.float64)
        wavelength = np.array(
            hdul["WAVELENGTH"].data.field(0), dtype=np.float64)

    return {
        "PHOT_MAP": phot_map,
        "NUM_DET_MAP": num_det_map,
        "INTEGRATION_TIME_MAP": int_time_map,
        "LATITUDE": lat,
        "LONGITUDE": lon,
        "SLIT_MASK_MAP": slit_mask_map,
        "WAVELENGTH": wavelength,
    }


# ------------------------------------------------------------
# Aurora oval helpers
# ------------------------------------------------------------
def _get_sav_variable(sav_data, *candidate_names):
    key_map = {key.lower(): key for key in sav_data.keys()}

    for name in candidate_names:
        if name.lower() in key_map:
            return np.array(sav_data[key_map[name.lower()]], dtype=np.float64)

    raise KeyError(f"Could not find any of the variables: {candidate_names}")


def _normalize_oval_array(oval_array):
    arr = np.array(oval_array, dtype=np.float64)

    if arr.ndim != 2:
        raise ValueError("Auroral oval array must be 2D.")

    if arr.shape[0] == 2:
        lon = arr[0, :]
        lat = arr[1, :]
    elif arr.shape[1] == 2:
        lon = arr[:, 0]
        lat = arr[:, 1]
    else:
        raise ValueError(f"Unexpected auroral oval array shape: {arr.shape}")

    lon = np.mod(lon, 360.0)

    return {
        "lon": lon,
        "lat": lat,
    }


def load_aurora_ovals_for_hemisphere(hemi_spin,
                                     sav_path="/Users/shin/Documents/Research/Juno/UVS/Code/PolarProjection/Input/OvalBonfond2017.sav"):
    hemi_key = hemi_spin.upper()

    if hemi_key in OVAL_CACHE:
        return OVAL_CACHE[hemi_key]

    sav_data = readsav(sav_path, python_dict=True)

    if hemi_key == "S":
        inner_raw = _get_sav_variable(sav_data, "NEW_INNEROVAL_S")
        outer_raw = _get_sav_variable(
            sav_data, "NEW_OUTEROVAL_S", "NEW_outeroval_S")
    else:
        inner_raw = _get_sav_variable(sav_data, "NEW_INNEROVAL_N")
        outer_raw = _get_sav_variable(
            sav_data, "NEW_OUTEROVAL_N", "NEW_outeroval_N")

    inner_oval = _normalize_oval_array(inner_raw)
    outer_oval = _normalize_oval_array(outer_raw)

    OVAL_CACHE[hemi_key] = {
        "inner": inner_oval,
        "outer": outer_oval,
    }

    return OVAL_CACHE[hemi_key]


# ------------------------------------------------------------
# Satellite footprint helpers
# ------------------------------------------------------------
def load_satellite_fp_for_hemisphere(hemi_spin,
                                     sav_path="/Users/shin/Documents/Research/Juno/UVS/Code/PolarProjection/Input/Satellite_FP_JRM33_LA.sav"):
    hemi_key = hemi_spin.upper()

    if hemi_key in SATELLITE_FP_CACHE:
        return SATELLITE_FP_CACHE[hemi_key]

    sav_data = readsav(sav_path, python_dict=True)

    if hemi_key == "S":
        ifp_raw = _get_sav_variable(
            sav_data, "IFP_CONTOUR_S_LA", "IFP_CONTOUR_S")
    else:
        ifp_raw = _get_sav_variable(
            sav_data, "IFP_CONTOUR_N_LA", "IFP_CONTOUR_N")

    ifp_raw = np.array(ifp_raw, dtype=np.float64)

    if ifp_raw.ndim != 2:
        raise ValueError(
            f"IFP_CONTOUR array must be 2D, got shape {ifp_raw.shape}")

    if ifp_raw.shape[1] == 3:
        io_sat_lon = np.array(ifp_raw[:, 0], dtype=np.float64)
        ifp_lon = np.array(ifp_raw[:, 1], dtype=np.float64)
        ifp_lat = np.array(ifp_raw[:, 2], dtype=np.float64)
    elif ifp_raw.shape[0] == 3:
        io_sat_lon = np.array(ifp_raw[0, :], dtype=np.float64)
        ifp_lon = np.array(ifp_raw[1, :], dtype=np.float64)
        ifp_lat = np.array(ifp_raw[2, :], dtype=np.float64)
    else:
        raise ValueError(f"Unexpected IFP_CONTOUR shape: {ifp_raw.shape}")

    io_sat_lon = np.mod(io_sat_lon, 360.0)
    ifp_lon = np.mod(ifp_lon, 360.0)

    SATELLITE_FP_CACHE[hemi_key] = {
        "io_sat_lon": io_sat_lon,
        "ifp_lon": ifp_lon,
        "ifp_lat": ifp_lat,
    }

    return SATELLITE_FP_CACHE[hemi_key]


def prepare_projection_overlay(
    hemi_spin,
    proj_lat,
    proj_lon,
    dlon=90.0,
    dlat=10.0,
    sav_path="/Users/shin/Documents/Research/Juno/UVS/Code/PolarProjection/Input/OvalBonfond2017.sav",
    satellite_fp_path="/Users/shin/Documents/Research/Juno/UVS/Code/PolarProjection/Input/Satellite_FP_JRM33_LA.sav",
):
    cache_key = (
        hemi_spin.upper(),
        float(proj_lat),
        float(proj_lon),
        float(dlon),
        float(dlat),
        sav_path,
        satellite_fp_path,
    )

    if cache_key in OVERLAY_CACHE:
        return OVERLAY_CACHE[cache_key]

    (
        xlat,
        ylat,
        xlon,
        ylon,
        nlon_grid,
        nlat_grid,
        latgrid_lon,
        latgrid_lat,
        longrid_lon,
        longrid_lat,
    ) = grid_ortho(dlon, dlat, proj_lat, proj_lon)

    oval_data = load_aurora_ovals_for_hemisphere(hemi_spin, sav_path=sav_path)

    inner_oval_lon = oval_data["inner"]["lon"]
    inner_oval_lat = oval_data["inner"]["lat"]
    outer_oval_lon = oval_data["outer"]["lon"]
    outer_oval_lat = oval_data["outer"]["lat"]

    satellite_fp = load_satellite_fp_for_hemisphere(
        hemi_spin, sav_path=satellite_fp_path)
    ifp_lon = satellite_fp["ifp_lon"]
    ifp_lat = satellite_fp["ifp_lat"]
    for i in range(len(ifp_lat)):
        if (ifp_lat[i] > 78.0) & (ifp_lat[i] < 79.5) & (ifp_lon[i] > 333.0):
            print('IFP:', ifp_lat[i], ifp_lon[i])

    x_inner_oval, y_inner_oval = ortho_latlon_to_xy(
        inner_oval_lat, inner_oval_lon, proj_lat, proj_lon)
    x_outer_oval, y_outer_oval = ortho_latlon_to_xy(
        outer_oval_lat, outer_oval_lon, proj_lat, proj_lon)
    x_ifp, y_ifp = ortho_latlon_to_xy(ifp_lat, ifp_lon, proj_lat, proj_lon)

    vis_inner = ortho_visible_mask(
        inner_oval_lat, inner_oval_lon, proj_lat, proj_lon)
    vis_outer = ortho_visible_mask(
        outer_oval_lat, outer_oval_lon, proj_lat, proj_lon)
    vis_ifp = ortho_visible_mask(ifp_lat, ifp_lon, proj_lat, proj_lon)

    x_inner_oval = np.where(vis_inner, x_inner_oval, np.nan)
    y_inner_oval = np.where(vis_inner, y_inner_oval, np.nan)
    x_outer_oval = np.where(vis_outer, x_outer_oval, np.nan)
    y_outer_oval = np.where(vis_outer, y_outer_oval, np.nan)
    x_ifp = np.where(vis_ifp, x_ifp, np.nan)
    y_ifp = np.where(vis_ifp, y_ifp, np.nan)

    OVERLAY_CACHE[cache_key] = {
        "xlat": xlat,
        "ylat": ylat,
        "xlon": xlon,
        "ylon": ylon,
        "nlon_grid": nlon_grid,
        "nlat_grid": nlat_grid,
        "x_inner_oval": x_inner_oval,
        "y_inner_oval": y_inner_oval,
        "x_outer_oval": x_outer_oval,
        "y_outer_oval": y_outer_oval,
        "x_ifp": x_ifp,
        "y_ifp": y_ifp,
    }

    return OVERLAY_CACHE[cache_key]


# ------------------------------------------------------------
# Brightness helpers
# ------------------------------------------------------------
def get_band_indices(wavelength, method):
    wavelength = np.asarray(wavelength, dtype=np.float64)

    if method == "default":
        return np.where((wavelength >= 155.0) & (wavelength <= 162.0))[0]

    if method in ("dual_band", "dual_band_raw"):
        iw1 = np.where((wavelength >= 115.0) & (wavelength <= 118.0))[0]
        iw2 = np.where((wavelength >= 125.0) & (wavelength <= 165.0))[0]
        return np.concatenate((iw1, iw2))

    raise ValueError(
        "method must be 'default', 'dual_band', or 'dual_band_raw'")


def compute_brightness_map(map_wc, map_exp, wavelength, method="default", fact_br=None):
    wavelength = np.asarray(wavelength, dtype=np.float64)
    iw = get_band_indices(wavelength, method)

    if iw.size == 0:
        raise ValueError(f"No wavelength indices found for method='{method}'")

    if fact_br is None:
        if method == "default":
            fact_br = 8.1
        elif method == "dual_band":
            fact_br = 1.84
        elif method == "dual_band_raw":
            fact_br = 1.0
        else:
            raise ValueError("Unsupported method")

    with np.errstate(divide="ignore", invalid="ignore"):
        br_w = 4.0 * np.pi * np.sum(map_wc[:, :, iw], axis=2) / map_exp / 1.0e9
        br_w *= fact_br

    br_w = np.nan_to_num(br_w, nan=0.0, posinf=0.0, neginf=0.0)
    return br_w


# ------------------------------------------------------------
# Geolocation helpers for the coadded mapped grid
# ------------------------------------------------------------
def finalize_geolocation_maps(map_lat_sum, map_lon_sin_sum, map_lon_cos_sum, map_geo_weight_sum):
    map_lat_mean = np.full_like(map_lat_sum, np.nan, dtype=np.float64)
    map_lon_mean = np.full_like(map_lat_sum, np.nan, dtype=np.float64)

    valid = map_geo_weight_sum > 0.0

    map_lat_mean[valid] = map_lat_sum[valid] / map_geo_weight_sum[valid]
    map_lon_mean[valid] = np.rad2deg(
        np.arctan2(map_lon_sin_sum[valid], map_lon_cos_sum[valid])
    )
    map_lon_mean[valid] = np.mod(map_lon_mean[valid], 360.0)

    return map_lat_mean, map_lon_mean


def estimate_surface_bin_area_km2(map_lat_mean, map_lon_mean, radius_km):
    nx, ny = map_lat_mean.shape
    area_km2 = np.full((nx, ny), np.nan, dtype=np.float64)

    for ix in range(nx):
        for iy in range(ny):
            if not np.isfinite(map_lat_mean[ix, iy]) or not np.isfinite(map_lon_mean[ix, iy]):
                continue

            dx_list = []
            dy_list = []

            if ix > 0 and np.isfinite(map_lat_mean[ix - 1, iy]) and np.isfinite(map_lon_mean[ix - 1, iy]):
                dx_list.append(
                    great_circle_distance_km(
                        map_lat_mean[ix, iy], map_lon_mean[ix, iy],
                        map_lat_mean[ix - 1, iy], map_lon_mean[ix - 1, iy],
                        radius_km,
                    )
                )

            if ix < nx - 1 and np.isfinite(map_lat_mean[ix + 1, iy]) and np.isfinite(map_lon_mean[ix + 1, iy]):
                dx_list.append(
                    great_circle_distance_km(
                        map_lat_mean[ix, iy], map_lon_mean[ix, iy],
                        map_lat_mean[ix + 1, iy], map_lon_mean[ix + 1, iy],
                        radius_km,
                    )
                )

            if iy > 0 and np.isfinite(map_lat_mean[ix, iy - 1]) and np.isfinite(map_lon_mean[ix, iy - 1]):
                dy_list.append(
                    great_circle_distance_km(
                        map_lat_mean[ix, iy], map_lon_mean[ix, iy],
                        map_lat_mean[ix, iy - 1], map_lon_mean[ix, iy - 1],
                        radius_km,
                    )
                )

            if iy < ny - 1 and np.isfinite(map_lat_mean[ix, iy + 1]) and np.isfinite(map_lon_mean[ix, iy + 1]):
                dy_list.append(
                    great_circle_distance_km(
                        map_lat_mean[ix, iy], map_lon_mean[ix, iy],
                        map_lat_mean[ix, iy + 1], map_lon_mean[ix, iy + 1],
                        radius_km,
                    )
                )

            if len(dx_list) == 0 or len(dy_list) == 0:
                continue

            dx_km = np.nanmean(dx_list)
            dy_km = np.nanmean(dy_list)

            if np.isfinite(dx_km) and np.isfinite(dy_km):
                area_km2[ix, iy] = dx_km * dy_km

    return area_km2


# ------------------------------------------------------------
# Auroral power helpers
# ------------------------------------------------------------
def compute_total_auroral_power_gw(
    map_wc,
    map_exp,
    wavelength,
    map_lat_mean,
    map_lon_mean,
    map_area_km2,
    center_lat_deg,
    center_lon_deg,
    radius_deg=1.0,
    power_scale_factor=2.04,
):
    if not np.isfinite(center_lat_deg) or not np.isfinite(center_lon_deg):
        return {
            "power_gw": np.nan,
            "distance_deg_map": np.full_like(map_lat_mean, np.nan, dtype=np.float64),
            "selection_mask": np.zeros_like(map_lat_mean, dtype=bool),
            "brightness_band_raw_kR": np.full_like(map_lat_mean, np.nan, dtype=np.float64),
        }

    brightness_band_raw_kR = compute_brightness_map(
        map_wc=map_wc,
        map_exp=map_exp,
        wavelength=wavelength,
        method="dual_band_raw",
    )

    distance_deg_map = great_circle_distance_deg(
        map_lat_mean,
        map_lon_mean,
        center_lat_deg,
        center_lon_deg,
    )

    selection_mask = (
        np.isfinite(distance_deg_map)
        & np.isfinite(map_area_km2)
        & np.isfinite(brightness_band_raw_kR)
        & (map_area_km2 > 0.0)
        & (distance_deg_map <= radius_deg)
    )

    if not np.any(selection_mask):
        return {
            "power_gw": np.nan,
            "distance_deg_map": distance_deg_map,
            "selection_mask": selection_mask,
            "brightness_band_raw_kR": brightness_band_raw_kR,
        }

    area_m2 = map_area_km2 * 1.0e6
    photons_per_sec_band = np.nansum(
        brightness_band_raw_kR[selection_mask] *
        1.0e13 * area_m2[selection_mask]
    )

    iw = get_band_indices(wavelength, "dual_band_raw")
    lambda_m = wavelength[iw] * 1.0e-9

    region_cube = map_wc[:, :, iw]
    region_weights = np.nansum(region_cube[selection_mask], axis=0)

    if np.all(~np.isfinite(region_weights)) or np.nansum(region_weights) <= 0.0:
        inv_lambda_weighted = np.nanmean(1.0 / lambda_m)
    else:
        inv_lambda_weighted = np.nansum(
            region_weights / lambda_m) / np.nansum(region_weights)

    h = 6.62607015e-34
    c = 2.99792458e8
    mean_photon_energy_j = h * c * inv_lambda_weighted

    power_w = power_scale_factor * photons_per_sec_band * mean_photon_energy_j
    power_gw = power_w / 1.0e9

    return {
        "power_gw": float(power_gw),
        "distance_deg_map": distance_deg_map,
        "selection_mask": selection_mask,
        "brightness_band_raw_kR": brightness_band_raw_kR,
    }


# ------------------------------------------------------------
# Convert selected power pixels to projected x/y points
# ------------------------------------------------------------
def selection_mask_to_projected_points(selection_mask, xbin, ybin):
    if selection_mask is None:
        return np.array([], dtype=np.float64), np.array([], dtype=np.float64)

    ix_sel, iy_sel = np.where(selection_mask)

    if len(ix_sel) == 0:
        return np.array([], dtype=np.float64), np.array([], dtype=np.float64)

    x_selected = np.asarray(xbin[ix_sel], dtype=np.float64)
    y_selected = np.asarray(ybin[iy_sel], dtype=np.float64)

    finite = np.isfinite(x_selected) & np.isfinite(y_selected)
    return x_selected[finite], y_selected[finite]


# ------------------------------------------------------------
# Plot helpers
# ------------------------------------------------------------
def format_map_for_display(map2d):
    return np.flipud(map2d.T)


def apply_plot_overlay(
    ax,
    overlay,
    x_ifp_spin=None,
    y_ifp_spin=None,
    x_ifp_first=None,
    y_ifp_first=None,
    x_ifp_last=None,
    y_ifp_last=None,
    x_selected_pixels=None,
    y_selected_pixels=None,
):
    for ilon in range(overlay["nlon_grid"]):
        ax.plot(
            overlay["xlat"][:, ilon],
            overlay["ylat"][:, ilon],
            color="white",
            linewidth=0.8,
            alpha=0.9,
        )

    for ilat in range(overlay["nlat_grid"]):
        ax.plot(
            overlay["xlon"][ilat, :],
            overlay["ylon"][ilat, :],
            color="white",
            linewidth=0.8,
            alpha=0.9,
        )

    ax.plot(
        overlay["x_inner_oval"],
        overlay["y_inner_oval"],
        color="orange",
        linewidth=1.2,
    )

    ax.plot(
        overlay["x_outer_oval"],
        overlay["y_outer_oval"],
        color="orange",
        linewidth=1.2,
    )

    if Shin is False:
        ax.plot(
            overlay["x_ifp"],
            overlay["y_ifp"],
            color="red",
            linewidth=0.8,
            alpha=0.5,
        )

    if Shin:
        if use_north:
            if current_pj == 9:
                _, _, _, _, _, _, moon_S3wlon0 = moonS3wlon_arr(
                    np.array([t0_n]), 'Io')
                fp_traced_arr = fp_traced(moon_S3wlon0[0])      # [deg]
                polar_fp_prediction_plot(ax, fp_traced_arr,
                                         moon_S3wlon0[0],
                                         -1)

        if use_south:
            0

    if Shin is not True:
        if x_selected_pixels is not None and y_selected_pixels is not None:
            if len(x_selected_pixels) > 0:
                ax.plot(
                    x_selected_pixels,
                    y_selected_pixels,
                    linestyle="None",
                    marker=".",
                    color="white",
                    markersize=3.0,
                    alpha=0.95,
                    zorder=6,
                )

    if x_ifp_spin is not None and y_ifp_spin is not None:
        ax.plot(
            x_ifp_spin,
            y_ifp_spin,
            linestyle="None",
            marker="+",
            color="red",
            markersize=6,
            markeredgewidth=1.2,
            zorder=7,
        )

    if x_ifp_first is not None and y_ifp_first is not None:
        ax.plot(
            x_ifp_first,
            y_ifp_first,
            linestyle="None",
            marker="+",
            color="red",
            markersize=6,
            markeredgewidth=1.2,
            zorder=8,
        )

    if x_ifp_last is not None and y_ifp_last is not None:
        ax.plot(
            x_ifp_last,
            y_ifp_last,
            linestyle="None",
            marker="+",
            color="w",
            markersize=6,
            markeredgewidth=1.2,
            zorder=8,
        )

    if Shin:
        print('x_ifp_last:', x_ifp_last)
        print('y_ifp_last:', y_ifp_last)


def style_axes(ax, hemi_spin, xlim, ylim):
    ax.set_xlim(xlim[0], xlim[1])

    # --------------------------------------------------------
    # Requested change:
    # reverse the Y axis only for the northern hemisphere.
    # Southern hemisphere is now plotted in the natural order.
    # --------------------------------------------------------
    if hemi_spin == "N":
        ax.set_ylim(ylim[1], ylim[0])
    else:
        ax.set_ylim(ylim[0], ylim[1])

    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.tick_params(axis="both", which="both", length=0)


def plot_brightness_and_exposure(
    br_w,
    map_exp,
    xbin,
    ybin,
    overlay,
    hemi_spin=None,
    save_png_path=None,
    title_line1=None,
    title_line2=None,
    x_ifp_spin=None,
    y_ifp_spin=None,
    x_ifp_first=None,
    y_ifp_first=None,
    x_ifp_last=None,
    y_ifp_last=None,
    custom_xlim=None,
    custom_ylim=None,
    annotation_text=None,
    x_selected_pixels=None,
    y_selected_pixels=None,
):
    br_display = np.where(br_w > 0, br_w, np.nan)
    exp_display = np.where(map_exp > 0, map_exp, np.nan)

    br_image = format_map_for_display(br_display)
    exp_image = format_map_for_display(exp_display)

    extent = [xbin[0], xbin[-1], ybin[0], ybin[-1]]
    plot_limits = get_plot_limits(hemi_spin)

    if custom_xlim is None:
        custom_xlim = (plot_limits["xmin"], plot_limits["xmax"])
    if custom_ylim is None:
        custom_ylim = (plot_limits["ymin"], plot_limits["ymax"])

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    if title_line1 is not None and title_line2 is not None:
        fig.suptitle(f"{title_line1}\n{title_line2}", fontsize=14)

    vmin, vmax = 1, 1000
    if Shin:
        vmin, vmax = 10, 400
    im0 = axes[0].imshow(
        br_image,
        origin="upper",
        extent=extent,
        cmap=IDL_BLUE_WHITE_CMAP,
        norm=LogNorm(vmin=vmin, vmax=vmax),
        aspect="equal",
    )
    cbar0 = plt.colorbar(im0, ax=axes[0])
    cbar0.set_label("Brightness (kR)")
    axes[0].set_title("Brightness")
    axes[0].set_facecolor(NO_DATA_COLOR)
    apply_plot_overlay(
        axes[0],
        overlay,
        x_ifp_spin=x_ifp_spin,
        y_ifp_spin=y_ifp_spin,
        x_ifp_first=x_ifp_first,
        y_ifp_first=y_ifp_first,
        x_ifp_last=x_ifp_last,
        y_ifp_last=y_ifp_last,
        x_selected_pixels=x_selected_pixels,
        y_selected_pixels=y_selected_pixels,
    )
    style_axes(axes[0], hemi_spin, custom_xlim, custom_ylim)

    if np.any(np.isfinite(exp_display)):
        vmin_exp = np.nanmin(exp_display)
        vmax_exp = np.nanmax(exp_display)
        if np.isclose(vmin_exp, vmax_exp):
            vmax_exp = vmin_exp + 1.0
        exp_norm = Normalize(vmin=vmin_exp, vmax=vmax_exp)
    else:
        exp_norm = Normalize(vmin=0.0, vmax=1.0)

    im1 = axes[1].imshow(
        exp_image,
        origin="upper",
        extent=extent,
        cmap=IDL_BLUE_WHITE_CMAP,
        norm=exp_norm,
        aspect="equal",
    )
    cbar1 = plt.colorbar(im1, ax=axes[1])
    cbar1.set_label("Exposure")
    axes[1].set_title("Exposure")
    axes[1].set_facecolor(NO_DATA_COLOR)
    apply_plot_overlay(
        axes[1],
        overlay,
        x_ifp_spin=x_ifp_spin,
        y_ifp_spin=y_ifp_spin,
        x_ifp_first=x_ifp_first,
        y_ifp_first=y_ifp_first,
        x_ifp_last=x_ifp_last,
        y_ifp_last=y_ifp_last,
        x_selected_pixels=None,
        y_selected_pixels=None,
    )
    style_axes(axes[1], hemi_spin, custom_xlim, custom_ylim)

    if annotation_text is not None:
        axes[0].text(
            0.02,
            0.98,
            annotation_text,
            transform=axes[0].transAxes,
            va="top",
            ha="left",
            fontsize=10,
            color="white",
            bbox=dict(boxstyle="round", facecolor="black",
                      alpha=0.55, edgecolor="none"),
        )

    plt.tight_layout(rect=[0, 0, 1, 0.92])

    if save_png_path is not None:
        os.makedirs(os.path.dirname(save_png_path), exist_ok=True)
        fig.savefig(save_png_path, dpi=200, bbox_inches="tight", format="png")
        print(f"Saved PNG: {save_png_path}")

    plt.show()
    plt.close(fig)


# ------------------------------------------------------------
# Mapping routine
# ------------------------------------------------------------
def map_uvs_data(
    pj_number,
    hemisphere_name,
    tstart,
    tend,
    selected_records,
    name_dir=None,
    power_integration_radius_deg=1.0,
):
    pjname = f"PJ{pj_number}"

    selected_records = filter_valid_fits_records(selected_records)

    if len(selected_records) == 0:
        print()
        print(
            f"No valid FITS files left for {pjname} | {hemisphere_name} | {tstart} -> {tend}")
        return None

    print()
    print("Proceeding to mapping routine")
    print(f"PJ: {pjname}")
    print(f"Hemisphere: {hemisphere_name}")
    print(f"Time window: {tstart} -> {tend}")
    print(f"Number of valid files: {len(selected_records)}")

    for record in selected_records:
        print(f"  {record['filename']}")

    nadir_times_utc = get_nadir_times_utc(selected_records)
    nadir_times_et = np.array([utc_to_et(t)
                              for t in nadir_times_utc], dtype=np.float64)
    nadir_altitude = np.array([compute_nadir_altitude(et)
                              for et in nadir_times_et], dtype=np.float64)

    req, rpol = get_jupiter_radii()

    hemi_spin, minx, maxx, miny, maxy, proj_lat, proj_lon = get_projection_setup_from_hemisphere(
        hemisphere_name
    )

    subsc_lat0 = compute_subspacecraft_latitude(nadir_times_et[0])

    spin_ifp = compute_spin_ifp_from_io_lon(
        hemi_spin=hemi_spin,
        time_et_array=nadir_times_et,
        satellite_fp_path="/Users/shin/Documents/Research/Juno/UVS/Code/PolarProjection/Input/Satellite_FP_JRM33_LA.sav",
    )

    x_ifp_spin_all, y_ifp_spin_all = ortho_latlon_to_xy(
        spin_ifp["ifp_lat_spin"],
        spin_ifp["ifp_lon_spin"],
        proj_lat,
        proj_lon,
    )
    vis_ifp_spin_all = ortho_visible_mask(
        spin_ifp["ifp_lat_spin"],
        spin_ifp["ifp_lon_spin"],
        proj_lat,
        proj_lon,
    )
    x_ifp_spin_all = np.where(vis_ifp_spin_all, x_ifp_spin_all, np.nan)
    y_ifp_spin_all = np.where(vis_ifp_spin_all, y_ifp_spin_all, np.nan)

    midpoint_ifp = compute_midpoint_ifp_position(
        hemi_spin=hemi_spin,
        tstart=tstart,
        tend=tend,
        proj_lat=proj_lat,
        proj_lon=proj_lon,
        satellite_fp_path="/Users/shin/Documents/Research/Juno/UVS/Code/PolarProjection/Input/Satellite_FP_JRM33_LA.sav",
    )

    avg_x_ifp_spin = midpoint_ifp["avg_x_ifp_spin"]
    avg_y_ifp_spin = midpoint_ifp["avg_y_ifp_spin"]
    ifp_lon_mid = midpoint_ifp["ifp_lon_mid"]
    ifp_lat_mid = midpoint_ifp["ifp_lat_mid"]

    overlay_static = prepare_projection_overlay(
        hemi_spin=hemi_spin,
        proj_lat=proj_lat,
        proj_lon=proj_lon,
        dlon=30.0,
        dlat=10.0,
        sav_path="/Users/shin/Documents/Research/Juno/UVS/Code/PolarProjection/Input/OvalBonfond2017.sav",
        satellite_fp_path="/Users/shin/Documents/Research/Juno/UVS/Code/PolarProjection/Input/Satellite_FP_JRM33_LA.sav",
    )

    print()
    print("Nadir times UTC:")
    for t in nadir_times_utc:
        print(f"  {t}")

    print()
    print("Nadir altitude [km]:")
    for alt in nadir_altitude:
        print(f"  {alt:.3f}")

    print()
    print(f"Requested projection hemisphere: {hemi_spin}")
    print(
        f"Diagnostic sub-spacecraft latitude of first file: {subsc_lat0:.3f} deg")
    print(f"Midpoint footprint lon/lat: {ifp_lon_mid:.3f}, {ifp_lat_mid:.3f}")
    print(
        f"Midpoint footprint x/y   : {avg_x_ifp_spin:.6f}, {avg_y_ifp_spin:.6f}")

    delta_xy = np.max(nadir_altitude) * np.tan(np.deg2rad(0.1)) / req

    xbin, ybin, nxmap, nymap = build_map_bins(minx, maxx, miny, maxy, delta_xy)

    print(f"delta_xy: {delta_xy:.8f}")
    print(f"nxmap: {nxmap}")
    print(f"nymap: {nymap}")

    first_products = load_fits_products(selected_records[0]["filepath"])
    wavelength = first_products["WAVELENGTH"]
    nw = len(wavelength)

    map_exp_coadd = np.zeros((nxmap, nymap), dtype=np.float64)
    map_wc_coadd = np.zeros((nxmap, nymap, nw), dtype=np.float64)
    map_phot_coadd = np.zeros((nxmap, nymap, nw), dtype=np.float64)

    map_lat_sum = np.zeros((nxmap, nymap), dtype=np.float64)
    map_lon_sin_sum = np.zeros((nxmap, nymap), dtype=np.float64)
    map_lon_cos_sum = np.zeros((nxmap, nymap), dtype=np.float64)
    map_geo_weight_sum = np.zeros((nxmap, nymap), dtype=np.float64)

    for irec, record in enumerate(selected_records):
        filepath = record["filepath"]
        filename = record["filename"]

        map_exp = np.zeros((nxmap, nymap), dtype=np.float64)
        map_wc = np.zeros((nxmap, nymap, nw), dtype=np.float64)
        map_phot = np.zeros((nxmap, nymap, nw), dtype=np.float64)

        try:
            products = load_fits_products(filepath)
        except Exception as e:
            print(f"ERROR: corrupted FITS file: {filepath}")
            print(f"       reason: {e}")
            continue

        phot_map = products["PHOT_MAP"]
        num_det_map = products["NUM_DET_MAP"]
        int_time_map = products["INTEGRATION_TIME_MAP"]
        lat = products["LATITUDE"]
        lon = products["LONGITUDE"]
        slit_mask_map = products["SLIT_MASK_MAP"]
        wavelength = products["WAVELENGTH"]

        print()
        print(
            f"Processing file {irec + 1}/{len(selected_records)}: {filename}")

        iok_w = np.where(
            (slit_mask_map != 0)
            & (lat != -999.0)
            & (lon != -999.0)
        )

        if len(iok_w[0]) == 0:
            print("  No usable data in this file.")
            continue

        x_wide, y_wide = ortho_latlon_to_xy(
            lat[iok_w], lon[iok_w], proj_lat, proj_lon)

        iok = np.where(
            (x_wide > minx) & (x_wide < maxx) &
            (y_wide > miny) & (y_wide < maxy)
        )[0]

        if len(iok) == 0:
            print("  No projected data in map bounds.")
            continue

        ind_y = iok_w[0][iok]
        ind_x = iok_w[1][iok]
        x_wide = x_wide[iok]
        y_wide = y_wide[iok]

        ix_wide = np.searchsorted(xbin, x_wide, side="right") - 1
        iy_wide = np.searchsorted(ybin, y_wide, side="right") - 1

        valid_bin = (
            (ix_wide >= 0) & (ix_wide < nxmap) &
            (iy_wide >= 0) & (iy_wide < nymap)
        )

        ix_wide = ix_wide[valid_bin]
        iy_wide = iy_wide[valid_bin]
        ind_y = ind_y[valid_bin]
        ind_x = ind_x[valid_bin]

        for j in range(len(ix_wide)):
            ix = ix_wide[j]
            iy = iy_wide[j]
            py = ind_y[j]
            px = ind_x[j]

            map_exp[ix, iy] += int_time_map[py, px]
            map_wc[ix, iy, :] += phot_map[:, py, px]
            map_phot[ix, iy, :] += num_det_map[:, py, px]

            map_exp_coadd[ix, iy] += int_time_map[py, px]
            map_wc_coadd[ix, iy, :] += phot_map[:, py, px]
            map_phot_coadd[ix, iy, :] += num_det_map[:, py, px]

            geo_w = max(float(int_time_map[py, px]), 0.0)
            if geo_w > 0.0:
                this_lat = float(lat[py, px])
                this_lon = float(np.mod(lon[py, px], 360.0))

                map_lat_sum[ix, iy] += geo_w * this_lat
                map_lon_sin_sum[ix, iy] += geo_w * np.sin(np.deg2rad(this_lon))
                map_lon_cos_sum[ix, iy] += geo_w * np.cos(np.deg2rad(this_lon))
                map_geo_weight_sum[ix, iy] += geo_w

        br_single = compute_brightness_map(
            map_wc, map_exp, wavelength, method="dual_band")

        x_ifp_this = x_ifp_spin_all[irec] if irec < len(
            x_ifp_spin_all) else np.nan
        y_ifp_this = y_ifp_spin_all[irec] if irec < len(
            y_ifp_spin_all) else np.nan

        """plot_brightness_and_exposure(
            br_w=br_single,
            map_exp=map_exp,
            xbin=xbin,
            ybin=ybin,
            overlay=overlay_static,
            hemi_spin=hemi_spin,
            title_line1=f"{pjname} | {hemisphere_name} | Spin",
            title_line2=filename,
            x_ifp_spin=x_ifp_this,
            y_ifp_spin=y_ifp_this,
        )"""

    br_coadd = compute_brightness_map(
        map_wc_coadd, map_exp_coadd, wavelength, method="dual_band")

    png_path = build_output_png_path(pj_number, tstart, tend, zoomed=False)
    zoom_png_path = build_output_png_path(pj_number, tstart, tend, zoomed=True)

    finite_spin = np.isfinite(x_ifp_spin_all) & np.isfinite(y_ifp_spin_all)

    if np.any(finite_spin):
        first_idx = np.where(finite_spin)[0][0]
        last_idx = np.where(finite_spin)[0][-1]

        x_ifp_first = float(x_ifp_spin_all[first_idx])
        y_ifp_first = float(y_ifp_spin_all[first_idx])
        x_ifp_last = float(x_ifp_spin_all[last_idx])
        y_ifp_last = float(y_ifp_spin_all[last_idx])
    else:
        x_ifp_first = None
        y_ifp_first = None
        x_ifp_last = None
        y_ifp_last = None

    map_lat_mean, map_lon_mean = finalize_geolocation_maps(
        map_lat_sum=map_lat_sum,
        map_lon_sin_sum=map_lon_sin_sum,
        map_lon_cos_sum=map_lon_cos_sum,
        map_geo_weight_sum=map_geo_weight_sum,
    )

    map_area_km2 = estimate_surface_bin_area_km2(
        map_lat_mean=map_lat_mean,
        map_lon_mean=map_lon_mean,
        radius_km=req,
    )

    power_result = compute_total_auroral_power_gw(
        map_wc=map_wc_coadd,
        map_exp=map_exp_coadd,
        wavelength=wavelength,
        map_lat_mean=map_lat_mean,
        map_lon_mean=map_lon_mean,
        map_area_km2=map_area_km2,
        center_lat_deg=ifp_lat_mid,
        center_lon_deg=ifp_lon_mid,
        radius_deg=power_integration_radius_deg,
        power_scale_factor=2.04,
    )

    total_power_gw = power_result["power_gw"]

    x_selected_power_pixels, y_selected_power_pixels = selection_mask_to_projected_points(
        selection_mask=power_result["selection_mask"],
        xbin=xbin,
        ybin=ybin,
    )

    print()
    print(
        f"Total auroral power within {power_integration_radius_deg:.2f} deg "
        f"of midpoint IFP = {total_power_gw:.6f} GW"
    )

    plot_brightness_and_exposure(
        br_w=br_coadd,
        map_exp=map_exp_coadd,
        xbin=xbin,
        ybin=ybin,
        overlay=overlay_static,
        hemi_spin=hemi_spin,
        save_png_path=png_path,
        title_line1=f"PJ{pj_number} | {hemisphere_name} | Co-added *",
        title_line2=f"{tstart} to {tend}",
        x_ifp_first=x_ifp_first,
        y_ifp_first=y_ifp_first,
        x_ifp_last=x_ifp_last,
        y_ifp_last=y_ifp_last,
    )

    zoom_xlim, zoom_ylim = get_zoom_limits(
        center_x=avg_x_ifp_spin,
        center_y=avg_y_ifp_spin,
        hemi_spin=hemi_spin,
        half_size=0.2,
    )

    zoom_annotation = (
        f"Midpoint IFP power\n"
        f"r <= {power_integration_radius_deg:.1f} deg\n"
        f"{total_power_gw:.3f} GW"
    )

    plot_brightness_and_exposure(
        br_w=br_coadd,
        map_exp=map_exp_coadd,
        xbin=xbin,
        ybin=ybin,
        overlay=overlay_static,
        hemi_spin=hemi_spin,
        save_png_path=zoom_png_path,
        title_line1=f"PJ{pj_number} | {hemisphere_name} | Co-added *",
        title_line2=f"{tstart} to {tend}",
        x_ifp_first=x_ifp_first,
        y_ifp_first=y_ifp_first,
        x_ifp_last=x_ifp_last,
        y_ifp_last=y_ifp_last,
        custom_xlim=zoom_xlim,
        custom_ylim=zoom_ylim,
        annotation_text=zoom_annotation,
        x_selected_pixels=x_selected_power_pixels,
        y_selected_pixels=y_selected_power_pixels,
    )

    return {
        "nadir_times_utc": nadir_times_utc,
        "nadir_times_et": nadir_times_et,
        "nadir_altitude": nadir_altitude,
        "req": req,
        "rpol": rpol,
        "hemi_spin": hemi_spin,
        "delta_xy": delta_xy,
        "xbin": xbin,
        "ybin": ybin,
        "map_exp_coadd": map_exp_coadd,
        "map_wc_coadd": map_wc_coadd,
        "map_phot_coadd": map_phot_coadd,
        "map_lat_mean": map_lat_mean,
        "map_lon_mean": map_lon_mean,
        "map_area_km2": map_area_km2,
        "wavelength": wavelength,
        "br_coadd": br_coadd,
        "spin_ifp": spin_ifp,
        "midpoint_ifp": midpoint_ifp,
        "avg_x_ifp_spin": avg_x_ifp_spin,
        "avg_y_ifp_spin": avg_y_ifp_spin,
        "auroral_power_radius_deg": power_integration_radius_deg,
        "auroral_power_gw": total_power_gw,
        "power_selection_mask": power_result["selection_mask"],
        "power_distance_deg_map": power_result["distance_deg_map"],
        "x_selected_power_pixels": x_selected_power_pixels,
        "y_selected_power_pixels": y_selected_power_pixels,
        "png_path": png_path,
        "zoom_png_path": zoom_png_path,
    }


# ------------------------------------------------------------
# Process all time intervals containing data
# ------------------------------------------------------------
def process_all_time_intervals(current_pj, hemisphere_windows, file_records):
    processed_count = 0
    hemisphere_png_paths = {
        "NORTH": {"average": [], "zoomed": []},
        "SOUTH": {"average": [], "zoomed": []},
    }

    for hemisphere_name, time_et in hemisphere_windows:
        if time_et.size < 2:
            continue

        time_utc = [et_to_utc(t) for t in time_et]

        for i in range(len(time_utc) - 1):
            tstart = time_utc[i]
            tend = time_utc[i + 1]

            selected_records = find_files_in_time_range(
                file_records, tstart, tend)

            print()
            print(f"{hemisphere_name} | {tstart} -> {tend}")
            print(f"Number of files in range: {len(selected_records)}")

            for record in selected_records:
                print(f"  {record['filename']}")

            if len(selected_records) == 0:
                continue

            result = map_uvs_data(
                pj_number=current_pj,
                hemisphere_name=hemisphere_name,
                tstart=tstart,
                tend=tend,
                selected_records=selected_records,
                power_integration_radius_deg=1.0,
            )

            if result is not None:
                processed_count += 1

                if result.get("png_path") is not None:
                    hemisphere_png_paths[hemisphere_name]["average"].append(
                        result["png_path"])

                if result.get("zoom_png_path") is not None:
                    hemisphere_png_paths[hemisphere_name]["zoomed"].append(
                        result["zoom_png_path"])

    return processed_count, hemisphere_png_paths


# ------------------------------------------------------------
# Target time
# ------------------------------------------------------------
def target_time():
    t0_n, t1_n = [0], [0]
    t0_s, t1_s = [0], [0]
    if current_pj == 3:
        if use_south:
            t0_s = [utc_to_et("2016-12-11 17:50:40"),
                    utc_to_et("2016-12-11 18:17:57")]
            t1_s = [utc_to_et("2016-12-11 17:51:40"),
                    utc_to_et("2016-12-11 18:18:57")]
    elif current_pj == 7:
        if use_south:
            t0_n = [utc_to_et("2017-07-11 02:53:14")]
            t1_n = [utc_to_et("2017-07-11 02:54:14")]
    elif current_pj == 9:
        if use_north:
            t0_n = [utc_to_et("2017-10-24 16:48:24")]
            t1_n = [utc_to_et("2017-10-24 16:49:24")]
        if use_south:
            t0_s = [utc_to_et("2017-10-24 19:05:29")]
            t1_s = [utc_to_et("2017-10-24 19:06:29")]
    elif current_pj == 11:
        if use_north:
            t0_n = [utc_to_et("2018-02-07 13:15:40")]
            t1_n = [utc_to_et("2018-02-07 13:21:45")]

    return t0_n, t1_n, t0_s, t1_s


# ------------------------------------------------------------
# Main routine
# ------------------------------------------------------------
def fp_traced(target_moon_s3_obs):
    """
    Args:
        target_moon_s3_obs (float): moon position at the time of the footprint observation [deg]

    Returns:
        _type_: _description_
    """
    filename = 'data_'+TARGET_MOON[0]+'FP_interp_map_' + \
        str(int(alt_ref[fp_alt_target]))+'km_'+retrieval+'.txt'
    interp = np.loadtxt('results/reflect_2/'+exname+'/'+filename)
    moon_s3_obs = interp[:, 0]      # [deg]
    idx = np.argmin(abs(moon_s3_obs-target_moon_s3_obs))

    positions = interp[idx, :]
    print('interp.shape:', interp.shape)
    print('positions.shape:', positions.shape)
    return positions


# ------------------------------------------------------------
# Generate the footpath of MAW
# ------------------------------------------------------------
def fp_path():
    filename = 'data_'+TARGET_MOON[0]+'FP_interp_map_' + \
        str(int(alt_ref[fp_alt_target]))+'km_'+retrieval+'.txt'
    interp = np.loadtxt('results/reflect_2/'+exname+'/'+filename)
    moon_s3_obs = interp[:, 0]

    # j=1: colatitude, j=2: w-longitude [rad]
    pos_N_MAW = interp[:, 1:3]
    pos_S_MAW = interp[:, 1+3*(3+reflections):3*(3+reflections)+3]

    pos_S_RAW1 = interp[:, 4:6]
    pos_N_RAW1 = interp[:, 4+3*(3+reflections):3*(3+reflections)+6]

    return moon_s3_obs, pos_N_MAW, pos_S_MAW, pos_N_RAW1, pos_S_RAW1


# ------------------------------------------------------------
# Instantaneous footprint position
# ------------------------------------------------------------
def instantaneous(target_moon_s3_obs, hem):
    Ai_best, ni_best, _, Hp_best = load_best_fit(exname, ni_num,
                                                 Ai_num, Ti_num,
                                                 Zi, Te,
                                                 retrieval)

    s3wlon_t0 = np.radians(target_moon_s3_obs)

    S_A0 = Wave.Awave().tracefield(r_moon,
                                   s3wlon_t0,
                                   0.0)

    # Initital trace
    # -> Instantaneous position at a selected altitude
    _, _, s3wlon, theta_s3, _, alt_flag = Wave.Awave().trace3_reflect(r_moon,
                                                                      s3wlon_t0,
                                                                      0.0,
                                                                      S_A0,
                                                                      Ai_best,
                                                                      ni_best,
                                                                      Hp_best,
                                                                      hem)
    non_0 = np.array(np.where(alt_flag != 0)[0])
    insta_fp_pos = np.zeros(2)
    insta_fp_pos[0] = theta_s3[non_0][fp_alt_target]  # Colatitude [rad]
    insta_fp_pos[1] = s3wlon[non_0][fp_alt_target]    # W.longitude [rad]

    return insta_fp_pos


def polar_fp_prediction_plot(ax, fp_traced_arr, target_moon_s3_obs, hem):
    if hem == -1:
        j_add = 0
    elif hem == 1:
        j_add = 1

    # MAW & RAW & TEB
    for j in range(3+reflections):
        j = 2*j + j_add
        colat = fp_traced_arr[3*j+1]    # [rad]
        wlon = fp_traced_arr[3*j+2]     # [rad]
        sign = -np.sign(0.5*np.pi-colat)
        x_fp = np.sin(colat)*np.cos(2*np.pi-wlon)
        y_fp = np.sin(colat)*np.sin(2*np.pi-wlon)
        if j in [3+reflections-2, 3+reflections-1, 2*(3+reflections)-2, 2*(3+reflections)-1]:
            marker = 'D'
        else:
            marker = 'o'
        ax.scatter(
            -y_fp,
            sign*x_fp,
            marker=marker,
            fc=UC.red, ec='w', s=15.0, zorder=2.0
        )

    # Foot path
    if hem == -1:
        _, pos_MAW, _, _, _ = fp_path()
    elif hem == 1:
        _, _, pos_MAW, _, _ = fp_path()
    sort = np.argsort(pos_MAW[:, 1])
    x_fpath = np.sin(pos_MAW[sort, 0])*np.cos(2*np.pi-pos_MAW[sort, 1])
    y_fpath = np.sin(pos_MAW[sort, 0])*np.sin(2*np.pi-pos_MAW[sort, 1])
    ax.plot(-y_fpath, sign*x_fpath, color=UC.red, zorder=1.0)

    # Instantaneous footprint positions
    insta_fp_pos = instantaneous(target_moon_s3_obs, hem)
    x_insta_fp = np.sin(insta_fp_pos[0])*np.cos(2*np.pi-insta_fp_pos[1])
    y_insta_fp = np.sin(insta_fp_pos[0])*np.sin(2*np.pi-insta_fp_pos[1])
    ax.scatter(
        -y_insta_fp,
        sign*x_insta_fp,
        marker='D', fc='k', ec='w', s=15.0,
    )
    return None


# ------------------------------------------------------------
# Main routine
# ------------------------------------------------------------
def main():
    dt = 1.2 * 60.0        # Time window size
    data_root = "/Users/shin/Documents/Research/Juno/UVS/Code/PolarProjection/Data/JunoUVS_SSI_Data"

    pj_time_utc = get_pj_time(current_pj)
    print(f"PJ{current_pj} UTC time: {pj_time_utc}")

    try:
        pjtime_et = utc_to_et(pj_time_utc)

        # This effectively does not clip the south window.
        south_start_override_et = doy_utc_to_et(2010, 197, 6, 3, 6)

        hemisphere_windows = generate_time_windows(
            pjtime_et,
            dt,
            use_north=use_north,
            use_south=use_south,
            south_start_override_et=south_start_override_et,
        )

        all_files = list_pj_files(data_root, current_pj)
        file_records = build_file_time_records(all_files)

        print(
            f"Total files found in {data_root}/PJ{current_pj}: {len(all_files)}")

        processed_count, hemisphere_png_paths = process_all_time_intervals(
            current_pj=current_pj,
            hemisphere_windows=hemisphere_windows,
            file_records=file_records,
        )

        for hemisphere_name, png_groups in hemisphere_png_paths.items():
            build_hemisphere_gifs(
                pj_number=current_pj,
                hemisphere_name=hemisphere_name,
                avg_png_paths=png_groups["average"],
                zoom_png_paths=png_groups["zoomed"],
                duration=0.8,
            )

        if processed_count == 0:
            print()
            print("No files found in any time interval.")
        else:
            print()
            print(f"Processed {processed_count} time interval(s) with data.")

    finally:
        unload_spice_kernels()


# ------------------------------------------------------------
# Entry point
# ------------------------------------------------------------
if __name__ == "__main__":
    # Name of execution
    exname = '003/20250516_054'

    # Input about Juno observation
    TARGET_MOON = 'Io'
    PJ_LIST = [9]
    TARGET_HEM = 'N'
    FLIP = False            # ALWAYS FALSE! Flip the flag (TEB <-> MAW)
    Ai_num = 3
    ni_num = 50
    Ti_num = 60
    Zi = 1.3                # Io: 1.3 / Eu: 1.4 / Ga: 1.3
    Te = 6.0                # Io: 6.0 [eV]/ Eu: 20.0 / Ga: 300.0
    reflections = 8         # fixed at 8
    alt_ref = [1500.0, 1400.0, 1300.0, 1200.0, 1100.0,
               1000.0, 900.0, 800.0, 700.0, 600.0,
               500.0, 400.0, 300.0, 200.0, 100.0,
               50.0, 10.0, 5.0]
    reflections = 8                     # fixed at 8
    reflect_alt_target = -len(alt_ref)  # ALWAYS NEGATIVE!!!
    fp_alt_target = -7                  # ALWAYS NEGATIVE!!!
    retrieval = 'cold5'                 # 'best', 'hot', 'dense'

    # Don't need to change below
    current_pj = PJ_LIST[0]

    use_north = False
    use_south = False
    if TARGET_HEM == 'N':
        use_north = True
    elif TARGET_HEM == 'S':
        use_south = True
    elif TARGET_HEM == 'both':
        use_north = True
        use_south = True

    meta_kernel = "./KERNELS/Meta_kernel_all.ker"
    load_spice_kernels(meta_kernel)

    t0_n_list, t1_n_list, t0_s_list, t1_s_list = target_time()
    time_index = 0
    t0_n, t1_n = t0_n_list[time_index], t1_n_list[time_index]
    t0_s, t1_s = t0_s_list[time_index], t1_s_list[time_index]

    # Orbital distance at the PJ time
    if TARGET_HEM == 'N':
        TARGET_ET = np.array([(t0_n+t1_n)*0.5])
    elif TARGET_HEM == 'S':
        TARGET_ET = np.array([(t0_s+t1_s)*0.5])
    _, _, _, r_moon_obs, _, _, s3wlon_moon_obs = moonS3wlon_arr(TARGET_ET,
                                                                TARGET_MOON)
    r_moon = r_moon_obs[0]
    print('Orbital distance [RJ]:', r_moon/RJ)

    main()
