#!/usr/bin/env python3
"""
Minor Planets Finder — fully local computation (no per-object HTTP requests)

Pipeline (like Stellarium):
  1. Download MPCORB.DAT.gz from MPC once (local cache, refresh every 7 days)
  2. Download CometEls.txt from MPC once (same cache policy)
  3. Load with pandas (only useful columns), pre-filter by H
  4. Propagate all orbits with Kepler's method (vectorised numpy)
  5. Compute geocentric position + V magnitude (H-G model / comet formula)
  6. Filter by V <= magnitude_limit
  7. Convert RA/Dec → topocentric Az/El with astropy
  8. Filter by sky window (azimuth/altitude range)

No HTTP request per object.

Usage:
    python minorplanetsfinder.py
    python minorplanetsfinder.py --date 2026-03-20 --time 22:00
    python minorplanetsfinder.py --body comets --sort vis
    python minorplanetsfinder.py --config other.json
"""

import argparse
import gzip
import json
import re
import sys
import warnings
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

warnings.filterwarnings("ignore")

try:
    import numpy as np
    import pandas as pd
    import requests
    import astropy.units as u
    from astropy.coordinates import (
        AltAz, EarthLocation, SkyCoord,
        get_body_barycentric, solar_system_ephemeris,
    )
    from astropy.time import Time
    from astropy.coordinates import get_sun
    from rich.console import Console
    from rich.panel import Panel
    from rich.progress import (
        Progress, SpinnerColumn, BarColumn, DownloadColumn,
        TransferSpeedColumn, TimeRemainingColumn, TextColumn,
    )
    from rich.prompt import Prompt
    from rich.table import Table
    from rich import box
except ImportError as e:
    print(f"Missing libraries: {e}")
    print("pip install numpy pandas requests astropy rich")
    sys.exit(1)

console = Console()

MPCORB_URL      = "https://minorplanetcenter.net/iau/MPCORB/MPCORB.DAT.gz"
COMET_URL       = "https://minorplanetcenter.net/iau/MPCORB/CometEls.txt"
CACHE_DIR       = Path.home() / ".minorplanetsfinder"
CACHE_FILE      = CACHE_DIR / "MPCORB.DAT.gz"
COMET_CACHE     = CACHE_DIR / "CometEls.txt"
MAX_CACHE_DAYS  = 7
ECLIPTIC_OBL    = np.radians(23.43929111)   # ecliptic obliquity J2000.0


# ── Download & cache ─────────────────────────────────────────────────────────────

def _cache_stale() -> bool:
    if not CACHE_FILE.exists():
        return True
    age = datetime.now() - datetime.fromtimestamp(CACHE_FILE.stat().st_mtime)
    return age.days >= MAX_CACHE_DAYS


def ensure_mpcorb():
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    if not _cache_stale():
        age = datetime.now() - datetime.fromtimestamp(CACHE_FILE.stat().st_mtime)
        console.print(f"[dim]MPCORB.DAT cached ({age.days} days ago).[/dim]")
        return

    console.print("[cyan]Downloading MPCORB.DAT.gz from MPC (~100 MB)...[/cyan]")
    r = requests.get(MPCORB_URL, stream=True, timeout=180)
    r.raise_for_status()
    total = int(r.headers.get("content-length", 0))

    with Progress(
        SpinnerColumn(),
        TextColumn("[cyan]{task.description}"),
        BarColumn(),
        DownloadColumn(),
        TransferSpeedColumn(),
        TimeRemainingColumn(),
    ) as prog:
        task = prog.add_task("MPCORB.DAT.gz", total=total)
        with open(CACHE_FILE, "wb") as f:
            for chunk in r.iter_content(chunk_size=131072):
                f.write(chunk)
                prog.advance(task, len(chunk))

    console.print(f"[green]Saved: {CACHE_FILE}[/green]")


def ensure_comets():
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    if COMET_CACHE.exists():
        age = datetime.now() - datetime.fromtimestamp(COMET_CACHE.stat().st_mtime)
        if age.days < MAX_CACHE_DAYS:
            console.print(f"[dim]CometEls.txt cached ({age.days} days ago).[/dim]")
            return
    console.print("[cyan]Downloading CometEls.txt from MPC (~300 KB)...[/cyan]")
    r = requests.get(COMET_URL, timeout=60)
    r.raise_for_status()
    COMET_CACHE.write_bytes(r.content)
    console.print(f"[green]Saved: {COMET_CACHE}[/green]")


# ── Parsing MPCORB ───────────────────────────────────────────────────────────────
# Fixed-width format, MPC docs:
# https://minorplanetcenter.net/iau/info/MPOrbitFormat.html

_COLSPECS = [
    (0,   7),    # designation (packed)
    (8,  13),    # H  – absolute magnitude
    (14, 19),    # G  – slope parameter
    (20, 25),    # epoch (packed)
    (26, 35),    # M  – mean anomaly at epoch (degrees)
    (37, 46),    # ω  – argument of perihelion (degrees)
    (48, 57),    # Ω  – longitude of ascending node (degrees)
    (59, 68),    # i  – inclination to ecliptic (degrees)
    (70, 79),    # e  – eccentricity
    (92, 103),   # a  – semimajor axis (AU)
    (166, 194),  # readable designation, e.g. "(1) Ceres" or "(12345)"
]
_COLNAMES = ["desig", "H", "G", "epoch_packed", "M", "omega", "Omega", "i", "e", "a", "name_raw"]


def _parse_name(desig, name_raw) -> str:
    try:
        s = str(name_raw).strip()
        if s and s not in ("nan", "None"):
            m = re.match(r'\(\d+\)\s+(.+)', s)
            if m:
                return m.group(1).strip()
    except Exception:
        pass
    d = str(desig).strip()
    try:
        return str(int(d))
    except ValueError:
        return d


# MPC packed epoch: K249B → 2024-Sep-11
_CENTURY  = {"I": 1800, "J": 1900, "K": 2000}
_PM       = {str(n): n for n in range(1, 10)}
_PM.update({"A": 10, "B": 11, "C": 12})
_PD       = {str(n): n for n in range(1, 10)}
_PD.update({chr(ord("A") + n): 10 + n for n in range(22)})


def _unpack_epoch(s) -> float:
    try:
        c = _CENTURY[s[0]]
        y = c + int(s[1:3])
        m = _PM[s[3]]
        d = _PD[s[4]]
        return Time(datetime(y, m, d), scale="tt").jd
    except Exception:
        return np.nan


def _epochs_to_jd_vectorized(epoch_packed: pd.Series) -> pd.Series:
    """Convert packed epoch column → Julian Date without Python loop (Meeus ch. 7)."""
    s = epoch_packed.str.strip()

    century = s.str[0].map({"I": 1800.0, "J": 1900.0, "K": 2000.0})
    yr2     = pd.to_numeric(s.str[1:3], errors="coerce")
    year    = century + yr2

    month = s.str[3].map({str(n): float(n) for n in range(1, 10)} |
                         {"A": 10.0, "B": 11.0, "C": 12.0})
    day   = s.str[4].map({str(n): float(n) for n in range(1, 10)} |
                         {chr(ord("A") + n): float(10 + n) for n in range(22)})

    y = np.where(month <= 2, year - 1, year)
    m = np.where(month <= 2, month + 12, month)
    d = day.to_numpy(dtype=float)

    A  = np.floor(y / 100.0)
    B  = 2.0 - A + np.floor(A / 4.0)
    jd = np.floor(365.25 * (y + 4716.0)) + np.floor(30.6001 * (m + 1.0)) + d + B - 1524.5

    valid = (
        pd.to_numeric(pd.Series(jd), errors="coerce").notna() &
        month.notna() & day.notna() & year.notna()
    )
    result = pd.Series(jd, index=epoch_packed.index, dtype=float)
    result[~valid] = np.nan
    return result


def _names_vectorized(desig: pd.Series, name_raw: pd.Series) -> pd.Series:
    """Extract readable names with str.extract() — no Python loop."""
    extracted = (
        name_raw.astype(str)
                .str.extract(r'\(\d+\)\s+(.+)', expand=False)
                .str.strip()
    )
    fallback = desig.str.strip().str.lstrip("0").replace("", pd.NA)
    return extracted.where(extracted.notna() & (extracted != ""), fallback).fillna(desig.str.strip())


def load_mpcorb(h_limit: float) -> pd.DataFrame:
    """
    Load MPCORB.DAT.gz in chunks with live progress,
    pre-filtered by H <= h_limit. All transforms are vectorised.
    """
    CHUNK = 60_000
    chunks: list[pd.DataFrame] = []
    total_read = 0

    # ── 1. Read ───────────────────────────────────────────────────────────────
    with Progress(SpinnerColumn(), TextColumn("[cyan]{task.description}[/cyan]"),
                  console=console) as prog:
        task = prog.add_task("Reading MPCORB.DAT...", total=None)
        with gzip.open(CACHE_FILE, "rt", encoding="ascii", errors="ignore") as f:
            for chunk in pd.read_fwf(
                f, colspecs=_COLSPECS, names=_COLNAMES,
                header=None, dtype=str, chunksize=CHUNK,
            ):
                total_read += len(chunk)
                prog.update(task, description=f"Reading MPCORB.DAT... {total_read:,} rows")
                chunk["H"] = pd.to_numeric(chunk["H"], errors="coerce")
                chunk = chunk[chunk["H"].notna() & (chunk["H"] <= h_limit)]
                if not chunk.empty:
                    chunks.append(chunk)

    if not chunks:
        return pd.DataFrame()

    # ── 2. Concat + numeric conversions ──────────────────────────────────────
    with console.status(f"[cyan]Merging {len(chunks)} chunks ({total_read:,} rows read)...[/cyan]"):
        df = pd.concat(chunks, ignore_index=True)
        df["G"] = pd.to_numeric(df["G"], errors="coerce").fillna(0.15)
        for col in ["M", "omega", "Omega", "i", "e", "a"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df = df.dropna(subset=["M", "omega", "Omega", "i", "e", "a"])

    # ── 3. Epoch → JD (vectorised, Meeus formula) ────────────────────────────
    with console.status(f"[cyan]Computing JD epochs for {len(df):,} objects...[/cyan]"):
        df["epoch_jd"] = _epochs_to_jd_vectorized(df["epoch_packed"])
        df = df.dropna(subset=["epoch_jd"])

    # ── 4. Readable names (vectorised, str.extract) ───────────────────────────
    with console.status(f"[cyan]Extracting names for {len(df):,} objects...[/cyan]"):
        df["name"] = _names_vectorized(df["desig"], df["name_raw"])

    df = df.reset_index(drop=True)
    console.print(f"[dim]{len(df):,} objects loaded (H ≤ {h_limit:.1f}).[/dim]")
    return df


# ── Parsing & loading comets (CometEls.txt) ──────────────────────────────────────
# MPC format: https://minorplanetcenter.net/iau/info/CometOrbitFormat.html
# Columns are 1-indexed in docs → here 0-indexed (exclusive end)

_COMET_COLSPECS = [
    (0,   4),    # num (periodic comet number, right-justified, blank if unnumbered)
    (4,   5),    # orbit_type (C, P, D, X, ...)
    (5,  13),    # desig_packed
    (14, 18),    # tp_year   (0-indexed; MPC 1-indexed cols 15-18)
    (19, 21),    # tp_month  (0-indexed; MPC 1-indexed cols 20-21)
    (21, 29),    # tp_day decimal (leading space stripped by pandas)
    (30, 39),    # q – perihelion distance (AU)
    (40, 49),    # e – eccentricity
    (50, 59),    # omega – argument of perihelion (°)
    (60, 69),    # Omega – longitude of ascending node (°)
    (70, 79),    # i – inclination (°)
    (91, 95),    # H – total absolute magnitude
    (96, 100),   # G – photometric index (n, typically ≈ 4)
    (102, 158),  # name/designation
]
_COMET_COLNAMES = [
    "num", "orbit_type", "desig_packed",
    "tp_year", "tp_month", "tp_day",
    "q", "e", "omega", "Omega", "i",
    "H", "G", "name",
]


def _tp_to_jd(y, m, d) -> float:
    """Convert year/month/day.decimal → JD (Meeus ch. 7)."""
    try:
        yi, mi, di = float(y), float(m), float(d)
        if not (np.isfinite(yi) and np.isfinite(mi) and np.isfinite(di)):
            return np.nan
        yi, mi = int(yi), int(mi)
        if mi <= 2:
            yi -= 1
            mi += 12
        A = int(yi / 100)
        B = 2 - A + int(A / 4)
        return int(365.25 * (yi + 4716)) + int(30.6001 * (mi + 1)) + di + B - 1524.5
    except Exception:
        return np.nan


def load_comets(h_limit: float) -> pd.DataFrame:
    """Load and pre-filter CometEls.txt."""
    with console.status("[cyan]Loading CometEls.txt...[/cyan]"):
        try:
            import io
            raw = COMET_CACHE.read_text(encoding="ascii", errors="ignore")
            # Remove MPC separator lines (all-dash rows) WITHOUT using comment="-"
            # (comment="-" also truncates names that contain a hyphen, e.g. "Hale-Bopp")
            cleaned = io.StringIO("".join(
                line for line in raw.splitlines(keepends=True)
                if not line.lstrip().startswith("---")
            ))
            df = pd.read_fwf(
                cleaned,
                colspecs=_COMET_COLSPECS,
                names=_COMET_COLNAMES,
                header=None,
                dtype=str,
            )
        except Exception as exc:
            console.print(f"[red]Error parsing CometEls.txt: {exc}[/red]")
            return pd.DataFrame()

        for col in ["q", "e", "omega", "Omega", "i"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df["H"] = pd.to_numeric(df["H"], errors="coerce")
        df["G"] = pd.to_numeric(df["G"], errors="coerce").fillna(4.0)

        tp_y = pd.to_numeric(df["tp_year"],  errors="coerce")
        tp_m = pd.to_numeric(df["tp_month"], errors="coerce")
        tp_d = pd.to_numeric(df["tp_day"],   errors="coerce")
        df["tp_jd"] = [_tp_to_jd(y, m, d) for y, m, d in zip(tp_y, tp_m, tp_d)]

        df = df.dropna(subset=["q", "e", "omega", "Omega", "i", "tp_jd", "H"])
        df = df[(df["e"] > 0) & (df["q"] > 0)]
        df = df[df["H"] <= h_limit].copy()

        # Readable designation
        num    = df["num"].astype(str).str.strip()
        otype  = df["orbit_type"].astype(str).str.strip()
        dpacked = df["desig_packed"].astype(str).str.strip()
        df["desig"] = np.where(num.str.len() > 0, num + otype, dpacked)
        df["name"]  = df["name"].astype(str).str.strip()
        df.loc[df["name"].isin(["nan", "", "None"]), "name"] = df["desig"]

    df = df.reset_index(drop=True)
    console.print(f"[dim]{len(df):,} comets loaded (H ≤ {h_limit:.1f}).[/dim]")
    return df


# ── Orbital mechanics (vectorised numpy) ─────────────────────────────────────────

def _kepler_solve(M: np.ndarray, e: np.ndarray) -> np.ndarray:
    """Solve M = E − e·sin(E) for E with Newton-Raphson (vectorised)."""
    E = M.copy()
    for _ in range(30):
        dE = (M - E + e * np.sin(E)) / (1.0 - e * np.cos(E))
        E += dE
        if np.max(np.abs(dE)) < 1e-10:
            break
    return E


def _ecl_to_eq(x, y, z):
    """Ecliptic J2000 → equatorial J2000 (ICRS)."""
    ce, se = np.cos(ECLIPTIC_OBL), np.sin(ECLIPTIC_OBL)
    return x, ce * y - se * z, se * y + ce * z


def _eq_to_ecl(x, y, z):
    """Equatorial J2000 (ICRS) → ecliptic J2000."""
    ce, se = np.cos(ECLIPTIC_OBL), np.sin(ECLIPTIC_OBL)
    return x, ce * y + se * z, -se * y + ce * z


def _propagate(df: pd.DataFrame, t_jd: float):
    """
    Propagate all orbital elements to t_jd.
    Returns (x, y, z) heliocentric ecliptic J2000 in AU.
    """
    k2 = 0.01720209895 ** 2          # Gaussian gravitational constant (AU³/day²)
    a  = df["a"].values
    e  = df["e"].values
    i  = np.radians(df["i"].values)
    w  = np.radians(df["omega"].values)   # ω
    W  = np.radians(df["Omega"].values)   # Ω
    M0 = np.radians(df["M"].values)
    t0 = df["epoch_jd"].values

    n = np.sqrt(k2 / a ** 3)              # mean motion (rad/day)
    M = (M0 + n * (t_jd - t0)) % (2 * np.pi)
    E = _kepler_solve(M, e)

    # Orbital plane
    x_orb = a * (np.cos(E) - e)
    y_orb = a * np.sqrt(np.maximum(0.0, 1 - e ** 2)) * np.sin(E)

    # Rotation to ecliptic (R₃(−Ω)·R₁(−i)·R₃(−ω))
    cw, sw = np.cos(w), np.sin(w)
    cW, sW = np.cos(W), np.sin(W)
    ci, si = np.cos(i), np.sin(i)

    x_ecl = (cw*cW - sw*sW*ci) * x_orb + (-sw*cW - cw*sW*ci) * y_orb
    y_ecl = (cw*sW + sw*cW*ci) * x_orb + (-sw*sW + cw*cW*ci) * y_orb
    z_ecl = (sw * si)           * x_orb + ( cw * si)           * y_orb
    return x_ecl, y_ecl, z_ecl


def _propagate_comets(df: pd.DataFrame, t_jd: float):
    """
    Propagate elliptic comet orbits (e < 0.9999) to t_jd.
    Parabolic/hyperbolic comets return NaN positions.
    """
    k2  = 0.01720209895 ** 2
    q   = df["q"].values
    e   = df["e"].values
    ell = e < 0.9999

    with np.errstate(invalid="ignore", divide="ignore"):
        a  = np.where(ell, q / np.maximum(1.0 - e, 1e-12), np.nan)
        n  = np.where(ell, np.sqrt(k2 / np.maximum(a**3, 1e-20)), np.nan)
        tp = df["tp_jd"].values
        M  = np.where(ell, (n * (t_jd - tp)) % (2 * np.pi), np.nan)

    E     = np.where(np.isfinite(M), M.copy(), np.nan)
    valid = ell & np.isfinite(M)
    if np.any(valid):
        E_v, e_v, M_v = E[valid], e[valid], M[valid]
        for _ in range(30):
            dE = (M_v - E_v + e_v * np.sin(E_v)) / (1.0 - e_v * np.cos(E_v))
            E_v += dE
            if np.max(np.abs(dE)) < 1e-10:
                break
        E[valid] = E_v

    x_orb = np.where(valid, a * (np.cos(E) - e),                               np.nan)
    y_orb = np.where(valid, a * np.sqrt(np.maximum(0.0, 1 - e**2)) * np.sin(E), np.nan)

    i  = np.radians(df["i"].values)
    w  = np.radians(df["omega"].values)
    W  = np.radians(df["Omega"].values)
    cw, sw = np.cos(w), np.sin(w)
    cW, sW = np.cos(W), np.sin(W)
    ci, si = np.cos(i), np.sin(i)

    x_ecl = (cw*cW - sw*sW*ci)*x_orb + (-sw*cW - cw*sW*ci)*y_orb
    y_ecl = (cw*sW + sw*cW*ci)*x_orb + (-sw*sW + cw*cW*ci)*y_orb
    z_ecl = (sw*si)*x_orb            + (cw*si)*y_orb
    return x_ecl, y_ecl, z_ecl


def _earth_helio_ecl(t: Time):
    """Heliocentric Earth position in ecliptic J2000 (AU)."""
    with solar_system_ephemeris.set("builtin"):
        eb = get_body_barycentric("earth", t)
        sb = get_body_barycentric("sun",   t)
    dx = (eb.x - sb.x).to(u.au).value
    dy = (eb.y - sb.y).to(u.au).value
    dz = (eb.z - sb.z).to(u.au).value
    return _eq_to_ecl(dx, dy, dz)


def _hg_magnitude(H, G, r, delta, phase_rad):
    """H-G magnitude model (Bowell et al. 1989). Fully vectorised."""
    th   = np.tan(np.clip(phase_rad, 0.0, np.pi * 0.9999) / 2)
    Phi1 = np.exp(-3.332  * th ** 0.631)
    Phi2 = np.exp(-1.862  * th ** 1.218)
    return H + 5.0 * np.log10(np.maximum(r * delta, 1e-6)) \
             - 2.5 * np.log10(np.maximum((1 - G) * Phi1 + G * Phi2, 1e-10))


def _comet_magnitude(H, G, r, delta):
    """m = H + 5·log10(Δ) + 2.5·n·log10(r), with n = G (default 4)."""
    n = np.where(np.isfinite(G) & (G > 0), G, 4.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        return (H
                + 5.0 * np.log10(np.maximum(delta, 1e-6))
                + 2.5 * n * np.log10(np.maximum(r, 1e-6)))


# ── Sky window filter ────────────────────────────────────────────────────────────

def _in_window(az: np.ndarray, el: np.ndarray, window: dict) -> np.ndarray:
    az0, az1 = window["azimuth_min"],  window["azimuth_max"]
    el0, el1 = window["altitude_min"], window["altitude_max"]
    if az0 <= az1:
        ok_az = (az >= az0) & (az <= az1)
    else:
        ok_az = (az >= az0) | (az <= az1)
    return ok_az & (el >= el0) & (el <= el1)


# ── Position computation for a single epoch ──────────────────────────────────────

def _positions_at(df: pd.DataFrame, t_jd: float, earth_ecl, t_astropy: Time,
                  earth_loc: EarthLocation, window: dict,
                  mag_limit: float, mag_min: float = 0.0) -> pd.DataFrame:
    """
    Propagate all orbits to t_jd, compute V magnitude and Az/El.
    Returns the visible subset within the sky window.
    """
    xe, ye, ze = earth_ecl

    xa, ya, za = _propagate(df, t_jd)
    r = np.sqrt(xa**2 + ya**2 + za**2)

    xg = xa - xe;  yg = ya - ye;  zg = za - ze
    delta = np.sqrt(xg**2 + yg**2 + zg**2)

    xs, ys, zs = -xe, -ye, -ze
    rs = np.sqrt(xs**2 + ys**2 + zs**2)
    cos_phase = np.clip((r**2 + delta**2 - rs**2) / (2.0 * r * delta + 1e-12), -1, 1)
    phase = np.arccos(cos_phase)

    V = _hg_magnitude(df["H"].values, df["G"].values, r, delta, phase)

    ok_mag = (V <= mag_limit) & (V >= mag_min)
    if not np.any(ok_mag):
        return pd.DataFrame()

    idx = np.where(ok_mag)[0]
    xg_f, yg_f, zg_f = xg[idx], yg[idx], zg[idx]

    xq_f, yq_f, zq_f = _ecl_to_eq(xg_f, yg_f, zg_f)
    ra_deg  = np.degrees(np.arctan2(yq_f, xq_f)) % 360.0
    dec_deg = np.degrees(np.arcsin(np.clip(zq_f / (delta[idx] + 1e-12), -1, 1)))

    coords = SkyCoord(ra=ra_deg * u.deg, dec=dec_deg * u.deg, frame="icrs")
    altaz  = coords.transform_to(AltAz(obstime=t_astropy, location=earth_loc))
    az = altaz.az.deg
    el = altaz.alt.deg

    ok_win = _in_window(az, el, window)
    if not np.any(ok_win):
        return pd.DataFrame()

    final_idx = idx[ok_win]
    result = df.iloc[final_idx][["desig", "name", "H"]].copy()
    result["V"]         = V[final_idx]
    result["az"]        = az[ok_win]
    result["el"]        = el[ok_win]
    result["ra"]        = ra_deg[ok_win]
    result["dec"]       = dec_deg[ok_win]
    result["r"]         = r[final_idx]
    result["delta"]     = delta[final_idx]
    result["body_type"] = "asteroid"
    return result


def _positions_at_comets(df: pd.DataFrame, t_jd: float, earth_ecl,
                         t_astropy: Time, earth_loc: EarthLocation,
                         window: dict, mag_limit: float,
                         mag_min: float = 0.0) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()

    xe, ye, ze = earth_ecl
    xa, ya, za = _propagate_comets(df, t_jd)

    valid_pos = np.isfinite(xa) & np.isfinite(ya) & np.isfinite(za)
    if not np.any(valid_pos):
        return pd.DataFrame()

    r     = np.where(valid_pos, np.sqrt(xa**2 + ya**2 + za**2), np.nan)
    xg    = xa - xe;  yg = ya - ye;  zg = za - ze
    delta = np.where(valid_pos, np.sqrt(xg**2 + yg**2 + zg**2), np.nan)

    V = _comet_magnitude(df["H"].values, df["G"].values, r, delta)

    ok_mag = valid_pos & (V <= mag_limit) & (V >= mag_min)
    if not np.any(ok_mag):
        return pd.DataFrame()

    idx = np.where(ok_mag)[0]
    xg_f, yg_f, zg_f = xg[idx], yg[idx], zg[idx]

    xq_f, yq_f, zq_f = _ecl_to_eq(xg_f, yg_f, zg_f)
    ra_deg  = np.degrees(np.arctan2(yq_f, xq_f)) % 360.0
    dec_deg = np.degrees(np.arcsin(np.clip(zq_f / (delta[idx] + 1e-12), -1, 1)))

    coords = SkyCoord(ra=ra_deg * u.deg, dec=dec_deg * u.deg, frame="icrs")
    altaz  = coords.transform_to(AltAz(obstime=t_astropy, location=earth_loc))
    az = altaz.az.deg
    el = altaz.alt.deg

    ok_win = _in_window(az, el, window)
    if not np.any(ok_win):
        return pd.DataFrame()

    final_idx = idx[ok_win]
    result = df.iloc[final_idx][["desig", "name", "H"]].copy()
    result["V"]         = V[final_idx]
    result["az"]        = az[ok_win]
    result["el"]        = el[ok_win]
    result["ra"]        = ra_deg[ok_win]
    result["dec"]       = dec_deg[ok_win]
    result["r"]         = r[final_idx]
    result["delta"]     = delta[final_idx]
    result["body_type"] = "comet"
    return result


# ── Config & CLI ─────────────────────────────────────────────────────────────────

def load_config(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def resolve_window(cfg: dict, override_date: str | None, override_time: str | None):
    obs = cfg["observation"]

    if override_date and override_date != "tonight":
        obs_date = datetime.strptime(override_date, "%Y-%m-%d").date()
    else:
        obs_date = date.today()

    time_str = override_time or obs.get("time_utc", "21:00")
    parts    = time_str.split(":")
    h, m     = int(parts[0]), int(parts[1]) if len(parts) > 1 else 0

    start_dt = datetime(obs_date.year, obs_date.month, obs_date.day, h, m,
                        tzinfo=timezone.utc)
    duration  = obs.get("duration_hours", 4)
    step_min  = int(obs.get("step_minutes", 30))
    n_steps   = int(duration * 60 / step_min) + 1

    epochs_dt = [start_dt + timedelta(minutes=i * step_min) for i in range(n_steps)]
    return epochs_dt, step_min


# ── Astronomical twilight ─────────────────────────────────────────────────────────

def find_twilight_utc(earth_loc: EarthLocation, obs_date: date):
    """
    Find astronomical twilight (Sun at −18°).
    Scans at 10-min steps; returns (evening, morning) as UTC datetime
    or None if not found (e.g. summer at high latitudes).
    """
    base  = datetime(obs_date.year, obs_date.month, obs_date.day, 12, 0, tzinfo=timezone.utc)
    steps = [base + timedelta(minutes=10 * i) for i in range(145)]
    t_arr = Time(steps)

    sun  = get_sun(t_arr)
    alts = sun.transform_to(AltAz(obstime=t_arr, location=earth_loc)).alt.deg

    evening = morning = None
    for i in range(len(alts) - 1):
        if alts[i] > -18.0 and alts[i + 1] <= -18.0 and evening is None:
            frac    = (alts[i] + 18.0) / (alts[i] - alts[i + 1])
            evening = steps[i] + timedelta(minutes=10 * frac)
        elif alts[i] <= -18.0 and alts[i + 1] > -18.0 and evening is not None and morning is None:
            frac    = (-18.0 - alts[i]) / (alts[i + 1] - alts[i])
            morning = steps[i] + timedelta(minutes=10 * frac)

    return evening, morning


def _parse_time(s: str) -> tuple[int, int] | None:
    """Convert 'HH:MM' or 'HH' to (h, m). Returns None if invalid."""
    try:
        parts = s.strip().split(":")
        return int(parts[0]) % 24, int(parts[1]) if len(parts) > 1 else 0
    except (ValueError, IndexError):
        return None


def _parse_date(s: str) -> date | None:
    for fmt in ("%d/%m/%Y", "%Y-%m-%d", "%d-%m-%Y"):
        try:
            return datetime.strptime(s.strip(), fmt).date()
        except ValueError:
            pass
    return None


def interactive_setup(cfg: dict, earth_loc: EarthLocation) -> tuple:
    """
    Ask user for body type, date, start/end time and magnitude range.
    Returns (epochs_dt, step_min, mag_min, mag_max, body_type).
    body_type: 'asteroids' | 'comets' | 'both'
    """
    step_min = int(cfg["observation"].get("step_minutes", 30))

    console.print()
    console.rule("[bold cyan]Observation setup[/bold cyan]")
    console.print()

    # ── Body type ──────────────────────────────────────────────────────────────
    console.print("  Body type:  "
                  "[bold cyan][1][/bold cyan] Asteroids  "
                  "[bold cyan][2][/bold cyan] Comets  "
                  "[bold cyan][3][/bold cyan] Both")
    raw_choice = Prompt.ask("  Choice", choices=["1", "2", "3"], default="1")
    body_type  = {"1": "asteroids", "2": "comets", "3": "both"}[raw_choice]
    console.print()

    # ── Date ───────────────────────────────────────────────────────────────────
    today = date.today()
    raw = Prompt.ask(
        "  [cyan]Date[/cyan] [dim](dd/mm/yyyy)[/dim]",
        default=today.strftime("%d/%m/%Y"),
    )
    obs_date = _parse_date(raw) or today
    if obs_date != today:
        console.print(f"  [dim]→ {obs_date.strftime('%d/%m/%Y')}[/dim]")

    # ── Compute twilight for defaults ──────────────────────────────────────────
    with console.status("[dim]  Computing astronomical twilight...[/dim]"):
        evening, morning = find_twilight_utc(earth_loc, obs_date)

    if evening:
        default_start = evening.strftime("%H:%M")
        default_end   = (evening + timedelta(hours=6)).strftime("%H:%M")
        console.print(
            f"  [dim]Astronomical twilight: "
            f"evening {evening.strftime('%H:%M')} UTC"
            + (f" — dawn {morning.strftime('%H:%M')} UTC" if morning else "")
            + "[/dim]"
        )
    else:
        default_start, default_end = "20:00", "02:00"

    # ── Start time ─────────────────────────────────────────────────────────────
    raw = Prompt.ask(
        "  [cyan]Start UTC[/cyan] [dim](HH:MM)[/dim]",
        default=default_start,
    )
    hm = _parse_time(raw) or _parse_time(default_start)
    start_dt = datetime(obs_date.year, obs_date.month, obs_date.day,
                        hm[0], hm[1], tzinfo=timezone.utc)

    # ── End time ───────────────────────────────────────────────────────────────
    raw = Prompt.ask(
        "  [cyan]End UTC[/cyan]   [dim](HH:MM)[/dim]",
        default=default_end,
    )
    hm_end = _parse_time(raw) or _parse_time(default_end)
    end_dt  = datetime(obs_date.year, obs_date.month, obs_date.day,
                       hm_end[0], hm_end[1], tzinfo=timezone.utc)
    if end_dt <= start_dt:
        end_dt += timedelta(days=1)   # crosses midnight

    n_steps   = max(2, int((end_dt - start_dt).total_seconds() / 60 / step_min) + 1)
    epochs_dt = [start_dt + timedelta(minutes=i * step_min) for i in range(n_steps)]

    # ── Magnitude ──────────────────────────────────────────────────────────────
    console.print()
    raw_max = Prompt.ask(
        "  [cyan]Max magnitude[/cyan] [dim](limit of your setup)[/dim]",
        default="16.0",
    )
    raw_min = Prompt.ask(
        "  [cyan]Min magnitude[/cyan]  [dim](0 = no lower limit)[/dim]",
        default="0",
    )
    try:
        mag_max = float(raw_max)
    except ValueError:
        mag_max = 16.0
    try:
        mag_min = float(raw_min)
    except ValueError:
        mag_min = 0.0

    console.print()
    return epochs_dt, step_min, mag_min, mag_max, body_type


# ── Output helpers ────────────────────────────────────────────────────────────────

_CARDINALS = [
    (337.5, 360.0, "N"),  (0.0,   22.5,  "N"),
    (22.5,  67.5,  "NE"), (67.5,  112.5, "E"),
    (112.5, 157.5, "SE"), (157.5, 202.5, "S"),
    (202.5, 247.5, "SW"), (247.5, 292.5, "W"),
    (292.5, 337.5, "NW"),
]

def _cardinal(az: float) -> str:
    for lo, hi, lbl in _CARDINALS:
        if lo <= az < hi:
            return lbl
    return "N"

def _magbar(mag: float) -> str:
    if mag <= 7.0:    return "[bold green]●●●●●[/bold green]"
    elif mag <= 9.0:  return "[green]●●●●○[/green]"
    elif mag <= 11.0: return "[yellow]●●●○○[/yellow]"
    elif mag <= 14.0: return "[orange3]●●○○○[/orange3]"
    else:             return "[red]●○○○○[/red]"

def _tool(mag: float) -> str:
    if mag <= 6.5:    return "[bold green]Naked eye[/bold green]"
    elif mag <= 9.0:  return "[green]Binoculars[/green]"
    elif mag <= 11.0: return "[yellow]Telescope[/yellow]"
    elif mag <= 14.0: return "[orange3]Med. scope[/orange3]"
    else:             return "[red]CCD/imaging[/red]"


# ── Main ──────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Minor planets visible from your sky window — local computation")
    parser.add_argument("--config",  default="config.json")
    parser.add_argument("--date",    default=None, help="YYYY-MM-DD")
    parser.add_argument("--time",    default=None, help="HH or HH:MM UTC")
    parser.add_argument("--refresh", action="store_true", help="Force re-download of catalogues")
    parser.add_argument("-y", "--yes", action="store_true",
                        help="Skip interactive prompts, use config/CLI values")
    parser.add_argument("--sort", default="mag",
                        choices=["mag", "vis", "alt", "az", "speed"],
                        help="Sort by: mag (default), vis, alt, az, speed")
    parser.add_argument("--body", default="asteroids",
                        choices=["asteroids", "comets", "both"],
                        help="Body type to search for (used with -y)")
    parser.add_argument("--limit", type=int, default=None, metavar="N",
                        help="Show only the top N results (after sorting)")
    args = parser.parse_args()

    # Config
    cfg_path = Path(args.config)
    if not cfg_path.exists():
        console.print(f"[red]config.json not found: {cfg_path}[/red]")
        sys.exit(1)

    cfg     = load_config(cfg_path)
    loc_cfg = cfg["location"]
    sky_window = cfg["sky_window"]
    obs_cfg = cfg["observation"]

    earth_loc = EarthLocation(
        lat    = loc_cfg["latitude"]    * u.deg,
        lon    = loc_cfg["longitude"]   * u.deg,
        height = loc_cfg["elevation_m"] * u.m,
    )

    # ── Interactive prompts (skippable with -y) ────────────────────────────────
    if args.yes:
        epochs_dt, step_min = resolve_window(cfg, args.date, args.time)
        mag_min   = 0.0
        mag_lim   = obs_cfg.get("magnitude_limit", 11.0)
        body_type = args.body
    else:
        epochs_dt, step_min, mag_min, mag_lim, body_type = interactive_setup(cfg, earth_loc)

    _BODY_LABELS = {
        "asteroids": "Asteroids",
        "comets":    "Comets",
        "both":      "Asteroids & Comets",
    }
    console.print()
    console.print(Panel(
        f"[bold cyan]{_BODY_LABELS[body_type]} — sky window[/bold cyan]\n"
        f"[dim]📍 {loc_cfg['name']}  "
        f"({loc_cfg['latitude']:.4f}°, {loc_cfg['longitude']:.4f}°)[/dim]\n"
        f"[dim]🌙 {epochs_dt[0].strftime('%d/%m/%Y')}  "
        f"from {epochs_dt[0].strftime('%H:%M')} "
        f"to {epochs_dt[-1].strftime('%H:%M')} UTC  "
        f"(step {step_min} min)[/dim]\n"
        f"[dim]🔭 Az {sky_window['azimuth_min']}°–{sky_window['azimuth_max']}° | "
        f"Alt {sky_window['altitude_min']}°–{sky_window['altitude_max']}° | "
        f"Mag {'≥'+str(mag_min)+' ' if mag_min > 0 else ''}≤ {mag_lim}[/dim]",
        border_style="cyan",
    ))
    console.print()

    # Download & load catalogues
    h_limit = mag_lim + 2.0
    if args.refresh:
        CACHE_FILE.unlink(missing_ok=True)
        COMET_CACHE.unlink(missing_ok=True)

    df_ast = pd.DataFrame()
    df_com = pd.DataFrame()

    if body_type in ("asteroids", "both"):
        ensure_mpcorb()
        df_ast = load_mpcorb(h_limit)
        if df_ast.empty and body_type == "asteroids":
            console.print("[red]No asteroids in the database for this magnitude limit.[/red]")
            sys.exit(1)

    if body_type in ("comets", "both"):
        ensure_comets()
        df_com = load_comets(h_limit)
        if df_com.empty and body_type == "comets":
            console.print("[red]No comets in the database for this magnitude limit.[/red]")
            sys.exit(1)

    total_objects = len(df_ast) + len(df_com)

    # Compute positions for each epoch
    best:          dict[str, dict]  = {}
    first_visible: dict[str, dict]  = {}
    epoch_counts:  dict[str, int]   = {}
    last_pos:      dict[str, tuple] = {}   # key -> (ra_deg, dec_deg, epoch_dt)
    velocity:      dict[str, float] = {}   # key -> arcsec/min

    with console.status("[cyan]Computing orbits...[/cyan]") as status:
        for i, dt in enumerate(epochs_dt):
            t_ast_time = Time(dt)
            t_jd       = t_ast_time.jd
            status.update(
                f"[cyan]Epoch {i+1}/{len(epochs_dt)}: {dt.strftime('%H:%M')} UTC  "
                f"({total_objects:,} objects)...[/cyan]"
            )
            earth_ecl = _earth_helio_ecl(t_ast_time)

            if not df_ast.empty:
                vis = _positions_at(df_ast, t_jd, earth_ecl, t_ast_time,
                                    earth_loc, sky_window, mag_lim, mag_min)
                for _, row in vis.iterrows():
                    key = "A:" + str(row["desig"]).strip()
                    epoch_counts[key] = epoch_counts.get(key, 0) + 1
                    if key not in best or row["V"] < best[key]["V"]:
                        best[key] = {**row.to_dict(), "epoch_dt": dt}
                    if key not in first_visible:
                        first_visible[key] = {
                            "az": float(row["az"]), "el": float(row["el"]),
                            "epoch_dt": dt,
                        }
                    new_ra, new_dec = float(row["ra"]), float(row["dec"])
                    if key in last_pos:
                        p_ra, p_dec, p_dt = last_pos[key]
                        dt_min = (dt - p_dt).total_seconds() / 60.0
                        if dt_min > 0:
                            cos_sep = (np.sin(np.radians(p_dec)) * np.sin(np.radians(new_dec))
                                       + np.cos(np.radians(p_dec)) * np.cos(np.radians(new_dec))
                                       * np.cos(np.radians(new_ra - p_ra)))
                            velocity[key] = np.degrees(np.arccos(np.clip(cos_sep, -1, 1))) * 3600.0 / dt_min
                    last_pos[key] = (new_ra, new_dec, dt)

            if not df_com.empty:
                vis = _positions_at_comets(df_com, t_jd, earth_ecl, t_ast_time,
                                           earth_loc, sky_window, mag_lim, mag_min)
                for _, row in vis.iterrows():
                    key = "C:" + str(row["desig"]).strip()
                    epoch_counts[key] = epoch_counts.get(key, 0) + 1
                    if key not in best or row["V"] < best[key]["V"]:
                        best[key] = {**row.to_dict(), "epoch_dt": dt}
                    if key not in first_visible:
                        first_visible[key] = {
                            "az": float(row["az"]), "el": float(row["el"]),
                            "epoch_dt": dt,
                        }
                    new_ra, new_dec = float(row["ra"]), float(row["dec"])
                    if key in last_pos:
                        p_ra, p_dec, p_dt = last_pos[key]
                        dt_min = (dt - p_dt).total_seconds() / 60.0
                        if dt_min > 0:
                            cos_sep = (np.sin(np.radians(p_dec)) * np.sin(np.radians(new_dec))
                                       + np.cos(np.radians(p_dec)) * np.cos(np.radians(new_dec))
                                       * np.cos(np.radians(new_ra - p_ra)))
                            velocity[key] = np.degrees(np.arccos(np.clip(cos_sep, -1, 1))) * 3600.0 / dt_min
                    last_pos[key] = (new_ra, new_dec, dt)

    # Add visibility duration and speed to each entry
    for key, entry in best.items():
        entry["vis_hours"] = epoch_counts.get(key, 1) * step_min / 60.0
        entry["speed"] = velocity.get(key)   # arcsec/min, or None if seen only once

    if not best:
        console.print(Panel(
            "[yellow]No objects found in the observation window.[/yellow]\n\n"
            "Try:\n"
            "  • Increasing [bold]magnitude_limit[/bold] in config\n"
            "  • Widening the azimuth/altitude window\n"
            "  • Changing time with [bold]--time HH:MM[/bold]",
            title="No results",
            border_style="yellow",
        ))
        return

    # Sorting
    sort_key = {
        "mag":   lambda r: r["V"],
        "vis":   lambda r: -r["vis_hours"],              # longest first
        "alt":   lambda r: -r["el"],                     # highest first
        "az":    lambda r: r["az"],
        "speed": lambda r: -(r.get("speed") or 0.0),    # fastest first
    }[args.sort]
    results = sorted(best.values(), key=sort_key)
    if args.limit and args.limit > 0:
        results = results[:args.limit]

    sort_labels = {"mag": "magnitude", "vis": "visibility duration",
                   "alt": "altitude",  "az":  "azimuth", "speed": "speed"}

    n_ast = sum(1 for r in results if r.get("body_type") == "asteroid")
    n_com = sum(1 for r in results if r.get("body_type") == "comet")
    if body_type == "both":
        counts_str = f"{n_ast} asteroid{'s' if n_ast != 1 else ''}, {n_com} comet{'s' if n_com != 1 else ''}"
    elif body_type == "comets":
        counts_str = f"{n_com} comet{'s' if n_com != 1 else ''}"
    else:
        counts_str = f"{n_ast} asteroid{'s' if n_ast != 1 else ''}"

    table = Table(
        title=(
            f"✨ {counts_str} visible  "
            f"[dim](H ≤ {h_limit:.0f} — "
            f"sorted by {sort_labels[args.sort]})[/dim]"
        ),
        box=box.ROUNDED,
        border_style="cyan",
        show_lines=True,
    )
    table.add_column("#",          style="dim",       width=4)
    if body_type == "both":
        table.add_column("Type",   justify="center",  width=3)
    table.add_column("Object",     style="bold white", min_width=12)
    table.add_column("V mag",      justify="center",   min_width=6)
    table.add_column("H abs",      justify="center",   min_width=6)
    table.add_column("Brightness", justify="center",   min_width=13)
    table.add_column("Visible",    justify="center",   min_width=9,
                     header_style="bold" if args.sort == "vis" else "")
    table.add_column("Az",         justify="center",   min_width=7,
                     header_style="bold" if args.sort == "az" else "")
    table.add_column("Alt",        justify="center",   min_width=7,
                     header_style="bold" if args.sort == "alt" else "")
    table.add_column("Dir.",       justify="center",   min_width=5)
    table.add_column("r (AU)",     justify="center",   min_width=7)
    table.add_column("Δ (AU)",     justify="center",   min_width=7)
    table.add_column("″/min",      justify="center",   min_width=7,
                     header_style="bold" if args.sort == "speed" else "")
    table.add_column("From (UTC)", justify="center",   min_width=8)
    table.add_column("Instrument", justify="center",   min_width=11)

    for i, r in enumerate(results, 1):
        # Readable name — guard against NaN values from pandas
        desig_str = str(r.get("desig", "")).strip()
        raw_name  = r.get("name", desig_str)
        if raw_name is None or (isinstance(raw_name, float) and np.isnan(raw_name)):
            name_str = desig_str
        else:
            name_str = str(raw_name).strip() or desig_str

        is_comet = r.get("body_type") == "comet"
        if is_comet:
            display = name_str
        else:
            try:
                num_i   = int(desig_str)
                display = f"{name_str} ({num_i})" if name_str != str(num_i) else str(num_i)
            except ValueError:
                display = name_str or desig_str

        # Visibility duration
        vh = r["vis_hours"]
        if vh < 1.0:
            vis_str = f"{int(vh * 60)}m"
        else:
            h_int = int(vh)
            m_int = int(round((vh - h_int) * 60))
            vis_str = f"{h_int}h{m_int:02d}m" if m_int else f"{h_int}h"

        # Az/Alt and first-visible time from first_visible dict
        key = ("C:" if is_comet else "A:") + desig_str
        fv  = first_visible.get(key, {"az": r.get("az", 0.0),
                                      "el": r.get("el", 0.0),
                                      "epoch_dt": r["epoch_dt"]})
        fv_az = float(fv["az"])
        fv_el = float(fv["el"])

        row_cells = [str(i)]
        if body_type == "both":
            row_cells.append("☄" if is_comet else "○")
        spd = r.get("speed")
        spd_str = f"{spd:.1f}" if spd is not None else "—"

        row_cells += [
            display,
            f"{r['V']:.1f}",
            f"{r['H']:.1f}",
            _magbar(r["V"]),
            vis_str,
            f"{fv_az:.1f}°",
            f"{fv_el:.1f}°",
            _cardinal(fv_az),
            f"{float(r['r']):.3f}",
            f"{float(r['delta']):.3f}",
            spd_str,
            fv["epoch_dt"].strftime("%H:%M"),
            _tool(r["V"]),
        ]
        table.add_row(*row_cells)

    console.print(table)
    console.print()
    console.print(
        f"[dim]ℹ  Asteroids: elliptic Kepler, H-G (Bowell 1989). "
        f"Comets: m=H+5·logΔ+2.5n·logr. Earth: JPL built-in ephemeris.[/dim]\n"
        f"[dim]💡 Az/Alt and ″/min shown at first-visible epoch. "
        f"Use [bold]--sort speed[/bold] to rank by fastest movers.[/dim]\n"
        f"[dim]🔄 To refresh catalogues: [bold]python minorplanetsfinder.py --refresh[/bold][/dim]\n"
    )


if __name__ == "__main__":
    main()
