"""
featurizer.py
─────────────
StoichML composition-based featurizer.

Design decisions:
  - All elemental properties fetched via mendeleev, patched with literature
    values where mendeleev's database has gaps for common elements.
  - Chemical hardness η = (I₁ − EA) / 2  [Parr & Pearson, JACS 1983]
    All three quantities (I₁, EA, η) are in eV — units are consistent.
  - vec() returns a dict (not a positional list) — prevents key/value misalignment.
  - stats() operates on valid entries only with renormalised weights.
  - miss fraction features removed — gaps are resolved by patches, so the
    signal is no longer meaningful for patched elements, and for truly exotic
    elements (superheavies, heavy actinides) the model should not see them
    in a real materials dataset anyway.
  - phys() uses valid-only weighted statistics (no None → 0 substitution).
  - unpaired electrons estimated via Hund's rule as a magnetic moment proxy.
  - dcnt restricted to valence d-shell (n == nmax-1) only — prevents
    overcounting for 4d/5d elements with filled 3d cores.

New properties vs. previous version:
  - wf      Work function [eV] — energy to remove an electron from the solid
            surface. Directly related to band alignment and gap magnitude.
            Wide-gap insulators (oxides, fluorides) have systematically higher
            work functions than narrow-gap semiconductors.
  - pcount  Valence p-electron count — sp-gaps are dominated by p-p orbital
            interactions. Complements dcount (d-electrons) which was already
            present. Extracted alongside dcount in valence_props().
  - period  Period number — captures relativistic effects in 5d/6p elements
            (s-orbital contraction, d-orbital expansion) that directly affect
            band gaps. Also cleanly separates 2p wide-gap insulators (MgO)
            from 3d narrow-gap semiconductors with identical val counts.
"""

import re
import numpy as np
import pandas as pd
from mendeleev import element  # type: ignore
from functools import lru_cache


# ══════════════════════════════════════════════════════════════════════════════
# Literature patch table
# ══════════════════════════════════════════════════════════════════════════════
PROPERTY_PATCHES = {
    # s-block
    "H":  {"magmom": 0.0},
    "He": {"chi": 0.0,  "magmom": 0.0},
    "Li": {"magmom": 0.0},
    "Be": {"magmom": 0.0},
    "Na": {"magmom": 0.0},
    "Mg": {"EA": 0.0,   "magmom": 0.0},
    "K":  {"magmom": 0.0},
    "Ca": {"kappa": 201.0, "magmom": 0.0},
    "Rb": {"magmom": 0.0},
    "Sr": {"kappa": 35.4,  "magmom": 0.0},
    "Cs": {"magmom": 0.0},
    "Ba": {"kappa": 18.4,  "magmom": 0.0},
    "Fr": {},
    "Ra": {},

    # p-block
    "B":  {"magmom": 0.0},
    "C":  {"magmom": 0.0},
    "N":  {"magmom": 0.0},
    "O":  {"magmom": 0.0},
    "F":  {"magmom": 0.0},
    "Ne": {"chi": 0.0,  "magmom": 0.0},
    "Al": {"magmom": 0.0},
    "Si": {"magmom": 0.0},
    "P":  {"Tm": 317.3, "Tb": 553.6, "kappa": 0.236, "magmom": 0.0},
    "S":  {"Tm": 388.4, "Tb": 717.8, "magmom": 0.0},
    "Cl": {"magmom": 0.0},
    "Ar": {"chi": 0.0,  "magmom": 0.0},
    "Ga": {"magmom": 0.0},
    "Ge": {"magmom": 0.0},
    "As": {"kappa": 50.2,  "magmom": 0.0},
    "Se": {"Tm": 494.0,    "magmom": 0.0},
    "Br": {"magmom": 0.0},
    "Kr": {"chi": 0.0,  "magmom": 0.0},
    "In": {"magmom": 0.0},
    "Sn": {"Tm": 505.1,    "magmom": 0.0},
    "Sb": {"magmom": 0.0},
    "Te": {"magmom": 0.0},
    "I":  {"kappa": 0.449, "magmom": 0.0},
    "Tl": {"magmom": 0.0},
    "Pb": {"magmom": 0.0},
    "Bi": {"magmom": 0.0},
    "Po": {"kappa": 20.0,  "magmom": 0.0},
    "At": {},

    # d-block
    "Sc": {"magmom": 0.0},
    "Ti": {"magmom": 0.0},
    "V":  {"magmom": 0.0},
    "Cr": {"magmom": 0.0},
    "Mn": {"EA": 0.0, "kappa": 7.81, "magmom": 0.0},
    "Cu": {"magmom": 0.0},
    "Zn": {"EA": 0.0,  "magmom": 0.0},
    "Y":  {"kappa": 17.2,  "magmom": 0.0},
    "Zr": {"magmom": 0.0},
    "Nb": {"magmom": 0.0},
    "Mo": {"kappa": 138.0, "magmom": 0.0},
    "Tc": {"magmom": 0.0},
    "Ru": {"magmom": 0.0},
    "Rh": {"magmom": 0.0},
    "Pd": {"magmom": 0.0},
    "Ag": {"magmom": 0.0},
    "Cd": {"EA": 0.0,  "magmom": 0.0},
    "Hf": {"magmom": 0.0},
    "Ta": {"magmom": 0.0},
    "W":  {"magmom": 0.0},
    "Re": {"magmom": 0.0},
    "Os": {"kappa": 87.6, "magmom": 0.0},
    "Ir": {"magmom": 0.0},
    "Pt": {"magmom": 0.0},
    "Au": {"magmom": 0.0},
    "Hg": {"EA": 0.0,  "magmom": 0.0},
    "Ac": {"kappa": 12.0, "magmom": 0.0},

    # f-block
    "La": {"magmom": 0.0},
    "Ce": {"magmom": 0.0},
    "Pr": {},
    "Nd": {"kappa": 16.5},
    "Pm": {},
    "Sm": {"kappa": 13.3},
    "Eu": {},
    "Gd": {"kappa": 10.6},
    "Tb": {},
    "Dy": {},
    "Ho": {"kappa": 16.2},
    "Er": {"kappa": 14.5},
    "Tm": {"kappa": 16.9},
    "Yb": {"magmom": 0.0},
    "Lu": {"kappa": 16.4, "magmom": 0.0},
    "Th": {"magmom": 0.0},
    "U":  {"magmom": 0.0},
}


def _apply_patches(symbol: str, prop_dict: dict) -> dict:
    """Overlay literature patches onto a property dict from vec()."""
    patched = dict(prop_dict)
    for prop, value in PROPERTY_PATCHES.get(symbol, {}).items():
        if prop in patched and patched[prop] is None:
            patched[prop] = value
    return patched


# ══════════════════════════════════════════════════════════════════════════════
# Basic helpers
# ══════════════════════════════════════════════════════════════════════════════

def fnum(x):
    """Safely cast to float; return None on failure or NaN."""
    try:
        v = float(x() if callable(x) else x)
        return None if np.isnan(v) else v
    except Exception:
        return None


@lru_cache(None)
def elem(sym):
    return element(sym)


# ══════════════════════════════════════════════════════════════════════════════
# Valence + d-shell + p-shell + unpaired electrons
# ══════════════════════════════════════════════════════════════════════════════

def valence_props(e):
    """
    Returns (val, vac, dcnt, pcnt, dhalf, unpaired).

      val      valence electron count
      vac      valence shell vacancies
      dcnt     valence d-electron count — restricted to n == nmax-1 only
               (prevents overcounting for 4d/5d elements with filled 3d cores)
      pcnt     valence p-electron count — sp-gaps are dominated by p-p orbital
               interactions; complements dcnt. Restricted to n == nmax only.
      dhalf    |dcnt - 5|  (distance from half-filled d-shell)
      unpaired Hund's rule estimate of unpaired electrons (magnetic proxy)
    """
    conf = e.econf or ""
    parts = re.findall(r'(\d+)([spdf])(\d+)', conf)
    if not parts:
        return 0.0, 0.0, 0.0, 0.0, 5.0, 0.0

    shells = [(int(n), o, int(k)) for n, o, k in parts]
    nmax   = max(n for n, _, _ in shells)
    block  = e.block
    val = cap = dcnt = pcnt = 0

    for n, o, k in shells:
        if block in ("s", "p"):
            if n == nmax:
                val += k
                cap += {"s": 2, "p": 6}.get(o, 0)
            # p-count: valence p-electrons at highest principal quantum number
            if o == "p" and n == nmax:
                pcnt += k

        elif block == "d":
            if (n == nmax and o == "s") or (n == nmax - 1 and o == "d"):
                val += k
                cap += {"s": 2, "d": 10}.get(o, 0)
            if o == "d" and n == nmax - 1:
                dcnt += k
            # d-block elements can have residual p-electrons (e.g. post-d sp)
            if o == "p" and n == nmax:
                pcnt += k

        elif block == "f":
            if (
                (n == nmax     and o == "s")
                or (n == nmax - 1 and o == "d")
                or (n == nmax - 2 and o == "f")
            ):
                val += k
                cap += {"s": 2, "d": 10, "f": 14}.get(o, 0)
            if o == "p" and n == nmax:
                pcnt += k

    vac   = max(cap - val, 0)
    dhalf = abs(dcnt - 5)

    # Hund's rule unpaired electron estimate
    if block == "d":
        unpaired = float(dcnt) if dcnt <= 5 else float(10 - dcnt)
    elif block == "f":
        f_cnt    = sum(k for n, o, k in shells if o == "f" and n == nmax - 2)
        unpaired = float(f_cnt) if f_cnt <= 7 else float(14 - f_cnt)
    else:
        unpaired = float(val % 2)

    return float(val), float(vac), float(dcnt), float(pcnt), float(dhalf), float(unpaired)


# ══════════════════════════════════════════════════════════════════════════════
# Element property vector  (dict-based, patched, hardness self-computed)
# ══════════════════════════════════════════════════════════════════════════════

@lru_cache(None)
def vec(sym):
    """
    Return a fully patched dict of elemental properties for symbol `sym`.

    New properties vs. previous version:
      wf      Work function [eV] from mendeleev. Directly relates to band
              alignment and gap magnitude — wide-gap insulators systematically
              higher than narrow-gap semiconductors.
      pcount  Valence p-electron count from valence_props(). Captures sp-gap
              physics that dcount (d-electrons) misses.
      period  Period number. Separates relativistic heavy elements (5d/6p)
              from lighter analogues with identical val/dcount.

    Chemical hardness (η) unit check:
        I1   [eV]  — mendeleev ionenergies in eV
        EA   [eV]  — mendeleev electron_affinity in eV
        η = (I1 - EA) / 2  →  [eV]   CONSISTENT.
    """
    e = elem(sym)
    val, vac, dcnt, pcnt, dhalf, unpaired = valence_props(e)

    I1   = fnum(e.ionenergies.get(1) if e.ionenergies else None)
    EA   = fnum(getattr(e, "electron_affinity",  None))
    hard = fnum(getattr(e, "hardness",           None))

    raw = {
        "Z":        fnum(e.atomic_number),
        "mass":     fnum(e.atomic_weight),
        "chi":      fnum(e.en_pauling),
        "radius":   fnum(e.covalent_radius or e.atomic_radius),
        "volume":   fnum(getattr(e, "atomic_volume",         None)),
        "polar":    fnum(getattr(e, "dipole_polarizability", None)),
        "hard":     hard,
        "val":      val,
        "vac":      vac,
        "dcount":   dcnt,
        "pcount":   pcnt,           # NEW: valence p-electron count
        "dhalf":    dhalf,
        "unpaired": unpaired,
        "EA":       EA,
        "I1":       I1,
        "Tm":       fnum(getattr(e, "melting_point",         None)),
        "Tb":       fnum(getattr(e, "boiling_point",         None)),
        "kappa":    fnum(getattr(e, "thermal_conductivity",  None)),
        "Ecoh":     fnum(getattr(e, "cohesive_energy",       None)),
        "magmom":   fnum(getattr(e, "magnetic_moment",       None)),
        "wf":       fnum(getattr(e, "work_function",         None)),  # NEW
        "period":   fnum(getattr(e, "period",                None)),  # NEW
    }

    # Step 1 — apply literature patches
    patched = _apply_patches(sym, raw)

    # Step 2 — compute hardness from Parr-Pearson if mendeleev returned None
    if patched["hard"] is None:
        i1_p = patched["I1"]
        ea_p = patched["EA"]
        if i1_p is not None and ea_p is not None:
            patched["hard"] = (i1_p - ea_p) / 2.0

    return patched


# ══════════════════════════════════════════════════════════════════════════════
# Weighted statistics  (valid-only, no miss feature)
# ══════════════════════════════════════════════════════════════════════════════

def stats(key, values, w):
    """
    Compute weighted descriptive statistics over elemental property `key`.

    Uses valid (non-None) entries only, with weights renormalised over
    valid entries.

    Statistics returned (8 per property):
      mean  weighted centroid
      std   weighted spread
      min   minimum element value (unweighted)
      max   maximum element value (unweighted)
      mad   weighted mean absolute deviation
      pos   (mean - min) / (max - min)
      hmean weighted harmonic mean (0.0 when any value ≤ 0)
      gmean weighted geometric mean (0.0 when any value = 0)

    Returns a flat dict with prefixed keys: {key}_mean, {key}_std, etc.
    """
    values = np.array(values, dtype=object)
    w      = np.array(w,      dtype=float)
    mask   = np.array([v is not None for v in values], dtype=bool)

    if mask.sum() == 0:
        return {
            f"{key}_mean":  0.0,
            f"{key}_std":   0.0,
            f"{key}_min":   0.0,
            f"{key}_max":   0.0,
            f"{key}_mad":   0.0,
            f"{key}_pos":   0.5,
            f"{key}_hmean": 0.0,
            f"{key}_gmean": 0.0,
        }

    xv = np.array([v for v in values[mask]], dtype=float)
    wv = w[mask]
    wv = wv / wv.sum()

    mean = float(np.sum(wv * xv))
    std  = float(np.sqrt(np.sum(wv * (xv - mean) ** 2)))
    vmin = float(xv.min())
    vmax = float(xv.max())
    mad  = float(np.sum(wv * np.abs(xv - mean)))
    rng  = vmax - vmin
    pos  = float((mean - vmin) / rng) if rng > 0 else 0.5

    hmean = float(1.0 / np.sum(wv / xv)) if np.all(xv > 0) else 0.0
    gmean = float(np.exp(np.sum(wv * np.log(np.abs(xv))))) if np.all(xv != 0) else 0.0

    return {
        f"{key}_mean":  mean,
        f"{key}_std":   std,
        f"{key}_min":   vmin,
        f"{key}_max":   vmax,
        f"{key}_mad":   mad,
        f"{key}_pos":   pos,
        f"{key}_hmean": hmean,
        f"{key}_gmean": gmean,
    }


# ══════════════════════════════════════════════════════════════════════════════
# Composition-level physics features
# ══════════════════════════════════════════════════════════════════════════════

def phys(elems, w, raw_counts=None):
    """
    Compute composition-level physics-inspired features.

    Args:
        elems      : list of element symbols
        w          : stoichiometric weights, normalised (sum = 1)
        raw_counts : raw stoichiometric counts before normalisation.
                     Required for n_atoms. If None, falls back to len(elems).
    """
    w      = np.array(w, dtype=float)
    w_norm = w / w.sum()

    chis,  chi_mask  = [], []
    radii, r_mask    = [], []
    masses, m_mask   = [], []
    vals, dhalf_arr, unpaired_arr = [], [], []
    orb_counts_list  = []

    for sym in elems:
        v_dict = vec(sym)
        e_obj  = elem(sym)
        _, _, _, _, dh, up = valence_props(e_obj)   # updated signature (6 values)

        chi  = v_dict["chi"]
        rad  = v_dict["radius"]
        mass = v_dict["mass"]
        v    = v_dict["val"]

        chis.append(chi);    chi_mask.append(chi  is not None)
        radii.append(rad);   r_mask.append(  rad  is not None)
        masses.append(mass); m_mask.append(  mass is not None)
        vals.append(v)
        dhalf_arr.append(dh)
        unpaired_arr.append(up)

        conf  = e_obj.econf or ""
        parts = re.findall(r'(\d+)([spdf])(\d+)', conf)
        shells = [(int(n), o, int(k)) for n, o, k in parts]
        block  = e_obj.block
        oc     = {"s": 0, "p": 0, "d": 0, "f": 0}

        if shells:
            nmax = max(n for n, _, _ in shells)
            for n, o, k in shells:
                if block in ("s", "p"):
                    if n == nmax:
                        oc[o] += k
                elif block == "d":
                    if (n == nmax and o == "s") or (n == nmax - 1 and o == "d"):
                        oc[o] += k
                elif block == "f":
                    if (
                        (n == nmax     and o == "s")
                        or (n == nmax - 1 and o == "d")
                        or (n == nmax - 2 and o == "f")
                    ):
                        oc[o] += k

        orb_counts_list.append(oc)

    vals         = np.array(vals,         dtype=float)
    dhalf_arr    = np.array(dhalf_arr,    dtype=float)
    unpaired_arr = np.array(unpaired_arr, dtype=float)

    def _wmean(arr, mask):
        arr, mask = np.array(arr, dtype=object), np.array(mask, dtype=bool)
        if mask.sum() == 0:
            return 0.0
        xv = np.array([v for v in arr[mask]], dtype=float)
        wv = w[mask]; wv = wv / wv.sum()
        return float(np.sum(wv * xv))

    def _wmad(arr, mask, mean):
        arr, mask = np.array(arr, dtype=object), np.array(mask, dtype=bool)
        if mask.sum() == 0:
            return 0.0
        xv = np.array([v for v in arr[mask]], dtype=float)
        wv = w[mask]; wv = wv / wv.sum()
        return float(np.sum(wv * np.abs(xv - mean)))

    def _wrng(arr, mask):
        arr, mask = np.array(arr, dtype=object), np.array(mask, dtype=bool)
        if mask.sum() == 0:
            return 0.0
        xv = np.array([v for v in arr[mask]], dtype=float)
        return float(xv.max() - xv.min())

    def _wstd(arr, mask, mean):
        arr, mask = np.array(arr, dtype=object), np.array(mask, dtype=bool)
        if mask.sum() == 0:
            return 0.0
        xv = np.array([v for v in arr[mask]], dtype=float)
        wv = w[mask]; wv = wv / wv.sum()
        return float(np.sqrt(np.sum(wv * (xv - mean) ** 2)))

    chi_mean = _wmean(chis,   chi_mask)
    r_mean   = _wmean(radii,  r_mask)
    m_mean   = _wmean(masses, m_mask)
    val_mean = float(np.sum(w_norm * vals))
    up_mean  = float(np.sum(w_norm * unpaired_arr))

    spin_arr = unpaired_arr / 2.0
    S_mag    = float(np.sum(w_norm * np.log(2.0 * spin_arr + 1.0)))

    total_orb = {o: 0.0 for o in "spdf"}
    for i, oc in enumerate(orb_counts_list):
        for o in "spdf":
            total_orb[o] += w_norm[i] * oc[o]
    total_elec = sum(total_orb.values())
    if total_elec > 0:
        p_orb = np.array([total_orb[o] / total_elec for o in "spdf"])
        p_orb = p_orb[p_orb > 0]
        S_orb = float(-np.sum(p_orb * np.log(p_orb)))
    else:
        S_orb = 0.0

    return {
        "n_elements":    float(len(elems)),
        "n_atoms":       float(raw_counts.sum()) if raw_counts is not None else float(len(elems)),
        "max_weight":    float(w_norm.max()),
        "conf_entropy":  float(-np.sum(w_norm * np.log(w_norm + 1e-12))),
        "S_mag":         S_mag,
        "S_orb":         S_orb,
        "chi_mad":       _wmad(chis,   chi_mask, chi_mean),
        "delta_chi":     _wrng(chis,   chi_mask),
        "pair_chi":      float(sum(
                             w_norm[i] * w_norm[j] *
                             (
                                 (chis[i] if chis[i] is not None else chi_mean) -
                                 (chis[j] if chis[j] is not None else chi_mean)
                             ) ** 2
                             for i in range(len(elems))
                             for j in range(i + 1, len(elems))
                         )),
        "r_mad":         _wmad(radii,  r_mask, r_mean),
        "mass_std":      _wstd(masses, m_mask, m_mean),
        "val_mean":      val_mean,
        "val_var":       float(np.sum(w_norm * (vals - val_mean) ** 2)),
        "dhalf_mean":    float(np.sum(w_norm * dhalf_arr)),
        "tm_frac":       float(np.sum(w_norm * np.array(
                             [elem(s).block == "d" for s in elems], dtype=float
                         ))),
        "f_frac":        float(np.sum(w_norm * np.array(
                             [elem(s).block == "f" for s in elems], dtype=float
                         ))),
        "unpaired_mean": up_mean,
        "unpaired_var":  float(np.sum(w_norm * (unpaired_arr - up_mean) ** 2)),
    }


# ══════════════════════════════════════════════════════════════════════════════
# Main featurizer
# ══════════════════════════════════════════════════════════════════════════════

def featurize(df, elements_col="elements", composition_col="composition"):
    """
    Featurize a DataFrame of materials by composition.

    Each row must have:
      elements_col    list of element symbols, e.g. ['Fe', 'O']
      composition_col list of stoichiometric counts, e.g. [1, 2]  (FeO2)

    Returns the original DataFrame concatenated with all feature columns.

    Feature count:
      22 properties × 8 stats = 176 elemental features
        Properties: Z, mass, chi, radius, volume, polar, hard,
                    val, vac, dcount, pcount, dhalf, unpaired,
                    EA, I1, Tm, Tb, kappa, Ecoh, magmom,
                    wf, period
        Stats per property: mean, std, min, max, mad, pos, hmean, gmean

      phys() returns 18 composition-level keys, but 5 of them —
      chi_mad, mass_std, val_mean, dhalf_mean, unpaired_mean — share
      their name with (and are numerically identical to) a stats()-derived
      elemental column for the same base property. feats.update(phys(...))
      therefore overwrites those 5 stats() columns in place rather than
      adding new ones. Net unique composition-level contribution: 13
      features:
        n_elements, n_atoms, max_weight,
        conf_entropy, S_mag, S_orb,
        delta_chi, pair_chi, r_mad,
        val_var, tm_frac, f_frac,
        unpaired_var

      = 176 + 13 = 189 features total.

      NOTE: this docstring previously stated 194 = 176 + 18, which counted
      phys()'s 18 return keys without accounting for the 5-key overlap
      above. That was a documentation error only — no feature values or
      model inputs changed; the pipeline has always produced 189 unique
      columns. The 5 overlapping phys() lines are dead code (their output
      is discarded by the overwrite) but are left in place here pending a
      decision on whether to remove or rename them.
    """
    out = []

    for _, row in df.iterrows():
        elems = row[elements_col]
        comp  = np.asarray(row[composition_col], dtype=float)

        s = comp.sum()
        if s <= 0:
            raise ValueError(
                f"Composition sums to zero for row: {dict(row)}"
            )
        w = comp / s

        elem_dicts = [vec(e) for e in elems]
        keys       = list(elem_dicts[0].keys())

        feats = {}

        for k in keys:
            col_values = [d[k] for d in elem_dicts]
            feats.update(stats(k, col_values, w))

        feats.update(phys(elems, w, comp))

        out.append(feats)

    return pd.concat(
        [df.reset_index(drop=True), pd.DataFrame(out)],
        axis=1,
    )