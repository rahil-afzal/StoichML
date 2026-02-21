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
"""

import re
import numpy as np
import pandas as pd
from mendeleev import element  # type: ignore
from functools import lru_cache


# ══════════════════════════════════════════════════════════════════════════════
# Literature patch table
# ══════════════════════════════════════════════════════════════════════════════
# Strategy:
#   EA = 0.0  for elements whose anions are thermodynamically unstable.
#             Convention used universally in descriptor databases (MAGPIE etc.)
#             when experimental EA < 0.  Source: Andersen (2004) Phys. Rep. 394.
#             Once EA is patched, hard = (I1 - EA)/2 is computable directly.
#
#   kappa     from CRC Handbook of Chemistry & Physics, 95th Ed. (Haynes 2014).
#
#   Tm / Tb   for allotropic elements (P, S, Se, Sn): stable phase at STP.
#             Carbon deliberately excluded — sublimates at 1 atm, no Tm exists.
#
#   chi = 0   for noble gases (He, Ne, Ar, Kr): Pauling EN undefined for
#             elements that form no stable bonds. Zero is the standard
#             placeholder in all major ML-for-materials frameworks.

PROPERTY_PATCHES = {
    # ─────────────────────────────────────────────────────────────────────
    # FORMAT: "Symbol": {"property": value, ...}
    #
    # EA = 0.0    anion unstable; standard convention when EA < 0
    #             Source: Andersen (2004) Phys. Rep. 394, 157-313
    # kappa       W/(m·K)  — CRC Handbook 95th Ed. (Haynes 2014)
    # Tm / Tb     K        — CRC Handbook, stable allotrope at STP
    # chi = 0.0   noble gases — Pauling EN undefined; placeholder used in
    #             MAGPIE and all major descriptor frameworks
    # magmom      μB (Bohr magnetons), solid-state ordered moment.
    #             None in mendeleev for non-magnetic elements = physically 0,
    #             not a data gap. Source: Kittel, ISSP 8th Ed.
    #             Only ferromagnetic / strongly magnetic elements are non-zero:
    #             Fe=2.22, Co=1.72, Ni=0.60, Gd=7.63, Dy=10.0 etc.
    # ─────────────────────────────────────────────────────────────────────

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
    "Fr": {},           # radioactive, sparse data — leave as-is
    "Ra": {},

    # p-block
    "B":  {"magmom": 0.0},
    "C":  {"magmom": 0.0},                          # Tm deliberately omitted (sublimates)
    "N":  {"magmom": 0.0},
    "O":  {"magmom": 0.0},
    "F":  {"magmom": 0.0},
    "Ne": {"chi": 0.0,  "magmom": 0.0},
    "Al": {"magmom": 0.0},
    "Si": {"magmom": 0.0},
    "P":  {"Tm": 317.3, "Tb": 553.6, "kappa": 0.236, "magmom": 0.0},   # white P
    "S":  {"Tm": 388.4, "Tb": 717.8, "magmom": 0.0},                    # rhombic S
    "Cl": {"magmom": 0.0},
    "Ar": {"chi": 0.0,  "magmom": 0.0},
    "Ga": {"magmom": 0.0},
    "Ge": {"magmom": 0.0},
    "As": {"kappa": 50.2,  "magmom": 0.0},
    "Se": {"Tm": 494.0,    "magmom": 0.0},                              # gray Se
    "Br": {"magmom": 0.0},
    "Kr": {"chi": 0.0,  "magmom": 0.0},
    "In": {"magmom": 0.0},
    "Sn": {"Tm": 505.1,    "magmom": 0.0},                              # white Sn
    "Sb": {"magmom": 0.0},
    "Te": {"magmom": 0.0},
    "I":  {"kappa": 0.449, "magmom": 0.0},
    "Tl": {"magmom": 0.0},
    "Pb": {"magmom": 0.0},
    "Bi": {"magmom": 0.0},
    "Po": {"kappa": 20.0,  "magmom": 0.0},
    "At": {},

    # d-block — transition metals
    # Magnetic elements (non-zero magmom) left to mendeleev where available
    # Non-magnetic ones patched to 0.0
    "Sc": {"magmom": 0.0},
    "Ti": {"magmom": 0.0},
    "V":  {"magmom": 0.0},
    "Cr": {"magmom": 0.0},                          # antiferromagnetic → 0 net
    "Mn": {"EA": 0.0, "kappa": 7.81, "magmom": 0.0},  # antiferromagnetic → 0 net
    # Fe, Co, Ni: ferromagnetic — mendeleev has their values, no patch needed
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

    # f-block — lanthanides and actinides
    # Strongly magnetic lanthanides left to mendeleev
    # Non-magnetic or closed-shell ones patched to 0
    "La": {"magmom": 0.0},
    "Ce": {"magmom": 0.0},                          # Ce is paramagnetic, ~0 ordered
    "Pr": {},                                        # magnetic — mendeleev handles
    "Nd": {"kappa": 16.5},                          # magnetic — leave magmom to mendeleev
    "Pm": {},
    "Sm": {"kappa": 13.3},                          # magnetic
    "Eu": {},                                        # magnetic
    "Gd": {"kappa": 10.6},                          # strongly magnetic (7.63 μB)
    "Tb": {},                                        # magnetic
    "Dy": {},                                        # strongly magnetic
    "Ho": {"kappa": 16.2},                          # magnetic
    "Er": {"kappa": 14.5},                          # magnetic
    "Tm": {"kappa": 16.9},                          # Thulium — magnetic
    "Yb": {"magmom": 0.0},                          # non-magnetic (4f14)
    "Lu": {"kappa": 16.4, "magmom": 0.0},           # non-magnetic (4f14 filled)
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
# Valence + d-shell + unpaired electrons
# ══════════════════════════════════════════════════════════════════════════════

def valence_props(e):
    """
    Returns (val, vac, dcnt, dhalf, unpaired).

      val      valence electron count
      vac      valence shell vacancies
      dcnt     valence d-electron count — restricted to n == nmax-1 only
               (prevents overcounting for 4d/5d elements with filled 3d cores)
      dhalf    |dcnt - 5|  (distance from half-filled d-shell)
      unpaired Hund's rule estimate of unpaired electrons (magnetic proxy)
    """
    conf = e.econf or ""
    parts = re.findall(r'(\d+)([spdf])(\d+)', conf)
    if not parts:
        return 0.0, 0.0, 0.0, 5.0, 0.0

    shells = [(int(n), o, int(k)) for n, o, k in parts]
    nmax   = max(n for n, _, _ in shells)
    block  = e.block
    val = cap = dcnt = 0

    for n, o, k in shells:
        if block in ("s", "p"):
            if n == nmax:
                val += k
                cap += {"s": 2, "p": 6}.get(o, 0)

        elif block == "d":
            if (n == nmax and o == "s") or (n == nmax - 1 and o == "d"):
                val += k
                cap += {"s": 2, "d": 10}.get(o, 0)
            if o == "d" and n == nmax - 1:          # valence d-shell only
                dcnt += k

        elif block == "f":
            if (
                (n == nmax     and o == "s")
                or (n == nmax - 1 and o == "d")
                or (n == nmax - 2 and o == "f")
            ):
                val += k
                cap += {"s": 2, "d": 10, "f": 14}.get(o, 0)

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

    return float(val), float(vac), float(dcnt), float(dhalf), float(unpaired)


# ══════════════════════════════════════════════════════════════════════════════
# Element property vector  (dict-based, patched, hardness self-computed)
# ══════════════════════════════════════════════════════════════════════════════

@lru_cache(None)
def vec(sym):
    """
    Return a fully patched dict of elemental properties for symbol `sym`.

    Chemical hardness (η) unit check:
        I1   [eV]  — mendeleev ionenergies in eV
        EA   [eV]  — mendeleev electron_affinity in eV
        η = (I1 - EA) / 2  →  [eV]   CONSISTENT.

    mendeleev computes hardness on the live element object using its own
    stored EA. Since we patch EA on our side (not on the mendeleev object),
    we must recompute η manually after patching — which is what the block
    at the end of this function does.
    """
    e = elem(sym)
    val, vac, dcnt, dhalf, unpaired = valence_props(e)

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
        "dhalf":    dhalf,
        "unpaired": unpaired,
        "EA":       EA,
        "I1":       I1,
        "Tm":       fnum(getattr(e, "melting_point",         None)),
        "Tb":       fnum(getattr(e, "boiling_point",         None)),
        "kappa":    fnum(getattr(e, "thermal_conductivity",  None)),
        # ── New properties ────────────────────────────────────────────────
        # Cohesive energy: energy to atomise the elemental solid [kJ/mol].
        # Directly links to formation enthalpy via Born-Haber cycle.
        # Every classical formation energy model (Miedema, CALPHAD) uses it.
        "Ecoh":     fnum(getattr(e, "cohesive_energy",       None)),
        # Solid-state magnetic moment [μB/atom].
        # Distinct from `unpaired` (free-atom Hund's rule estimate):
        # this is the actual ordered moment of the elemental solid,
        # reflecting crystal-field quenching. Fe=2.22, Co=1.72, Ni=0.60.
        # Non-magnetic elements patched to 0.0 in PROPERTY_PATCHES.
        "magmom":   fnum(getattr(e, "magnetic_moment",       None)),
    }

    # Step 1 — apply literature patches (EA is patched here if needed)
    patched = _apply_patches(sym, raw)

    # Step 2 — compute hardness from Parr-Pearson if mendeleev returned None.
    #           Units: (eV - eV) / 2 = eV  — consistent throughout.
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
    valid entries. The miss fraction feature is omitted — property gaps
    are resolved by the patch table for all common elements.

    Statistics returned (5 per property):
      mean  weighted centroid
      std   weighted spread
      min   lightest/smallest extreme element value (unweighted — the extreme
            element matters regardless of its stoichiometric fraction)
      max   heaviest/largest extreme element value  (same rationale)
      mad   weighted mean absolute deviation — more robust than std for
            small compositions; diverges from std when distribution is skewed
      pos   (mean - min) / (max - min) — position of the weighted mean within
            the property range. Independent of mean/std/min/max individually.
            Answers: is the composition's centroid near its light or heavy
            end? E.g. pos(chi) near 1 → high-EN element dominates by weight.

    NOTE: rng (= max - min) is intentionally excluded — it is fully
    determined by min and max and adds no information to the feature set.

    Returns a flat dict with prefixed keys: {key}_mean, {key}_std, etc.
    """
    values = np.array(values, dtype=object)
    w      = np.array(w,      dtype=float)
    mask   = np.array([v is not None for v in values], dtype=bool)

    if mask.sum() == 0:
        return {
            f"{key}_mean": 0.0,
            f"{key}_std":  0.0,
            f"{key}_min":  0.0,
            f"{key}_max":  0.0,
            f"{key}_mad":  0.0,
            f"{key}_pos":  0.5,   # neutral: mean at midpoint when all values absent
        }

    xv = np.array([v for v in values[mask]], dtype=float)
    wv = w[mask]
    wv = wv / wv.sum()                          # renormalise over valid entries

    mean = float(np.sum(wv * xv))
    std  = float(np.sqrt(np.sum(wv * (xv - mean) ** 2)))
    vmin = float(xv.min())
    vmax = float(xv.max())
    mad  = float(np.sum(wv * np.abs(xv - mean)))
    rng  = vmax - vmin
    pos  = float((mean - vmin) / rng) if rng > 0 else 0.5

    return {
        f"{key}_mean": mean,
        f"{key}_std":  std,
        f"{key}_min":  vmin,
        f"{key}_max":  vmax,
        f"{key}_mad":  mad,
        f"{key}_pos":  pos,
    }


# ══════════════════════════════════════════════════════════════════════════════
# Composition-level physics features
# ══════════════════════════════════════════════════════════════════════════════

def phys(elems, w):
    """
    Compute composition-level physics-inspired features.

    All weighted statistics use valid-only entries with renormalised
    weights — no None-to-zero substitution anywhere.

    Features added vs. previous version:
      n_elements   number of distinct elements — strong structural prior;
                   binary and quinary compounds with the same weighted-mean
                   chi are physically very different materials.
      max_weight   stoichiometric fraction of the majority element — captures
                   whether the composition is near-pure (max_weight → 1)
                   or well-mixed (max_weight → 1/n_elements).
                   Together with conf_entropy this fully characterises the
                   stoichiometric distribution shape.
    """
    w      = np.array(w, dtype=float)
    w_norm = w / w.sum()

    chis,  chi_mask  = [], []
    radii, r_mask    = [], []
    masses, m_mask   = [], []
    vals, dhalf_arr, unpaired_arr = [], [], []

    for sym in elems:
        v_dict = vec(sym)                       # patched values for chi/radius/mass
        _, _, _, dh, up = valence_props(elem(sym))

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

    vals         = np.array(vals,         dtype=float)
    dhalf_arr    = np.array(dhalf_arr,    dtype=float)
    unpaired_arr = np.array(unpaired_arr, dtype=float)

    # ── local helpers ─────────────────────────────────────────────────────

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

    # ── compute ───────────────────────────────────────────────────────────

    chi_mean = _wmean(chis,   chi_mask)
    r_mean   = _wmean(radii,  r_mask)
    m_mean   = _wmean(masses, m_mask)
    val_mean = float(np.sum(w_norm * vals))
    up_mean  = float(np.sum(w_norm * unpaired_arr))

    return {
        # ── Stoichiometric structure ──────────────────────────────────────
        # Number of distinct elements — binary vs quinary is a strong prior
        "n_elements":    float(len(elems)),

        # Majority element stoichiometric fraction
        # Near 1.0 → near-pure or dilute doping; near 1/n → equiatomic
        "max_weight":    float(w_norm.max()),

        # Configurational entropy (high-entropy alloy formalism)
        # Redundant with max_weight for simple binaries but captures
        # full distribution shape for multicomponent compositions
        "conf_entropy":  float(-np.sum(w_norm * np.log(w_norm + 1e-12))),

        # ── Electronegativity mismatch → bond ionicity proxy ──────────────
        # delta_chi: raw span from most electropositive to most electronegative
        # element in the compound — directly analogous to Phillips ionicity
        # and Pettifor structure maps. Unweighted because the extreme elements
        # determine the bond polarity ceiling regardless of stoichiometry.
        # chi_mad: stoichiometry-weighted average deviation — complements
        # delta_chi by capturing how spread out the distribution is.
        "chi_mad":       _wmad(chis,   chi_mask, chi_mean),
        "delta_chi":     _wrng(chis,   chi_mask),

        # ── Atomic size mismatch → lattice strain proxy ───────────────────
        "r_mad":         _wmad(radii,  r_mask, r_mean),

        # ── Mass dispersion ───────────────────────────────────────────────
        "mass_std":      _wstd(masses, m_mask, m_mean),

        # ── Valence electron statistics ───────────────────────────────────
        "val_mean":      val_mean,
        "val_var":       float(np.sum(w_norm * (vals - val_mean) ** 2)),

        # ── d-shell half-filling → proximity to Hund's maximum ───────────
        "dhalf_mean":    float(np.sum(w_norm * dhalf_arr)),

        # ── Transition metal (d-block) fraction ──────────────────────────
        "tm_frac":       float(np.sum(w_norm * np.array(
                             [elem(s).block == "d" for s in elems], dtype=float
                         ))),

        # ── f-block (lanthanide / actinide) fraction ──────────────────────
        # Analogous to tm_frac. Rare-earth magnets (SmCo, NdFeB analogues)
        # and heavy-fermion systems are characterised by high f_frac.
        # Without this, all-lanthanide and all-d-block compounds look
        # identical to the model from the tm_frac perspective.
        "f_frac":        float(np.sum(w_norm * np.array(
                             [elem(s).block == "f" for s in elems], dtype=float
                         ))),

        # ── Unpaired electron statistics → magnetic moment proxy ──────────
        # Directly relevant to half-metal / spintronic classification.
        # unpaired_var captures whether all elements contribute equally
        # to spin (low var) or one element dominates (high var).
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
    Feature count: 19 properties x 5 stats = 95 elemental features
                   (mean, std, min, max, mad, pos — rng removed as redundant)
                   Properties: Z, mass, chi, radius, volume, polar, hard,
                   val, vac, dcount, dhalf, unpaired, EA, I1, Tm, Tb,
                   kappa, Ecoh, magmom
                 + 15 physics features:
                   n_elements, max_weight, conf_entropy,
                   chi_mad, delta_chi, r_mad, mass_std,
                   val_mean, val_var, dhalf_mean,
                   tm_frac, f_frac,
                   unpaired_mean, unpaired_var
                 = 109 features total
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

        # All property dicts cached after first call per element
        elem_dicts = [vec(e) for e in elems]
        keys       = list(elem_dicts[0].keys())

        feats = {}

        # Weighted statistics per elemental property  (5 stats × 19 props = 95)
        for k in keys:
            col_values = [d[k] for d in elem_dicts]
            feats.update(stats(k, col_values, w))

        # Composition-level physics features  (14 features)
        feats.update(phys(elems, w))

        out.append(feats)

    return pd.concat(
        [df.reset_index(drop=True), pd.DataFrame(out)],
        axis=1,
    )