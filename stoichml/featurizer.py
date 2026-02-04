import numpy as np
import pandas as pd
from mendeleev import element
from functools import lru_cache
import math
import re

# ---------- Helpers ----------
def _to_float_safe(x, default=0.0):
    try:
        if callable(x):
            x = x()
        if x is None or (isinstance(x, float) and np.isnan(x)):
            return default
        return float(str(x)) if not isinstance(x, (int, float)) else float(x)
    except:
        return default

@lru_cache(None)
def get_elem(sym):
    return element(sym)

# ---------- Valence + d-electron ----------
def valence_shell_vacancy_and_dfrac(e):
    econf_pattern = re.compile(r'(\d+)([spdf])(\d+)')
    shells = [(int(n), orb, int(occ)) for n, orb, occ in econf_pattern.findall(e.econf or "")]
    if not shells:
        return 0.0, 0.0, 0.0

    n_max = max(s[0] for s in shells)
    block = e.block
    val_e, cap, d_count = 0, 0, 0

    for n, orb, occ in shells:
        if block in ("s", "p"):
            if n == n_max:
                val_e += occ
                cap += {"s":2,"p":6}[orb]
        elif block == "d":
            if (n==n_max and orb=="s") or (n==n_max-1 and orb=="d"):
                val_e += occ
                cap += {"s":2,"d":10}[orb]
            if orb=="d":
                d_count += occ
        elif block == "f":
            if ((n==n_max and orb=="s") or (n==n_max-1 and orb=="d") or (n==n_max-2 and orb=="f")):
                val_e += occ
                cap += {"s":2,"d":10,"f":14}[orb]

    vacancy = max(cap - val_e, 0.0)
    d_frac = d_count / max(val_e,1e-12)
    return float(val_e), float(vacancy), float(d_frac)

# ---------- Packed vector ----------
PROP_KEYS = [
    "Z","atomic_mass","chi","radius","atomic_volume",
    "polarizability","hardness",
    "NValence","NValenceVacancy","d_frac",
    "EA","I1","Tm","Tb","kappa"
]

@lru_cache(None)
def packed_vector(sym: str):
    e = get_elem(sym)
    val_e, vac, d_frac = valence_shell_vacancy_and_dfrac(e)

    raw = [
        e.atomic_number,
        e.atomic_weight,
        e.en_pauling,
        e.covalent_radius or e.atomic_radius,
        getattr(e,"atomic_volume",0.0),
        getattr(e,"dipole_polarizability",0.0),
        getattr(e,"hardness",0.0),
        val_e,
        vac,
        d_frac,
        getattr(e,"electron_affinity",0.0),
        e.ionenergies.get(1) if e.ionenergies else 0.0,
        getattr(e,"melting_point",0.0),
        getattr(e,"boiling_point",0.0),
        getattr(e,"thermal_conductivity",0.0)
    ]

    return np.array([_to_float_safe(x,0.0) for x in raw],dtype=float)

# ---------- Safe weighted statistics ----------
def safe_weighted_stats(values, weights):
    vals = values
    w = weights / weights.sum()
    mean = np.sum(w*vals)
    var = np.sum(w*(vals-mean)**2)
    return {
        "mean": mean,
        "std": np.sqrt(var),
        "min": float(vals.min()),
        "max": float(vals.max()),
        "range": float(np.ptp(vals))
    }

# ---------- Physics-informed stoichiometric features ----------
def stoich_physics_features(elems, fractions):
    chis = np.array([_to_float_safe(get_elem(e).en_pauling,0.0) for e in elems])
    radii = np.array([_to_float_safe(get_elem(e).covalent_radius or get_elem(e).atomic_radius,0.0) for e in elems])
    Nval = np.array([valence_shell_vacancy_and_dfrac(get_elem(e))[0] for e in elems])
    dfrac = np.array([valence_shell_vacancy_and_dfrac(get_elem(e))[2] for e in elems])

    chi_bar = np.sum(fractions*chis)
    r_bar   = np.sum(fractions*radii)
    nval_bar = np.sum(fractions*Nval)

    entropy = -np.sum(fractions*np.log(fractions+1e-12))

    return {
        "stoich_entropy": entropy,
        "chi_mismatch": np.sum(fractions*np.abs(chis-chi_bar)),
        "radius_mismatch": np.sum(fractions*np.abs(radii-r_bar)),
        "Nval_mean": nval_bar,
        "Nval_var": np.sum(fractions*(Nval-nval_bar)**2),
        "dfrac_mean": np.sum(fractions*dfrac),
        "TM_fraction": np.sum(fractions*[get_elem(e).block=="d" for e in elems])
    }

# ---------- Featurizer ----------
def featurize(df, elements_col="elements", composition_col="composition"):
    rows = []

    for _, row in df.iterrows():
        elems = row[elements_col]
        comp = np.array(row[composition_col],dtype=float)
        fractions = comp / comp.sum()

        mat = np.vstack([packed_vector(e) for e in elems])
        feats = {}

        # Element-property statistics
        for idx,key in enumerate(PROP_KEYS):
            stats = safe_weighted_stats(mat[:,idx], fractions)
            for k,v in stats.items():
                feats[f"{key}_{k}"] = v

        # Physics-informed stoichiometric features
        phys = stoich_physics_features(elems, fractions)
        feats.update(phys)

        rows.append(feats)

    return pd.concat([df.reset_index(drop=True), pd.DataFrame(rows)], axis=1)
