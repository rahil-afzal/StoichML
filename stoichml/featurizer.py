### **stoichml/featurizer.py**
import numpy as np
import pandas as pd
from mendeleev import element
from functools import lru_cache

# ---------- Helpers ----------
def _to_float_safe(x):
    try:
        if callable(x):
            x = x()
        if x is None:
            return np.nan
        return float(x)
    except:
        return np.nan

@lru_cache(None)
def get_elem(sym):
    return element(sym)

# ---------- Valence + d-electron ----------
def valence_shell_vacancy_and_dfrac(e):
    # Parse electronic configuration
    import re
    econf_pattern = re.compile(r'(\d+)([spdf])(\d+)')
    shells = [(int(n), orb, int(occ)) for n, orb, occ in econf_pattern.findall(e.econf)]
    if not shells:
        return 0.0, 0.0, 0.0
    n_max = max(s[0] for s in shells)
    block = e.block
    val_e = 0
    cap = 0
    d_count = 0
    for n, orb, occ in shells:
        if block in ("s","p"):
            if n == n_max:
                val_e += occ
                cap += {"s":2,"p":6}[orb]
        elif block=="d":
            if (n==n_max and orb=="s") or (n==n_max-1 and orb=="d"):
                val_e += occ
                cap += {"s":2,"d":10}[orb]
            if orb=="d":
                d_count += occ
        elif block=="f":
            if ((n==n_max and orb=="s") or (n==n_max-1 and orb=="d") or (n==n_max-2 and orb=="f")):
                val_e += occ
                cap += {"s":2,"d":10,"f":14}[orb]
    vacancy = max(cap - val_e, 0.0)
    d_frac = d_count / max(val_e,1e-12)
    return float(val_e), float(vacancy), float(d_frac)

# ---------- Packed vector ----------
PROP_KEYS = [
    "Z", "atomic_mass", "chi", "radius", "atomic_volume",
    "polarizability","hardness",
    "NValence","NValenceVacancy","d_frac",
    "EA","I1"
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
        getattr(e,"atomic_volume",np.nan),
        getattr(e,"dipole_polarizability",np.nan),
        getattr(e,"hardness",np.nan),
        val_e,
        vac,
        d_frac,
        getattr(e,"electron_affinity",np.nan),
        e.ionenergies.get(1) if e.ionenergies else np.nan
    ]
    return np.array([_to_float_safe(x) for x in raw],dtype=float)

# ---------- Safe weighted statistics ----------
def safe_weighted_stats(values, weights):
    mask = ~np.isnan(values)
    if not mask.any(): return {}
    vals = values[mask]
    w = weights[mask]
    ws = w.sum()
    if ws <= 1e-12: return {}
    w = w/ws
    mean = np.sum(w*vals)
    var = np.sum(w*(vals-mean)**2)
    return {
        "mean": mean,
        "std": np.sqrt(var),
        "min": float(vals.min()),
        "max": float(vals.max()),
        "range": float(vals.ptp())
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
        for idx,key in enumerate(PROP_KEYS):
            stats = safe_weighted_stats(mat[:,idx], fractions)
            for k,v in stats.items():
                feats[f"{key}_{k}"] = v
        rows.append(feats)
    return pd.concat([df.reset_index(drop=True), pd.DataFrame(rows)],axis=1)