import numpy as np
import pandas as pd
from mendeleev import element  # type: ignore
from functools import lru_cache
import re

# =====================
# Basic helpers
# =====================

def fnum(x):
    try:
        if callable(x):
            x = x()
        v = float(x)
        if np.isnan(v):
            return None
        return v
    except:
        return None


@lru_cache(None)
def elem(sym):
    return element(sym)


# =====================
# Valence + d-shell
# =====================

def valence_props(e):
    conf = e.econf or ""
    parts = re.findall(r'(\d+)([spdf])(\d+)', conf)
    if not parts:
        return 0.0, 0.0, 0.0, 5.0

    shells = [(int(n), o, int(k)) for n, o, k in parts]
    nmax = max(n for n, _, _ in shells)
    block = e.block

    val = 0
    cap = 0
    dcnt = 0

    for n, o, k in shells:
        if block in ("s", "p"):
            if n == nmax:
                val += k
                cap += {"s": 2, "p": 6}.get(o, 0)

        elif block == "d":
            if (n == nmax and o == "s") or (n == nmax - 1 and o == "d"):
                val += k
                cap += {"s": 2, "d": 10}.get(o, 0)
            if o == "d":
                dcnt += k

        elif block == "f":
            if (
                (n == nmax and o == "s")
                or (n == nmax - 1 and o == "d")
                or (n == nmax - 2 and o == "f")
            ):
                val += k
                cap += {"s": 2, "d": 10, "f": 14}.get(o, 0)

    vac = max(cap - val, 0)
    d_half = abs(dcnt - 5)

    return float(val), float(vac), float(dcnt), float(d_half)


# =====================
# Element vector
# =====================

KEYS = [
    "Z",
    "mass",
    "chi",
    "radius",
    "volume",
    "polar",
    "hard",
    "val",
    "vac",
    "dcount",
    "dhalf",
    "EA",
    "I1",
    "Tm",
    "Tb",
    "kappa",
]


@lru_cache(None)
def vec(sym):
    e = elem(sym)
    val, vac, dcnt, dhalf = valence_props(e)

    raw = [
        e.atomic_number,
        e.atomic_weight,
        e.en_pauling,
        e.covalent_radius or e.atomic_radius,
        getattr(e, "atomic_volume", None),
        getattr(e, "dipole_polarizability", None),
        getattr(e, "hardness", None),
        val,
        vac,
        dcnt,
        dhalf,
        getattr(e, "electron_affinity", None),
        e.ionenergies.get(1) if e.ionenergies else None,
        getattr(e, "melting_point", None),
        getattr(e, "boiling_point", None),
        getattr(e, "thermal_conductivity", None),
    ]

    return [fnum(x) for x in raw]


# =====================
# Missing-safe stats
# =====================

def stats(x, w):
    x = np.array(x, dtype=object)
    w = np.array(w, dtype=float)

    mask = np.array([v is not None for v in x])
    miss = float(1.0 - mask.mean())

    if mask.sum() == 0:
        return dict(
            mean=0.0, std=0.0, min=0.0, max=0.0,
            rng=0.0, mad=0.0, miss=1.0
        )

    xv = np.array([v for v in x[mask]], float)
    wv = w[mask]
    wv = wv / wv.sum()

    mean = float(np.sum(wv * xv))
    std = float(np.sqrt(np.sum(wv * (xv - mean) ** 2)))

    # neutral imputation for extrema + MAD
    xf = np.array([mean if v is None else v for v in x], float)

    return dict(
        mean=mean,
        std=std,
        min=float(xf.min()),
        max=float(xf.max()),
        rng=float(xf.max() - xf.min()),
        mad=float(np.sum(w * np.abs(xf - mean))),
        miss=miss,
    )




# =====================
# Stoichiometric physics
# =====================

def phys(elems, w):
    chis, radii, masses, vals, dhalf = [], [], [], [], []

    for e in elems:
        el = elem(e)
        chis.append(fnum(el.en_pauling))
        radii.append(fnum(el.covalent_radius or el.atomic_radius))
        masses.append(fnum(el.atomic_weight))
        v, _, _, dh = valence_props(el)
        vals.append(v)
        dhalf.append(dh)

    chis = np.array([c if c is not None else 0 for c in chis])
    radii = np.array([r if r is not None else 0 for r in radii])
    masses = np.array([m if m is not None else 0 for m in masses])
    vals = np.array(vals)
    dhalf = np.array(dhalf)

    chi_m = np.sum(w * chis)
    r_m = np.sum(w * radii)
    m_m = np.sum(w * masses)

    return {
        "conf_entropy": -np.sum(w * np.log(w + 1e-12)),
        "chi_mad": np.sum(w * np.abs(chis - chi_m)),
        "chi_rng": chis.max() - chis.min(),
        "r_mad": np.sum(w * np.abs(radii - r_m)),
        "mass_std": np.sqrt(np.sum(w * (masses - m_m) ** 2)),
        "val_mean": np.sum(w * vals),
        "val_var": np.sum(w * (vals - np.sum(w * vals)) ** 2),
        "dhalf_mean": np.sum(w * dhalf),
        "tm_frac": np.sum(w * np.array([elem(e).block == "d" for e in elems], float)),
    }


# =====================
# Featurizer
# =====================

def featurize(df, elements_col="elements", composition_col="composition"):
    out = []

    for _, row in df.iterrows():
        elems = row[elements_col]
        comp = np.asarray(row[composition_col], float)

        s = comp.sum()
        if s <= 0:
            raise ValueError("Composition sums to zero")

        w = comp / s

        mat = list(zip(*[vec(e) for e in elems]))
        feats = {}

        for k, col in zip(KEYS, mat):
            sdict = stats(col, w)
            for n, v in sdict.items():
                feats[f"{k}_{n}"] = v

        feats.update(phys(elems, w))
        out.append(feats)

    return pd.concat(
        [df.reset_index(drop=True), pd.DataFrame(out)],
        axis=1
    )