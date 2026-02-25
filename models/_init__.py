# /mount/src/fci-streamlit-app/models/__init__.py

import numpy as np

# ---- NumPy compatibility shims for older libraries (e.g., pysheds) ----
# NumPy 2.4+ removed np.in1d; pysheds may still call it.
if not hasattr(np, "in1d"):
    np.in1d = np.isin

# Older libs sometimes still reference these deprecated aliases.
_deprecated_aliases = {
    "bool": bool,
    "int": int,
    "float": float,
    "complex": complex,
    "object": object,
    "str": str,
}
for _name, _pytype in _deprecated_aliases.items():
    if not hasattr(np, _name):
        setattr(np, _name, _pytype)
