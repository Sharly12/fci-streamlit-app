# models/__init__.py
"""
Hydrological model modules.

Currently implemented:
- fci_model: Flow Corridor Importance

Placeholders (to be implemented):
- hsr_model: Hydrological Storage Role
- pec_model: Parcel Elevation Context
- uds_model: Upstream–Downstream Sensitivity
- sei_model: Surrounding Exposure Index
"""

import numpy as np

# NumPy 2.4+ removed np.in1d; some pysheds versions still call it.
# This alias restores compatibility without changing analysis behavior.
if not hasattr(np, "in1d"):
    np.in1d = np.isin
