"""
Compatibility layer for numpy versions:
- Gym (SlimeVolleyGym environment) requires older numpy with 'bool8'.
- OpenCV (SlimeVolleyGym rendering) needs newer numpy without 'bool8'.

This module bridges the gap by defining 'bool8' for both numpy versions.
"""

import numpy as np

def ensure_bool8_compatibility():
    """Ensure 'bool8' is available in numpy."""
    if not hasattr(np, 'bool8'):
        np.bool8 = np.bool_

# Run the compatibility function
ensure_bool8_compatibility()

