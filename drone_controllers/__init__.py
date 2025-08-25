"""Implementations of onboard drone controllers in Python.

All controllers are implemented using the array API standard. This means that every controller is
agnostic to the choice of framework and supports e.g. NumPy, JAX, or PyTorch. We also implement all
controllers as pure functions to ensure that users can jit-compile them. All controllers use
broadcasting to support batching of arbitrary leading dimensions.
"""

import os
import sys
from typing import Callable

# SciPy array API check. We use the most recent array API features, which require the
# SCIPY_ARRAY_API environment variable to be set to "1". This flag MUST be set before importing
# scipy, because scipy's C extensions cannot be unloaded once they have been imported. Therefore, we
# have to error out if the flag is not set. Otherwise, we immediately import scipy to ensure that no
# other package sets the flag to a different value before importing scipy.

if "scipy" in sys.modules and os.environ.get("SCIPY_ARRAY_API") != "1":
    msg = """scipy has already been imported and the 'SCIPY_ARRAY_API' environment variable has not
    been set. Please restart your Python session and set SCIPY_ARRAY_API="1" before importing any
    packages that depend on scipy, or import this package first to automatically set the flag."""
    raise RuntimeError(msg)

os.environ["SCIPY_ARRAY_API"] = "1"
import scipy  # noqa: F401, ensure scipy uses array API features

from drone_controllers.core import parametrize
from drone_controllers.mellinger import attitude2force_torque as mellinger_attitude2force_torque
from drone_controllers.mellinger import state2attitude as mellinger_state2attitude

available_controller: dict[str, Callable] = {
    "mellinger_state2attitude": mellinger_state2attitude,
    "mellinger_attitude2force_torque": mellinger_attitude2force_torque,
}

__all__ = ["parametrize"]
