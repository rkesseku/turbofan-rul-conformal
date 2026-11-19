"""Uncertainty-aware remaining-useful-life prediction on NASA C-MAPSS."""

# On Windows, importing torch after scikit-learn has initialised its OpenMP runtime can fail
# to load torch's DLLs, so torch is imported first, before any submodule is used.
import torch  # noqa: F401

__version__ = "0.1.0"
