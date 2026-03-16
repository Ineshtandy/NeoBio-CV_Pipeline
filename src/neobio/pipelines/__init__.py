"""
Pipeline orchestrators - combines blot detection stages.
"""

from .blot_pipeline import run_blot_pipeline
from .ocr_prep_pipeline import run_ocr_prep_pipeline

__all__ = ["run_blot_pipeline", "run_ocr_prep_pipeline"]
