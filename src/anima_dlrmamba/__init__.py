"""DLRMamba ANIMA module — Low-Rank SS2D + Structure-Aware Distillation for RGB-IR detection."""

from .config import AppConfig, load_config
from .models.model import DLRMambaDetector

__all__ = ["AppConfig", "load_config", "DLRMambaDetector"]
