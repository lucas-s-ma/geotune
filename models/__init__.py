"""
GeoTune New models package init
"""
from .geotune_esm_model import load_esm_with_lora, GeoTuneESMModel, create_lora_config

# Don't import gearnet_model here to avoid torchdrug import conflicts with transformers
# Import directly when needed: from models.gearnet_model import GearNet

__all__ = [
    'load_esm_with_lora',
    'GeoTuneESMModel',
    'create_lora_config',
]