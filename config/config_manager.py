"""
Configuration Management System

Loads and validates YAML/JSON configuration files for safe zones and model parameters.
"""

from dataclasses import dataclass
from typing import List, Dict, Any, Optional
import json
import yaml
from pathlib import Path
from core.exceptions import InvalidConfigException
from utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class DangerZone:
    """Defines a danger zone with specific safety requirements"""
    name: str
    region: tuple  # (x1, y1, x2, y2) in normalized coordinates (0-1)
    required_ppe: List[str]  # ['gloves', 'helmet', 'vest']
    banned_tools: List[str] = None  # Tools not allowed in this zone
    allowed_tools: List[str] = None  # If specified, ONLY these tools allowed
    alert_level: str = 'warning'  # 'warning' or 'critical'
    
    def contains_point(self, x: float, y: float) -> bool:
        """Check if point (x, y) in normalized coords is in this zone"""
        x1, y1, x2, y2 = self.region
        return x1 <= x <= x2 and y1 <= y <= y2
    
    def contains_bbox(self, bbox: tuple, threshold: float = 0.5) -> bool:
        """Check if bounding box overlaps with zone (normalized coords)"""
        bx1, by1, bx2, by2 = bbox
        zx1, zy1, zx2, zy2 = self.region
        
        # Calculate intersection area
        inter_x1 = max(bx1, zx1)
        inter_y1 = max(by1, zy1)
        inter_x2 = min(bx2, zx2)
        inter_y2 = min(by2, zy2)
        
        if inter_x1 >= inter_x2 or inter_y1 >= inter_y2:
            return False
        
        inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
        bbox_area = (bx2 - bx1) * (by2 - by1)
        
        return inter_area / bbox_area >= threshold


@dataclass
class SafeZone:
    """Defines a safe zone where certain activities are encouraged"""
    name: str
    region: tuple  # (x1, y1, x2, y2) in normalized coordinates


class VisionSystemConfig:
    """Central configuration manager for vision system"""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize configuration
        
        Args:
            config_path: Path to YAML/JSON config file. If None, uses defaults.
        """
        self.config_path = config_path
        self.raw_config: Dict[str, Any] = {}
        
        self.danger_zones: List[DangerZone] = []
        self.safe_zones: List[SafeZone] = []
        
        # Model configuration
        self.detection_model_path: str = "yolov8m.pt"
        self.gesture_model_path: str = "Models/hand.pt"
        self.safety_model_path: str = "Models/safety.pt"
        
        # Inference parameters
        self.detection_conf_threshold: float = 0.45
        self.gesture_conf_threshold: float = 0.4
        self.safety_conf_threshold: float = 0.5
        
        # Frame skipping for performance
        self.detection_frame_skip: int = 1  # Run on every frame
        self.gesture_frame_skip: int = 1
        self.safety_frame_skip: int = 3  # Run every 3 frames
        
        # ROI parameters for gesture model
        self.gesture_roi_padding: int = 20  # pixels
        self.gesture_roi_min_size: int = 32  # minimum region size
        
        # Load configuration if provided
        if config_path:
            self.load_from_file(config_path)
        else:
            self._setup_default_config()
        
        logger.info("Configuration loaded successfully")
    
    def load_from_file(self, config_path: str) -> None:
        """Load configuration from YAML or JSON file"""
        config_path = Path(config_path)
        
        if not config_path.exists():
            logger.warning(f"Config file not found: {config_path}, using defaults")
            self._setup_default_config()
            return
        
        try:
            if config_path.suffix == '.yaml' or config_path.suffix == '.yml':
                with open(config_path, 'r') as f:
                    self.raw_config = yaml.safe_load(f) or {}
            elif config_path.suffix == '.json':
                with open(config_path, 'r') as f:
                    self.raw_config = json.load(f)
            else:
                raise InvalidConfigException(
                    'config_file',
                    'YAML or JSON',
                    config_path.suffix
                )
            
            self._parse_config()
            logger.info(f"Configuration loaded from {config_path}")
            
        except Exception as e:
            logger.error(f"Failed to load config: {e}")
            self._setup_default_config()
    
    def _parse_config(self) -> None:
        """Parse raw config dict into typed objects"""
        
        # Parse danger zones
        if 'danger_zones' in self.raw_config:
            for zone_dict in self.raw_config['danger_zones']:
                self.danger_zones.append(DangerZone(
                    name=zone_dict['name'],
                    region=tuple(zone_dict['region']),
                    required_ppe=zone_dict.get('required_ppe', []),
                    banned_tools=zone_dict.get('banned_tools'),
                    allowed_tools=zone_dict.get('allowed_tools'),
                    alert_level=zone_dict.get('alert_level', 'warning')
                ))
        
        # Parse safe zones
        if 'safe_zones' in self.raw_config:
            for zone_dict in self.raw_config['safe_zones']:
                self.safe_zones.append(SafeZone(
                    name=zone_dict['name'],
                    region=tuple(zone_dict['region'])
                ))
        
        # Parse model paths
        models_config = self.raw_config.get('models', {})
        self.detection_model_path = models_config.get(
            'detection', self.detection_model_path
        )
        self.gesture_model_path = models_config.get(
            'gesture', self.gesture_model_path
        )
        self.safety_model_path = models_config.get(
            'safety', self.safety_model_path
        )
        
        # Parse inference parameters
        inference_config = self.raw_config.get('inference', {})
        self.detection_conf_threshold = inference_config.get(
            'detection_conf_threshold', self.detection_conf_threshold
        )
        self.gesture_conf_threshold = inference_config.get(
            'gesture_conf_threshold', self.gesture_conf_threshold
        )
        self.safety_conf_threshold = inference_config.get(
            'safety_conf_threshold', self.safety_conf_threshold
        )
        
        # Parse frame skipping
        performance_config = self.raw_config.get('performance', {})
        self.detection_frame_skip = performance_config.get(
            'detection_frame_skip', self.detection_frame_skip
        )
        self.gesture_frame_skip = performance_config.get(
            'gesture_frame_skip', self.gesture_frame_skip
        )
        self.safety_frame_skip = performance_config.get(
            'safety_frame_skip', self.safety_frame_skip
        )
        
        # Parse ROI parameters
        roi_config = self.raw_config.get('gesture_roi', {})
        self.gesture_roi_padding = roi_config.get(
            'padding', self.gesture_roi_padding
        )
        self.gesture_roi_min_size = roi_config.get(
            'min_size', self.gesture_roi_min_size
        )
    
    def _setup_default_config(self) -> None:
        """Setup default configuration"""
        logger.info("Using default configuration")
        
        # Default danger zones (construction site example)
        self.danger_zones = [
            DangerZone(
                name="High-Power Tool Zone",
                region=(0.0, 0.6, 1.0, 1.0),  # Bottom third of frame
                required_ppe=['gloves', 'helmet'],
                banned_tools=[],
                alert_level='critical'
            ),
            DangerZone(
                name="Electrical Zone",
                region=(0.0, 0.0, 0.3, 0.3),  # Top-left quadrant
                required_ppe=['gloves', 'vest'],
                alert_level='warning'
            )
        ]
        
        self.safe_zones = [
            SafeZone(
                name="Rest Area",
                region=(0.7, 0.7, 1.0, 1.0)
            )
        ]
    
    def get_danger_zones_for_bbox(self, bbox: tuple) -> List[DangerZone]:
        """Get all danger zones that contain this bounding box"""
        normalized_bbox = self._normalize_bbox(bbox)
        return [z for z in self.danger_zones if z.contains_bbox(normalized_bbox)]
    
    def _normalize_bbox(self, bbox: tuple) -> tuple:
        """Convert pixel coords to normalized (0-1) coords - override with frame size"""
        # This is a placeholder - actual implementation needs frame dimensions
        return bbox
    
    def to_dict(self) -> Dict[str, Any]:
        """Export configuration as dictionary"""
        return {
            'danger_zones': [
                {
                    'name': z.name,
                    'region': z.region,
                    'required_ppe': z.required_ppe,
                    'alert_level': z.alert_level
                }
                for z in self.danger_zones
            ],
            'safe_zones': [
                {'name': z.name, 'region': z.region}
                for z in self.safe_zones
            ],
            'models': {
                'detection': self.detection_model_path,
                'gesture': self.gesture_model_path,
                'safety': self.safety_model_path
            },
            'inference': {
                'detection_conf_threshold': self.detection_conf_threshold,
                'gesture_conf_threshold': self.gesture_conf_threshold,
                'safety_conf_threshold': self.safety_conf_threshold
            },
            'performance': {
                'detection_frame_skip': self.detection_frame_skip,
                'gesture_frame_skip': self.gesture_frame_skip,
                'safety_frame_skip': self.safety_frame_skip
            }
        }
