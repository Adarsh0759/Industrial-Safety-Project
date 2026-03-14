"""
SystemState: Unified data structure combining all three AI pipelines

Represents the complete state of the vision system for a single frame.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from enum import Enum
import numpy as np
from datetime import datetime


class SafetyLevel(Enum):
    """Safety alert severity levels"""
    SAFE = "safe"
    WARNING = "warning"
    CRITICAL = "critical"


@dataclass
class Detection:
    """Single object detection result"""
    class_id: int
    class_name: str
    confidence: float
    bbox: tuple  # (x1, y1, x2, y2)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'class_id': self.class_id,
            'class_name': self.class_name,
            'confidence': float(self.confidence),
            'bbox': list(self.bbox)
        }


@dataclass
class HandGesture:
    """Hand gesture detection result"""
    gesture_id: int
    gesture_name: str
    confidence: float
    bbox: tuple  # (x1, y1, x2, y2)
    hand_side: Optional[str] = None  # 'left' or 'right'
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'gesture_id': self.gesture_id,
            'gesture_name': self.gesture_name,
            'confidence': float(self.confidence),
            'bbox': list(self.bbox),
            'hand_side': self.hand_side
        }


@dataclass
class SafetyAlert:
    """Safety compliance violation alert"""
    alert_type: str  # e.g., 'PPE_VIOLATION', 'DANGER_ZONE_ENTRY'
    severity: SafetyLevel
    message: str
    affected_regions: List[tuple] = field(default_factory=list)
    recommended_action: str = ""
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'alert_type': self.alert_type,
            'severity': self.severity.value,
            'message': self.message,
            'affected_regions': self.affected_regions,
            'recommended_action': self.recommended_action,
            'timestamp': self.timestamp.isoformat()
        }


@dataclass
class DetectionPipelineOutput:
    """Output from Detection Model pipeline"""
    detections: List[Detection] = field(default_factory=list)
    people_count: int = 0
    tools_detected: List[str] = field(default_factory=list)
    processing_time_ms: float = 0.0
    has_hand_in_roi: bool = False  # Important for gesture model activation
    hand_bbox: Optional[tuple] = None  # Bounding box of detected hand (if any)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'detections': [d.to_dict() for d in self.detections],
            'people_count': self.people_count,
            'tools_detected': self.tools_detected,
            'processing_time_ms': float(self.processing_time_ms),
            'has_hand_in_roi': self.has_hand_in_roi,
            'hand_bbox': list(self.hand_bbox) if self.hand_bbox else None
        }


@dataclass
class GesturePipelineOutput:
    """Output from Hand Gesture Model pipeline"""
    gestures: List[HandGesture] = field(default_factory=list)
    was_triggered: bool = False  # Whether gesture model ran on this frame
    trigger_reason: str = ""  # Why model was or wasn't triggered
    processing_time_ms: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'gestures': [g.to_dict() for g in self.gestures],
            'was_triggered': self.was_triggered,
            'trigger_reason': self.trigger_reason,
            'processing_time_ms': float(self.processing_time_ms)
        }


@dataclass
class SafetyPipelineOutput:
    """Output from Safety Compliance Model pipeline"""
    alerts: List[SafetyAlert] = field(default_factory=list)
    overall_safety_level: SafetyLevel = SafetyLevel.SAFE
    ppe_violations: List[str] = field(default_factory=list)  # Missing PPE items
    danger_zone_entries: List[tuple] = field(default_factory=list)  # (person_id, zone_name)
    processing_time_ms: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'alerts': [a.to_dict() for a in self.alerts],
            'overall_safety_level': self.overall_safety_level.value,
            'ppe_violations': self.ppe_violations,
            'danger_zone_entries': self.danger_zone_entries,
            'processing_time_ms': float(self.processing_time_ms)
        }


@dataclass
class SystemState:
    """
    Complete unified state of the vision system for a single frame.
    
    Combines outputs from all three AI pipelines with orchestrator metadata.
    """
    
    # Frame metadata
    frame_id: int
    timestamp: datetime = field(default_factory=datetime.now)
    frame_width: int = 0
    frame_height: int = 0
    
    # Pipeline outputs
    detection_output: DetectionPipelineOutput = field(default_factory=DetectionPipelineOutput)
    gesture_output: GesturePipelineOutput = field(default_factory=GesturePipelineOutput)
    safety_output: SafetyPipelineOutput = field(default_factory=SafetyPipelineOutput)
    
    # Orchestrator decisions
    orchestrator_decisions: Dict[str, Any] = field(default_factory=dict)
    
    # Performance metrics
    total_processing_time_ms: float = 0.0
    fps: float = 0.0
    
    # System status
    is_error_state: bool = False
    error_message: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert entire system state to dictionary for JSON serialization"""
        return {
            'frame_id': self.frame_id,
            'timestamp': self.timestamp.isoformat(),
            'frame_size': {
                'width': self.frame_width,
                'height': self.frame_height
            },
            'detection_pipeline': self.detection_output.to_dict(),
            'gesture_pipeline': self.gesture_output.to_dict(),
            'safety_pipeline': self.safety_output.to_dict(),
            'orchestrator_decisions': self.orchestrator_decisions,
            'performance': {
                'total_processing_time_ms': float(self.total_processing_time_ms),
                'fps': float(self.fps)
            },
            'system_status': {
                'is_error': self.is_error_state,
                'error_message': self.error_message
            }
        }
    
    def get_highest_alert_severity(self) -> SafetyLevel:
        """Get the highest severity alert from safety pipeline"""
        if not self.safety_output.alerts:
            return SafetyLevel.SAFE
        return max(alert.severity for alert in self.safety_output.alerts)
    
    def has_alerts(self) -> bool:
        """Check if there are any safety alerts"""
        return len(self.safety_output.alerts) > 0
    
    def has_critical_alerts(self) -> bool:
        """Check if there are critical safety alerts"""
        return any(
            alert.severity == SafetyLevel.CRITICAL
            for alert in self.safety_output.alerts
        )
