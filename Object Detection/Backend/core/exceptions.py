"""
Custom Exception Classes for Vision System

Organized by severity and category for precise error handling.
"""

from typing import Optional


class VisionSystemException(Exception):
    """Base exception for all vision system errors"""
    
    def __init__(self, message: str, error_code: str = "UNKNOWN"):
        self.message = message
        self.error_code = error_code
        super().__init__(f"[{error_code}] {message}")


class ModelException(VisionSystemException):
    """Base exception for model-related errors"""
    pass


class ModelLoadException(ModelException):
    """Raised when model fails to load"""
    
    def __init__(self, model_name: str, path: str, reason: Optional[str] = None):
        message = f"Failed to load {model_name} from {path}"
        if reason:
            message += f": {reason}"
        super().__init__(message, "MODEL_LOAD_FAILED")
        self.model_name = model_name
        self.model_path = path


class ModelInferenceException(ModelException):
    """Raised when model inference fails"""
    
    def __init__(self, model_name: str, reason: Optional[str] = None):
        message = f"Inference failed for {model_name}"
        if reason:
            message += f": {reason}"
        super().__init__(message, "MODEL_INFERENCE_FAILED")
        self.model_name = model_name


class HardwareException(VisionSystemException):
    """Base exception for hardware-related errors"""
    pass


class CameraException(HardwareException):
    """Raised when camera initialization or capture fails"""
    
    def __init__(self, reason: Optional[str] = None):
        message = "Camera error"
        if reason:
            message += f": {reason}"
        super().__init__(message, "CAMERA_FAILED")


class DataException(VisionSystemException):
    """Base exception for data-related errors"""
    pass


class InvalidConfigException(DataException):
    """Raised when configuration is invalid"""
    
    def __init__(self, config_key: str, expected_type: str, actual_value=None):
        message = f"Invalid config '{config_key}': expected {expected_type}"
        if actual_value is not None:
            message += f", got {type(actual_value).__name__}"
        super().__init__(message, "INVALID_CONFIG")
        self.config_key = config_key


class SafetyAlertException(VisionSystemException):
    """Raised for safety violations - not an error, but an alert"""
    
    def __init__(self, alert_type: str, details: dict):
        message = f"SAFETY ALERT: {alert_type}"
        super().__init__(message, f"SAFETY_{alert_type.upper()}")
        self.alert_type = alert_type
        self.details = details


class FrameProcessingException(VisionSystemException):
    """Raised when frame processing encounters an error"""
    
    def __init__(self, stage: str, reason: Optional[str] = None):
        message = f"Frame processing failed at {stage}"
        if reason:
            message += f": {reason}"
        super().__init__(message, "FRAME_PROCESSING_FAILED")
        self.stage = stage


class OrchestrationException(VisionSystemException):
    """Raised when orchestrator encounters a critical error"""
    
    def __init__(self, reason: str):
        super().__init__(f"Orchestration error: {reason}", "ORCHESTRATION_FAILED")
