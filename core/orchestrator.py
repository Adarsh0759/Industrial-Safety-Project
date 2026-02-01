"""
InferenceOrchestrator: Central Mediator Pattern Implementation

Controls all three AI pipelines and prevents model overlap/conflicts.
- Models do NOT communicate directly
- All data flows through Orchestrator
- Orchestrator makes conflict resolution decisions
- Uses ROI strategy to reduce computational load
"""

import cv2
import numpy as np
import threading
import time
from typing import Optional, Dict, Any, Tuple
from dataclasses import dataclass
from enum import Enum
from queue import Queue, Empty

from core.system_state import (
    SystemState, Detection, HandGesture, SafetyAlert, SafetyLevel,
    DetectionPipelineOutput, GesturePipelineOutput, SafetyPipelineOutput
)
from core.exceptions import OrchestrationException, FrameProcessingException
from config.config_manager import VisionSystemConfig
from utils.logger import get_logger

logger = get_logger(__name__)


class OrchestratorState(Enum):
    """State of the orchestrator"""
    IDLE = "idle"
    PROCESSING = "processing"
    ERROR = "error"
    SHUTDOWN = "shutdown"


@dataclass
class ModelThreadTask:
    """Task to be processed by model thread"""
    frame: np.ndarray
    frame_id: int
    timestamp: float
    roi: Optional[Tuple[int, int, int, int]] = None  # (x1, y1, x2, y2)


class ModelThread(threading.Thread):
    """
    Dedicated thread for running a single model.
    
    This prevents frame-rate drops by running models asynchronously.
    """
    
    def __init__(self, name: str, model_loader_func, confidence_threshold: float):
        super().__init__(daemon=True, name=f"{name}_Thread")
        
        self.model_name = name
        self.model = model_loader_func()
        self.confidence_threshold = confidence_threshold
        
        self.task_queue: Queue[ModelThreadTask] = Queue(maxsize=2)
        self.result_queue: Queue[Dict[str, Any]] = Queue(maxsize=2)
        
        self.is_running = False
        self._stop_event = threading.Event()
    
    def run(self) -> None:
        """Main thread loop"""
        self.is_running = True
        logger.info(f"{self.model_name} thread started")
        
        while not self._stop_event.is_set():
            try:
                # Wait for task with timeout
                task = self.task_queue.get(timeout=1.0)
                
                try:
                    # Run inference
                    result = self._run_inference(task)
                    
                    # Put result (discard oldest if queue full)
                    try:
                        self.result_queue.put_nowait(result)
                    except:
                        try:
                            self.result_queue.get_nowait()
                        except:
                            pass
                        self.result_queue.put_nowait(result)
                
                except Exception as e:
                    logger.error(f"{self.model_name} inference error: {e}")
                    self.result_queue.put({
                        'error': str(e),
                        'frame_id': task.frame_id
                    })
            
            except Empty:
                continue
    
    def _run_inference(self, task: ModelThreadTask) -> Dict[str, Any]:
        """Override in subclass"""
        raise NotImplementedError
    
    def stop(self) -> None:
        """Stop the thread gracefully"""
        self._stop_event.set()
        self.join(timeout=2.0)
        self.is_running = False
        logger.info(f"{self.model_name} thread stopped")


class DetectionModelThread(ModelThread):
    """Thread for running detection model"""
    
    def _run_inference(self, task: ModelThreadTask) -> Dict[str, Any]:
        start_time = time.time()
        
        results = self.model.predict(
            source=task.frame,
            conf=self.confidence_threshold,
            verbose=False,
            device='cpu',
            imgsz=416
        )
        
        detections = []
        people_count = 0
        tools_detected = []
        has_hand_in_roi = False
        hand_bbox = None
        
        if results and len(results) > 0:
            result = results[0]
            
            if result.boxes is not None and len(result.boxes) > 0:
                h, w = task.frame.shape[:2]
                
                for box in result.boxes:
                    cls_id = int(box.cls[0].item())
                    conf = float(box.conf[0].item())
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    
                    # Normalize to 0-1
                    nx1, ny1, nx2, ny2 = x1/w, y1/h, x2/w, y2/h
                    
                    class_name = self.model.names.get(cls_id, f"Object_{cls_id}")
                    
                    detections.append(Detection(
                        class_id=cls_id,
                        class_name=class_name,
                        confidence=conf,
                        bbox=(nx1, ny1, nx2, ny2)
                    ))
                    
                    if class_name == 'person':
                        people_count += 1
                    
                    # Check if it's a hand
                    if class_name in ['hand', 'person']:  # Could be hand detection
                        has_hand_in_roi = True
                        hand_bbox = (x1, y1, x2, y2)
                    
                    # Track tools
                    if class_name in ['hammer', 'drill', 'saw', 'wrench', 'chainsaw']:
                        tools_detected.append(class_name)
        
        processing_time = (time.time() - start_time) * 1000
        
        return {
            'frame_id': task.frame_id,
            'detections': detections,
            'people_count': people_count,
            'tools_detected': tools_detected,
            'has_hand_in_roi': has_hand_in_roi,
            'hand_bbox': hand_bbox,
            'processing_time_ms': processing_time
        }


class GestureModelThread(ModelThread):
    """Thread for running hand gesture model"""
    
    def _run_inference(self, task: ModelThreadTask) -> Dict[str, Any]:
        start_time = time.time()
        
        # If ROI provided, use only that region
        frame_to_infer = task.frame
        if task.roi is not None:
            x1, y1, x2, y2 = task.roi
            frame_to_infer = task.frame[y1:y2, x1:x2]
        
        gestures = []
        was_triggered = bool(task.roi is not None)
        trigger_reason = "ROI provided" if was_triggered else "Gesture model disabled"
        
        if was_triggered and frame_to_infer.size > 0:
            try:
                results = self.model.predict(
                    source=frame_to_infer,
                    conf=self.confidence_threshold,
                    verbose=False,
                    device='cpu',
                    imgsz=416
                )
                
                if results and len(results) > 0:
                    result = results[0]
                    
                    if result.boxes is not None and len(result.boxes) > 0:
                        x1, y1, x2, y2 = task.roi
                        h, w = task.frame.shape[:2]
                        
                        for box in result.boxes:
                            cls_id = int(box.cls[0].item())
                            conf = float(box.conf[0].item())
                            bx1, by1, bx2, by2 = map(int, box.xyxy[0].tolist())
                            
                            # Adjust to full frame coordinates
                            frame_x1 = x1 + bx1
                            frame_y1 = y1 + by1
                            frame_x2 = x1 + bx2
                            frame_y2 = y1 + by2
                            
                            # Normalize
                            nx1 = frame_x1 / w
                            ny1 = frame_y1 / h
                            nx2 = frame_x2 / w
                            ny2 = frame_y2 / h
                            
                            gesture_name = self.model.names.get(cls_id, f"Gesture_{cls_id}")
                            
                            gestures.append(HandGesture(
                                gesture_id=cls_id,
                                gesture_name=gesture_name,
                                confidence=conf,
                                bbox=(nx1, ny1, nx2, ny2)
                            ))
            
            except Exception as e:
                logger.debug(f"Gesture inference error: {e}")
        
        processing_time = (time.time() - start_time) * 1000
        
        return {
            'frame_id': task.frame_id,
            'gestures': gestures,
            'was_triggered': was_triggered,
            'trigger_reason': trigger_reason,
            'processing_time_ms': processing_time
        }


class InferenceOrchestrator:
    """
    Central Mediator for all three AI pipelines.
    
    Controls:
    - Model execution order and frequency (frame skipping)
    - ROI strategy (only run gesture model when hand detected)
    - Conflict resolution (prevent overlapping alerts)
    - Threading for async processing
    
    Architecture:
    Frame → Detection (always) → [Hand detected?] → Gesture (conditional)
                              → Safety (every N frames)
    """
    
    def __init__(self, config: VisionSystemConfig):
        self.config = config
        self.state = OrchestratorState.IDLE
        
        # Frame counters for frame skipping
        self.detection_frame_counter = 0
        self.gesture_frame_counter = 0
        self.safety_frame_counter = 0
        
        # Frame tracking
        self.current_frame_id = 0
        self.frame_timestamps = {}
        
        # Model threads (will be initialized later)
        self.detection_thread: Optional[DetectionModelThread] = None
        self.gesture_thread: Optional[GestureModelThread] = None
        
        # Performance tracking
        self.frame_times = []
        self.max_frame_history = 30
        
        logger.info("InferenceOrchestrator initialized")
    
    def initialize_models(self, detection_model, gesture_model) -> None:
        """Initialize model threads"""
        try:
            self.detection_thread = DetectionModelThread(
                "Detection",
                lambda: detection_model,
                self.config.detection_conf_threshold
            )
            self.detection_thread.model = detection_model
            self.detection_thread.start()
            
            self.gesture_thread = GestureModelThread(
                "Gesture",
                lambda: gesture_model,
                self.config.gesture_conf_threshold
            )
            self.gesture_thread.model = gesture_model
            self.gesture_thread.start()
            
            self.state = OrchestratorState.IDLE
            logger.info("Model threads initialized and running")
        
        except Exception as e:
            self.state = OrchestratorState.ERROR
            logger.error(f"Failed to initialize model threads: {e}")
            raise OrchestrationException(f"Model initialization failed: {e}")
    
    def process_frame(self, frame: np.ndarray) -> SystemState:
        """
        Main orchestration logic - processes frame through all pipelines.
        
        Returns:
            SystemState: Complete system state for this frame
        """
        
        frame_id = self.current_frame_id
        self.current_frame_id += 1
        start_time = time.time()
        
        h, w = frame.shape[:2]
        
        try:
            # === STEP 1: Always run Detection Model ===
            detection_output = self._orchestrate_detection(frame, frame_id)
            
            # === STEP 2: Conditionally run Gesture Model ===
            # KEY INSIGHT: Only activate gesture model if detection found a hand in ROI
            gesture_output = self._orchestrate_gesture(
                frame, frame_id,
                detection_output.has_hand_in_roi,
                detection_output.hand_bbox
            )
            
            # === STEP 3: Conditionally run Safety Model ===
            # CONFLICT RESOLUTION: Check if we can safely perform the detected gesture
            safety_output = self._orchestrate_safety(
                frame, frame_id,
                detection_output,
                gesture_output
            )
            
            # === STEP 4: Create unified system state ===
            system_state = SystemState(
                frame_id=frame_id,
                frame_width=w,
                frame_height=h,
                detection_output=detection_output,
                gesture_output=gesture_output,
                safety_output=safety_output
            )
            
            # === STEP 5: Make orchestrator decisions ===
            system_state.orchestrator_decisions = self._make_orchestration_decisions(
                system_state
            )
            
            # Performance tracking
            total_time = (time.time() - start_time) * 1000
            system_state.total_processing_time_ms = total_time
            
            self.frame_times.append(total_time)
            if len(self.frame_times) > self.max_frame_history:
                self.frame_times.pop(0)
            
            system_state.fps = 1000 / (sum(self.frame_times) / len(self.frame_times)) \
                if self.frame_times else 0
            
            return system_state
        
        except Exception as e:
            logger.error(f"Frame processing error: {e}")
            system_state = SystemState(frame_id=frame_id, frame_width=w, frame_height=h)
            system_state.is_error_state = True
            system_state.error_message = str(e)
            return system_state
    
    def _orchestrate_detection(
        self,
        frame: np.ndarray,
        frame_id: int
    ) -> DetectionPipelineOutput:
        """
        Orchestrate detection model execution.
        
        Runs on configurable frame skip.
        """
        
        self.detection_frame_counter += 1
        
        if self.detection_frame_counter % self.config.detection_frame_skip != 0:
            # Skip this frame - return empty output
            return DetectionPipelineOutput()
        
        try:
            # Send task to detection thread
            task = ModelThreadTask(frame, frame_id, time.time())
            self.detection_thread.task_queue.put_nowait(task)
            
            # Try to get result
            try:
                result = self.detection_thread.result_queue.get(timeout=0.5)
                
                if 'error' in result:
                    logger.warning(f"Detection error: {result['error']}")
                    return DetectionPipelineOutput()
                
                return DetectionPipelineOutput(
                    detections=result.get('detections', []),
                    people_count=result.get('people_count', 0),
                    tools_detected=result.get('tools_detected', []),
                    processing_time_ms=result.get('processing_time_ms', 0),
                    has_hand_in_roi=result.get('has_hand_in_roi', False),
                    hand_bbox=result.get('hand_bbox')
                )
            
            except Empty:
                logger.debug("Detection result not ready yet")
                return DetectionPipelineOutput()
        
        except Exception as e:
            logger.error(f"Detection orchestration error: {e}")
            return DetectionPipelineOutput()
    
    def _orchestrate_gesture(
        self,
        frame: np.ndarray,
        frame_id: int,
        hand_detected: bool,
        hand_bbox: Optional[tuple]
    ) -> GesturePipelineOutput:
        """
        Orchestrate gesture model execution.
        
        KEY DECISION POINT: Only run gesture model if:
        1. Hand was detected by detection model
        2. Frame skip allows it
        3. ROI is valid
        
        This prevents unnecessary computation.
        """
        
        self.gesture_frame_counter += 1
        
        # Check frame skip
        if self.gesture_frame_counter % self.config.gesture_frame_skip != 0:
            return GesturePipelineOutput(
                was_triggered=False,
                trigger_reason="Frame skip"
            )
        
        # CRITICAL: Only activate if hand detected AND ROI valid
        if not hand_detected or hand_bbox is None:
            return GesturePipelineOutput(
                was_triggered=False,
                trigger_reason="No hand detected in ROI"
            )
        
        try:
            x1, y1, x2, y2 = hand_bbox
            h, w = frame.shape[:2]
            
            # Add padding
            padding = self.config.gesture_roi_padding
            x1 = max(0, x1 - padding)
            y1 = max(0, y1 - padding)
            x2 = min(w, x2 + padding)
            y2 = min(h, y2 + padding)
            
            # Check minimum size
            roi_width = x2 - x1
            roi_height = y2 - y1
            
            if roi_width < self.config.gesture_roi_min_size or \
               roi_height < self.config.gesture_roi_min_size:
                return GesturePipelineOutput(
                    was_triggered=False,
                    trigger_reason="ROI too small"
                )
            
            # Send task to gesture thread with ROI
            task = ModelThreadTask(frame, frame_id, time.time(), roi=(x1, y1, x2, y2))
            self.gesture_thread.task_queue.put_nowait(task)
            
            # Try to get result
            try:
                result = self.gesture_thread.result_queue.get(timeout=0.5)
                
                if 'error' in result:
                    logger.warning(f"Gesture error: {result['error']}")
                    return GesturePipelineOutput(
                        was_triggered=True,
                        trigger_reason=f"Gesture error: {result['error']}"
                    )
                
                return GesturePipelineOutput(
                    gestures=result.get('gestures', []),
                    was_triggered=result.get('was_triggered', True),
                    trigger_reason=result.get('trigger_reason', 'Hand ROI valid'),
                    processing_time_ms=result.get('processing_time_ms', 0)
                )
            
            except Empty:
                logger.debug("Gesture result not ready yet")
                return GesturePipelineOutput(
                    was_triggered=True,
                    trigger_reason="Processing..."
                )
        
        except Exception as e:
            logger.error(f"Gesture orchestration error: {e}")
            return GesturePipelineOutput(
                was_triggered=False,
                trigger_reason=f"Error: {str(e)}"
            )
    
    def _orchestrate_safety(
        self,
        frame: np.ndarray,
        frame_id: int,
        detection_output: DetectionPipelineOutput,
        gesture_output: GesturePipelineOutput
    ) -> SafetyPipelineOutput:
        """
        Orchestrate safety model execution and alert generation.
        
        CONFLICT RESOLUTION:
        - If a tool is detected in a danger zone without proper PPE → CRITICAL alert
        - If a gesture is detected while unsafe tool is nearby → Block the action
        """
        
        self.safety_frame_counter += 1
        
        alerts = []
        overall_safety_level = SafetyLevel.SAFE
        ppe_violations = []
        danger_zone_entries = []
        
        # Run safety checks every N frames
        if self.safety_frame_counter % self.config.safety_frame_skip == 0:
            
            # Check for danger zone violations
            for detection in detection_output.detections:
                for danger_zone in self.config.danger_zones:
                    if danger_zone.contains_bbox(detection.bbox):
                        
                        # Check for required PPE
                        if detection.class_name == 'person':
                            # Here you would check if person has required PPE
                            # For now, simple heuristic
                            alert = SafetyAlert(
                                alert_type='DANGER_ZONE_ENTRY',
                                severity=SafetyLevel.WARNING,
                                message=f"Person entered {danger_zone.name}",
                                affected_regions=[detection.bbox],
                                recommended_action="Ensure proper PPE"
                            )
                            alerts.append(alert)
                            danger_zone_entries.append(danger_zone.name)
                            
                            if danger_zone.alert_level == 'critical':
                                overall_safety_level = SafetyLevel.CRITICAL
            
            # CONFLICT RESOLUTION: Check if gesture should be blocked
            if gesture_output.gestures and detection_output.tools_detected:
                # If person is using a tool, block certain gestures
                for gesture in gesture_output.gestures:
                    alert = SafetyAlert(
                        alert_type='UNSAFE_GESTURE',
                        severity=SafetyLevel.WARNING,
                        message=f"Gesture '{gesture.gesture_name}' detected while tools active",
                        affected_regions=[gesture.bbox],
                        recommended_action="Complete current task before using gestures"
                    )
                    alerts.append(alert)
        
        return SafetyPipelineOutput(
            alerts=alerts,
            overall_safety_level=overall_safety_level,
            ppe_violations=ppe_violations,
            danger_zone_entries=danger_zone_entries,
            processing_time_ms=0.0
        )
    
    def _make_orchestration_decisions(self, state: SystemState) -> Dict[str, Any]:
        """
        Make final orchestration-level decisions based on all pipeline outputs.
        
        Returns dict with decisions that can trigger actions.
        """
        
        decisions = {
            'should_trigger_gesture_action': False,
            'gesture_action_blocked_reason': None,
            'emergency_stop': False,
            'alert_user': False,
            'recommended_action': None
        }
        
        # Decision 1: Can we execute a detected gesture?
        if state.gesture_output.gestures and not state.safety_output.has_critical_alerts():
            decisions['should_trigger_gesture_action'] = True
        elif state.gesture_output.gestures and state.safety_output.has_critical_alerts():
            decisions['gesture_action_blocked_reason'] = 'Critical safety alert'
        
        # Decision 2: Emergency stop?
        if state.safety_output.has_critical_alerts():
            decisions['emergency_stop'] = True
            decisions['alert_user'] = True
        
        # Decision 3: Recommended action
        if state.safety_output.alerts:
            decisions['recommended_action'] = state.safety_output.alerts[0].recommended_action
        
        return decisions
    
    def shutdown(self) -> None:
        """Shutdown orchestrator and threads gracefully"""
        logger.info("Shutting down InferenceOrchestrator")
        
        self.state = OrchestratorState.SHUTDOWN
        
        if self.detection_thread:
            self.detection_thread.stop()
        
        if self.gesture_thread:
            self.gesture_thread.stop()
        
        logger.info("InferenceOrchestrator shutdown complete")
