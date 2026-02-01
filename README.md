# Industrial-Safety-Project


Overview
This project focuses on real-time object detection for industrial safety using deep learning and computer vision. The goal is to automatically identify safe and unsafe behaviors in industrial environments from video streams, providing timely alerts and supporting workplace safety initiatives.

Features
YOLOv8-based Object Detection:
Utilizes Ultralytics YOLOv8 for high-performance detection of safety-related actions and objects.

Custom Dataset Support:
Designed to work with both public datasets (e.g., from Mendeley Data) and custom-labeled frames extracted from industrial videos.

Training and Fine-Tuning:
Supports training from scratch or fine-tuning existing YOLO .pt weights on new safety datasets.

Real-Time Video Analytics:
Processes video feeds or image sequences for live safety monitoring and alerting.

Automation and Alerting:
YAML-based configuration for easy customization of detection classes and safety alert rules.

Workflow
Data Preparation

Download and extract video datasets (e.g., Mendeley Data: Safe and Unsafe Behaviours).

Extract frames from videos and annotate them for object detection.

Organize data into YOLO-compatible format.

Model Training

Train YOLOv8 models on prepared datasets.

Fine-tune using existing best.pt or last.pt weights for improved accuracy.

Evaluation and Inference

Validate model performance on test sets.

Run inference on new videos or images to detect safety violations in real time.

Deployment

Integrate trained models into industrial monitoring systems.

Configure safety alert rules using YAML files.
