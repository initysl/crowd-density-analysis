# Crowd Density & Flow Analysis System

A real-time computer vision system that monitors crowd density, detects bottlenecks, and identifies reverse flow patterns to prevent crowd disasters at events, stadiums, and public gatherings.

![Python](https://img.shields.io/badge/Python-3.11-blue)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-green)
![OpenCV](https://img.shields.io/badge/OpenCV-4.8-red)
![License](https://img.shields.io/badge/License-MIT-yellow)

<video src="docs/assets/demotwo.mp4" controls="controls" style="max-width: 100%;">
</video>

## Project Overview

This system analyzes surveillance footage to:

- **Detect & Track** individuals in crowded environments
- **Calculate density** per zone in people per square meter
- **Identify bottlenecks** where high density meets low movement
- **Flag reverse flow** as an early panic indicator
- **Generate real-time alerts** for critical conditions

Built with YOLOv8 for person detection, ByteTrack for multi-object tracking, and custom flow analysis algorithms.

---

> **Sample Output:** Aerial surveillance footage showing real-time density heatmap overlay with color-coded zones (green = safe, yellow = caution, red = critical), bottleneck detection (magenta borders), and reverse flow warnings (orange borders).

---

## Features

### Detection & Tracking

- **YOLOv8n** person detection with configurable confidence thresholds
- **ByteTrack** for stable ID assignment across frames
- **Inference Slicer** for improved small object detection at high camera angles
- Optimized for overhead/aerial surveillance angles

### Density Analysis

- Frame divided into 8×8 grid zones
- Density calculated as people per square meter
- Three risk levels: Safe (<2 p/m²), Caution (2-4 p/m²), Critical (>4 p/m²)
- Calibratable for different camera heights and angles

### Flow Analysis

- Movement vector tracking across 30-frame history
- **Bottleneck Detection:** High density + low movement speed
- **Reverse Flow Detection:** People moving against dominant direction (>20% threshold)
- Zone-based throughput calculation

### Alert System

- Real-time alert generation for:
  - Critical density zones
  - Sustained bottlenecks
  - Reverse flow patterns (panic indicator)
- Severity classification (low/medium/high)
- Exportable alert logs with timestamps

---

## Use Cases

### Event Safety

- Music festivals, concerts, sporting events
- Real-time crowd monitoring for event organizers
- Automated alerts to security teams

### Infrastructure Management

- Airport/train station queue management
- Stadium exit route optimization
- Subway platform congestion monitoring

### Research & Planning

- Post-event analysis for venue improvements
- Crowd simulation validation
- Emergency evacuation planning

## Acknowledgments

- **Ultralytics YOLOv8** - Object detection framework
- **Roboflow Supervision** - Computer vision utilities
- **ByteTrack** - Multi-object tracking algorithm
- Inspired by crowd safety research following the Astroworld tragedy

** Disclaimer:** This system is for research and development purposes. Deployment in safety-critical environments requires thorough testing, validation, and integration with human oversight protocols.

## License

This project is licensed under the MIT License.
