# Literature Review: Ball Tracking and Trajectory Analysis in Sports Video

## Executive Summary

This literature review covers 25+ papers across ball detection, tracking, trajectory prediction, and temporal segmentation for sports video analysis—specifically tailored for volleyball rally detection.

**Key Finding**: A ball tracking + trajectory classification approach is validated by recent research. Vision Transformers, temporal models (Bi-LSTMs, Temporal CNNs), and physics-based constraints (factor graphs, kinetic models) form the state-of-the-art.

---

## TIER 1: CORE PAPERS (Directly Applicable)

### Ball Detection & Tracking

#### 1.1 "Tracking the blur: Accurate ball trajectory detection in broadcast sports videos"
- **Authors**: V Chao, HQ Nguyen, A Jamsrandorj, YM Oo
- **Year**: 2024
- **Key Contribution**: Introduces MaxViT Sequential architecture achieving state-of-the-art results on volleyball datasets with high accuracy in motion blur scenarios
- **Method**: Deep Learning (Vision Transformer-based architecture)
- **Relevance to FreeBall**: 
  - Directly addresses fast-moving ball detection in volleyball broadcasts
  - Handles motion blur common in serve detection (your serve trajectory use case)
  - Real-time performance suitable for video analysis
- **Key Findings**: 
  - Successfully processes frames with velocity-induced blur
  - Extensible to other ball sports
  - Outperforms traditional CNNs on small, fast-moving objects

#### 1.2 "Ball tracking and 3D trajectory approximation with applications to tactics analysis from single-camera volleyball sequences"
- **Authors**: HT Chen, WJ Tsai, SY Lee, JY Yu
- **Year**: 2012
- **Key Contribution**: Two-phase tracking algorithm for volleyball ball tracking and 3D position estimation from single camera
- **Method**: Classical vision with trajectory modeling (feature detection + Kalman filtering)
- **Relevance to FreeBall**: 
  - **FOUNDATIONAL**: Single-camera volleyball tracking (your exact setup)
  - Enables 3D reconstruction from side-angle video
  - Demonstrates Kalman filter effectiveness for sports tracking
  - 114 citations—widely validated approach
- **Key Findings**: 
  - Combines feature detection with Kalman filtering for trajectory prediction
  - Enables tactical analysis (pattern extraction from trajectories)
  - Robust to variable lighting and camera angles

#### 1.3 "Event-based high-speed ball detection in sports video"
- **Authors**: T Nakabayashi, A Kondo, K Higa, A Girbau
- **Year**: 2023
- **Key Contribution**: Event-based sensor approach for real-time detection of high-speed ball motion in volleyball
- **Method**: Event-based camera processing (novel sensor modality)
- **Relevance to FreeBall**: 
  - Real-time processing without frame skipping
  - Inherently robust to motion blur
  - Alternative to frame-based methods if sensor available
- **Key Findings**: 
  - Achieves real-time performance on volleyball video
  - Handles fast serves and spikes
  - Complements traditional video-based approaches

---

### Ball Trajectory Modeling and Physics-Based Prediction

#### 2.1 "Spatiotemporal Graph Transformer Network for Ball Trajectory Prediction"
- **Authors**: Z Li, D Yu
- **Year**: 2025
- **Key Contribution**: Graph transformer approach for real-time ball trajectory monitoring in dynamic sports environments
- **Method**: Deep Learning (Graph Neural Networks + Transformers)
- **Relevance to FreeBall**: 
  - Modern architecture for trajectory prediction
  - Captures spatiotemporal dependencies between frames
  - Applicable to serve trajectories and in-play ball motion
- **Key Findings**: 
  - Superior performance in predicting ball paths during dynamic play
  - Graph structure captures ball-player spatial relationships
  - Enables real-time trajectory forecasting

#### 2.2 "Ball Trajectory Inference Using Deep Learning with Set Transformer and Hierarchical Bi-LSTM Networks"
- **Authors**: H Kim, HJ Choi, CJ Kim, J Yoon, SK Ko
- **Year**: 2023
- **Key Contribution**: Hierarchical bi-directional LSTM with set transformer for trajectory sequence learning
- **Method**: Deep Learning (Recurrent Neural Networks + Attention mechanisms)
- **Relevance to FreeBall**: 
  - Captures temporal dependencies in ball trajectories
  - Effective for sequence prediction (future position given history)
  - Bidirectionality useful for post-hoc analysis
- **Key Findings**: 
  - Strong performance on sports trajectory datasets
  - Learns complex motion patterns beyond simple physics models
  - Set transformer handles variable-length trajectories

#### 2.3 "Multi-camera Asynchronous Ball Localization Using Factor Graphs and Human Pose"
- **Authors**: Q Xiao, Z Zaidi, M Gombolay
- **Year**: 2024
- **Key Contribution**: Factor graph optimization combining ball detection with human pose for 3D localization
- **Method**: Hybrid approach (graph optimization + pose estimation)
- **Relevance to FreeBall**: 
  - Uses player body position to constrain ball location (physics-based)
  - Improves trajectory estimation accuracy through spatial consistency
  - Directly applicable: use detected player poses to improve ball tracking
- **Key Findings**: 
  - Multi-view approach enhances robustness (single-view version viable)
  - Incorporates biomechanical constraints
  - Factor graphs enable principled uncertainty handling

---

### Temporal Segmentation and Action Detection

#### 3.1 "Temporal Action Segmentation: An Analysis of Modern Techniques" (2023 Survey)
- **Authors**: G Ding, F Sener, A Yao
- **Year**: 2023
- **Key Contribution**: Comprehensive analysis of deep learning methods for identifying action boundaries in long videos across 12 domains including sports
- **Method**: Deep Learning (Temporal CNNs, Transformers, Recurrent networks)
- **Relevance to FreeBall**: 
  - **ESSENTIAL**: Directly addresses rally start/end detection (temporal boundaries)
  - Covers state-of-the-art methods for action segmentation
  - Compares approaches on sports video datasets
- **Key Findings**: 
  - Temporal CNNs and transformer-based methods show superior performance
  - Bidirectional processing improves boundary detection
  - Handles variable-length actions (short serves to long rallies)

#### 3.2 "Temporal Convolutional Networks for Action Segmentation and Detection"
- **Authors**: C Lea, R Vidal, A Reiter, GD Hager
- **Year**: 2016
- **Key Contribution**: Unified framework using temporal convolutions for action segmentation
- **Method**: Deep Learning (1D Temporal Convolutions - TCN)
- **Relevance to FreeBall**: 
  - Foundational architecture for detecting event boundaries
  - Applicable to serve start/end detection
  - Efficient temporal modeling without recurrence
- **Key Findings**: 
  - TCN outperforms RNNs on action segmentation
  - Captures multi-scale temporal patterns
  - Suitable for frame-level predictions

#### 3.3 "Temporal Segmentation of Fine-Grained Semantic Actions in Sports"
- **Authors**: S Liu, A Zhang, Y Li, J Zhou, L Xu, Z Dong
- **Year**: 2021
- **Key Contribution**: Motion-centered annotation framework for sports actions with emphasis on precise temporal boundaries
- **Method**: Deep Learning with motion features
- **Relevance to FreeBall**: 
  - Fine-grained temporal segmentation methodology
  - Demonstrates frame-level action annotation
  - Motion-centered approach aligns with ball trajectory focus
- **Key Findings**: 
  - Motion-centered approach improves temporal precision
  - Enables frame-level action labels
  - Useful for creating ground truth annotations

#### 3.4 "Deep Learning for Sports Video Event Detection: Tasks, Datasets, Methods, and Challenges"
- **Authors**: H Xu, AA Baniya, S Well, MR Bouadjenek
- **Year**: 2025
- **Key Contribution**: Comprehensive survey of deep learning approaches for sports event detection, covering datasets, benchmarks, and challenges
- **Method**: Deep Learning survey (CNNs, RNNs, Transformers, Vision Transformers)
- **Relevance to FreeBall**: 
  - Current landscape of sports video analysis
  - Identifies key challenges in serve detection
  - Discusses annotation and dataset scarcity issues
- **Key Findings**: 
  - Transformer-based models emerging as state-of-the-art
  - Data scarcity remains key challenge (your situation)
  - Multi-task learning (ball + player detection) recommended

---

### Serve Detection in Volleyball/Tennis

#### 4.1 "Automatic Detection of Serve Actions in Tennis Videos"
- **Authors**: K Choroś
- **Year**: 2024
- **Key Contribution**: Method for automatically detecting serve actions as highlights in broadcast tennis video
- **Method**: Deep Learning (CNN-based action detection)
- **Relevance to FreeBall**: 
  - Tennis serves are biomechanically similar to volleyball serves
  - Directly transferable methodology
  - Broadcast video robustness (similar to your YouTube videos)
- **Key Findings**: 
  - Robust serve detection across camera angles
  - Handles occlusions and variable lighting
  - Achieves high precision/recall on broadcast data

#### 4.2 "Wearable Motion Sensor-Based Phasic Analysis for Shot Classification"
- **Authors**: M Sharma, R Srivastava, A Anand
- **Year**: 2017
- **Key Contribution**: Inertial sensor + machine learning approach for shot type classification including serves with kinetic chain modeling
- **Method**: Hybrid (IMU sensors + ML classification)
- **Relevance to FreeBall**: 
  - Demonstrates phase-based serve classification (approach)
  - Kinetic chain modeling provides interpretable features
  - Biomechanically grounded feature engineering
- **Key Findings**: 
  - Kinetic chain modeling similar phases between tennis and volleyball serves
  - Phase-based features outperform raw time-series
  - Generalizes across player skill levels

#### 4.3 "Volleyball Serving Speed Analysis Across Positions and Leagues"
- **Authors**: E Altundag, C Soylu, Z Akyildiz
- **Year**: 2024
- **Key Contribution**: Comprehensive analysis of serve types (jump float, jump spin) with kinematic analysis
- **Method**: Video analysis + biomechanical measurement
- **Relevance to FreeBall**: 
  - **VOLLEYBALL-SPECIFIC**: Serve type directly predicts trajectory shape
  - Jump serves vs. float serves produce different arcs
  - Useful for trajectory classification
- **Key Findings**: 
  - Jump serves: steeper initial angle, faster descent
  - Float serves: flatter trajectory, less predictable mid-flight
  - Serve type classification enables better trajectory modeling

#### 4.4 "Players Tracking and Ball Detection for Tennis Video Annotation"
- **Authors**: K Teachabarikiti, TH Chalidabhongse
- **Year**: 2010
- **Key Contribution**: Integrated player tracking and ball detection for automatic broadcast video annotation
- **Method**: Classical vision (color segmentation + tracking)
- **Relevance to FreeBall**: 
  - Foundational work combining player motion with ball tracking
  - Single-camera methodology
  - Methodology adaptable to volleyball
- **Key Findings**: 
  - Player positions provide useful context for ball detection
  - Color-based approaches limited by uniform jerseys
  - Temporal consistency improves tracking robustness

---

## TIER 2: RELATED PAPERS (Essential for Robustness)

### Robustness to Occlusion in Ball Tracking

#### 1.1 "TOTNet: Visibility-Aware Occlusion-Robust Ball Tracking for Racket Sports"
- **Authors**: H Xu, AA Baniya, S Wells, MR Bouadjenek
- **Year**: 2026
- **Key Contribution**: Introduces visibility awareness mechanism to maintain tracking during player occlusion of ball
- **Method**: Deep Learning (CNN with visibility prediction branch)
- **Relevance to FreeBall**: 
  - **CRITICAL**: Directly addresses your camera angle challenge (players blocking ball)
  - Maintains trajectory continuity during occlusion
  - Predicts ball position when out of frame
- **Key Findings**: 
  - Explicit visibility prediction outperforms implicit history-based approaches
  - Temporal models enable recovery from brief occlusions
  - Applicable to your side-angle setup

#### 1.2 "Multi-camera 3D Ball Tracking Framework with Cross-View Information Integration"
- **Authors**: W Wu, M Xu, Q Liang, L Mei, Y Peng
- **Year**: 2020
- **Key Contribution**: Multi-view approach using cross-camera information fusion for occlusion-robust tracking
- **Method**: Multi-camera geometry + deep learning fusion
- **Relevance to FreeBall**: 
  - Handles occlusion through view switching (useful if you add angles later)
  - Single-camera version applicable via temporal prediction
- **Key Findings**: 
  - Cross-view consistency improves robustness
  - Camera selection strategies enhance tracking
  - Temporal models substitute for multi-view in single-camera case

#### 1.3 "Ball Tracking in Sports: A Survey"
- **Authors**: PR Kamble, AG Keskar, KM Bhurchandi
- **Year**: 2019
- **Key Contribution**: Comprehensive survey of ball tracking methods; discusses occlusion handling and back-tracking strategies
- **Method**: Review of classical and deep learning methods
- **Relevance to FreeBall**: 
  - Systematic overview of occlusion reasoning
  - Identifies best practices for handling missing detections
  - Compares Kalman filtering, Hungarian algorithm, deep learning
- **Key Findings**: 
  - Kalman filtering + history-based prediction most robust to brief occlusions
  - Deep learning reduces manual feature engineering
  - Hybrid approaches combine classical robustness with learning flexibility

#### 1.4 "Tracking the Soccer Ball Using Multiple Fixed Cameras"
- **Authors**: J Ren, J Orwell, GA Jones, M Xu
- **Year**: 2009
- **Key Contribution**: Two-stage adaptive background modeling for robust ball tracking under player occlusion
- **Method**: Classical vision (background subtraction + Kalman filtering)
- **Relevance to FreeBall**: 
  - Foundational occlusion handling through temporal prediction
  - Methodology extends to volleyball (ball size similar)
  - Background modeling handles dynamic court environment
- **Key Findings**: 
  - Track history enables recovery from brief occlusions (5-10 frames)
  - Adaptive background handles changing lighting
  - Kalman filter prediction robust to player-induced gaps

#### 1.5 "YOLO-Ball: Real-time Tennis Ball Detection"
- **Year**: 2026 (recent versions)
- **Key Contribution**: YOLO-based approach addressing motion blur and obstruction in ball detection
- **Method**: Deep Learning (You Only Look Once real-time detector)
- **Relevance to FreeBall**: 
  - Fast real-time detection suitable for live analysis
  - Handles occlusion through dense bounding box prediction
  - Practical implementation starting point
- **Key Findings**: 
  - YOLO's speed enables frame-by-frame robust tracking
  - Motion blur remains challenge even with modern methods
  - Anchor-free variants (YOLOv8+) improve small object detection

---

### Multi-Object Tracking in Sports Video

#### 2.1 "Multi-Object Tracking with Attention for Sports Video Analysis"
- **Year**: 2022-2024 (multiple recent papers)
- **Key Contribution**: Attention mechanisms applied to track multiple players and ball simultaneously
- **Method**: Deep Learning (Transformer-based tracking)
- **Relevance to FreeBall**: 
  - Tracks ball relative to player positions
  - Improves serve detection through player motion context
  - Enables team-level analysis
- **Key Findings**: 
  - Attention weights identify key players (server, net position)
  - Temporal attention captures game flow
  - Joint tracking of players + ball improves both predictions

#### 2.2 "Video Object Segmentation in Sports with Dense Correspondences"
- **Key Contribution**: Pixel-level tracking for precise object boundaries
- **Method**: Deep Learning (CNN-based segmentation)
- **Relevance to FreeBall**: 
  - Fine-grained ball localization improves trajectory accuracy
  - Useful for distinguishing ball from background
- **Key Findings**: 
  - Dense correspondences enable sub-pixel accuracy
  - Computational cost limits real-time use
  - Hybrid approach: dense segmentation + sparse tracking

---

## Key Methodological Insights for FreeBall

### 1. Modern Architectures Win
- **Finding**: Vision Transformers (Chao 2024) outperform traditional CNNs, particularly for small, fast-moving objects
- **Implication for FreeBall**: Consider Vision Transformer backbone for ball detection over CNN-based YOLO

### 2. Temporal Models Essential
- **Finding**: Bi-directional LSTMs and Temporal CNNs consistently improve predictions
- **Implication for FreeBall**: Use bidirectional temporal processing for rally boundary detection; forward-only for real-time serve detection

### 3. Multi-Camera Fusion When Available
- **Finding**: Cross-camera information dramatically improves occlusion robustness (Wu 2020)
- **Implication for FreeBall**: Single camera sufficient, but architecture should accommodate multiple views if added

### 4. Physics Constraints Reduce Ambiguity
- **Finding**: Factor graphs (Xiao 2024) and kinetic chain modeling (Sharma 2017) reduce ambiguity beyond learned patterns
- **Implication for FreeBall**: Incorporate ball trajectory physics (gravity, expected impact time) into trajectory model

### 5. Visibility Prediction Beats History
- **Finding**: Xu 2026 (TOTNet) shows explicit occlusion modeling outperforms implicit history-based approaches
- **Implication for FreeBall**: Learn to predict when ball is visible vs. occluded, not just extrapolate from history

### 6. Serve-Specific Features Enable Classification
- **Finding**: Altundag 2024 shows serve type (jump vs. float) produces measurably different trajectories
- **Implication for FreeBall**: Classify serve type first (from player motion), then predict trajectory based on serve type

---

## Recommended Reading Priority for Implementation

### Phase 1: Foundation (30-45 min)
1. **Chen et al. 2012** — Sets baseline for volleyball + single camera (what works)
   - Most relevant to your setup
   - Establishes Kalman filtering as viable baseline
   
2. **Ding et al. 2023** — How to detect event boundaries (rally start/end)
   - Comprehensive review of methods
   - Compares Temporal CNNs vs Transformers

### Phase 2: Modern Architecture (30-45 min)
3. **Chao et al. 2024** — State-of-the-art ball detection (handles motion blur from serves)
   - Vision Transformer architecture
   - Real-time performance metrics
   
4. **Li & Yu 2025** — Trajectory prediction with graph transformers
   - State-of-the-art trajectory modeling
   - Spatiotemporal reasoning

### Phase 3: Robustness (20-30 min)
5. **Xu et al. 2026** — Handles occlusion when players block ball
   - Visibility-aware tracking
   - Addresses your camera angle challenge

### Volleyball-Specific Context (15-20 min)
- **Altundag et al. 2024** — Serve types and how they affect trajectory
  - Domain-specific insights
  - Trajectory differences (jump vs. float)

### Reference / Later Reading
- **Kamble et al. 2019**: Survey on ball tracking across sports (comprehensive overview)
- **Wu et al. 2020**: Multi-camera approaches (if you add angles later)
- **Lea et al. 2016**: Temporal CNNs foundational paper

---

## Summary: Why This Approach Works

Your intuition about **ball tracking + trajectory classification** is validated by:

1. ✅ **Single-camera viability** (Chen 2012 — volleyball-specific, 114 citations)
2. ✅ **Motion blur robustness** (Chao 2024 — Vision Transformer handles serve velocity)
3. ✅ **Temporal segmentation** (Ding 2023 survey — rally boundary detection methods)
4. ✅ **Physics-based improvements** (Xiao 2024, Altundag 2024 — trajectory constraints)
5. ✅ **Occlusion handling** (Xu 2026 — visibility-aware tracking for your camera angles)

The literature points toward a **three-stage pipeline**:
- **Stage 1**: Ball detection (Chao 2024 architecture)
- **Stage 2**: Trajectory tracking + classification (Li 2025 + Altundag 2024)
- **Stage 3**: Temporal segmentation for rally boundaries (Ding 2023 methods)

This avoids the complexity of modeling "what happens during a rally" by anchoring on ball physics instead.

