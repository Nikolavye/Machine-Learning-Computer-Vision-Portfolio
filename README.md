# Machine Learning & Computer Vision Portfolio

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-blue.svg" alt="Python">
  <img src="https://img.shields.io/badge/Ultralytics-YOLOv8-yellow.svg" alt="YOLO">
  <img src="https://img.shields.io/badge/Machine%20Learning-Scikit--Learn-orange.svg" alt="Scikit-Learn">
  <img src="https://img.shields.io/badge/Architecture-Data%20Flywheel-success.svg" alt="Architecture">
</p>

This repository contains end-to-end Machine Learning and Computer Vision projects. It demonstrates problem-solving under data constraints and scalable deployment architectures.

---

## 1. Sensor Anomaly Detection
**Focus**: Semi-Supervised Learning | Active Learning | System Architecture  
**Directory**: [`sensor_anomaly_detection/`](./sensor_anomaly_detection/)

### The Challenge
In specific industrial use cases, collecting sensor data is accessible, but labeling it requires domain experts. In this project, **2.5% of the data was labeled**. The goal was to build a classifier using this limited supervision.

### The Solution: Active Learning Data Flywheel
A multi-model consensus pipeline is used to scale labels for high-confidence data, routing low-confidence edge cases to domain experts for manual review.

```mermaid
graph TD
    classDef offline fill:#e3f2fd,stroke:#1565c0,stroke-width:2px;
    classDef online fill:#e8f5e9,stroke:#2e7d32,stroke-width:2px;
    classDef human fill:#fff3e0,stroke:#e65100,stroke-width:2px;

    subgraph Offline Batch Pipeline [1. Offline Consensus Pipeline]
        RawData[(Unlabeled<br>Sensor Data)]
        M1[K-Means]
        M2[Random Forest]
        M3[Label Propagation]
        M4[Label Spreading]
        Vote{100%<br>Consensus?}
        
        RawData --> M1
        RawData --> M2
        RawData --> M3
        RawData --> M4
        M1 & M2 & M3 & M4 --> Vote
    end

    subgraph Active Learning Loop [3. Active Learning Loop]
        Expert[Human Expert Review]
        LowConf[Low Confidence Score<br>Disputed Data]
    end

    subgraph Online Inference [2. Real-Time Online Inference]
        RF[Surrogate<br>Random Forest]
        Kafka((Kafka Stream))
        Output(Real-time Predictions)
    end

    Vote -- Yes --> ProdDB[(Production<br>Dataset)]
    Vote -- No --> LowConf
    LowConf -.-> Expert
    Expert -- Ground Truth --> ProdDB
    
    ProdDB ==>|Daily Retrain| RF
    Kafka --> RF
    RF --> Output
    
    class M1,M2,M3,M4,Vote,RawData offline;
    class RF,Kafka,Output online;
    class Expert,LowConf human;
```

### Key Components
1. **Automated Label Scaling:** Uses a 4-model consensus to generate pseudo-labels, progressively expanding the training set.
2. **Optimized Inference:** Graph-based models (Label Propagation) are kept offline. A Random Forest model is trained on the expanded dataset and deployed to handle Kafka streams.
3. **Active Learning:** Extracts low-confidence edge cases based on model agreement, routing them for manual expert review.

---

## 2. Sports Player Tracking & Analytics
**Focus**: Computer Vision | Multi-Object Tracking | Temporal Smoothing  
**Directory**: [`sports_player_tracking/`](./sports_player_tracking/)

### The Challenge
Tracking and classifying players and referees in sports broadcasts featuring moving cameras, motion blur, and occlusion.

### The Solution: YOLOv8 + BoT-SORT + Consensus Voting
A tracking pipeline designed to achieve stable team classification across consecutive frames.

```mermaid
flowchart LR
    classDef core fill:#ede7f6,stroke:#4527a0,stroke-width:2px;
    classDef filter fill:#ffebee,stroke:#c62828,stroke-width:2px;

    Frame[Raw Video Frame] --> YOLO[YOLOv8 Det & BoT-SORT]
    YOLO --> Filter[Spatial Heuristic Filter<br>is_on_pitch]
    
    Filter -- Discard --> Garbage[Audience & Ads]
    Filter -- Keep --> HSV[Color Topology Analysis]
    
    HSV --> Cache[(20-Frame Temporal Cache)]
    Cache --> Vote{Track-Level<br>Majority Voting}
    
    Vote --> Red[Red Team]
    Vote --> White[White Team]
    Vote --> GK[Goalkeepers<br>Yellow/Blue]

    class Frame,YOLO,Cache,Vote core;
    class Filter,Garbage filter;
```

### Key Components
1. **Filtering:** An `is_on_pitch` heuristic filter to omit bounding boxes corresponding to audience members and visual artifacts.
2. **Track-Level Voting Mechanism:** A temporal history buffer maintains classifications over a 20-frame window. Results are determined by track-level majority voting to reduce frame-to-frame flickering.
3. **Color-Based Classification:** An HSV and BGR clustering system to differentiate team colors and goalkeeper kits.
4. **Hardware Portability:** Code structured to support edge deployment, with portability to TensorRT or Quantized INT8/FP16 formats.

---

## 3. Deployment Strategy

The repository includes a deployment architecture utilizing Docker and Serverless technologies.

### A. Containerized Microservices
Each component can be independently containerized:
- **Offline Consensus Factory (`sensor_anomaly_detection/Dockerfile`)**: Packaged as a batch job. It can be orchestrated via **Apache Airflow** or **Kubernetes CronJobs** to pull unlabeled data, run the 4-model consensus voting, and update the dataset.
- **Online Surveillance Engine (`sports_player_tracking/Dockerfile`)**: An inference container with OpenCV and Ultralytics dependencies, designed to consume RTSP/Kafka video streams.

### B. Serverless Execution
- **AWS Lambda / GCP Cloud Functions:** The Random Forest model from the Sensor Anomaly project can be packaged with a FastAPI wrapper inside a Lambda function for scalable event-driven processing.
- **AWS SageMaker Asynchronous Inference:** The Sports Tracking YOLO pipeline can be deployed as an autoscaling SageMaker endpoint, provisioning GPU instances during active processing.
