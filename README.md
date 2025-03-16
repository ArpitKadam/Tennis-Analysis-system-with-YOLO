# 🎾 Tennis Analysis System with YOLO

## 📌 Overview
This project utilizes **YOLO (You Only Look Once)** for **tennis match analysis**, including player tracking, ball tracking, and court line detection. It processes input videos to generate insights into player movements and ball trajectory.

![GIF Preview](https://github.com/ArpitKadam/Tennis-Analysis-system-with-YOLO/blob/main/output_video/output_video.gif)

Watch video tutorial at [🎥 Watch Video Tutorial](https://youtu.be/L23oIHZE14w?si=r6EaArqAdW91d2KR)

The coder in the YouTube video explains the implementation in a clear and structured manner, making complex concepts easy to understand. Their step-by-step approach ensures that viewers can follow along and replicate the project effortlessly.

**Credits:** Special thanks to [Code in a Jiffy](https://www.youtube.com/@codeinajiffy) for the amazing tutorial!

---

## 📂 Directory Structure
```
└── arpitkadam-tennis-analysis-system-with-yolo/
    ├── README.md                         # Project documentation
    ├── main.py                           # Main script to run analysis
    ├── yolo_inference.py                 # YOLO inference script
    ├── yolo_inference_with_trained_model.py
    ├── yolov8x.pt                         # YOLO model weights
    ├── player_stats_data.csv             # Player statistics data
    ├── Boxes_of_images.txt
    ├── Boxes_of_video.txt
    ├── LICENSE
    ├── analysis/
    │   └── ball_analysis.ipynb           # Ball movement analysis
    ├── constants/
    ├── court_line_detector/
    │   ├── court_line_detector.py        # Court line detection module
    ├── input_videos/                     # Store input videos
    ├── minicourt/
    ├── output_video/
    │   └── output_video.avi              # Processed output video
    ├── runs/
    │   └── detect/
    ├── tracker/
    │   ├── ball_tracker.py               # Ball tracking script
    │   ├── player_tracker.py             # Player tracking script
    ├── tracker_stubs/
    ├── training/
    │   ├── tennis_ball_detector_training.ipynb
    │   ├── tennis_court_keypoint_training.ipynb
    ├── utils/
    │   ├── bbox_utils.py                 # Bounding box utilities
    │   ├── video_utils.py                # Video processing utilities
```

---

## 🚀 Features
✅ **Player Tracking:** Detects and tracks player movements using YOLO.
✅ **Ball Tracking:** Identifies ball position and trajectory.
✅ **Court Line Detection:** Recognizes court boundaries for better analysis.
✅ **Real-time Processing:** Can be optimized for real-time tracking.
✅ **Deep Learning Models:** YOLOv8 models for improved accuracy.

---

## ⚙️ Installation & Setup
### 1️⃣ Clone the Repository
```bash
git clone https://github.com/ArpitKadam/Tennis-Analysis-system-with-YOLO.git
cd Tennis-Analysis-system-with-YOLO
```

### 2️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

### 3️⃣ Run the Model
```bash
python main.py --input input_videos/sample.mp4 --output output_video/output_video.avi
```

---

## 📊 Usage
- Place input videos in the `input_videos/` folder.
- Run `main.py` to process and analyze the game.
- Output videos will be saved in `output_video/`.

---

## 🤖 Model Training
To train your own model, refer to the notebooks in the `training/` directory:
```bash
jupyter notebook training/tennis_ball_detector_training.ipynb
```

---

## 📜 License
This project is licensed under the **MIT License**. Feel free to use and modify it as needed.

---

## 📬 Contact
📧 Email: [arpitkadam922@gmail.com](mailto:arpitkadam922@gmail.com)  
🔗 LinkedIn: [Arpit Kadam](https://www.linkedin.com/in/arpitkadam)  
🐙 GitHub: [@ArpitKadam](https://github.com/arpitkadam)  
🎥 Video Tutorial: [🎥 Link](https://youtu.be/L23oIHZE14w?si=r6EaArqAdW91d2KR)

---

### 🎾 Let's analyze tennis like never before! 🚀
