# AI Moderator - YOLOv11 Enhanced Face Detection with Speech Analysis

An advanced AI-powered meeting moderator system featuring YOLOv11 face detection, person tracking, and real-time speech analysis through mouth movement detection.

## 🌟 Features

### 🎯 Advanced Face Detection
- **YOLOv11 Integration**: Latest YOLO model for superior face detection accuracy
- **Multi-Method Support**: Switch between YOLOv11, OpenCV DNN, and Haar Cascade
- **Distant Face Recognition**: Better detection of faces at various distances
- **Real-time Performance**: Optimized for live camera feeds

### 🎤 Speech Detection & Analysis
- **Mouth Movement Analysis**: Detects speaking through facial landmark tracking
- **Speaking Intensity**: Measures how actively someone is speaking
- **Temporal Smoothing**: Reduces false positives with intelligent filtering
- **Visual Indicators**: Real-time visual feedback for speaking status

### 👥 Person Identity Management  
- **Persistent Tracking**: Remembers individuals across frame gaps
- **Face Recognition**: Uses facial features for person identification
- **Entry/Exit Detection**: Tracks when people join or leave
- **Appearance Statistics**: Counts total appearances per person

### 📊 Analytics & Visualization
- **Real-time Statistics**: Live metrics display
- **Session Analytics**: Track meeting participation over time
- **Export Capabilities**: JSON export of all tracking data
- **Visual Overlays**: Rich visual feedback with bounding boxes and labels

## 🚀 Quick Start

### Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd ai-moderator
   ```

2. **Install dependencies**:
   ```bash
   pip install ultralytics opencv-python numpy torch torchvision
   ```

3. **Run the GUI application**:
   ```bash
   python main_yolov11_gui.py
   ```

4. **Or run the command-line demo**:
   ```bash
   python yolov11_demo.py --method yolo --camera 0
   ```

### First Run

On first launch, YOLOv11 models will be automatically downloaded:
- Face detection model (~6MB)  
- Pose detection model for facial landmarks (~6MB)

## 📁 File Structure

```
ai-moderator/
├── yolov11_face_detector.py      # YOLOv11 face detection implementation
├── enhanced_face_tracker.py      # Multi-method face tracker with person ID
├── main_yolov11_gui.py           # Full-featured GUI application
├── yolov11_demo.py               # Command-line camera demo
├── test_yolov11.py               # Test suite and feature demo
├── integrated_tracker.py         # Integration with existing activity tracking
└── main_enhanced.py              # Enhanced main application
```

## 🎮 Usage

### GUI Application

```bash
python main_yolov11_gui.py
```

**Features**:
- Detection method switching buttons
- Real-time statistics display
- People tracking list with speech status
- Session data export
- Camera controls

### Command-Line Demo

```bash
# Basic usage with YOLOv11
python yolov11_demo.py --method yolo

# Use different camera
python yolov11_demo.py --camera 1 --method yolo

# Custom resolution
python yolov11_demo.py --width 1280 --height 720 --method yolo

# Run without display window
python yolov11_demo.py --no-gui --method yolo
```

### Test Suite

```bash
python test_yolov11.py
```

Demonstrates all features without camera dependency.

## ⌨️ Keyboard Controls

When using camera applications:

- **1** - Switch to Haar Cascade detection
- **2** - Switch to OpenCV DNN detection  
- **3** - Switch to YOLOv11 detection
- **R** - Reset all tracking data
- **S** - Save screenshot (demo mode)
- **Q** - Quit application

## 🔧 Configuration

### Detection Methods

```python
from enhanced_face_tracker import EnhancedFaceTracker

# Initialize with specific method
tracker = EnhancedFaceTracker(detection_method='yolo')

# Check available methods
methods = tracker.get_available_methods()

# Switch methods at runtime
tracker.set_detection_method('yolo')
```

### YOLOv11 Parameters

```python
from yolov11_face_detector import YOLOv11FaceDetector

detector = YOLOv11FaceDetector()

# Adjust detection thresholds
detector.confidence_threshold = 0.3  # Lower for distant faces
detector.speech_threshold = 0.02     # Speech detection sensitivity
detector.smoothing_factor = 0.7      # Temporal smoothing
```

## 📊 Speech Detection Algorithm

The speech detection system uses multiple indicators:

1. **Mouth Aspect Ratio (MAR)**: Ratio of mouth height to width
2. **Temporal Changes**: Rapid mouth movement detection  
3. **Pixel Analysis**: Texture changes in mouth region
4. **Landmark Tracking**: Precise mouth corner detection

### Speech Detection Pipeline

```
Frame Input → YOLOv11 Face Detection → Facial Landmarks → 
Mouth Region Analysis → MAR Calculation → Temporal Smoothing → 
Speaking Detection → Intensity Measurement → Visual Output
```

## 🎯 Performance Comparison

| Method | Speed | Accuracy | Distance | Speech | Best Use Case |
|--------|-------|----------|----------|---------|---------------|
| **YOLOv11** | 80ms | Excellent | Excellent | ✅ Yes | Production, distant faces |
| **OpenCV DNN** | 60ms | Good | Good | ❌ No | Balanced performance |
| **Haar Cascade** | 20ms | Fair | Fair | ❌ No | Low-resource systems |

## 🧠 Person Tracking

### Identity Management
- Faces are encoded using histogram-based features
- Cosine similarity matching for person recognition
- Temporal tracking handles brief occlusions
- Configurable similarity thresholds

### Tracking Parameters
```python
tracker = EnhancedFaceTracker(
    max_people=8,                    # Maximum people to track
    recognition_threshold=0.5,       # Face similarity threshold
    absent_timeout=2.0,              # Seconds before marking absent
    distance_threshold=60            # Pixel distance for tracking
)
```

## 📈 Analytics & Export

### Real-time Metrics
- People present count
- Currently speaking count  
- Total people tracked
- Session duration
- Detection FPS

### Data Export
```python
# Export session data
app.export_data()  # Creates timestamped JSON file
```

Export includes:
- Session metadata
- Person appearance statistics
- Speech activity data (YOLOv11)
- Detection method performance

## 🔍 Advanced Features

### Multi-Method Architecture
- Hot-swappable detection methods
- Automatic fallback selection
- Performance profiling
- Method-specific optimizations

### Speech Analysis (YOLOv11 Only)
- Mouth landmark detection
- Movement intensity calculation
- Speaking streak tracking
- False positive reduction

### Visual Feedback
- Color-coded bounding boxes
- Speaking indicators (yellow borders)
- Confidence scores
- Real-time statistics overlay

## 🛠️ Development

### Testing
```bash
# Run comprehensive test suite
python test_yolov11.py

# Test specific components
python -c "from enhanced_face_tracker import EnhancedFaceTracker; t=EnhancedFaceTracker('yolo'); print('Available:', t.get_available_methods())"
```

### Extending Detection Methods
```python
class CustomDetector:
    def detect_faces_with_speech(self, frame):
        # Implement detection logic
        return detections
    
    def is_available(self):
        return True

# Integrate with EnhancedFaceTracker
tracker.custom_detector = CustomDetector()
```

## 📋 Requirements

### System Requirements
- Python 3.8+
- OpenCV 4.0+
- 4GB+ RAM
- Webcam or video input

### Python Dependencies
```
ultralytics>=8.0.0
opencv-python>=4.5.0
numpy>=1.20.0
torch>=1.9.0
torchvision>=0.10.0
tkinter (usually included with Python)
```

### Optional GPU Support
For improved performance, install CUDA-compatible PyTorch:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

## 🐛 Troubleshooting

### Common Issues

**Camera not opening**:
```bash
# Try different camera indices
python yolov11_demo.py --camera 1
```

**YOLOv11 models not downloading**:
```bash
# Manual download
python -c "from ultralytics import YOLO; YOLO('yolo11n.pt')"
```

**Poor face detection**:
- Ensure adequate lighting
- Try different detection methods
- Adjust confidence thresholds
- Check camera resolution

**Speech detection not working**:
- Only available with YOLOv11 method
- Requires clear view of mouth area
- Adjust speech sensitivity threshold

### Debug Mode
```bash
# Enable verbose output
YOLO_VERBOSE=1 python yolov11_demo.py --method yolo
```

## 🔄 Version History

### v2.0 - YOLOv11 Integration
- Added YOLOv11 face detection
- Implemented speech analysis
- Enhanced person tracking
- Added GUI application
- Performance optimizations

### v1.0 - Basic Implementation  
- OpenCV DNN and Haar Cascade
- Basic person tracking
- Command-line interface

## 📄 License

MIT License - see LICENSE file for details.

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -am 'Add amazing feature'`)  
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

## 📞 Support

- Issues: GitHub Issues page
- Documentation: This README
- Examples: `test_yolov11.py` and demo files

---

**Made with ❤️ for better meeting experiences**