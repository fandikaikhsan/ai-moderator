# AI Discussion Moderator

An intelligent discussion monitoring system that uses computer vision and AI to analyze group discussions, track participation patterns, and provide real-time insights into discussion dynamics.

## Features

### 🎯 Core Capabilities

- **Real-time Face Detection & Tracking**: Identifies and tracks multiple participants in discussions
- **Mouth Movement Analysis**: Detects speaking activity by analyzing facial landmarks and lip movements
- **Speaking Activity Tracking**: Records who's talking, for how long, and identifies dominant vs. quiet participants
- **Discussion Analytics**: Analyzes turn-taking patterns, interruptions, and engagement levels
- **Live GUI Interface**: Real-time visualization of participant activity and discussion metrics
- **Comprehensive Reporting**: Generates detailed reports with insights and recommendations

### 📊 Analytics & Insights

- **Participation Matrix**: Shows speaking time distribution and activity levels for each participant
- **Discussion Quality Score**: Evaluates overall discussion quality based on balance and dynamics
- **Turn-taking Analysis**: Identifies smooth transitions, interruptions, and overlapping speech
- **Engagement Patterns**: Tracks participation peaks, valleys, and overall engagement trends
- **Recommendations**: Provides actionable suggestions for improving discussion quality

### 💾 Data Management

- **Session Recording**: Automatically saves discussion sessions with detailed metrics
- **Multiple Export Formats**: JSON, CSV, and specialized analysis formats
- **SQLite Database**: Persistent storage for historical analysis and comparison
- **Timeline Visualization**: Visual representation of discussion flow and activity patterns

## Installation

### Prerequisites

- Python 3.8 or higher
- Camera/webcam for video input
- Sufficient processing power for real-time computer vision

### Setup Instructions

1. **Clone and Navigate to Project**

   ```bash
   cd /path/to/ai-moderator
   ```

2. **Create Virtual Environment** (Recommended)

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate  # On macOS/Linux
   # or
   .venv\Scripts\activate     # On Windows
   ```

3. **Install Dependencies**

   ```bash
   pip install -r requirements.txt
   ```

4. **Verify Camera Access**
   ```bash
   python main.py --camera 0 --no-gui
   ```

## Usage

### Basic Usage

**Start with GUI (Recommended)**

```bash
python main.py
```

**Run without GUI (Headless mode)**

```bash
python main.py --no-gui
```

**Use specific camera**

```bash
python main.py --camera 1
```

**Export session data**

```bash
python main.py --export my_session.json
```

### Using the GUI

1. **Start Monitoring**: Click "Start Monitoring" to begin analyzing the discussion
2. **View Live Stats**: Monitor real-time participant activity in the statistics panel
3. **Check Analytics**: Review discussion quality and patterns in the analytics tabs
4. **Export Data**: Use "Export Data" to save session information for later analysis
5. **Reset Session**: Click "Reset Session" to start fresh

### Command Line Options

```bash
python main.py [options]

Options:
  --camera INDEX    Camera device index (default: 0)
  --no-gui         Run without graphical interface
  --export FILE    Export session data to specified file
  -h, --help       Show help message
```

## Understanding the Output

### Participation Levels

- **Dominant**: Speaking significantly more than average (high participation)
- **Balanced**: Speaking around the average amount (healthy participation)
- **Quiet**: Speaking less than average but still contributing (needs encouragement)
- **Silent**: Very little or no speaking activity (requires attention)

### Discussion Quality Metrics

- **Turn-taking Balance**: How well participants take turns speaking (0-1 scale)
- **Engagement Score**: Overall level of active participation
- **Quality Score**: Comprehensive assessment of discussion health
- **Interruption Rate**: Frequency of overlapping or interrupting speech

### Real-time Indicators

- **Green Box**: Currently speaking participant
- **Activity Bars**: Speaking intensity levels
- **Participation %**: Percentage of total discussion time each person has spoken
- **Status Labels**: Current speaking state for each participant

## Project Structure

```
ai-moderator/
├── main.py                    # Main application entry point
├── face_tracker.py           # Face detection and tracking
├── mouth_tracker.py          # Mouth movement analysis
├── activity_tracker.py       # Speaking activity tracking
├── discussion_analytics.py   # Advanced discussion analytics
├── moderator_gui.py          # Real-time GUI interface
├── data_persistence.py       # Data storage and reporting
├── requirements.txt          # Python dependencies
├── README.md                 # This file
└── run.sh                    # Quick start script
```

## Technical Details

### AI Components

- **MediaPipe**: Face detection and facial landmark analysis
- **OpenCV**: Computer vision and video processing
- **Custom Algorithms**: Mouth movement detection and speech activity analysis
- **Real-time Processing**: Optimized for live video analysis (~30 FPS)

### Data Processing

- **Face Tracking**: Assigns consistent IDs to participants across frames
- **Landmark Analysis**: Uses 468 facial landmarks for precise mouth tracking
- **Movement Detection**: Calculates Mouth Aspect Ratio (MAR) and lip movement
- **Activity Classification**: Determines speaking/quiet states with confidence scoring

### Analytics Engine

- **Turn-taking Analysis**: Identifies conversation patterns and transitions
- **Interruption Detection**: Recognizes when participants interrupt or overlap
- **Engagement Tracking**: Measures participation intensity over time
- **Quality Assessment**: Evaluates discussion balance and effectiveness

## Troubleshooting

### Common Issues

**Camera not detected**

- Check camera permissions and connection
- Try different camera indices (0, 1, 2, etc.)
- Ensure no other applications are using the camera

**Poor face detection**

- Ensure good lighting conditions
- Check camera positioning and focus
- Verify faces are clearly visible and not too small

**High CPU usage**

- Reduce camera resolution in code if needed
- Close other resource-intensive applications
- Consider using a more powerful machine for larger groups

**Import errors**

- Ensure all dependencies are installed: `pip install -r requirements.txt`
- Check Python version compatibility (3.8+)
- Verify virtual environment is activated

### Performance Optimization

- **Lighting**: Good, even lighting improves detection accuracy
- **Camera Position**: Position camera to capture all participants clearly
- **Background**: Minimize busy backgrounds that might interfere with detection
- **Group Size**: Optimal performance with 2-8 participants

## Development

### Contributing

1. Fork the repository
2. Create a feature branch
3. Implement your changes
4. Add tests if applicable
5. Submit a pull request

### Extending the System

- **Custom Analytics**: Add new metrics in `discussion_analytics.py`
- **UI Enhancements**: Modify the GUI in `moderator_gui.py`
- **Export Formats**: Add new export options in `data_persistence.py`
- **Detection Improvements**: Enhance algorithms in face/mouth tracking modules

## License

This project is provided as-is for educational and research purposes.

## Support

For issues, questions, or feature requests, please refer to the project documentation or create an issue in the repository.

---

**Built with ❤️ for better discussions and more inclusive conversations**

### Notes (macOS, Windows, Linux)

- If the webcam window is black or permission is denied on macOS:
  System Settings → Privacy & Security → Camera → allow your terminal/IDE.
- If `cv2.imshow` errors on a headless server, install `opencv-python-headless`
  and skip GUI features.
- If you use Conda:
  ```bash
  conda create -n cv-hello python=3.11 -y
  conda activate cv-hello
  pip install -r requirements.txt
  ```
