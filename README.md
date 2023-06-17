# SafeTrack
SafeTrack is an ML-based application uses computer vision techniques and deep learning to detect drowsiness in real-time from a webcam feed. It detects human faces, identifies the eyes, and determines if the eyes are open or closed. If closed eyes are detected for an extended period, it triggers an alert sound indicating drowsiness.

## Requirements

Make sure you have the following dependencies installed:

- pygame==2.0.1
- opencv-python==4.5.3.56
- numpy==1.21.0
- tensorflow==2.5.0

You can install them by running the following command: `pip install -r requirements.txt`


## Usage

1. Clone this repository or download the source code.

2. Install the required dependencies using the command mentioned above.

3. Run the program using the following command: `python drowsiness_detection.py`

4. The webcam feed will open in a new window, and the drowsiness detection will start automatically.

5. If the program detects closed eyes for an extended period (15 frames in this case), it will display an alert on the screen and play an audio file to alert the user about drowsiness.

6. Press the 'q' key to exit the program.