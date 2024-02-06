# Emojiman 🤠
Motion tracking avatar 

## Overview
Emojiman is a fun project that overlays emojis and images on a person's face and body in real-time using computer vision techniques.

This project uses the Mediapipe library along with OpenCV to detect and track the pose of a person from a live video feed. It then overlays emojis and images onto specific body parts detected in the video stream.

## Features

- Real-time pose detection and tracking.
- Overlay emojis on the face.
- Overlay images of hands and boots on corresponding body parts.
- Detect when a cigarette is close to the face and display smoke effect.

## Usage

To use this project, make sure you have Python installed along with the necessary dependencies. You can install the required packages by running:

pip install -r requirements.txt 
Then, run the `emojiman.py` script:
python emojiman.py

Press 'q' to exit the application.

## Dependencies

- OpenCV
- Mediapipe

## TODO

1. Implement emoji scaling based on shoulder width.
2. Refactor the emoji overlay code into a function that takes specific emojis as parameters.
3. Add an option to overlay on recorded video. 

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

