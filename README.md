# Mouse Control with Face & Hand (OpenCV)

## Desription

This project allows controlling the computer mouse using a webcam.

- cursor movement → controlled by head position
- mouse click → triggered by detecting a hand in a specific area

The application uses real-time image processing.

## How It Works
The webcam captures video frames
A face is detected using Haar Cascade
The center of the face is mapped to screen coordinates
The cursor follows the head movement
When a hand (or object) is detected in the ROI area, a click is performed
