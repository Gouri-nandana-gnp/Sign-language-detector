# SIGN LANGUAGE DETECTION GAME



This project is a game that uses a webcam to detect American Sign Language (ASL) gestures in real-time using OpenCV and a pre-trained machine learning model. The objective of the game is to correctly sign letters to spell out words displayed on the screen.



## Features
- Real-Time Sign Language Detention: The game uses a pre-trained model to detect ASL signs 
  through your webcam.
- Interactive Gameplay: Players are given words to spell by signing the corresponding letters 
  in ASL.
- Dynamic Feedback: The game provides instant feedback, showing progress and updating the 
  score based on correct or incorrect signs.



## Requirements

- Python 3.x
- TensorFlow
- Opencv
- NumPy

## Installation 

1. Clone the repository:
   ```bash
   git clone https://github.com/Gouri-nandana-gnp/Sign-language-detector.git
   cd Sign-language-detector
2. Install the required dependencies:
  ```bash
  pip install -r requirements.txt
  ```
3. Download the pre-trained model and labels:

  - Place the model (keras_model.h5) in the ./Model directory.
  - Place the label file (labels.txt) in the ./Model directory.


## Usage

1. Ensure your webcam is connected.
2. Run the game:
  ```bash
    python combined.py
```
3. Follow the on-screen instructions to spell out the words displayed by signing the 
   corresponding ASL letters.
4. The game ends when all words are completed. Your final score will be displayed.

## How It Works

- Model Loading: The game loads a pre-trained ASL model (keras_model.h5) to recognize hand 
  gestures.
- Webcam Input: The game captures images from your webcam and preprocesses them for the model.
- Prediction: The model predicts the letter being signed based on the input image, and the game 
  logic determines if the letter is correct.
- Feedback: The game updates the displayed word and score based on your input.

## Contributing
Contributions are welcome! Feel free to open a pull request or report issues.

## Acknowledgements
  Special thanks to the contributors!

