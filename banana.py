import cv2
import numpy as np
import time
from tensorflow.keras.models import load_model
import threading
import tensorflow as tf
import queue

# Check for GPU availability
print(tf.config.list_physical_devices())
if tf.config.list_physical_devices('GPU'):
    print("GPU is available.")
else:
    print("GPU is not available.")

# Parameters
MIN_WORD_LENGTH = 3
COUNT_PANGRAMS = 1
TOTAL_LETTER_COUNT = 7

# Load the ASL model
model = load_model("./Model/keras_model.h5", compile=False)
class_names = open("./Model/labels.txt", "r").readlines()

# Initialize the webcam
camera = cv2.VideoCapture(0)
camera.set(cv2.CAP_PROP_FRAME_WIDTH, 480)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)
camera.set(cv2.CAP_PROP_FPS, 15)  # Set the frame rate to 15 fps

# List of words for the game
word_list = ["CAT", "DUCK", "BIRD", "FISH", "PAN"]

# Queue for communication between threads
prediction_queue = queue.Queue()
stop_event = threading.Event()

def process_frame():
    while not stop_event.is_set():
        # Grab the webcam's image
        ret, image = camera.read()
        if not ret:
            continue

        # Resize the raw image into (224-height,224-width) pixels
        image_resized = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)

        # Make the image a numpy array and reshape it to the model's input shape
        image_array = np.asarray(image_resized, dtype=np.float32).reshape(1, 224, 224, 3)

        # Normalize the image array
        image_array = (image_array / 127.5) - 1

        # Predict with the model
        prediction = model.predict(image_array, verbose=0)
        index = np.argmax(prediction)
        class_name = class_names[index].strip()  # Strip any extra whitespace
        confidence_score = prediction[0][index]

        # Debug: Print prediction and confidence score
        print(f"Prediction: {class_name}, Confidence: {confidence_score}")

        # Return the guessed letter if confidence is high enough
        if confidence_score > 0.8:  # Adjust the confidence threshold as needed
            guessed_letter = class_name[2:].strip()
            prediction_queue.put((guessed_letter, confidence_score))

def display_game_frame(image, word, guessed_letters, score):
    overlay = image.copy()
    alpha = 0.7

    # Display the current word to spell
    cv2.putText(overlay, f"Spell the word: {word}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Display the current progress
    partial_word = ' '.join([letter if letter in guessed_letters else '_' for letter in word])
    cv2.putText(overlay, f"Current Progress: {partial_word}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Display the current score
    cv2.putText(overlay, f"Current Score: {score}", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Blend the overlay with the original image
    cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)

def play():
    score = 0
    frame_delay = 0.1  # Delay in seconds to reduce frame rate

    # Start the frame processing thread
    threading.Thread(target=process_frame, daemon=True).start()

    for word in word_list:
        guessed_letters = []

        while True:
            start_time = time.time()

            # Grab the webcam's image
            ret, image = camera.read()
            if not ret:
                continue

            # Get ASL input from the prediction queue
            try:
                guess, _ = prediction_queue.get(timeout=1)  # Timeout to prevent blocking
                print(f"Guessed letter from queue: {guess}")
            except queue.Empty:
                guess = None

            if guess and guess in guessed_letters:
                print('You already signed this letter:', guess, '\n')
                continue

            if guess and guess in word:
                guessed_letters.append(guess)
                print('Correct letter:', guess)

                # Check if the word is fully guessed
                if all(letter in guessed_letters for letter in word):
                    print(f"Well done! You spelled the word {word}.")
                    score += 1
                    break
            elif guess:
                print('Incorrect letter. Try again.', '\n')

            # Display the current word, progress, and score
            display_game_frame(image, word, guessed_letters, score)
            cv2.imshow("Webcam Image", image)

            # Listen to the keyboard for presses
            keyboard_input = cv2.waitKey(1)

            # 27 is the ASCII for the esc key on your keyboard
            if keyboard_input == 27:
                stop_event.set()  # Stop the prediction thread
                break

            # Ensure the loop runs at the desired frame rate
            elapsed_time = time.time() - start_time
            if elapsed_time < frame_delay:
                time.sleep(frame_delay - elapsed_time)

    print(f"Game Over! Your final score is {score}.")

def main():
    play()

if __name__ == "__main__":
    main()
