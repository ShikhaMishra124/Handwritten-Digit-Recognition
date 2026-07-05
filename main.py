import cv2
import numpy as np
import tensorflow as tf
from keras.models import load_model
import tkinter as tk
from tkinter import Canvas

# Load the pre-trained MNIST digit recognition model.
# This model is used to predict handwritten digits drawn on the canvas.
model = load_model("mnist.h5")


class DigitRecognizer:
    """
    DigitRecognizer creates a GUI-based handwritten digit recognition application.

    Features:
    - Provides a drawing canvas for the user.
    - Allows clearing the canvas.
    - Segments multiple handwritten digits.
    - Preprocesses each digit into MNIST format.
    - Predicts the digits using the trained CNN model.
    """

    def __init__(self, root):
        """
        Initializes the application.

        What this function does:
        ------------------------
        1. Creates the main application window.
        2. Adds a black drawing canvas.
        3. Creates Clear and Predict buttons.
        4. Registers mouse movement events for drawing.
        5. Creates a NumPy image that stores the drawing for OpenCV processing.
        """

        self.root = root
        self.root.title("Handwritten Digit Recognition")
        self.canvas = Canvas(root, width=560, height=280, bg="black")
        self.canvas.pack()
        self.btn_clear = tk.Button(root, text="Clear", command=self.clear_canvas)
        self.btn_clear.pack(side="left")
        self.btn_predict = tk.Button(root, text="Predict", command=self.predict_digits)
        self.btn_predict.pack(side="right")
        self.canvas.bind("<B1-Motion>", self.draw)
        self.image = np.zeros((280, 560), dtype=np.uint8)

    def draw(self, event):
        """
        Draws on both the Tkinter canvas and the NumPy image.

        What this function does:
        ------------------------
        1. Gets the current mouse coordinates.
        2. Draws a white circle on the visible Tkinter canvas.
        3. Draws the same circle on the NumPy image using OpenCV.
        4. The NumPy image is later used for image processing and prediction.
        """

        x, y = event.x, event.y
        self.canvas.create_oval(x, y, x + 8, y + 8, fill="white", outline="white")
        cv2.circle(self.image, (x, y), 8, 255, -1)

    def clear_canvas(self):
        """
        Clears everything drawn by the user.

        What this function does:
        ------------------------
        1. Removes all drawings from the Tkinter canvas.
        2. Resets the NumPy image to all black pixels.
        3. Allows the user to start drawing again.
        """

        self.canvas.delete("all")
        self.image.fill(0)

    def segment_digits(self):
        """
        Detects and separates multiple handwritten digits.

        What this function does:
        ------------------------
        1. Copies the original drawing.
        2. Applies dilation so broken strokes become connected.
        3. Finds contours (connected white regions).
        4. Removes tiny contours considered as noise.
        5. Computes a bounding box around each digit.
        6. Sorts the bounding boxes from left to right.
        7. Returns the list of detected digit regions.
        """

        img = self.image.copy()

        kernel = np.ones((15, 15), np.uint8)
        dilated = cv2.dilate(img, kernel, iterations=1)

        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return []

        boxes = []
        for c in contours:
            x, y, w, h = cv2.boundingRect(c)

            # Ignore very small blobs that are likely noise.
            if w * h < 100:
                continue

            boxes.append((x, y, w, h))

        # Sort digits from left to right.
        boxes.sort(key=lambda b: b[0])
        return boxes

    def prepare_digit(self, x, y, w, h):
        """
        Converts one detected digit into MNIST format.

        What this function does:
        ------------------------
        1. Crops the digit from the original image.
        2. Adds padding around the digit.
        3. Removes extra empty space.
        4. Preserves the digit's aspect ratio.
        5. Resizes the digit to fit inside a 20x20 box.
        6. Places the digit at the center of a 28x28 image.
        7. Applies slight Gaussian blur to resemble MNIST images.
        8. Returns the processed 28x28 image ready for prediction.
        """
        pad = 10
        x0 = max(0, x - pad)
        y0 = max(0, y - pad)
        x1 = min(self.image.shape[1], x + w + pad)
        y1 = min(self.image.shape[0], y + h + pad)
        digit = self.image[y0:y1, x0:x1]

        coords = cv2.findNonZero(digit)
        if coords is None:
            return None

        bx, by, bw, bh = cv2.boundingRect(coords)
        digit = digit[by:by + bh, bx:bx + bw]

        if bw > bh:
            new_w = 20
            new_h = max(1, int(bh * (20.0 / bw)))
        else:
            new_h = 20
            new_w = max(1, int(bw * (20.0 / bh)))

        digit_resized = cv2.resize(digit, (new_w, new_h), interpolation=cv2.INTER_AREA)

        canvas28 = np.zeros((28, 28), dtype=np.uint8)
        x_off = (28 - new_w) // 2
        y_off = (28 - new_h) // 2
        canvas28[y_off:y_off + new_h, x_off:x_off + new_w] = digit_resized

        canvas28 = cv2.GaussianBlur(canvas28, (3, 3), 0)

        return canvas28

    def predict_digits(self):
        """
        Predicts all handwritten digits present on the canvas.

        What this function does:
        ------------------------
        1. Segments all handwritten digits.
        2. Processes each digit into MNIST format.
        3. Normalizes pixel values to the range [0,1].
        4. Reshapes the image for CNN input.
        5. Uses the trained model to predict each digit.
        6. Calculates prediction confidence.
        7. Combines all predicted digits into one number.
        8. Computes average confidence across all digits.
        9. Displays the final recognized number in a popup window.
        """

        boxes = self.segment_digits()
        if not boxes:
            return

        predicted_string = ""
        confidences = []

        for (x, y, w, h) in boxes:
            processed = self.prepare_digit(x, y, w, h)
            if processed is None:
                continue

            img_input = processed.astype("float32") / 255.0
            img_input = img_input.reshape(1, 28, 28, 1)

            prediction = model.predict(img_input, verbose=0)
            digit = np.argmax(prediction)
            confidence = float(np.max(prediction)) * 100

            predicted_string += str(digit)
            confidences.append(confidence)

        avg_conf = sum(confidences) / len(confidences) if confidences else 0

        result_window = tk.Toplevel(self.root)
        result_window.title("Prediction")
        tk.Label(
            result_window,
            text=f"Recognized Number: {predicted_string}\nAvg Confidence: {avg_conf:.1f}%",
            font=("Arial", 20),
        ).pack()


# Entry point of the application.
# Creates the main Tkinter window and starts the GUI event loop.
if __name__ == "__main__":
    root = tk.Tk()
    app = DigitRecognizer(root)
    root.mainloop()