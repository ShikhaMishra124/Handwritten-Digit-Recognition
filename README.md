# Handwritten Digit Recognition — System Documentation

## Overview
This project recognizes handwritten digits (single and multi-digit) drawn by a
user on a Tkinter canvas. It uses a Convolutional Neural Network (CNN) trained
on the MNIST dataset, with OpenCV handling image preprocessing before and
after prediction.

The system has two parts:
1. **`train_model.py`** — trains the CNN and saves it as `mnist.h5`
2. **`main.py`** — loads `mnist.h5` and predicts digits drawn by the user

---

## Setup & Usage

**1. Install dependencies:**
```bash
pip install -r requirements.txt
```

**2. Train the model** (run once — creates `mnist.h5`):
```bash
python train_model.py
```

**3. Run the app** (draw digits and predict):
```bash
python main.py
```

> Note: Skip step 2 on future runs once `mnist.h5` already exists in the folder —
> only re-run training if you change the model architecture or want to retrain.

---

## 1. Training (`train_model.py`)

**Dataset:** [MNIST](http://yann.lecun.com/exdb/mnist/) — loaded directly via
`keras.datasets.mnist.load_data()`.
- 60,000 training images, 10,000 test images
- Each image: 28×28 grayscale, single handwritten digit (0–9)
- Downloaded automatically on first run and cached locally (no manual dataset setup needed)

**Preprocessing:**
- Images reshaped to `(28, 28, 1)` (single channel)
- Pixel values normalized from `0–255` to `0–1`
- Labels one-hot encoded into 10 classes using `to_categorical`

**Model architecture (CNN):**
```
Conv2D(32, 5x5) → MaxPooling2D
Conv2D(64, 5x5) → MaxPooling2D
Flatten
Dense(128) → Dropout(0.3)
Dense(64)  → Dropout(0.4)
Dense(10, softmax)
```

**Training:**
- Optimizer: Adam
- Loss: categorical cross-entropy
- Batch size: 128, Epochs: 10
- Validated against the MNIST test set after each epoch

**Output:**
- Prints final test loss and accuracy
- Saves the trained model to **`mnist.h5`** — this file is the only thing
  `main.py` needs; the raw dataset is not required afterward

---

## 2. Prediction (`main.py`)

**Loading the model:**
```python
model = load_model("mnist.h5")
```
`mnist.h5` must be in the same directory as `main.py`.

**Drawing input:**
- User draws on a Tkinter `Canvas` (white strokes on black, matching MNIST's format)
- Each stroke is mirrored into a NumPy array (`self.image`) in parallel with
  the visible canvas, so the same drawing exists both on-screen and as
  pixel data OpenCV can process

**Preprocessing before prediction (per digit):**
To match the format the model was trained on, each drawn digit is:
1. Cropped to its bounding box (removes empty canvas space)
2. Scaled so its longest side fits a 20×20 box
3. Centered inside a 28×28 frame (padding added around it)
4. Lightly Gaussian-blurred to mimic MNIST's anti-aliased strokes
5. Normalized to 0–1 and reshaped to `(1, 28, 28, 1)`

This step matters because MNIST digits are centered and consistently sized —
skipping it (e.g. just resizing the whole canvas) is the most common cause of
wrong predictions.

**Multi-digit support:**
- The canvas is scanned for separate digit blobs (using dilation +
  `cv2.findContours`), filtered to remove noise, and sorted left to right
- Each blob is preprocessed and predicted independently
- Results are concatenated into a single string, e.g. `"1"` + `"2"` → `"12"`

**Output:**
- Displays the recognized digit(s) and the model's confidence in a popup window

---

## Flow Summary

```
MNIST dataset
     │
     ▼
train_model.py  →  trains CNN  →  saves mnist.h5
     │
     ▼
main.py  →  loads mnist.h5
     │
     ▼
User draws digit(s) on canvas
     │
     ▼
OpenCV preprocessing (crop → resize → center → blur)
     │
     ▼
Model predicts each digit
     │
     ▼
Result shown in popup (digit(s) + confidence)
```

---

## Notes / Limitations
- The model only recognizes digits **0–9**; multi-digit numbers work by
  segmenting and predicting each digit separately, not as true multi-digit OCR
- Recognition accuracy depends heavily on how closely the drawn digit matches
  MNIST's style (centered, reasonably sized, single stroke thickness)
- Digits drawn too close together may be merged into one blob during
  segmentation; digits with disconnected strokes may be split into two —
  tune the dilation kernel size in `segment_digits()` if this happens