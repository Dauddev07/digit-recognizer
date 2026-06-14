# Handwritten Digit Recognition Using CNN

A web-based handwritten digit recognition system powered by a Convolutional Neural Network (CNN) trained on the MNIST dataset. Users can draw digits (0–9) on an interactive canvas and receive real-time predictions with confidence scores.

---

## Features

- **Real-time prediction** — draw a digit on the HTML5 canvas and get instant classification results.
- **Confidence scores** — displays prediction probability for all 10 digit classes.
- **Intelligent preprocessing** — automatic colour inversion, bounding-box cropping, aspect-ratio-preserving padding, and LANCZOS resampling bridge the gap between canvas input and MNIST format.
- **Data augmentation** — random rotation, zoom, and translation during training improve robustness to natural handwriting variations.
- **Lightweight** — compact model (~1.1 MB) runs on standard hardware without a GPU.

---

## Demo

1. Launch the Flask server (see [Usage](#usage)).
2. Open `http://127.0.0.1:5000` in your browser.
3. Draw any digit (0–9) on the canvas.
4. Click **Predict** to see the recognised digit and its confidence score.
5. Click **Clear** to reset the canvas and try another digit.

---

## Project Structure

```
digit_recognizer/
├── app.py                          # Flask web server (inference API)
├── train_model.py                  # CNN training script
├── model/
│   └── digit_recognizer.keras      # Pre-trained model file
├── static/
│   ├── style.css                   # Frontend styling
│   └── script.js                   # Canvas drawing & API logic
├── templates/
│   └── index.html                  # Web interface template
├── venv/                           # Python virtual environment
├── .gitignore                      # Git ignore rules
└── README.md                       # Setup and run instructions
```

---

## Prerequisites

- **Python** 3.10 or higher
- **pip** (Python package manager)
- **Git** (optional, for cloning)

---

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/digit_recognizer.git
cd digit_recognizer
```

### 2. Create a Virtual Environment

```bash
python -m venv venv
```

### 3. Activate the Virtual Environment

**Windows:**
```bash
venv\Scripts\activate
```

**macOS / Linux:**
```bash
source venv/bin/activate
```

### 4. Install Dependencies

```bash
pip install tensorflow numpy flask pillow
```

> **Note:** On machines without a dedicated GPU, TensorFlow will automatically use CPU mode. No additional GPU drivers are required for inference.

---

## Usage

### Training the Model

If you want to retrain the model from scratch:

```bash
python train_model.py
```

This will:
1. Download the MNIST dataset via the Keras API (automatic on first run).
2. Train the CNN for up to 20 epochs with early stopping.
3. Save the trained model to `model/digit_recognizer.keras`.

Training takes approximately 5–10 minutes on CPU or 1–2 minutes on a CUDA-enabled GPU.

> **Tip:** The repository already includes a pre-trained model, so you can skip this step and go directly to running the web application.

### Running the Web Application

```bash
python app.py
```

The server starts on `http://127.0.0.1:5000` by default. Open this URL in any modern browser to use the application.

---

## Model Architecture

The CNN consists of three convolutional blocks followed by fully connected layers:

| Layer                     | Output Shape   | Details                    |
|---------------------------|----------------|----------------------------|
| Conv2D (32 filters, 3×3)  | 26 × 26 × 32  | ReLU activation            |
| BatchNormalization        | 26 × 26 × 32  | Normalises activations     |
| MaxPooling2D (2×2)        | 13 × 13 × 32  | Spatial downsampling       |
| Conv2D (64 filters, 3×3)  | 11 × 11 × 64  | ReLU activation            |
| BatchNormalization        | 11 × 11 × 64  | Normalises activations     |
| MaxPooling2D (2×2)        | 5 × 5 × 64    | Spatial downsampling       |
| Conv2D (128 filters, 3×3) | 3 × 3 × 128   | ReLU activation            |
| BatchNormalization        | 3 × 3 × 128   | Normalises activations     |
| Flatten                   | 1152           | Converts to 1D vector      |
| Dropout (0.4)             | 1152           | Regularisation             |
| Dense (128)               | 128            | ReLU activation            |
| Dropout (0.3)             | 128            | Regularisation             |
| Dense (10)                | 10             | Softmax (output)           |

**Data augmentation** (applied during training only): RandomRotation (±18°), RandomZoom (15%), RandomTranslation (10%).

---

## Performance

| Metric        | Score   |
|---------------|---------|
| Accuracy      | 99.31%  |
| Precision     | 99.30%  |
| Recall        | 99.30%  |
| F1-Score      | 99.30%  |

Evaluated on the standard MNIST test set (10,000 images).

---

## Technologies Used

| Category        | Technology                              |
|-----------------|-----------------------------------------|
| Language        | Python 3.12                             |
| Deep Learning   | TensorFlow / Keras                      |
| Web Framework   | Flask                                   |
| Frontend        | HTML5, CSS3, JavaScript (Canvas API)    |
| Libraries       | NumPy, Pillow (PIL)                     |
| IDE             | VS Code                                |

---

## Team Members

| Name              | Roll Number       | Role         |
|-------------------|--------------------|--------------|
| Daud Khalil       | F23BDOCS1M01073   | Group Leader |
| Malaika Samreez   | F23BDOCS1M01064   | Member       |
| Areeb Rehman      | F23BDOCS1M01171   | Member       |

**Course:** Computer Vision
**Session:** SP'26
**Institution:** Islamia University of Bahawalpur — Department of Computer Science

---

## License

This project was developed as an academic submission for the Computer Vision course at the Islamia University of Bahawalpur. All rights reserved.
