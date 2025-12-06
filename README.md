# Image Classification Gradio App

Interactive Gradio demo that classifies an uploaded image as **dog**, **cat**, or **snake** using a custom CNN trained in PyTorch. The app overlays the predicted label on the image and returns both the label text and the annotated image.

## 🚀 Try the Live Demo

You can try this app directly in your browser without any setup:

**[https://huggingface.co/spaces/aboomar26/classification-gradio-app](https://huggingface.co/spaces/aboomar26/classification-gradio-app)**

Simply upload an image and get instant predictions!

## Project Structure
- `app.py` — Gradio interface wiring the upload UI to the predictor.
- `core/predict.py` — `CustomCNNClassifier` definition and `ImageClassifier` wrapper for preprocessing, inference, and label overlay.
- `model/cnn_model.pth` — Trained weights for the 3‑class CNN (loaded on CPU or CUDA if available).
- `notebook/custom_dataset & custom CNN Architecture.ipynb` — Training and experimentation notebook.
- Sample images: `dog.jpg`, `labels_image.jpg` (example output), `uploaded_image.jpg` (latest upload).

## Requirements
- Python 3.9+ recommended
- Dependencies listed in `requirements.txt`:
  - torch, torchvision, gradio, opencv-python, Pillow

## Setup
1) (Optional) create and activate a virtual environment.  
2) Install dependencies:
```bash
pip install -r requirements.txt
```

## Run the app

### Option 1: Try the Live Demo
Visit the hosted version on Hugging Face Spaces: **[https://huggingface.co/spaces/aboomar26/classification-gradio-app](https://huggingface.co/spaces/aboomar26/classification-gradio-app)**

### Option 2: Run Locally
```bash
python app.py
```
This launches a local Gradio server and prints a URL. Open it in your browser to upload an image. The app saves the upload to `uploaded_image.jpg`, runs inference, writes the annotated result to `labels_image.jpg`, and displays both the predicted label and the labeled image.

## Training / Re-training
- Use the notebook `notebook/custom_dataset & custom CNN Architecture.ipynb` to inspect data preprocessing, model architecture, and training.  
- After training, export weights to `model/cnn_model.pth` (matching the architecture in `core/predict.py`), then rerun the app.

## Notes
- CUDA is used automatically if available; otherwise inference runs on CPU.
- Images are resized to 128×128 and normalized with ImageNet means/std before inference.
- Update the `class_name` mapping in `app.py` or `core/predict.py` if you retrain on new classes.

# Gradio_Classification_app
