# Autism Detection and Interpretation System

This project aims to build an *Autism Detection and Interpretation System* that utilizes *spectrograms, **eye-tracking heatmaps, and **EfficientNetB0* for detecting Autism Spectrum Disorder (ASD) in adults. The system leverages deep learning models for classification, computer vision techniques for data preprocessing, and heatmap visualization for interpretation.

## 🔍 Project Highlights

* *Input Data*: Spectrograms and eye-tracking heatmaps derived from stimuli.
* *Model*: EfficientNetB0 (pre-trained on ImageNet, fine-tuned on ASD dataset).
* *Task*: Binary classification (ASD vs. Non-ASD).
* *Interpretability*: Grad-CAM heatmaps for visual explanations.
* *Tools*: TensorFlow, OpenCV, NumPy, Matplotlib, and Seaborn.
* *Deployment*: Can be extended for web applications or clinical decision support systems.

---

## 📦 Libraries and Frameworks Used

| Library/Framework | Purpose                                              |
| ----------------- | ---------------------------------------------------- |
| TensorFlow/Keras  | Model training, loading EfficientNetB0               |
| NumPy             | Numerical operations                                 |
| Matplotlib        | Plotting graphs and visualizations                   |
| Seaborn           | Heatmap visualizations                               |
| OpenCV            | Image preprocessing and manipulation                 |
| scikit-learn      | Evaluation metrics (accuracy, classification report) |
| pandas            | Data handling (if used in the notebook)              |
| PIL (Pillow)      | Image processing                                     |

---

## 🧠 Algorithms and Techniques

* *Convolutional Neural Networks (CNNs)*: For image classification.
* *EfficientNetB0*: A state-of-the-art CNN architecture for efficient and accurate image classification.
* *Transfer Learning*: Fine-tuning pre-trained EfficientNetB0 on custom dataset.
* *Grad-CAM*: For model interpretability, showing heatmaps of important regions in the input images.
* *Image Preprocessing*: Resizing, normalization, and augmentation of input images.
* *Evaluation Metrics*: Accuracy, Precision, Recall, F1-score, and Confusion Matrix.

---

## 🚀 How to Run

1. *Clone the Repository*:

   bash
   git clone https://github.com/yourusername/autism-detection.git
   cd autism-detection
   

2. *Install Dependencies*:

   bash
   pip install -r requirements.txt
   

3. *Prepare Data*:

   * Place spectrogram and eye-tracking heatmap images in the data/ folder.

4. *Run the Notebook*:
   Open SRM_MINOR_PROJECT.ipynb and execute all cells sequentially.

5. *View Results*:

   * Model performance metrics.
   * Grad-CAM heatmaps for interpretation.

---

## 📊 Results

* *Model Accuracy*: \~XX% (replace with actual accuracy from your project)
* *Interpretation*: Heatmaps indicate areas of interest based on model predictions.

---

## 📄 Future Work

* Extend model to multi-class classification.
* Enhance dataset with more diverse samples.
* Deploy as a web application using Flask or Streamlit.

---
