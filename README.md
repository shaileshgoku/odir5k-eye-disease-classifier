# ODIR-5K Eye Disease Classification

An AI-powered retinal image classification application using **EfficientNetB0 and clinical features**, built with TensorFlow and deployed using Streamlit Community Cloud.

## Live Demo

🚀 Streamlit Application: https://odir5k-eye-disease-classifier-kpupbpdrgmv3aznbcprcen.streamlit.app/
---

## 1. Project Overview

Eye diseases can cause significant vision impairment if they are not identified and managed appropriately.

This project explores the use of deep learning and clinical information to classify retinal fundus images into eight categories.

The model combines:
- Retinal fundus images.
- Patient age.
- Patient gender.

An EfficientNetB0-based deep learning architecture processes the retinal images, while clinical features provide additional input to the classification model.

The final application allows users to upload a retinal image, enter clinical information, and view the model's predicted disease category and class scores.

**Project objective:** Build and deploy an end-to-end deep learning research prototype for multi-class retinal image classification.

## 2. Dataset

**Dataset:** ODIR-5K (Ocular Disease Intelligent Recognition)

The ODIR-5K dataset contains retinal fundus images and associated patient information.

This project uses the dataset for an eight-class classification task.

### Disease Categories

| Label | Category |
|---|---|
| N | Normal |
| D | Diabetes |
| G | Glaucoma |
| C | Cataract |
| A | Age-related Macular Degeneration |
| H | Hypertension-related findings |
| M | Myopia |
| O | Other retinal abnormalities |

**Dataset used:** 6,392 image records.

The dataset was explored and prepared before model development.

## 3. Technology Stack

| Component | Technology |
|---|---|
| Programming Language | Python |
| Deep Learning Framework | TensorFlow |
| Neural Network Architecture | EfficientNetB0 |
| Image Processing | OpenCV, Pillow |
| Data Analysis | Pandas, NumPy |
| Model Format | Keras |
| Application Framework | Streamlit |
| Version Control | Git and GitHub |
| Deployment | Streamlit Community Cloud |
| Development Environment | Google Colab |

## 4. Model Architecture

The project uses a multi-input deep learning architecture.

It combines image features and clinical features to generate predictions.

### Input 1: Retinal Image

- Input size: 224 × 224 pixels.
- Color format: RGB.
- Image type: Fundus image.
- Feature extraction: EfficientNetB0.

### Input 2: Clinical Features

Three clinical features are provided to the model:

1. Standardized patient age.
2. Female indicator.
3. Male indicator.

Age is standardized using the training dataset's mean and standard deviation.

### Model Output

The model produces eight class scores corresponding to the eight disease categories.

The class with the highest predicted score is displayed as the model's prediction.

## 5. Model Pipeline

```text
              Retinal Fundus Image
                       |
                       v
              Image Preprocessing
                224 x 224 RGB
                       |
                       v
                 EfficientNetB0
                       |
                       v
                Image Features
                       |
                       |
                       +----------------+
                                        |
Patient Age                             |
    |                                   |
    v                                   |
Standardization                         |
    |                                   |
    + Gender Indicators                 |
    |                                   |
    v                                   |
Clinical Features                       |
    |                                   |
    +-----------------------------------+
                       |
                       v
                Feature Fusion
                       |
                       v
               Classification Layer
                       |
                       v
                 8 Class Scores
                       |
                       v
               Predicted Category
```

## 6. Model Performance

The final model achieved the following test-set results:

| Metric | Result |
|---|---:|
| Test Accuracy | 52.48% |
| Macro F1-score | 0.4369 |

The results indicate that the model has room for improvement, particularly in its performance across individual disease categories.

These metrics represent experimental performance on the project's test dataset and should not be interpreted as evidence of clinical effectiveness.

## 7. Application Features

The deployed Streamlit application provides:

- Retinal fundus image upload.
- Patient age input.
- Patient gender selection.
- Image preprocessing.
- Clinical feature preprocessing.
- Model inference.
- Predicted disease category.
- Class-wise prediction scores.
- Visual presentation of prediction results.

## 8. Project Structure

```text
odir5k-eye-disease-classifier/
│
├── app.py
│
├── odir5k_model.keras
│
├── requirements.txt
│
├── README.md
│
└── .gitignore
```

### File Description

| File | Purpose |
|---|---|
| `app.py` | Streamlit application |
| `odir5k_model.keras` | Trained deep learning model |
| `requirements.txt` | Python dependencies |
| `README.md` | Project documentation |
| `.gitignore` | Files excluded from Git |

## 9. Installation and Execution

### Step 1: Clone the Repository

```bash
git clone https://github.com/ShaileshGoku/odir5k-eye-disease-classifier.git
```

### Step 2: Navigate to the Project

```bash
cd odir5k-eye-disease-classifier
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Run the Streamlit Application

```bash
streamlit run app.py
```

The application will open in your browser.

## 10. Deployment

The application is deployed using **Streamlit Community Cloud**.

Deployment workflow:

1. Upload project files to GitHub.
2. Connect the GitHub repository to Streamlit Community Cloud.
3. Select `app.py` as the application entry point.
4. Configure the Python environment.
5. Install dependencies from `requirements.txt`.
6. Deploy the application.

The deployed application loads the trained Keras model and performs predictions through its user interface.

## 11. Key Learnings

This project provided practical experience in:

- Exploring a real-world medical imaging dataset.
- Preparing image and clinical features.
- Building a multi-input deep learning model.
- Working with EfficientNetB0.
- Integrating clinical information with image-based features.
- Evaluating multi-class classification performance.
- Saving and loading trained Keras models.
- Building an interactive Streamlit application.
- Managing Python and TensorFlow dependency compatibility.
- Deploying a deep learning application to the cloud.

## 12. Future Improvements

Potential improvements include:

- Improving classification performance across disease categories.
- Investigating class imbalance.
- Experimenting with alternative image architectures.
- Applying explainability techniques such as Grad-CAM.
- Evaluating model calibration and prediction confidence.
- Conducting more rigorous external validation.
- Improving application monitoring and error handling.

## 13. Important Disclaimer

**This project is an educational and research prototype.**

It is not a clinically validated medical diagnostic system and must not be used to diagnose, treat, or make medical decisions.

The model's predictions may be inaccurate. Always consult a qualified ophthalmologist or healthcare professional for medical evaluation.

## 14. Acknowledgements

- ODIR-5K dataset and its contributors.
- TensorFlow and Keras.
- EfficientNet architecture.
- Streamlit Community Cloud.

---

**Built as a hands-on deep learning and deployment project.**

*Exploring how AI can help solve real-world problems through data and engineering.*
