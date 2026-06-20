# Diabetic Retinopathy Detector

A deep learning model that classifies retina scan images as showing signs of diabetic retinopathy or not, built using transfer learning and deployed as an interactive web app.

## The Problem

Diabetic retinopathy is a leading cause of preventable blindness, caused by diabetes-related damage to blood vessels in the retina. Early detection through retina screening allows for treatment before vision loss occurs, but access to specialist screening is limited in many regions, including much of Kenya and East Africa. This project explores whether a relatively lightweight, transfer-learning based model can provide a useful first-pass screening signal.

## What This Tool Does

Upload a retina scan image, and the model classifies it as either:
- **DR** — signs of diabetic retinopathy detected
- **No_DR** — no signs detected

Along with a confidence score for the prediction.

## Live App

[Link to your deployed Streamlit app — add once deployed]

## Model Details

| Detail | Value |
|---|---|
| Architecture | MobileNetV2 (transfer learning, base frozen) |
| Input size | 224 x 224 RGB |
| Training images | 2,076 (1,050 DR / 1,026 No_DR) |
| Validation images | 531 |
| Test images | 231 |
| Validation accuracy | 91.9% |
| Test accuracy | 89.18% |
| DR recall (test set) | 89% |
| No_DR recall (test set) | 89% |

### Confusion Matrix (Test Set)

|              | Predicted DR | Predicted No_DR |
|--------------|:---:|:---:|
| **Actual DR**     | 101 | 12 |
| **Actual No_DR**  | 13  | 105 |

Out of 113 true DR cases, the model correctly identified 101 (89% recall). Out of 118 true No_DR cases, it correctly identified 105 (89% recall). Performance is balanced across both classes, with no systematic bias toward either prediction.

## Dataset

[Diagnosis of Diabetic Retinopathy](https://www.kaggle.com/datasets/pkdarabi/diagnosis-of-diabetic-retinopathy) — a binary-labeled, pre-split (train/valid/test) retina image dataset sourced from Kaggle.

## Tech Stack

- **Training**: Google Colab (T4 GPU), TensorFlow/Keras, MobileNetV2 pretrained on ImageNet
- **App**: Streamlit, Python 3.11
- **Image handling**: Pillow, NumPy

## How It Was Built

1. **Data exploration and verification** — confirmed class balance and inspected sample images before training
2. **Transfer learning** — loaded MobileNetV2 with frozen ImageNet weights, added a custom classification head (GlobalAveragePooling2D → Dropout → Dense sigmoid output)
3. **Training** — 10 epochs, Adam optimizer, binary crossentropy loss; only 1,281 parameters were trainable (the new head), keeping training fast and reducing overfitting risk on a relatively small dataset
4. **Evaluation** — assessed on a held-out test set never used during training or validation tuning, with full classification report and confusion matrix
5. **Deployment** — model exported in native Keras format, loaded into a Streamlit app for interactive predictions

Full training and evaluation code is in [`notebooks/`](notebooks/).

## Project Structure
