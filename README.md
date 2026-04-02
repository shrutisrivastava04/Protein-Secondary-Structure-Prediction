# 🧬 Protein-Secondary-Structure-Prediction

This is a Deep Learning project that predicts protein secondary structures from amino acid sequences using a **Bidrectional LSTM (BiLSTM)** model. The system supports both **Q3 (3-class)** and **Q8 (8-class)** structure prediction and includes an end-to-end pipeline from preprocessing to deployment.

## Overview

Proteins play a critical role in biological systems, and their function is highly dependent on their structure. Predicting secondary structure from sequence data is a fundamental problem in bio-informatics.

This project builds an end-to-end solution that:
- Processes raw protein sequences
- Learning sequence patterns using deep learning
- Predicts structural classes at the residue level
- Provides an interactive interface for real-time predictions

## Objectives

- Predict protein secondary structure using:
    - **Q3 Classification:** Helix (H), Sheet (E), Coil (C)
    - **Q8 Classification:** 8 detailed structural states
- Implement and train BiLSTM-based sequence model
- Build a clean, modular ML pipeline
- Deploy a user-friendly web application

## Model Architecture

The model is based on a **Bidirectional LSTM (BiLSTM)** network that captures both forward and backward dependencies in protein sequences.

### Architecture Flow

Input Sequence -> Embedding Layer -> BiLSTM Encoder -> Dual Heads

### Key Components
- **Embedding Layer:** Converts amino acids into dense vector representations
- **BiLSTM Encoder:** Captures contextual dependencies in sequences
- **Dual Heads:** 
    - Q3 classifier (3 classes)
    - Q8 classifier (8 classes)

## Dataset

The dataset consists of protein sequences and their annotated secondary structures.

### Features

- **seq:** Amino Acid Sequence
- **sst3:** Q3 Labels
- **sst8:** Q8 Labels

## Preprocessing Pipeline

- Cleaning sequences (handling invalid residues)
- Tokenization (character-level encoding)
- Vocabulary mapping
- Padding sequences for batching 
- Masking padded tokens during training

## Tech Stack
Python | PyTorch | PyTorch Lightning | Streamlit | Pandas | NumPy | Scikit-Learn

## App URL

https://protein-secondary-structure-prediction-ss21.streamlit.app/

## Use Case

This project demonstrates how deep learning can accelerate structural analysis from sequence data. It can be used in:
- Drug Discovery & Development
- Disease Research
- Protein Engineering
- Genomics & Functional Annotation
- Education & Research Tool

## Conclusion

This project reinforces core deep learning concepts and showcases their impact in solving meaningful problems in computational biology.

## Author

**Shruti Srivastava**

*srivastavashruti218@gmail.com*