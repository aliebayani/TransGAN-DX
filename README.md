# TransGAN-DX: A Hybrid Transformer-GAN Approach for Enhanced Cardiovascular Disease Diagnosis

## Abstract
Cardiovascular disease remains one of the leading causes of mortality worldwide, highlighting the urgent need for more accurate and reliable diagnostic tools. **TransGAN-DX** is a novel deep learning framework that integrates Generative Adversarial Networks (GANs) and Transformer-based architectures to enhance cardiovascular disease detection.

The framework:
- Leverages GANs to generate synthetic data, improving training diversity and addressing data imbalance challenges.
- Utilizes a Transformer-based classifier trained on the enriched dataset to predict disease presence with high precision.
- Incorporates correlation and feature importance analyses for interpretability, aiding clinical decision-making.
- Addresses model overconfidence by calibrating predictions using calibration curves and expected calibration error (ECE).

Evaluated on the UCI Heart Disease dataset, **TransGAN-DX** achieved:
- **Accuracy:** 89%
- **F1-Score:** 86%
- **ROC-AUC Score:** 88%

## Features
- **Data Preprocessing**: Scales, normalizes, and handles data imbalance.
- **GAN Module**: Generates synthetic samples to augment the dataset.
- **Transformer Classifier**: Predicts disease presence with high precision.
- **Evaluation Tools**: Provides accuracy, F1-score, ROC curve, PR curve, and confusion matrix.
- **Visualization**: Includes correlation heatmaps, SHAP analysis, and loss plots.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

© ALI BAYANI

## Repository Structure
```plaintext
project/
│
├── data_preprocessing.py      # Data loading and preprocessing logic
├── gan_model.py               # Generator, Discriminator, and GAN training loop
├── transformer_classifier.py  # TransformerClassifier model and training
├── evaluation.py              # Accuracy, F1, ROC, PR curve, and confusion matrix
├── visualization.py           # Correlation heatmap, SHAP, and loss plots
├── main.py                    # Main script orchestrating the modules
└── requirements.txt           # Dependencies for the project
