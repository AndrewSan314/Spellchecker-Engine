# Vietnamese Spelling Error Detection

This project implements a Vietnamese Spelling Error Detection system using the VSEC dataset. The system compares classical machine learning models (Logistic Regression, Random Forest, Naive Bayes) and a reimplementation of the VSEC model with some adaptations.

## Setup Instructions

Install dependencies:

```
pip install -r requirements.txt
```

## How to Run the Project

Run the main script:

```
python main.py
```

This script performs the following:

* Preprocesses and splits the data into train, dev, and test sets.
* Extracts features for classical baseline models.
* Trains and evaluates Logistic Regression, Random Forest, and Naive Bayes models.
* Fine-tunes the VSEC Transformer-based model and evaluates its performance.

## Key Files

Here is a list of the key files in this project and their roles:

* **preprocess.py**: Loads the data and preprocesses it into train, dev, and test splits.
* **feature_engineering.py**: Extracts features with context for the classical baseline models.
* **train_baselines.py**: Trains the classical machine learning models (Logistic Regression, Random Forest, Naive Bayes) and saves them.
* **evaluate.py**: Evaluates models using key metrics like Precision, Recall, F1 Score, Accuracy, and Balanced Accuracy.
* **vsec.py**: Implements the VSEC Transformer-based model for spelling error detection.
* **main.py**: The main script that orchestrates preprocessing, training, evaluation, and generating final results.
* **utils.py**: Utility functions to load and manage the dataset.

## Project Report

For detailed analysis, results, and discussions, refer to the [project report](https://github.com/halannhile/vietnamese-spelling-error-detection/blob/main/nhihlle_report.pdf).

## Reference

Do, Dinh-Truong, et al. *"VSEC: Transformer-based Model for Vietnamese Spelling Correction"*. [arXiv:2111.00640](https://arxiv.org/abs/2111.00640)

GitHub page for the VSEC dataset can be found [here](https://github.com/VSEC2021/VSEC/tree/main).
