# Fake Reviews Detection Project

## Overview

This project aims to detect fake reviews in Amazon product datasets using classical machine learning and modern ensemble methods. 
The workflow includes data cleaning, feature engineering, heuristic labeling, model training (Logistic Regression, LightGBM), and evaluation.

## Project Structure
- dataset used https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023 
- `modeling.ipynb` — Main notebook for data processing, feature engineering, modeling, and evaluation.
- `export.ipynb` — Data loading, EDA, and export of cleaned datasets.
- `analysis.ipynb` — In-depth EDA and visualization.
- `.gitignore` — Ensures `.pkl` files and other large files are not tracked by Git.


## Features Used

- TF-IDF vectorized review text
- Review length
- Sentiment polarity
- Helpfulness votes
- Verified purchase status
- Duplicate review detection
- User behaviour heuristics
  
## Key Features & Analysis

- **Text Cleaning:** Lowercasing, punctuation removal, and stopword filtering.
    
- **Heuristic Labeling:** Combines metadata and behavioral signals to label likely fake reviews.
- **EDA:**  
  - Review length distribution (with and without outliers)
  - Top words in fake vs. genuine reviews (stopwords removed)
  - Sentiment analysis by rating
  - Helpful votes and verified purchase analysis
  - Duplicate review and top product analysis

## Models

- Logistic Regression (with class balancing)
- LightGBM (gradient boosting, handles large/sparse data efficiently)
- (Optionally, no success) SGDClassifier, RandomForest, SVM

## Results

- LightGBM achieved the best F1 score (~0.61) using current features and heuristic labels.

## How to Run

1. Install dependencies:
    ```
    pip install -r requirements.txt
    ```
    *(or manually install: pandas, numpy, scikit-learn, lightgbm, matplotlib, textblob, datasets)*

2. Run `export.ipynb` to prepare and clean the dataset.
3. Run `modeling.ipynb` to train and evaluate models.
4. Run `analysis.ipynb` for advanced EDA and visualizations.

## Example EDA Visualizations

- Distribution of review lengths (zoomed and log-scaled)
- Top 10 products by number of reviews (bar plot)
- Boxplots for helpful votes and sentiment by label

## Future Plans

- **Feature Engineering:**  
  Add more features such as:
  - Exclamation/question mark counts
  - Uppercase ratio
  - Average word length
  - Time-based features (if available)
  - Product/user-level statistics

- **Model Improvements:**  
  - Hyperparameter tuning (GridSearchCV/RandomizedSearchCV)
  - Try deep learning models (BERT or similar, if resources allow)
  - Ensemble multiple models

- **Label Quality:**  
  - Improve heuristic labeling
  - Incorporate manual or crowdsourced annotation for ground truth


## License

MIT License

---

*This project is a work in progress. Contributions and suggestions are welcome!*
