# Fake Reviews Detection Project

## Overview

This project aims to detect fake reviews in Amazon product datasets using classical machine learning and modern ensemble methods. 
The workflow includes data cleaning, feature engineering, heuristic labeling, model training (Logistic Regression, LightGBM), and evaluation.

## Project Structure

- `modeling.ipynb` — Main notebook for data processing, feature engineering, modeling, and evaluation.
- `export.ipynb` — Data loading, EDA, and export of cleaned datasets.
- `.gitignore` — Ensures `.pkl` files and other large files are not tracked by Git.

## Features Used

- TF-IDF vectorized review text
- Review length
- Sentiment polarity
- Helpfulness votes
- Verified purchase status
- Duplicate review detection
- User behavior heuristics

## Models

- Logistic Regression (with class balancing)
- LightGBM (gradient boosting, handles large/sparse data efficiently)
- (Optionally) SGDClassifier, RandomForest, SVM

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

- **Data Expansion:**  
  - Apply the pipeline to other product categories (e.g., Beauty, Children)
  - Combine multiple datasets for better generalization

- **Label Quality:**  
  - Improve heuristic labeling
  - Incorporate manual or crowdsourced annotation for ground truth

- **Deployment:**  
  - Package the best model for API or batch inference
  - Prepare for deployment on cloud platforms (e.g., Azure, AWS)

- **Monitoring & Maintenance:**  
  - Monitor model performance on new data
  - Regularly retrain with updated datasets

## License

MIT License

---

*This project is a work in progress. Contributions and suggestions are welcome!*
