# Airbnb Price Classification

A Python machine learning project comparing Decision Tree and K-Nearest Neighbors (KNN) classifiers to predict nightly price categories for New York City Airbnb listings.

## Dataset
Use the [New York City Airbnb Open Data dataset](https://www.kaggle.com/datasets/dgomonov/new-york-city-airbnb-open-data), file `AB_NYC_2019.csv`. Download it separately and place it in the project folder. The dataset is not included in this repository.

## Method
- Define budget ($100 or less), midrange (over $100 through $200), and premium (over $200) categories. These are project-defined thresholds, not official Airbnb categories.
- Exclude price from input features to prevent target leakage; use location, room type, availability, minimum nights and review activity.
- Split data into 80% training and 20% testing using stratification.
- Impute missing values, scale numeric features and one-hot encode categorical features inside a pipeline.
- Tune both classifiers using three-fold cross-validation and macro F1 scoring on training data only.
- Report held-out accuracy, per-class precision, recall, F1 and confusion matrices.

## Run locally
```bash
pip install -r requirements.txt
python train.py AB_NYC_2019.csv
```

## Results
Run the script on the real dataset and record the printed results here before claiming performance metrics. No real-data results are reported yet.

## Limitations
The data is historical, and listing prices are not necessarily actual booked prices. Results may not generalize to current listings or other cities. The model predicts price bands, not exact prices.
