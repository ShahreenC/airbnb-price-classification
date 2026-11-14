"""Classify NYC Airbnb listing prices using listing attributes."""
import argparse
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

NUM = ['latitude', 'longitude', 'minimum_nights', 'number_of_reviews', 'reviews_per_month', 'availability_365']
CAT = ['neighbourhood_group', 'room_type']

def main(path):
    df = pd.read_csv(path)
    df['price'] = pd.to_numeric(df['price'], errors='coerce')
    df = df.loc[df['price'] > 0].copy()
    df['price_category'] = pd.cut(df['price'], bins=[0, 100, 200, float('inf')], labels=['budget', 'midrange', 'premium'])
    X, y = df[NUM + CAT], df['price_category']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=42, stratify=y)
    prep = ColumnTransformer([
        ('numeric', Pipeline([('impute', SimpleImputer(strategy='median')), ('scale', StandardScaler())]), NUM),
        ('category', Pipeline([('impute', SimpleImputer(strategy='most_frequent')), ('encode', OneHotEncoder(handle_unknown='ignore'))]), CAT)
    ])
    models = {
        'Decision Tree': (DecisionTreeClassifier(random_state=42), {'model__max_depth': [5, 10, 20], 'model__min_samples_leaf': [5, 20]}),
        'KNN': (KNeighborsClassifier(), {'model__n_neighbors': [5, 11, 21], 'model__weights': ['uniform', 'distance']})
    }
    print(f'Listings: {len(df)} | Train: {len(X_train)} | Test: {len(X_test)}')
    print('Class counts:\n', y.value_counts())
    for name, (model, params) in models.items():
        search = GridSearchCV(Pipeline([('prep', prep), ('model', model)]), params, cv=3, scoring='f1_macro', n_jobs=-1)
        search.fit(X_train, y_train)
        pred = search.predict(X_test)
        print(f'\n{name}\nBest parameters: {search.best_params_}\nAccuracy: {accuracy_score(y_test, pred):.3f}')
        print(classification_report(y_test, pred, zero_division=0))
        print('Confusion matrix (budget, midrange, premium):\n', confusion_matrix(y_test, pred, labels=['budget','midrange','premium']))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('csv', help='Path to AB_NYC_2019.csv')
    main(parser.parse_args().csv)
