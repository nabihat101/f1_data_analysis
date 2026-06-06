# F1 Data Analysis and Predictor 2026
This project uses machine learning, python libraries, and the Fast F1 API to create accurate predictions on race outcomes for the F1 2026 season.

## Project Overview 
This project builds a full machine learning pipeline for Formula 1 race prediction:

- Data collection using the FastF1 API
- Feature engineering from:
  - FP1 / FP2 / FP3 practice sessions
  - Qualifying results
  - Race results (training labels)
  - Weather conditions
- Model training using Gradient Boosting Regression
- Live prediction system for race weekends

## How it Works
1. Training 
- Loads historical F1 seasons (2018–2025)
- Extracts driver-level features per race
- Cleans and encodes data
- Trains Gradient Boosting model
- Saves model using joblib
2. Live Prediction 
- Loads upcoming race weekend sessions
- Extracts FP1 + FP2 + qualifying data
- Builds features same as training
- Predicts finishing order
- Outputs ranked race prediction

## Python Libraries
- numPy
- scikit learn
- pandas
- fastf1 API


