Mumbai House Price Prediction
Project Overview
This project aims to build a robust machine learning model to predict house prices in Mumbai, India, using a dataset containing various property attributes. The goal is to provide accurate price estimations by analyzing key features of residential properties.

Features
Data Loading & Initial Exploration: Loading the dataset and understanding its structure.
Data Preprocessing:
Standardizing price units (converting 'Cr' and 'L' to a consistent 'Lakhs' unit).
Handling categorical variables through One-Hot Encoding.
Exploratory Data Analysis (EDA) & Visualization:
Correlation analysis between numerical features and the target variable.
Visualizing distributions of key numerical features (bhk, area, price).
Exploring relationships between categorical features (type, region, status, age) and house prices using box plots.
Machine Learning Model:
Utilizing LightGBM Regressor, an efficient and high-performance gradient boosting framework.
Implementing Pipeline for streamlined data preprocessing and model training.
Performing Hyperparameter Tuning using GridSearchCV with cross-validation to optimize model accuracy.
Model Evaluation: Assessing model performance using standard regression metrics: R-squared, Mean Absolute Error (MAE), Mean Squared Error (MSE), and Root Mean Squared Error (RMSE).
Dataset
The project uses the Mumbai House Prices.csv dataset, which includes the following columns:

bhk: Number of bedrooms, halls, and kitchens.
type: Type of property (e.g., Apartment).
locality: Specific locality of the property.
area: Area of the property.
price: Quoted price of the property.
price_unit: Unit of the price ('Cr' for Crores, 'L' for Lakhs).
region: Broader region in Mumbai.
status: Property status (e.g., Ready to move, Under Construction).
age: Age of the property (e.g., New).
Methodology
Data Loading: The Mumbai House Prices.csv file is loaded into a pandas DataFrame.
Price Normalization: The price column is converted to a uniform unit (Lakhs) to ensure consistency.
Feature Engineering: The dataset is prepared for modeling by identifying numerical and categorical features.
Exploratory Data Analysis: Visualizations and correlation matrices are generated to understand data patterns and relationships.
Data Splitting: The dataset is split into training and testing sets to evaluate the model's generalization capability.
Preprocessing Pipeline: A ColumnTransformer combined with OneHotEncoder is used within a Pipeline to automate the preprocessing of numerical and categorical features.
Model Training: A LightGBM Regressor is trained on the preprocessed training data.
Hyperparameter Tuning: GridSearchCV is employed to find the optimal hyperparameters for the LightGBM model, enhancing its predictive accuracy.
Model Evaluation: The trained model's performance is assessed on the unseen test data using various regression metrics.
Results
After hyperparameter tuning, the model achieved the following performance on the test set:

R-squared (R 
2
 ): 0.8953
Mean Absolute Error (MAE): 27.13
Mean Squared Error (MSE): 4783.63
Root Mean Squared Error (RMSE): 69.16
These results indicate that the model explains approximately 89.53% of the variance in house prices and provides relatively accurate predictions.

Usage / How to Run
Clone the repository (if hosted on GitHub) or download the project files.
Ensure you have Jupyter Notebook/Lab installed.
Install the necessary libraries:
Bash

pip install pandas numpy matplotlib seaborn scikit-learn lightgbm
Place the Mumbai House Prices.csv file in the same directory as the Jupyter notebook (Mumbai_House_Price_Prediction_with_Analysis.ipynb).
Open the Jupyter Notebook:
Bash

jupyter notebook
Navigate to and open Mumbai_House_Price_Prediction_with_Analysis.ipynb.
Run all cells in the notebook.
Dependencies
pandas
numpy
matplotlib
seaborn
scikit-learn
lightgbm
Future Enhancements
More Advanced Feature Engineering: Explore creating more sophisticated features (e.g., polynomial features, interaction terms, or geographical features if latitude/longitude data were available).
Explore Other Models: Experiment with other regression algorithms (e.g., XGBoost, RandomForest, CatBoost) and ensemble methods.
Deep Learning Models: Investigate neural network architectures for price prediction.
Outlier Detection and Handling: Implement robust outlier detection and treatment strategies.
Data Collection: Incorporate more diverse data sources, if available, such as amenities, proximity to public transport, schools, etc.
