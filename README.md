# Alpha-Signal-Predition-with-XGBoost
XGBoost Model-based Alpha Signal Prediction using Microblogging Data
Welcome to the Alpha Signal Prediction project! This repository demonstrates a full workflow for predicting alpha signals in stock data using XGBoost, leveraging microblogging and social media-derived features. The project includes data loading, exploratory analysis, preprocessing, outlier handling, visualization, and machine learning modeling.

# Project Overview

This project aims to classify alpha signals (1–4) for stocks based on a set of engineered features (SF1–SF7) extracted from microblogging data. The workflow includes:

Data loading and inspection

Exploratory Data Analysis (EDA)

Data preprocessing and cleaning

Outlier detection and treatment

Feature engineering

Data visualization

Model training with XGBoost

Evaluation and reporting

📂 Dataset
The dataset (dataset.csv) contains the following columns:

Column	Description
Id	Unique row identifier
date	Date of observation
ticker	Stock ticker symbol
SF1–SF7	Engineered features
alpha	Target label (1–4)
Example:

Id	date	ticker	SF1	SF2	...	SF7	alpha
1	21/08/18	$NTAP	-0.62865	0.98889	...	-0.99553	2
🛠️ Installation
Install the required Python packages:

bash
pip install pandas numpy matplotlib seaborn missingno scikit-learn xgboost joblib feature-engine
📊 Exploratory Data Analysis
Shape: 27,006 rows × 11 columns

No missing values detected.

Alpha Classes: 1, 2, 3, 4

Feature Distributions: Visualized using histograms and boxplots for all SF1–SF7 features.

Example: Count Plot of Labels
![Count Plot](assets/label_countplot.png Preprocessing

Date Splitting: Extracted day and month from the date column.

Ticker Cleaning: Removed $ prefix from ticker symbols.

Categorical Conversion: Converted date, ticker, Day, and Month to categorical or integer types as appropriate.

📉 Outlier Detection & Treatment
Outliers in SF1–SF7 are visualized using boxplots and treated using the Winsorizer from feature-engine:

python
from feature_engine.outliers import Winsorizer

for col in ['SF1','SF2','SF3','SF4','SF5','SF6','SF7']:
    wi = Winsorizer(capping_method='iqr', tail='both', fold=1.5)
    data[col] = wi.fit_transform(data[[col]])
📈 Data Visualization
Distribution plots for each feature (SF1–SF7)

Boxplots for outlier visualization

Example:

![Feature Distribution](assets/feature_distributionots](assets/feature_boxplots Model Training: XGBoost

The workflow includes splitting the data, training an XGBoost classifier, and evaluating performance:

python
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Feature selection
features = ['SF1','SF2','SF3','SF4','SF5','SF6','SF7','Day','Month','ticker']
X = data[features]
y = data['alpha']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model training
model = xgb.XGBClassifier()
model.fit(X_train, y_train)

# Evaluation
y_pred = model.predict(X_test)
print('Accuracy:', accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
📋 Outputs & Results
Accuracy Score: Printed after model evaluation

Classification Report: Precision, recall, F1-score for each alpha class

Confusion Matrix: For detailed error analysis

📑 Project Structure
text
.
├── dataset.csv
├── alpha_signal_prediction.ipynb
├── README.md
└── assets/
    ├── label_countplot.png
    ├── feature_distribution.png
    └── feature_boxplots.png
🖼️ Example Visualizations
Count Plot of Alpha Labels
![Count Plot](assets/label_countplot.png Stock Factors*
![Feature Distribution](assets/feature_distributionots of Features*
![Box Plots](assets/feature_boxplots License

This project is licensed under the MIT License.

🙌 Acknowledgments
Pandas

XGBoost

Feature-engine

Seaborn

Scikit-learn


For any questions or suggestions, feel free to open an issue or submit a pull request!
