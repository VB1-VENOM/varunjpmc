# Census Bureau Income Classification and Customer Segmentation
##Author: Varun Satheesh

This project implements two machine learning models for a retail business client:
1. **Classification Model**: Predicts whether a person earns more than $50,000 or less than/equal to $50,000
2. **Segmentation Model**: Creates customer segments for targeted marketing

## Project Structure

```
.
├── census-bureau.data          # Census data file
├── census-bureau.columns        # Column names file
├── load_data.py                # Data loading and preprocessing utilities
├── classification_model.py     # Classification model implementation
├── segmentation_model.py       # Segmentation/clustering model implementation
├── run_all.py                  # Main script to run both models
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

## Requirements

- Python 3.7 or higher
- Required packages (see `requirements.txt`):
  - pandas
  - numpy
  - scikit-learn
  - matplotlib
  - seaborn
  - joblib

## Installation

1. Clone or download this repository

2. Install required packages:
```bash
pip install -r requirements.txt
```

## Usage

### Option 1: Run Both Models (Recommended)

Run both the classification and segmentation models sequentially:

```bash
python run_all.py
```

### Option 2: Run Models Individually

**Run Classification Model Only:**
```bash
python classification_model.py
```

**Run Segmentation Model Only:**
```bash
python segmentation_model.py
```

## Output Files

After running the models, the following files will be generated:

### Classification Model Outputs:
- `classification_model.pkl`: Trained classification model (Random Forest)
- `classification_results.csv`: Model performance metrics (accuracy, precision, recall, F1-score, ROC-AUC)
- `feature_importance.csv`: Top 20 most important features for income prediction
- `label_encoders.pkl`: Label encoders for categorical features
- `scaler.pkl`: StandardScaler for numerical features
- `feature_names.pkl`: List of feature names

### Segmentation Model Outputs:
- `segmentation_model.pkl`: Trained K-means clustering model
- `cluster_analysis.csv`: Detailed characteristics of each customer segment
- `cluster_visualization.png`: 2D visualization of clusters using PCA
- `marketing_insights.txt`: Marketing recommendations for each segment
- `data_with_clusters.csv`: Original data with assigned cluster labels
- `segmentation_label_encoders.pkl`: Label encoders for categorical features
- `segmentation_scaler.pkl`: StandardScaler for features
- `segmentation_feature_names.pkl`: List of feature names

## Model Details

### Classification Model

The classification model uses a **Random Forest Classifier** to predict income levels. Key features:

- **Algorithm**: Random Forest with 100 trees
- **Class Balancing**: Uses `class_weight='balanced'` to handle imbalanced data (93.8% <=$50k, 6.2% >$50k)
- **Preprocessing**:
  - Handles missing values (replaces '?' and NaN)
  - Encodes categorical variables using Label Encoding
  - Scales numerical features using StandardScaler
  - Removes weight and year columns (not predictive features)
- **Evaluation**: Provides accuracy, precision, recall, F1-score, ROC-AUC, and confusion matrix
- **Feature Importance**: Identifies top 20 most important features for income prediction

### Segmentation Model

The segmentation model uses **K-means Clustering** to create customer segments. Key features:

- **Algorithm**: K-means clustering
- **Optimal Clusters**: Automatically determines optimal number of clusters using silhouette score
- **Preprocessing**:
  - Samples 20,000 records for computational efficiency (can be adjusted)
  - Same preprocessing as classification model
- **Analysis**:
  - Analyzes key demographic and employment features for each segment
  - Calculates income distribution within each segment
  - Generates marketing insights and recommendations
- **Visualization**: Creates 2D PCA visualization of clusters

## Data Information

The dataset contains:
- **199,523 observations** with 40 demographic and employment variables
- **Target variable**: Income label (`- 50000.` for <=$50k, `50000+.` for >$50k)
- **Weight column**: Population representation weights (not used as a feature)
- **Year column**: Survey year (not used as a feature)

## Key Features Analyzed

The models analyze various demographic and employment features including:
- Age
- Education level
- Marital status
- Race and ethnicity
- Sex
- Employment status
- Capital gains/losses
- Weeks worked per year
- Wage per hour
- Industry and occupation codes
- And many more...

## Marketing Applications

The segmentation model provides actionable insights for marketing:

1. **Segment Identification**: Identifies distinct customer groups based on demographics and employment
2. **Income Distribution**: Shows income distribution within each segment
3. **Targeting Recommendations**: Suggests appropriate products/services for each segment:
   - Premium products for high-income segments
   - Mid-range products for moderate-income segments
   - Budget-friendly products for lower-income segments
4. **Personalization**: Enables personalized marketing campaigns based on segment characteristics

## Performance Notes

- The classification model handles class imbalance using balanced class weights
- The segmentation model samples data for efficiency but can be adjusted in the code
- Both models use cross-validation and standard evaluation metrics
- All preprocessing steps are saved for future predictions on new data

## Customization

You can customize the models by modifying:

- **Number of clusters**: Change `max_k` parameter in `segmentation_model.py`
- **Sample size**: Adjust `n_samples` in `preprocess_for_clustering()` function
- **Model parameters**: Modify hyperparameters in `train_classification_model()` function
- **Features analyzed**: Update `key_features` list in `analyze_clusters()` function

## Troubleshooting

If you encounter memory issues:
- Reduce the sample size in `segmentation_model.py` (currently 20,000)
- Reduce `max_k` for cluster optimization

If you encounter import errors:
- Ensure all packages in `requirements.txt` are installed
- Check Python version (3.7+ required)


