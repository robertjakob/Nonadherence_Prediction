# Vivira & Manoa Nonadherence Prediction Repository

This repository contains analysis code for studying non-adherence and churn prediction models in two mobile health interventions (Manoa & Vivira).

## Repository Structure

```
.
├── Manoa Code/
│   ├── manoa_nonadherence.py
│   └── manoa_churn.py
├── Vivira Code/
    ├── vivira_nonadherence.py
    └── vivira_churn.py

```

## Data Description

### Input Data Files
- `manoa.csv`: Contains complete user data for Manoa application (used for churn analysis)
- `manoa_filtered.csv`: Contains user data for Manoa application users who completed the initial blood pressure measurement week (used for non-adherence analysis)
- `vivira.csv`: Contains user data for Vivira application (used for both non-adherence and churn analysis)

### Key Data Points

#### User Activity Metrics
- **Manoa Features**:
  - Login Data: Daily login counts (`logins1` to `loginsN`)
  - Active Usage: Daily active usage indicators (`active1` to `activeN`)

- **Vivira Features**:
  - Exercise Data: Daily exercise counts (`exercise1` to `exerciseN`)
  - Active Usage: Daily active usage indicators (`active1` to `activeN`)

#### Target Variables
- **Nonadherence Prediction**:
  - Manoa: `NonAdhMonth2` to `NonAdhMonth6`: Binary indicators for non-adherence in months 2-6
  - Vivira: `NonAdhWeek2` to `NonAdhWeek13`: Binary indicators for non-adherence in weeks 2-13

- **Churn Prediction**:
  - Manoa: `Churn8` to `Churn187`: Binary indicators for churn from day 8 to day 187
  - Vivira: `Churn8` to `Churn90`: Binary indicators for churn from day 8 to day 90
  - Each churn indicator represents whether a user has churned by that specific day

## Prediction Components

### Non-adherence Prediction
- **Manoa**:
  - Predicts user nonadherence over months 2-6
  - Uses login and active usage patterns as features
  - Only features before the prediction window are used

- **Vivira**:
  - Predicts user nonadherence over weeks 2-13
  - Uses exercise and active usage patterns as features
  - Only features before the prediction window are used

### Churn Prediction
- **Manoa**:
  - Analyzes user churn patterns on a daily basis from day 8 to day 187
  - Uses login and active usage patterns as features
  - Only features before the prediction window are used

- **Vivira**:
  - Analyzes user churn patterns on a daily basis from day 8 to day 90
  - Uses exercise and active usage patterns as features
  - Only features before the prediction window are used

### Common Prediction Features
- Both applications use Random Forest Classifier with hyperparameter tuning
- Both implement Tomek Links undersampling for balanced classification
- Both output comprehensive metrics including accuracy, F1 score, AUC, precision, and recall
- Both categorize results by user status:
  - Churned users (Last Login before prediction window)
  - Churning users (Last Login within prediction window)
  - Active users (Last Login after prediction window)

## Technical Details

### Dependencies
- Python 3.x
- pandas
- numpy
- scikit-learn
- imbalanced-learn

### Model Features
- Preprocessing: Square root transformation and standardization
- Cross-validation: 10-fold stratified cross-validation
- Hyperparameter tuning: Randomized search with 20 iterations
- Evaluation metrics: Accuracy, F1 Score, AUC, Precision, Recall

### Output
- Detailed metrics for each prediction period
- Confusion matrices
- Feature importance rankings
- Results categorized by user activity status (Churned, Churning, Active)

## Usage

1. Ensure all required Python packages are installed
2. Place the input data files in the appropriate directories:
   - `manoa.csv` for Manoa churn analysis
   - `manoa_filtered.csv` for Manoa non-adherence analysis
   - `vivira.csv` for Vivira analysis (both non-adherence and churn)
3. Run the analysis scripts:
   ```bash
   python Manoa\ Code/manoa_nonadherence.py
   python Manoa\ Code/manoa_churn.py
   python Vivira\ Code/vivira_nonadherence.py
   python Vivira\ Code/vivira_churn.py
   ```

## Results

Results are saved in text files:
- `ManoaNonadherence.txt`: Contains non-adherence analysis results for months 2-6
- `ManoaChurn.txt`: Contains churn analysis results for days 8-187
- `ViviraNonadherence.txt`: Contains non-adherence analysis results for weeks 2-13
- `ViviraChurn.txt`: Contains churn analysis results for days 8-90

Each output file contains:
- Detailed metrics for each prediction period
- Confusion matrices
- Feature importance rankings
- Results categorized by user activity status (Churned, Churning, Active)
- Best model parameters from hyperparameter tuning 
