# Binomial Confidence Intervals Visualizer

A Streamlit app for visualizing confidence intervals for TPR (True Positive Rate) and TNR (True Negative Rate) across different sample sizes.

## Features

- Interactive sliders to adjust:
  - TPR and TNR values
  - Prevalence
  - Sample size range
  - Confidence level (via alpha)
  - Number of points to calculate

- Three-panel visualization:
  - TPR confidence intervals
  - TNR confidence intervals
  - Width of confidence intervals

## Installation

1. Clone this repository
2. Install the required packages:

```
pip install -r requirements.txt
```

## Usage

Run the Streamlit app:

```
streamlit run app.py
```

Then access the app in your web browser (typically at http://localhost:8501).

## Parameters

- **TPR**: True Positive Rate (Sensitivity) - The proportion of actual positives correctly identified
- **TNR**: True Negative Rate (Specificity) - The proportion of actual negatives correctly identified
- **Prevalence**: The proportion of positive cases in the population
- **Sample Size**: Min and max total sample size to visualize
- **Significance Level (α)**: Determines the confidence level (1-α) * 100%
- **Number of Sample Points**: Number of points to calculate along the sample size range 