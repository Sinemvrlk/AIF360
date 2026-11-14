# Assignment 4 – Adversarial Debiasing

This project evaluates **adversarial debiasing**, the bias-mitigation algorithm introduced by Zhang, Lemoine and Mitchell (2018). The experiments use the existing implementation from IBM's **AI Fairness 360 (AIF360)** library.

The experiment notebook is in the main folder of this repository: **[Assignment_4.ipynb](Assignment_4.ipynb)**.

## Datasets

The notebook uses three datasets:

- **Adult Income:** predicts income, with sex as the protected attribute.
- **German Credit:** predicts credit outcomes, with age as the protected attribute.
- **COMPAS:** predicts recidivism-related outcomes, with race as the protected attribute.

## How to run

1. Open [Assignment_4.ipynb in Google Colab](https://colab.research.google.com/github/Sinemvrlk/AIF360/blob/main/Assignment_4.ipynb).
2. Run the installation cell first. Restart the Colab session if error is given.
3. Run the remaining cells in order, or choose **Runtime → Run all** after installation.
4. View the results table at the end of the notebook.

The notebook needs Python, AIF360, TensorFlow, pandas, NumPy and scikit-learn. The installation cell contains the package commands used for the experiment.

## Method

First, the notebook calculates fairness from the original dataset labels. It then splits the data into training and test sets, using an 80/20 split. For each dataset, it trains a model without debiasing and another with adversarial debiasing. The predictions are evaluated on the held-out test set.

The evaluation uses four measures:

- **DP_Diff:** difference in positive outcome rates between protected groups; closer to 0 is better.
- **DIR:** ratio of positive outcome rates; closer to 1 is better.
- **Consistency:** similarity of outcomes for people with similar features; closer to 1 is better.
- **Accuracy:** percentage of correct predictions.

## Results and discussion

The notebook compares the original labels, baseline predictions and debiased predictions for all three datasets. In the results discussed for this assignment, demographic parity improved after mitigation, particularly for the Adult dataset, while consistency did not necessarily improve. This illustrates that improving group fairness does not automatically improve individual fairness.

The original paper's main Adult experiment focused on **equalized odds**. This notebook instead uses the AIF360 demographic-parity implementation because the assignment specifically requires DP_Diff and DIR. 

## References

- Zhang, B. H., Lemoine, B., & Mitchell, M. (2018). *Mitigating Unwanted Biases with Adversarial Learning*. https://arxiv.org/abs/1801.07593
- IBM AI Fairness 360: https://github.com/Trusted-AI/AIF360
