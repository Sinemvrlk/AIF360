# Adversarial Debiasing — Task 2

This project tests **adversarial debiasing**, an in-processing fairness method from [Zhang et al. (2018)](https://arxiv.org/abs/1801.07593). It uses the existing [AIF360 implementation](https://github.com/Trusted-AI/AIF360) in [my fork](https://github.com/Sinemvrlk/AIF360). The method trains a predictor while an adversary tries to identify the protected group from its output.

We use the **demographic parity** version of the algorithm. The paper's main Adult experiment instead focused on **equalized odds**, so our results are not a direct reproduction.

## Datasets

| Dataset | Prediction task | Protected attribute | Privileged group |
| --- | --- | --- | --- |
| Adult | Income above $50K | Sex | Male |
| German Credit | Credit outcome | Age | Older group |
| COMPAS | Recidivism-related outcome | Race | Caucasian |

## Running the notebook

1. Open the experiment notebook in [Google Colab](https://colab.research.google.com/).
2. Run the installation cell:

   ```python
   %pip install -q "aif360[AdversarialDebiasing]==0.6.1" "scikit-learn==1.5.2" pandas numpy
   ```

3. Restart the session if needed, then run the remaining cells in order.
4. Download `fairness_results.csv` from the last cell.

## Experiment

For each dataset, we calculate fairness on the **original labels**, then make an **80/20 stratified train/test split** (seed 42). After preparing the features, we train a baseline model and a model with adversarial debiasing, using 30 epochs and batch size 128. Both models are evaluated on the same held-out test set.

We report **DP_Diff** (best near 0), **DIR** (best near 1), **consistency** (best near 1, based on five nearest neighbors), and **accuracy** (higher is better). The original-label results use the full dataset, while model results use only the test set.

## Results

| Dataset | Stage | DP_Diff | DIR | Consistency | Accuracy |
| --- | --- | ---: | ---: | ---: | ---: |
| Adult | Original | -0.1989 | 0.3635 | 0.8175 | — |
| Adult | Baseline | -0.1884 | 0.3228 | 0.9341 | 0.8506 |
| Adult | Mitigated | -0.0945 | 0.5815 | 0.9339 | 0.8438 |
| German | Original | -0.1494 | 0.7948 | 0.6020 | — |
| German | Baseline | -0.0858 | 0.8923 | 0.7880 | 0.8000 |
| German | Mitigated | -0.0689 | 0.9085 | 0.7810 | 0.8100 |
| COMPAS | Original | -0.1321 | 0.7830 | 0.5873 | — |
| COMPAS | Baseline | -0.2069 | 0.6919 | 0.8129 | 0.6626 |
| COMPAS | Mitigated | -0.1983 | 0.7118 | 0.8064 | 0.6654 |

## Conclusion and limitations

Compared with the baseline, demographic parity improved on all three datasets. Adult improved the most, with a small accuracy decrease. German also gained accuracy, while COMPAS showed only a small fairness improvement. Consistency fell slightly in all three cases. This supports the paper's general idea that adversarial training can reduce group disparities, although the fairness objective differs.

These are the notebook's reported results, **not independently verified**. The positive-label mappings and original-label consistency calculation still need checking. Results may also change with another random split, and the algorithm does not directly optimize consistency.

**Credit:** The algorithm and library come from Zhang et al. and AIF360. The experiment and evaluation are additions for this assignment; the original project's license and notices are kept.
