# Method note

Compare probability quality and a hypothetical review policy using the UCI Wisconsin Diagnostic Breast Cancer data. Source: https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic, Wolberg, Mangasarian, Street and Street (1993), DOI 10.24432/C5DW2B, CC BY 4.0. This is a statistical teaching exercise on existing anonymized numeric records, not a clinical decision system.

## 02 · Disjoint roles and paired model search

Malignancy is the positive label in this historical diagnostic-image dataset. The split reserves separate data for fitting, sigmoid calibration, policy choice and final evaluation. IDs are excluded. The three families receive the same five stratified training folds and three hyperparameter candidates each (15 fits per family), with average precision (AP) as the selection metric. This equal fit-count budget is not equal computational cost. Logistic regression is the strong simple baseline; tree ensembles are alternatives, not presumed improvements.

## 03 · Calibration without reuse of policy/test data

Each selected candidate is fitted on training rows, then a sigmoid maps its log-odds to probabilities using the calibration rows alone. The sigmoid is implemented by a near-unpenalized one-variable logistic model; it is not chosen from several calibration methods. No training or calibration refit uses the policy or test partitions. Calibration can worsen a small-sample estimate; both raw and calibrated probabilities are retained.

## 04 · Paired uncertainty and a ranking counterexample

Stratified paired bootstrap draws preserve the held-out positive and negative counts and reuse identical sampled rows across models and probability versions. Percentile intervals describe test-sample variability conditional on fitted models and this class balance; they exclude training/tuning uncertainty. Differences are model minus Logistic, so smaller Brier/ECE is better but larger AP/AUC is better. ECE depends on bin choice. An intentionally sharpened probability map preserves rank while changing calibration and probability loss.

## 05 · Cost, review capacity and threshold sensitivity

All costs are hypothetical teaching units, not medical utilities. A missed positive costs 5 and a false alarm costs 1; at most 30% of a batch can be reviewed. On the decision partition, enumerate score thresholds including reject-all, then minimize cost subject to the review cap, breaking ties toward fewer reviews. Lock that rule before test scoring. A changed score distribution can exceed capacity at the fixed threshold: the deployable rule also caps flagged cases to the highest scores, with stable row-order ties. Sensitivity changes only decision-set costs/capacity.

## 06 · Error slices and claim boundary

Mean-radius quartile boundaries are frozen from training features. Error rates include denominators, because the held-out sample is small. These are morphology slices, not demographic fairness evidence. High discrimination on historical biopsy images does not establish safe clinical use, general population screening performance or capacity in a different institution. The threshold is an illustrative resource allocation exercise, and its small decision partition creates substantial policy uncertainty.
