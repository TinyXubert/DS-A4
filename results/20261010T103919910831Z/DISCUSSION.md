## Interpretation of this run

The strong simple baseline, Logistic, wins the development comparison. Its calibrated Brier score (0.0248) is higher than its raw score (0.0209); calibration did not improve this held-out probability loss. The calibrated policy is retained because it was specified before test evaluation. Replacing it after this result would require new evaluation data.

Sharpening leaves ROC AUC unchanged but changes Brier from 0.0248 to 0.0278 and log loss from 0.0766 to 0.1268. Good ranking therefore does not certify trustworthy probabilities.

The decision-set threshold is high (0.999711) and the final policy misses 11 positives out of 32, despite excellent ranking. The hypothetical review constraint is consequential, and small-sample threshold uncertainty is not captured by model-metric bootstrap intervals. A fixed threshold can also leave capacity unused; the predefined rule caps excess demand but does not force the batch to fill every review slot. Neither this example nor its costs justify a clinical workflow.
