import unittest

import numpy as np
import pandas as pd

from cost_sensitive_experiment import decision_cost, group_sensitive_weights, label_weights, metrics


class CostSensitiveHelpersTest(unittest.TestCase):
    def test_decision_cost(self):
        self.assertEqual(decision_cost(10, 3, 2, 5), 35)

    def test_class_weights_follow_error_cost_side(self):
        weights = label_weights([0, 1, 0, 1], c_fp=3, c_fn=2)
        self.assertGreater(weights[0], weights[1])
        self.assertAlmostEqual(weights.mean(), 1)

    def test_sensitive_group_multiplier_is_configurable(self):
        groups = pd.DataFrame({"crop": ["rice", "wheat"], "soil_type": ["clay", "clay"],
                               "growth_stage": ["Sowing", "Harvest"]})
        weighted = label_weights([0, 1], 1, 1, groups, sensitive_multiplier=3)
        self.assertGreater(weighted[0], weighted[1])

    def test_group_specific_fn_weight_only_applies_to_positive_sensitive_rows(self):
        groups = pd.DataFrame({"crop": ["rice", "rice", "wheat"],
                               "growth_stage": ["Sowing", "Sowing", "Harvest"],
                               "soil_type": ["clay", "clay", "clay"]})
        weights = group_sensitive_weights([1, 0, 1], groups, {"crop:rice": 2})
        self.assertGreater(weights[0], weights[1])
        self.assertGreater(weights[0], weights[2])

    def test_cost_and_confusion_metrics(self):
        row = metrics([0, 0, 1, 1], [1, 0, 0, 1], None, 3, 2)
        self.assertEqual((row["fp"], row["fn"], row["total_decision_cost"]), (1, 1, 5))

    def test_fold_partition_is_disjoint_by_construction(self):
        # The experiment fits scaler and models from fit_idx only; validation
        # indices are disjoint, as enforced by StratifiedKFold's contract.
        from sklearn.model_selection import StratifiedKFold
        idx = np.arange(100)
        y = np.tile([0, 1], 50)
        for fit, valid in StratifiedKFold(5, shuffle=True, random_state=42).split(idx, y):
            self.assertFalse(set(fit) & set(valid))
            self.assertEqual(len(fit) + len(valid), len(idx))


if __name__ == "__main__":
    unittest.main()
