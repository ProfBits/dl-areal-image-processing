import cv2
import numpy as np
from numpy import ndarray
from collections import Counter

TRUE_POSITIVE = (True, True)
TRUE_NEGATIVE = (True, False)
FALSE_POSITIVE = (False, True)
FALSE_NEGATIVE = (False, False)


def evaluate_mask(base_mask: ndarray, actual: ndarray) -> Counter:
    # The weights are carefully chosen such that the 4 possible sums result in values that
    # can be differentiated by looking at one specific bit.
    # For False 0 is used for True base uses 54 and actual 99
    # true_negative  ->  0 +  0 = 0b0000_0000 + 0b0000_0000 = 0b0000_0000 -> all zero -> equals count - non_zeros
    # true_positive  -> 54 + 99 = 0b0011_0110 + 0b0110_0011 = 0b1001_1001 -> unique 1 at bit 4
    # false_positive ->  0 + 99 = 0b0000_0000 + 0b0110_0011 = 0b0110_0011 -> unique 1 at bit 7
    # false_negative -> 54 +  0 = 0b0011_0110 + 0b0000_0000 = 0b0011_0110 -> unique 1 at bit 3
    weighted_sum = cv2.addWeighted(base_mask, 0.211765, actual, 0.388235, 0)

    # use the unique bits from above to filter with bitwise & and count none zero values
    # this is done instead of np.unique(array, return_counts=True) as unique count is very slow and all values
    # are known in advance.
    return Counter({
        TRUE_NEGATIVE: np.prod(weighted_sum.shape) - np.count_nonzero(weighted_sum),
        TRUE_POSITIVE:  np.count_nonzero(np.bitwise_and(weighted_sum, 0b0000_1000)),
        FALSE_POSITIVE: np.count_nonzero(np.bitwise_and(weighted_sum, 0b0100_0000)),
        FALSE_NEGATIVE: np.count_nonzero(np.bitwise_and(weighted_sum, 0b0000_0100))
    })


def evaluate_mask_file(base_mask_path: str, predicted_mask_path: str) -> Counter:
    expected = cv2.imread(base_mask_path, cv2.IMREAD_GRAYSCALE)
    actual = cv2.imread(predicted_mask_path, cv2.IMREAD_GRAYSCALE)
    return evaluate_mask(expected, actual)


def print_metrics(counter: Counter):
    # Formulas based on wikipedia pages binary classification and f-score
    TPR = sensitivity = recall = counter[TRUE_POSITIVE] / (counter[TRUE_POSITIVE] + counter[FALSE_NEGATIVE])
    FNR = counter[FALSE_NEGATIVE] / (counter[TRUE_POSITIVE] + counter[FALSE_NEGATIVE])

    TNR = specificity = counter[TRUE_NEGATIVE] / (counter[TRUE_NEGATIVE] + counter[FALSE_POSITIVE])
    FPR = independent_of_prevalence = counter[FALSE_POSITIVE] / (counter[TRUE_NEGATIVE] + counter[FALSE_POSITIVE])

    PPV = precision = counter[TRUE_POSITIVE] / (counter[TRUE_POSITIVE] + counter[FALSE_POSITIVE])
    FDR = counter[FALSE_POSITIVE] / (counter[TRUE_POSITIVE] + counter[FALSE_POSITIVE])

    NPV = counter[TRUE_NEGATIVE] / (counter[TRUE_NEGATIVE] + counter[FALSE_NEGATIVE])
    FOR = dependence_on_prevalence = counter[FALSE_NEGATIVE] / (counter[TRUE_NEGATIVE] + counter[FALSE_NEGATIVE])

    ACC = accuracy = (counter[TRUE_POSITIVE] + counter[TRUE_NEGATIVE]) / counter.total()
    f1 = 2 * precision * recall / (precision + recall)

    print(f'Results: ')
    print(f'   accuracy: {accuracy * 100:.3f} %')
    print(f'         f1:  {f1:.3f}')
    print(f'     recall:  {recall:.3f} (or sensitivity)')
    print(f'  precision:  {precision:.3f}')
    print(f'specificity:  {specificity:.3f}')
    print(f'')
    print(f'TPR: {TPR:.3f} | FNR: {FNR:.3f}')
    print(f'TNR: {TNR:.3f} | FPR: {FPR:.3f}')
    print(f'PPV: {PPV:.3f} | FDR: {FDR:.3f}')
    print(f'NPV: {NPV:.3f} | FOR: {FOR:.3f}')


if __name__ == '__main__':
    # One mask and one prediction:
    # res = evaluate_mask_file("base_mask.png", "actual_mask.png")

    # Multi masks with predictions:
    data = [
        ("base_mask.png", "actual_mask.png"),
        ("base_mask.png", "actual_mask.png"),
        ("base_mask.png", "actual_mask.png"),
        ("base_mask.png", "actual_mask.png")
    ]
    res = sum([evaluate_mask_file(base, actual) for base, actual in data], Counter())
    print_metrics(res)
