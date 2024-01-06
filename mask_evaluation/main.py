import cv2
import numpy as np
from numpy import ndarray
from collections import Counter
from dataclasses import dataclass


TRUE_POSITIVE = (True, True)
TRUE_NEGATIVE = (True, False)
FALSE_POSITIVE = (False, True)
FALSE_NEGATIVE = (False, False)


@dataclass(init=True, repr=True, eq=True, frozen=True)
class Evaluation:
    """The full evaluation result of all input masks"""
    TPR: float
    """True positive rate or sensitivity or recall in range 0 to 1"""
    FNR: float
    """False negative rate in range 0 to 1"""
    TNR: float
    """True negative rate or specificity in range 0 to 1"""
    FPR: float
    """False positive rate in range 0 to 1"""
    PPV: float
    """Positive predictive value or precision in range 0 to 1"""
    FDR: float
    """False discovery rate in range 0 to 1"""
    NPV: float
    """Negative predictive value in range 0 to 1"""
    FOR: float
    """False omission rate in range 0 to 1"""
    ACC: float
    """Accuracy in range 0 to 1"""
    F1: float
    """F1-Score in range 0 to 1"""
    sensitivity: float
    """Sensitivity in range 0 to 1"""
    recall: float
    """Recall in range 0 to 1"""
    specificity: float
    """Specificity in range 0 to 1"""
    precision: float
    """Precision in range 0 to 1"""
    accuracy: float
    """Accuracy in range 0 to 1"""


def _evaluate(counter: Counter) -> Evaluation:
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

    return Evaluation(TPR, FNR, TNR, FPR, PPV, FDR, NPV, FOR, ACC, f1, sensitivity, recall, specificity, precision, accuracy)


def _count_mask(base_mask: ndarray, actual: ndarray) -> Counter:
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


def _count_file(base_mask_path: str, predicted_mask_path: str) -> Counter:
    expected = cv2.imread(base_mask_path, cv2.IMREAD_GRAYSCALE)
    actual = cv2.imread(predicted_mask_path, cv2.IMREAD_GRAYSCALE)
    return _count_mask(expected, actual)


def evaluate_mask(base_mask: ndarray, predicted_mask: ndarray) -> Evaluation:
    """Evaluates 2 in memory images"""
    counts = _count_mask(base_mask, predicted_mask)
    return _evaluate(counts)


def evaluate_file(base_mask_path: str, predicted_mask_path: str) -> Evaluation:
    """Evaluates 2 image files"""
    counts = _count_file(base_mask_path, predicted_mask_path)
    return _evaluate(counts)


def evaluate_masks(masks: list[tuple[ndarray, ndarray]]) -> Evaluation:
    """
    Evaluates a list of in memory images. Tuples are (base_mask, predicted_mask)

    Parameters:
        masks (list[tuple[ndarray, ndarray]]): a list of image tuples (base_mask, predicted_mask)
    """
    counts = sum([_count_mask(tup[0], tup[1]) for tup in masks], start=Counter())
    return _evaluate(counts)


def evaluate_files(masks: list[tuple[str, str]]) -> Evaluation:
    """
    Evaluates a list images. Tuples are (base_mask_path, predicted_mask_path)

    Parameters:
        masks (list[tuple[ndarray, ndarray]]): a list of image tuples (base_mask_path, predicted_mask_path)
    """
    counts = sum([_count_file(tup[0], tup[1]) for tup in masks], start=Counter())
    return _evaluate(counts)


def print_metrics(metrics: Evaluation, file=None):
    output = []
    output.append(f'Results: ')
    output.append(f'   accuracy: {metrics.accuracy * 100:.3f} %')
    output.append(f'         f1:  {metrics.F1:.3f}')
    output.append(f'     recall:  {metrics.recall:.3f} (or sensitivity)')
    output.append(f'  precision:  {metrics.precision:.3f}')
    output.append(f'specificity:  {metrics.specificity:.3f}')
    output.append('')
    output.append(f'TPR: {metrics.TPR:.3f} | FNR: {metrics.FNR:.3f}')
    output.append(f'TNR: {metrics.TNR:.3f} | FPR: {metrics.FPR:.3f}')
    output.append(f'PPV: {metrics.PPV:.3f} | FDR: {metrics.FDR:.3f}')
    output.append(f'NPV: {metrics.NPV:.3f} | FOR: {metrics.FOR:.3f}')

    if file is not None:
        for line in output:
            print(line, file=file)
    else:
        for line in output:
            print(line)


if __name__ == '__main__':
    # Examples:

    # One mask and one prediction:
    # res = evaluate_mask_file("base_mask.png", "actual_mask.png")

    # Multi masks with predictions:
    data = [
        ("base_mask.png", "actual_mask.png"),
        ("base_mask.png", "actual_mask.png"),
        ("base_mask.png", "actual_mask.png"),
        ("base_mask.png", "actual_mask.png")
    ]
    res = evaluate_files(data)
    print_metrics(res)
