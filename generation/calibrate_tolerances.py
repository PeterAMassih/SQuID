"""Computes human-agreement statistics from the 500 NAIP annotations.

For each numeric question family (percentage, proximity, count) it reports the
per-item median absolute deviation (MAD), its normalized form for counts
(MADc = MAD / median), ICC(2,k) and Krippendorff's alpha. The ``tolerances``
entry holds the values ``build_benchmark.py`` uses: the median MADs for
percentages, and the adjusted MADc of 0.19 for counts, as in the paper.

Usage:
    python generation/calibrate_tolerances.py    # -> generation/agreement_metrics.json
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
COUNT_MADC = 0.19


def question_family(question):
    """Returns 'proximity', 'percentage', 'count', 'binary' or 'other'."""
    q = str(question).lower()
    if 'within' in q and 'm of' in q:
        return 'proximity'
    if 'what percentage' in q:
        return 'percentage'
    if 'how many' in q:
        return 'count'
    if q.startswith('is'):
        return 'binary'
    return 'other'


def mad(values):
    """Returns median(|x - median(x)|), or NaN for fewer than two values."""
    values = np.asarray(values, float)
    values = values[~np.isnan(values)]
    if values.size < 2:
        return np.nan
    return np.median(np.abs(values - np.median(values)))


def krippendorff_alpha(ratings):
    """Krippendorff's alpha for interval data.

    Args:
        ratings: ``(items, raters)`` array with NaN for missing ratings.

    Returns:
        ``1 - D_o / D_e``, the observed over expected disagreement.
    """
    sq_diff, pairs = 0.0, 0
    for row in ratings:
        values = row[~np.isnan(row)]
        if values.size >= 2:
            diffs = (values[:, None] - values[None, :])[np.triu_indices(values.size, 1)] ** 2
            sq_diff += float(diffs.sum())
            pairs += diffs.size
    if pairs == 0:
        return np.nan
    pooled = ratings.ravel()
    pooled = pooled[~np.isnan(pooled)]
    if pooled.size < 2:
        return np.nan
    expected = 2.0 * float(np.mean((pooled - pooled.mean()) ** 2))
    if expected == 0:
        return 1.0 if sq_diff == 0 else 0.0
    return float(1.0 - (sq_diff / pairs) / expected)


def icc_2k(ratings):
    """ICC(2,k): two-way random effects, absolute agreement, average measures.

    Follows Koo & Li (2016). Items and raters with fewer than two ratings are
    dropped first.

    Args:
        ratings: ``(items, raters)`` array with NaN for missing ratings.

    Returns:
        The ICC, or NaN when fewer than two items or raters remain.
    """
    df = pd.DataFrame(ratings).dropna(thresh=2).dropna(axis=1, thresh=2)
    n, k = df.shape
    if n < 2 or k < 2:
        return np.nan
    grand = df.stack().mean()
    ss_rows = k * ((df.mean(axis=1) - grand) ** 2).sum()
    ss_cols = n * ((df.mean(axis=0) - grand) ** 2).sum()
    ss_error = ((df.fillna(grand) - grand).to_numpy() ** 2).sum() - ss_rows - ss_cols
    ms_rows = ss_rows / (n - 1)
    ms_cols = ss_cols / (k - 1)
    ms_error = ss_error / ((n - 1) * (k - 1))
    denom = ms_rows + (k - 1) * ms_error + (k / n) * (ms_cols - ms_error)
    return np.nan if denom == 0 else float((ms_rows - ms_error) / denom)


def family_metrics(df, family):
    """Computes the agreement statistics of one question family.

    Args:
        df: Numeric annotations, one row per (item, annotator).
        family: Question family, see :func:`question_family`.

    Returns:
        A dict of statistics, or None if the family has too few items.
    """
    subset = df[df['Input.question'].apply(question_family) == family].copy()
    if subset.empty:
        return None
    subset['item'] = subset['Input.image'].astype(str) + ' | ' + subset['Input.question']
    pivot = subset.pivot_table(index='item', columns='Turkle.Username', values='Answer.answer', aggfunc='mean')
    if pivot.shape[0] < 2:
        return None

    mads, madcs, medians = [], [], []
    for _, row in pivot.iterrows():
        values = row.dropna().values
        if len(values) >= 3:
            item_mad, item_median = mad(values), np.median(values)
            mads.append(item_mad)
            medians.append(item_median)
            if item_median > 0:
                madcs.append(item_mad / item_median)
    if not mads:
        return None

    ratings = pivot.to_numpy()
    result = {
        'n_items': int(pivot.shape[0]),
        'n_raters': int(pivot.shape[1]),
        'mad_mean': float(np.mean(mads)),
        'mad_median': float(np.median(mads)),
        'mad_std': float(np.std(mads)),
        'median_value': float(np.median(medians)),
        'icc_2k': icc_2k(ratings),
        'krippendorff_alpha': krippendorff_alpha(ratings),
    }
    if family == 'count' and madcs:
        result.update(madc_mean=float(np.mean(madcs)), madc_median=float(np.median(madcs)),
                      madc_std=float(np.std(madcs)))
    return result


def main():
    df = pd.read_csv(HERE / 'human_annotations_anonymized.csv')
    df = df[df['Answer.answer_type'] == 'numeric'].copy()
    df['Answer.answer'] = pd.to_numeric(df['Answer.answer'], errors='coerce')
    df = df.dropna(subset=['Answer.answer'])

    results = {}
    print(f"{'family':<11} {'items':>5} {'MAD med':>8} {'ICC(2,k)':>9} {'alpha':>7}")
    for family in ['percentage', 'proximity', 'count']:
        metrics = family_metrics(df, family)
        if metrics:
            results[family] = metrics
            print(f"{family:<11} {metrics['n_items']:>5} {metrics['mad_median']:>8.3f} "
                  f"{metrics['icc_2k']:>9.3f} {metrics['krippendorff_alpha']:>7.3f}")

    results['tolerances'] = {'percentage': results['percentage']['mad_median'],
                             'proximity': results['proximity']['mad_median'],
                             'count_madc': COUNT_MADC}
    with open(HERE / 'agreement_metrics.json', 'w') as f:
        json.dump(results, f, indent=2, default=float)


if __name__ == '__main__':
    main()
