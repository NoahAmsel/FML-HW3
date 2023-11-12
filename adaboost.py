from itertools import chain, pairwise

# import matplotlib.pyplot as plt

# from joblib import Parallel, delayed
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import KFold
from tqdm import tqdm


class Stumps:
    def __init__(self, X_df, y):
        """
        X: DataFrame of features. Each row is a point
        y: array of boolean labels
        """
        y_sign = 2 * y - 1  # {-1, +1}
        Xy_df = X_df.copy()
        col_names = list(Xy_df)
        assert "y_sign" not in col_names
        Xy_df["y_sign"] = y_sign
        self.col_sorts = {
            col: Xy_df[[col, "y_sign"]].sort_values(col) for col in col_names
        }

    def _threshold_error_iterable(self, col, weights):
        col_sorts = self.col_sorts[col]
        weights = weights[col_sorts.index]  # reorder
        threshold = -np.inf  # initially, label everything -1
        total_weight = weights.sum()
        error_left = (col_sorts.y_sign == 1) @ weights  # weight of +1 labeled points
        yield (col, threshold, 1), error_left
        yield (col, threshold, -1), total_weight - error_left
        for ((threshold, y_sign_i), (next_threshold, _)), w_i in zip(
            pairwise(
                chain(col_sorts.itertuples(index=False, name=None), [(np.inf, np.nan)])
            ),
            weights,
        ):
            # we are now labeling x_i with a 1
            # if this is correct, subtract a (weighted) error
            # otherwise add one
            error_left -= y_sign_i * w_i
            # if two thresholds tie, just ignore it until we've seen all
            if threshold < next_threshold:
                yield (col, threshold, 1), error_left
                yield (col, threshold, -1), total_weight - error_left

    def best_hypothesis(self, weights):
        def best_threshold(col):
            return min(self._threshold_error_iterable(col, weights), key=lambda x: x[1])

        # return min(
        #     Parallel(n_jobs=1, return_as="generator")(
        #         delayed(best_threshold)(col) for col in self.col_sorts.keys()
        #     ),
        #     key=lambda x: x[1],
        # )
        return min(
            (best_threshold(col) for col in self.col_sorts.keys()),
            key=lambda x: x[1],
        )

    @staticmethod
    def eval_hypothesis(hypothesis, X):
        col, threshold, left_sign = hypothesis
        return (2 * (X[col] <= threshold) - 1) * left_sign


class AdaBoost:
    def __init__(self, X_df, y):
        self.X_df = X_df
        self.y_sign = 2 * y - 1
        self.stumps = Stumps(X_df, y)
        self.weights = np.full(len(X_df), 1 / len(X_df))
        self.alpha = []
        self.hs = []

    def step(self):
        h, error = self.stumps.best_hypothesis(self.weights)
        alpha = 0.5 * np.log((1 - error) / error)
        self.hs.append(h)
        self.alpha.append(alpha)
        self.weights *= np.exp(
            -alpha
            * (self.y_sign * self.stumps.eval_hypothesis(h, self.X_df)).astype(float)
        )
        self.weights /= sum(self.weights)

    def __call__(self, test_X_df):
        return sum(
            self.alpha[i] * self.stumps.eval_hypothesis(self.hs[i], test_X_df)
            for i in range(len(self.hs))
        )


if __name__ == "__main__":
    df = pd.read_csv("spambase.data.shuffled.csv", header=None)
    train_df = df[:3450].sample(frac=1).reset_index(drop=True)
    test_df = df[3450:]

    X_cols = list(df)[:-1]
    y_col = list(df)[-1]
    train_X = train_df[X_cols]
    train_y = train_df[y_col]
    test_X = test_df[X_cols]
    test_y = test_df[y_col]

    # normalize data
    # the question requires this but I don't see how it can possibly affect
    # performance of stumps
    train_X_norm = (train_X - train_X.mean()) / train_X.std()
    test_X_norm = (test_X - train_X.mean()) / train_X.std()

    validation_df = pd.DataFrame()
    for fold_i, (train_ixs, val_ixs) in enumerate(
        KFold(n_splits=10).split(train_X_norm, train_y)
    ):
        ada = AdaBoost(
            train_X_norm.loc[train_ixs].reset_index(drop=True),
            train_y.loc[train_ixs].reset_index(drop=True),
        )
        results = []
        for k in tqdm(range(10**2)):
            ada.step()
            val_preds = ada(train_X_norm.loc[val_ixs]) > 0
            results.append(sum(val_preds != train_y[val_ixs]) / len(val_ixs))
        validation_df[f"Val Err {fold_i}"] = pd.Series(results, dtype=float)

    validation_df.to_csv("cross_validation.csv", index=False)
    validation_df = pd.read_csv("cross_validation.csv")

    validation_long = pd.wide_to_long(
        validation_df.reset_index(), "Val Err ", "index", "Fold"
    ).reset_index()
    sns.lineplot(data=validation_long, x="index", y="Val Err ", errorbar="sd")


if False:
    X = pd.DataFrame(
        dict(
            x1=[0, 0, 2, 2, 4, 4, 6, 7, 7, 9, 10], x2=[2, 9, 5, 7, 4, 10, 1, 5, 7, 0, 2]
        )
    )
    y = np.array(
        [True, True, True, False, True, False, True, True, False, False, False]
    )

    # plt.scatter(x=X["x1"], y=X["x2"], c=y)

    ada = AdaBoost(X, y)
    print("step")
    ada.step()
    print("step")
    ada.step()

    # print(ada.hs)
