from cost_functions import *
from tqdm import tqdm
import pandas as pd


class TrainModel(ABC):
    @abstractmethod
    def __call__(self, X):
        """

        :param X: features to predict
        :return: the prediction result
        """

    @abstractmethod
    def fit(self, X, y):
        """

        :param X: data to learn from
        :param y: True values
        """
    @abstractmethod
    def predict(self, X):
        """

        :param X: data to predict
        :return: the new model prediction
        """

class RandomModel(TrainModel):
    def __init__(self):
        self.coef = np.random.normal(size=6)
        self.intercept = np.random.normal(size=1)

    def fit(self, X, y):
        return

    def predict(self, X):
        return self(X)

    def __call__(self, X):
        def return_single_prediction(single_float_score):
            return 1 if single_float_score >= 0 else -1

        return np.vectorize(return_single_prediction)(X @ self.coef + self.intercept)


class HardtAlgo(TrainModel):
    def __init__(self, separable_cost: SeparableCost):
        self.min_si = None
        self.separable_cost = separable_cost
        self.coef_ = (separable_cost.a, None)
        self.intercept_ = None

    def __call__(self, X):
        def apply_single_prdictive(x):
            return 1 if self.separable_cost.apply_cost2(x) >= self.min_si else -1

        if self.min_si is None:
            print("model hasn't trained yet. please train first")
            return

        if isinstance(X, np.ndarray):
            return np.apply_along_axis(apply_single_prdictive, 1, X)
        return X.apply(apply_single_prdictive, axis=1)

    def predict(self, X):
        return self(X)


    def fit(self, X: pd.DataFrame, y):
        def apply_cost_with_thresh(x):
            return 1 if self.separable_cost.apply_cost1(x) >= thresh else -1
        min_err_si = np.inf
        S = X.apply(self.separable_cost.apply_cost2, axis=1) + 2

        with tqdm(total=len(S)) as t:
            for i, s_i in enumerate(S):
                thresh = s_i - 2
                err_si = np.sum(y != X.apply(apply_cost_with_thresh, axis=1)) / len(X)
                if min_err_si > err_si:
                    min_err_si = err_si
                    self.min_si = s_i
                t.update(1)
            print(f'min_err_si: {min_err_si}')
            self.intercept_ = -self.min_si / self.separable_cost.cost_factor










