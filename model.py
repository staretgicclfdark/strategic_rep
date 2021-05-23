from cost_functions import *
from tqdm import tqdm
import pandas as pd


class TrainModel(ABC):
    @abstractmethod
    def __call__(self, X):
        """
        :param X: Vector features to predict
        :return: The prediction result
        """

    @abstractmethod
    def fit(self, X, y):
        """
        :param X: Data to learn from
        :param y: True values
        """
    @abstractmethod
    def predict(self, X):
        """
        :param X: Data to predict
        :return: The new model prediction
        """


class HardtAlgo(TrainModel):

    '''
    The Hardt model is implemented as it was described in paper Strategic Classification (Hardt et al)
    '''

    def __init__(self, separable_cost: SeparableCost):
        '''

        :param separable_cost: The cost separable function.
        '''
        self.min_si = None
        self.separable_cost = separable_cost
        self.coef_ = (separable_cost.a, None)
        self.intercept_ = None

    def __call__(self, X):
        def apply_single_prdictive(x):
            return 1 if self.separable_cost.apply_cost2(x) >= self.min_si else -1

        if self.min_si is None:
            print("The model hasn't trained yet. please train first")
            return

        if isinstance(X, np.ndarray):
            return np.apply_along_axis(apply_single_prdictive, 1, X)
        return X.apply(apply_single_prdictive, axis=1)

    def predict(self, X):
        return self(X)


    def fit(self, X: pd.DataFrame, y):
        def apply_cost_with_thresh(x):
            return 1 if self.separable_cost.apply_cost1(x) >= thresh else -1
        print("training hardt model it might take a while..")
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
            self.intercept_ = -self.min_si / self.separable_cost.cost_factor










