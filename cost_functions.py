from abc import ABC, abstractmethod
import numpy as np
import cvxpy as cp


class CostFunction(ABC):
    @abstractmethod
    def __call__(self, z: np.array, x: np.array):
        '''

        :param z: Feature vector that player might want to have
        :param x: Feature that player now has.
        :return: the cost that player pays to become z
        '''
        pass

    def maximize_features_against_binary_model(self, x: np.array, trained_model, use_spare_cost=False):
        '''

        :param x: current vector features.
        :param trained_model: binary model that is trained and player want to get positive score on it.
        :param use_spare_cost: if we want to use some of the spare cost in order to improve player score on the trained model
        :return: vector features  that has minimum cost and get positive score on trained_model.
        '''
        pass


class SeparableCost(CostFunction):
    @abstractmethod
    def apply_cost1(self, x: np.array):
        pass

    @abstractmethod
    def apply_cost2(self, x: np.array):
        pass


class WeightedLinearCostFunction(SeparableCost):
    """
    This class represents weighted linear cost function. That means calculation in the form of:
    max {<a, z-x>, 0} where a is the weights vector and z-x is the changed that player pays on.
    """

    def __init__(self, weighted_vector: np.array, cost_factor=6):
        '''
        :param weighted_vector: Weights vector. Each entry i in the vector represents the payment of moving
        one unit in the i'th feature.
        :param cost_factor: This parameter determines the scale of the cost function. This is a const that
        multiply the cost result.
        '''
        self.a = weighted_vector
        self.cost_factor = cost_factor

    def __call__(self, z: np.array, x: np.array):
        return max(self.a.T @ (z - x), 0)

    def maximize_features_against_binary_model(self, x: np.array, trained_model, tolerance=1e-9):
        x_tag = cp.Variable(len(x))

        func_to_solve = cp.Minimize(cp.maximum(self.cost_factor * self.a.T @ (x_tag - x), 0))
        constrains = [x_tag @ trained_model.coef_[0] >= -trained_model.intercept_ + tolerance]
        prob = cp.Problem(func_to_solve, constrains)
        prob.solve()
        cost_result = cp.maximum(self.cost_factor * self.a.T @ (x_tag.value - x), 0)
        if x_tag is None:
            print("couldn't solve this problem")
            return
        if trained_model.predict(x_tag.value.reshape(1, -1))[0] == 1 and cost_result.value < 2:
            return x_tag.value
        else:
            return x

    def apply_cost1(self, x: np.array):
        if isinstance(x, float):
            return self.cost_factor * self.a.T * x
        else:
            return self.cost_factor * self.a.T @ x

    def apply_cost2(self, x: np.array):
        if isinstance(x, float):
            return self.cost_factor * self.a.T * x
        else:
            return self.cost_factor * self.a.T @ x


class MixWeightedLinearSumSquareCostFunction(CostFunction):
    """
    This class represents cost function that Consists of two parts. First part is weighted linear function
    and the second part is sum square cost function. That means calculation in the form of:
    (1-epsilon) * max {<a, z-x>, 0} + epsilon * square(norm2(z-x)).
    where a is the weights vector, z-x is the changed that player pays on and epsilon is the weight of the
    l2 cost function.
    """
    def __init__(self, weighted_vector: np.array, epsilon=0.3, cost_factor=7, spare_cost=0.2):
        '''

        :param weighted_vector: Weights vector. Each entry i in the vector represents the payment of moving
        one unit in the i'th feature.
        :param epsilon: The weight of the l2 cost function.
        :param cost_factor: This parameter determines the scale of the cost function. This is a const that
        multiply the cost result.
        :param spare_cost: How much palyer agree to pay more in order to be beyond the classifier bound.
        '''
        self.a = weighted_vector
        self.epsilon = epsilon
        self.cost_factor = cost_factor
        self.spare_cost = spare_cost

    def __call__(self, z: np.array, x: np.array):
        cost_value = (1 - self.epsilon) * self.a.T @ (z - x) + self.epsilon * np.sum((z - x) ** 2)
        return max(self.cost_factor * cost_value, 0)

    def solve_problem_min_cost_s_t_model(self, model, x, tol):
        '''
        Fucntion solve the optimization problem find the x' that minimize the cost function with constrain of
        the model predict x' as positive.
        :param model: The model that the player wants to get positive prediction.
        :param x: Current player's vector features.
        :param tol: Small tolerance for optimization needs. That indicates how much x' should be above the model
        intercept.
        :return: Tuple: (x', cost(x, x')).
        '''
        x_t = cp.Variable(len(x))
        func_to_solve = cp.Minimize(
            self.cost_factor * (cp.maximum((1 - self.epsilon) * self.a.T @ (x_t - x), 0) + self.epsilon *
                                cp.sum((x_t - x) ** 2)))
        constrains = [x_t @ model.coef_[0] >= -model.intercept_ + tol]

        prob = cp.Problem(func_to_solve, constrains)
        try:
            prob.solve()
            if x_t is None:
                print("couldn't solve this problem")
                return
            cost = cp.maximum((1 - self.epsilon) * self.a.T @ (x_t - x), 0) + self.epsilon * cp.sum(
                (x_t - x) ** 2)
            cost *= self.cost_factor
            return x_t, cost
        except:
            print('solver failed')
            return x, None

    def solve_problem_max_model_s_t_cost(self, model, x, spare_cost, tol=0.00001):
        x_t = cp.Variable(len(x))
        func_to_solve = cp.Maximize(
            x_t @ model.coef_[0] + model.intercept_)

        constrains = [self.cost_factor * (cp.maximum((1 - self.epsilon) * self.a.T @ (x_t - x), 0) + self.epsilon *
                                          cp.sum((x_t - x) ** 2)) <= spare_cost - tol]
        prob = cp.Problem(func_to_solve, constrains)
        prob.solve()
        if x_t.value is None:
            print("couldn't solve this problem")
            return None
        cost = cp.maximum((1 - self.epsilon) * self.a.T @ (x_t - x), 0) + self.epsilon * cp.sum(
            (x_t - x) ** 2)
        cost *= self.cost_factor
        return x_t, cost


    def check_change_condition(self, trained_model, x_tag, cost_result):
        return trained_model.predict(x_tag.value.reshape(1, -1))[0] == 1 and cost_result.value < 2

    def maximize_features_against_binary_model(self, x: np.array, trained_model, tolerance=0.00001,
                                               use_spare_cost=False):
        x_tag, cost_result = self.solve_problem_min_cost_s_t_model(trained_model, x, tolerance)
        if cost_result is None:
            return x_tag
        if use_spare_cost and self.spare_cost != 0:
            if self.check_change_condition(trained_model, x_tag, cost_result):
                spare_cost = min(self.spare_cost + cost_result.value, 2)
                x_cost_tup = self.solve_problem_max_model_s_t_cost(trained_model, x, spare_cost)
                if x_cost_tup is not None:
                    x_tag, cost_result = x_cost_tup
        if self.check_change_condition(trained_model, x_tag, cost_result):
            return x_tag.value
        else:
            return x
