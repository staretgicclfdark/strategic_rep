from abc import ABC, abstractmethod
import numpy as np
import cvxpy as cp


class CostFunction(ABC):
    @abstractmethod
    def __call__(self, z: np.array, x: np.array):
        '''

        :param z: Feature vector that player might want to have
        :param x: Feature that player now have.
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
    def __init__(self, weighted_vector: np.array, cost_factor=6):
        self.a = weighted_vector
        self.cost_factor = cost_factor

    def __call__(self, z: np.array, x: np.array):
        return max(self.a.T @ (z - x), 0)

    def maximize_features_against_binary_model(self, x: np.array, trained_model, tolerance=1e-9):
        x_tag = cp.Variable(len(x))

        func_to_solve = cp.Minimize(cp.maximum(self.a.T @ (x_tag - x), 0))
        constrains = [x_tag @ trained_model.coef_[0] >= -trained_model.intercept_ + tolerance]
        prob = cp.Problem(func_to_solve, constrains)
        prob.solve()
        cost_result = cp.maximum(self.a.T @ (x_tag.value - x), 0)
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

def check_result(trained_model, new_x, cost):
    return trained_model.predict(new_x.value.reshape(1, -1))[0] == 1 and cost.value < 2


class MixWeightedLinearSumSquareCostFunction(CostFunction):
    def __init__(self, weighted_vector: np.array, epsilon=0.3, cost_factor=7, spare_cost=0.2):
        self.a = weighted_vector
        self.epsilon = epsilon
        # some values for statistic and debugging:
        self.num_changed = 0  # the number of example that his changed because of solving the minimization problem
        self.num_examples = 0
        self.num_above_trash = 0
        self.trash = 0.001
        self.cost_factor = cost_factor
        self.max_cost, self.max_separable_cost = -np.inf, -np.inf
        self.spare_cost = spare_cost


    def __call__(self, z: np.array, x: np.array):
        return max((1 - self.epsilon) * self.a.T @ (z - x) + self.epsilon * np.sum((z - x) ** 2), 0)

    def solve_problem_min_cost_s_t_model(self, model, x, tol):
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
            print('solver faild')
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

    def update_statistic_and_return_correct_x(self, x: np.ndarray, x_tag: cp.Variable, cost_result, trained_model):
        self.max_cost = max(self.max_cost, cost_result.value)
        self.max_separable_cost = max(self.max_separable_cost, (1 - self.epsilon) * self.a.T @ (x_tag.value - x))
        self.num_examples += 1
        if self.check_change_condition(trained_model, x_tag, cost_result):
            self.num_changed += 1
            # if self.f.predict(x_tag.value.reshape(1, -1))[0] == -1:
            #     self.num_changed_on_f_hat_not_f += 1
            #     self.cost_left_avg += 2 - cost_result.value
            #     self.sub_f_res_f_hat_res += (x_tag.value @ self.f.coef_[0] + self.f.intercept_ - (
            #             x_tag.value @ trained_model.coef_[0] + trained_model.intercept_))
            # if self.a.T @ (x_tag.value - x) > self.trash:
            #     self.num_above_trash += 1
            return x_tag.value
        else:

            return x


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
        return self.update_statistic_and_return_correct_x(x, x_tag, cost_result, trained_model)


    def get_number_example_that_moved(self):
        return self.num_changed

    def get_statistic_on_num_change(self):
        calc_percent = lambda x: 100 * x / self.num_examples
        print(
            f'number of examples that has changed: {self.num_changed} and the percent is {calc_percent(self.num_changed)}'
            f'the number of examples above {self.trash} is : {self.num_above_trash} which are {calc_percent(self.num_above_trash)}% '
            f'max cost func is:{self.max_cost} and the max separable cost is: {self.max_separable_cost} \n'
        )

