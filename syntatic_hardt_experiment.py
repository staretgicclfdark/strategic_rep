import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from model import HardtAlgo
from cost_functions import WeightedLinearCostFunction, MixWeightedLinearSumSquareCostFunction
from strategic_players import strategic_modify_using_known_clf, strategic_modify_learn_from_friends
from create_synthetic_data import create_member_friends_dict
from utills_and_consts import evaluate_model_on_test_set, result_folder_path, plot_graph, safe_create_folder, save_model, load_model
import json
from sklearn.svm import LinearSVC
import random


def from_numpy_to_panda_df(data):
    data = pd.DataFrame(data=data[0:, 0:], index=[i for i in range(data.shape[0])],
                 columns=['f' + str(i) for i in range(data.shape[1])])
    return data



def create_dataset(data_size, covariance=None, d=1):
    def map_sum_one_minus_one(sum_value):
        return 1 if sum_value >=0 else -1

    if covariance is None:
        covariance = np.eye(d)
    means = np.zeros(shape=d)
    data = np.random.multivariate_normal(mean=means, cov=covariance, size=data_size)
    data = from_numpy_to_panda_df(data)
    # memberkeys:
    member_keys = [f's{i}' for i in range(data_size)]
    data.insert(len(data.columns), 'MemberKey', member_keys, allow_duplicates=True)
    labels = list(data.sum(axis=1).apply(map_sum_one_minus_one))
    # using LoanStatus as label to prevent bugs
    data.insert(len(data.columns), 'LoanStatus', labels, allow_duplicates=True)
    return data




def plot_hist(data, title: str = '', save_path=None):
    data_pos = data[data['label'] == 1]
    data_neg = data[data['label'] == -1]
    plt.hist(data_pos['f0'], 20, label='positive data', alpha=0.7)
    plt.hist(data_neg['f0'], 20, label='negetive data', alpha=0.7)
    plt.legend(loc='upper right')
    plt.title(title)
    plt.savefig(save_path)
    plt.show()


def change_test_datasets_f_info(f_model_list, test_lists, feature_list, epsilon, cost_factor, a_tag, spare_cost=0):
    strategic_modify_tests_list = list()
    for (test_set, f_model) in zip(test_lists, f_model_list):
        cost_func_for_gaming = MixWeightedLinearSumSquareCostFunction(a_tag, epsilon=epsilon, cost_factor=cost_factor,
                                                                  spare_cost=spare_cost)
        strategic_modify_tests_list.append(strategic_modify_using_known_clf(test_set, f_model, feature_list, cost_func_for_gaming))
    return strategic_modify_tests_list



def get_test_data_sets(test_size, num_data_sets_to_create, covariance=None, d=1, seed=42):
    np.random.seed(seed)
    return [create_dataset(test_size, covariance, d=d) for _ in range(num_data_sets_to_create)]


def create_firends_data_set(m, f, feature_list, covariance=None, d=1, seed=42):

    friends_set, hardt_label_friends = None, None
    friends_label = set()
    f_labels_friends = None
    while len(friends_label) != 2:
        friends_set = create_dataset(m, covariance=covariance, d=d)
        f_labels_friends = f.predict(pd.DataFrame(friends_set[feature_list]))
        friends_label = set(f_labels_friends)
    return friends_set, f_labels_friends


def get_trained_hardt_models(train_size, exp_path, num_to_train, a_tag, cost_factor, covariance=None, d=1, force_to_create=False):
    hardt_models_dir = safe_create_folder(exp_path, 'hardt_models')
    hardt_models_to_return = list()
    for i in range(num_to_train):
        feature_list = [f'f{i}' for i in range(d)]
        model_path = os.path.join(hardt_models_dir, f'hardt_{i}')
        if force_to_create or os.path.exists(model_path) is False:
            print(f' training hardt number: {i}')
            train_set = create_dataset(train_size, covariance, d=d)
            hardt_model = HardtAlgo(WeightedLinearCostFunction(a_tag, cost_factor))
            hardt_model.fit(pd.DataFrame(train_set[feature_list]), train_set['LoanStatus'])
            save_model(hardt_model, model_path)
        else:
            hardt_model = load_model(model_path)
        hardt_models_to_return.append(hardt_model)
    return hardt_models_to_return


def m_exp():
    np.random.seed(42)
    base_folder = safe_create_folder(result_folder_path, 'oneD_synthetic_hardt_exp')
    m_exp_path = safe_create_folder(base_folder, 'm_exp')
    m_list = [4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]

    a_tag = np.array([1])
    epsilon = 0.0000001
    cost_factor = 1
    test_size = 1000
    train_size =4000
    num_splits = 5
    repeat_on_same_model_exp = 50

    f_models_list = get_trained_hardt_models(train_size, m_exp_path, num_splits, a_tag, cost_factor, force_to_create=False)
    test_data_sets_list = get_test_data_sets(test_size, num_splits)
    tests_full_info_changed = change_test_datasets_f_info(f_models_list, test_data_sets_list, ['f0'], epsilon, cost_factor, a_tag)
    err_f_on_x_f_list = [1 - evaluate_model_on_test_set(test, f, ['f0']) for (f, test) in zip(f_models_list, tests_full_info_changed)]
    test_f_pred_no_change = [f.predict(pd.DataFrame(test['f0'])) for (f, test) in zip(f_models_list, test_data_sets_list)]
    f_hat_ne_f_err_list, f_hat_ne_f_err_var_list = list(), list()
    pop_list, pop_var_list = list(), list()



    for m in m_list:
        splits_pop_list, f_hat_ne_f_err_split_list = list(), list()
        # for each split the test is fixed and f is fixed
        for i in range(num_splits):
            print(f' m: {m} split number: {i}')
            f_model, test_set = f_models_list[i], test_data_sets_list[i]
            err_f_on_x_f = err_f_on_x_f_list[i]
            # todo: f_hat trained more that one 10..
            for _ in range(repeat_on_same_model_exp):
                friends_set, friends_hardt_labels = create_firends_data_set(m, f_model, ['f0'])
                f_hat = LinearSVC(C=1000, random_state=42, max_iter=100000)
                f_hat.fit(pd.DataFrame(friends_set['f0']), friends_hardt_labels)
                test_f_hat_pred = f_hat.predict(pd.DataFrame(test_set['f0']))
                f_hat_ne_f_err_split_list.append(np.sum(test_f_hat_pred != test_f_pred_no_change[i]).item() / len(test_set))
                cost_func_for_gaming = MixWeightedLinearSumSquareCostFunction(a_tag, epsilon=epsilon,
                                                                              cost_factor=cost_factor,
                                                                              spare_cost=0)
                test_known_f_hat_changed = strategic_modify_using_known_clf(test_set, f_hat, ['f0'], cost_func_for_gaming)
                err_f_on_x_f_hat = 1 - evaluate_model_on_test_set(test_known_f_hat_changed, f_model, ['f0'])
                splits_pop_list.append((err_f_on_x_f_hat - err_f_on_x_f).item())

        f_hat_ne_f_err_list.append(sum(f_hat_ne_f_err_split_list) / len(f_hat_ne_f_err_split_list))
        f_hat_ne_f_err_var_list.append(sum([(f_hat_ne_f_err_val - f_hat_ne_f_err_list[-1]) ** 2 for f_hat_ne_f_err_val in f_hat_ne_f_err_list]) / len(f_hat_ne_f_err_list))
        pop_list.append(sum(splits_pop_list) / len(splits_pop_list))
        pop_var_list.append(sum([(pop_val - pop_list[-1])**2 for pop_val in pop_list]) / len(pop_list))
    # example: sum((val - mean) ** 2 for val in data_to_return[key_list_name]) / len(data_to_return[key_list_name])
    # mean_err_f_f_hardt = sum(err_f_on_x_f_list) / len(err_f_on_x_f_list)
    # err_f_hardt_f_f_hardt_var = sum([(err_f_on_x_f - mean_err_f_f_hardt)**2 for err_f_on_x_f in mean_err_f_f_hardt]) / len(mean_err_f_f_hardt)
    # err_f_f_hardt = [mean_err_f_f_hardt for _ in m_list]
    # var_err_f_f_hardt = [err_f_hardt_f_f_hardt_var for _ in m_list]

    data_graph_path = os.path.join(m_exp_path, 'm_graph_data.json')
    with open(data_graph_path, 'w+') as f:
        data = dict()
        data['f_hat_ne_f_err_list'] = f_hat_ne_f_err_list
        data['f_hat_ne_f_err_var_list'] = f_hat_ne_f_err_var_list
        data['pop_list'] = pop_list
        data['pop_var_list'] = pop_var_list
        json.dump(data, f, indent=4)

    saving_path = os.path.join(m_exp_path, 'm_graph.png')
    plot_graph(title='err vs m', x_label='m', y_label='err', x_data_list=[m_list, m_list],
               y_data_list=[f_hat_ne_f_err_list, pop_list], saving_path=saving_path,
               graph_label_list=[r'$E[1\{f(x) \neq \^{f}\}]$', 'POP'], var_lists=[f_hat_ne_f_err_list, pop_var_list], SE=True, num_samples=num_splits*repeat_on_same_model_exp)



m_exp()
