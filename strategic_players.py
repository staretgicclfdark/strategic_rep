import pandas as pd
import os
from cost_functions import *
from tqdm import tqdm
from sklearn.svm import LinearSVC
from sklearn.model_selection import LeaveOneOut, KFold
from create_synthetic_data import apply_transform_creditgrade_loan_returned
from utills_and_consts import *
import matplotlib.pyplot as plt
import json


def get_angle_between_two_vectors(vec1, vec2, result_in_degree=True):
    angle = np.arccos(vec1 @ vec2 / (np.linalg.norm(vec1) * np.linalg.norm(vec2))).item()
    if result_in_degree:
        angle *= 180 / np.pi
    return angle


def visualize_projected_changed_df(clf_name, before_change_df, after_change_df, features_to_project, title, f_weights, f_inter,
                                   label='LoanStatus',
                                   num_point_to_plot=100, dir_for_projection_images: str = '2D_projection_images',
                                   to_save=True, dir_name_for_saving_visualize=None):
    def apply_transform_for_2D(df: pd.DataFrame):
        transform_matrix = np.array([[f_weights[0], 0], [f_weights[1], 0], [f_weights[2], 0],
                                     [0.5 * f_weights[3], 0.5 * f_weights[3]], [0, f_weights[4]], [0, f_weights[5]]])
        return df @ transform_matrix

    os.makedirs(dir_name_for_saving_visualize, exist_ok=True)
    dir_for_projection_images = os.path.join(dir_name_for_saving_visualize, dir_for_projection_images)
    df_before_loan_status, df_before = before_change_df[label], before_change_df[features_to_project]
    df_after = after_change_df[features_to_project]
    projected_df_before, projected_df_after = apply_transform_for_2D(df_before), apply_transform_for_2D(df_after)
    fig, ax = plt.subplots(1, 1)
    ax.scatter(projected_df_before[0][:num_point_to_plot], projected_df_before[1][:num_point_to_plot], s=10)
    ax.scatter(projected_df_after[0][:num_point_to_plot], projected_df_after[1][:num_point_to_plot], s=10)

    if clf_name == 'SVM':
        left_bound, right_bound = 1, 3
        bottom_bound, up_bound = 0.2, 1.4
        head_length = 0.03
        head_width = 0.02
    elif clf_name == 'Hardt':
        left_bound, right_bound = 0.5, 2.5
        bottom_bound, up_bound = 0, 1.2
        head_length = 0.03
        head_width = 0.02
    elif clf_name == 'SVM_real_network':
        left_bound, right_bound = 20, 60
        bottom_bound, up_bound = 10, 35
        head_length = 1
        head_width = 0.5
    else:
        print('clf_name should be Hardt or SVM returning without plot')
        return

    for i, (before_row_tup, after_row_tup, before_full, after_full) in enumerate(
            zip(projected_df_before.iterrows(), projected_df_after.iterrows(), df_before.iterrows(),
                df_after.iterrows())):
        before_row, after_row = before_row_tup[1], after_row_tup[1]
        plt.arrow(before_row[0], before_row[1], after_row[0] - before_row[0], after_row[1] - before_row[1],
                  shape='full', color='black', length_includes_head=True,
                  zorder=0, head_length=head_length, head_width=head_width)
        if i > num_point_to_plot:
            break


    t = np.arange(left_bound, right_bound, 0.2)
    plt.plot(t, -t - f_inter, color='blue')
    plt.xlim([left_bound, right_bound])
    plt.ylim([bottom_bound, up_bound])
    plt.title(title)
    if to_save:
        saving_path = os.path.join(dir_for_projection_images, title + '.png')
        os.makedirs(dir_for_projection_images, exist_ok=True)
        plt.savefig(saving_path)
    plt.show()


def get_f_star_loan_status_real_train_val_test_df(force_create_train=True, force_create_val=True,
                                                  force_create_test=True):
    # this function creates datasets like the real dataset but with different loan status. the f_star loan status.
    def get_real_f_star_loan_status_real_df(force_create, orig_df_path, orig_df_f_star_loan_status):
        if os.path.exists(orig_df_f_star_loan_status) is False or force_create:
            orig_real_df = pd.read_csv(orig_df_path)

            orig_real_df['LoanStatus'] = orig_real_df['CreditGrade'].apply(apply_transform_creditgrade_loan_returned)
            orig_real_df.to_csv(orig_df_f_star_loan_status)
        else:
            orig_real_df = pd.read_csv(orig_df_f_star_loan_status)
        return orig_real_df

    train_df = get_real_f_star_loan_status_real_df(force_create_train, real_train_path,
                                                   real_train_f_star_loan_status_path)
    val_df = get_real_f_star_loan_status_real_df(force_create_val, real_val_path,
                                                 real_val_f_star_loan_status_path)
    test_df = get_real_f_star_loan_status_real_df(force_create_test, real_test_path,
                                                  real_test_f_star_loan_status_path)
    train_val_df = pd.concat([train_df, val_df], ignore_index=True)
    train_val_df.to_csv(real_train_val_f_star_loan_status_path)
    return train_df, val_df, test_df, train_val_df


def train_loan_return_svm_model(list_features_for_pred, binary_trained_model_path, target_label='LoanStatus',
                                use_cv=False):
    train_df, val_df, test_df, train_val_df = get_f_star_loan_status_real_train_val_test_df()
    if use_cv:
        print('training loan return model using cross validation')
        linear_model = train_loan_status_model_with_cv(train_val_df[list_features_for_pred], train_val_df[target_label])
    else:
        linear_model = LinearSVC(C=0.01, penalty='l2', random_state=42)
        linear_model.fit(train_val_df[list_features_for_pred], train_val_df[target_label])
    y_test_pred = linear_model.predict(test_df[list_features_for_pred])
    acc = np.sum(y_test_pred == test_df[target_label]) / len(y_test_pred)
    print(f'acc on not modify real test: {acc}')
    pickle.dump(linear_model, open(binary_trained_model_path, 'wb'))
    return linear_model


def strategic_modify_using_known_clf(orig_df: pd.DataFrame, f, feature_list, cost_func: CostFunction):
    modify_data = orig_df[feature_list].copy()
    with tqdm(total=len(orig_df)) as t:
        for index, ex in orig_df[feature_list].iterrows():
            x = np.array(ex)
            if f.predict(x.reshape(1, -1))[0] == -1:
                z = cost_func.maximize_features_against_binary_model(x, f)
                modify_data.loc[index] = z if len(z) > 1 else z[0]
            else:
                modify_data.loc[index] = x if len(x) > 1 else x[0]  # todo: chek if there is problem
            t.update(1)
    # insert other features that are not used for prediction but yet important:
    for col_name in filter(lambda c: c not in modify_data.columns, orig_df.columns):
        modify_data.insert(len(modify_data.columns), col_name, orig_df[col_name], True)

    return modify_data


def create_strategic_data_sets_using_known_clf(dir_name_for_saving_visualize, cost_factor, epsilon, f,
                                               save_visualize_projected_changed=True, clf_name='',
                                               features_to_use=six_most_significant_features):
    real_test_f_star_loan_status = pd.read_csv(real_test_f_star_loan_status_path)
    f_weights, f_inter = f.coef_[0], f.intercept_

    weighted_linear_cost = MixWeightedLinearSumSquareCostFunction(a, cost_factor=cost_factor, epsilon=epsilon)
    modify_full_information_test = strategic_modify_using_known_clf(real_test_f_star_loan_status, f, features_to_use,
                                                                    weighted_linear_cost)
    visualize_projected_changed_df(clf_name, real_test_f_star_loan_status, modify_full_information_test, features_to_use,
                                   'real test move by gaming against ' + clf_name,
                                   to_save=save_visualize_projected_changed,
                                   dir_name_for_saving_visualize=dir_name_for_saving_visualize, f_weights=f_weights,
                                   f_inter=f_inter)

    acc_modify_full_information_test_modify = evaluate_model_on_test_set(modify_full_information_test, f,
                                                                         features_to_use)
    print(
        f'the accuracy on the test set when it clf is ' + clf_name + f' trained on not modify train {acc_modify_full_information_test_modify}')
    print(
        f'angle: {get_angle_between_two_vectors(a, f.coef_[0])}')  # that is only for debugging. we can delete it later
    weighted_linear_cost.get_statistic_on_num_change()
    return modify_full_information_test


def train_loan_status_model_with_cv(training_set: pd.DataFrame, training_labels: pd.Series):
    num_samples = len(training_set)
    if num_samples <= 10:
        splitter = LeaveOneOut()
        k = len(training_set)
    elif num_samples < 1000:
        k = 10
        splitter = KFold(n_splits=k)
    else:
        k = 3
        splitter = KFold(n_splits=k)

    c_valuse = [0.01, 0.05, 0.1, 0.5, 1, 5, 10, 100, 1000]
    best_c, best_acc = -1, -np.inf
    for c in c_valuse:
        sum_current_c_acc = 0
        for train_index, test_index in splitter.split(training_set):
            X_train, X_test = training_set.iloc[train_index, :], training_set.iloc[test_index, :]
            y_train, y_test = training_labels.iloc[train_index], training_labels.iloc[test_index]
            current_f = LinearSVC(C=c, random_state=42, max_iter=100000).fit(X_train, y_train)
            sum_current_c_acc += np.sum(current_f.predict(X_test) == y_test) / len(y_test)
        if best_acc < sum_current_c_acc / k:
            best_acc, best_c = sum_current_c_acc/k, c
    print(f'picked C: {best_c} with cross validation')
    f_hat = LinearSVC(C=best_c, random_state=42, max_iter=100000).fit(training_set, training_labels)
    return f_hat


def get_svm_loan_return_model(model_loan_returned_path, features_to_use, retrain_model_loan_return=False):
    if retrain_model_loan_return or os.path.exists(model_loan_returned_path) is False:
        f = train_loan_return_svm_model(features_to_use, model_loan_returned_path)
    else:
        f = load_model(model_loan_returned_path)
        real_test_f_star_loan_status = pd.read_csv(real_test_f_star_loan_status_path)
        acc_on_not_modify = evaluate_model_on_test_set(real_test_f_star_loan_status, f, features_to_use)
        print(f'acc on not modify real test:{acc_on_not_modify}')
    return f


def init_data_to_return_dict():
    data_to_return = dict()

    data_to_return['number_moved'] = 0
    data_to_return['acc_list'] = list()
    data_to_return['l2_norm_list'] = list()
    data_to_return['angle_f_hat_f_list'] = list()
    return data_to_return


def update_dicts_for_f_hat_result(f_hat_data_dict, data_to_return, member_key, orig_df, f_hat, feature_to_learn_list, orig_df_f_loan_status,
                                  models_f_hat_path, f_vec, did_changed, prediction_f_hat_on_x):
    f_hat_data_dict[member_key] = dict()
    saving_model_path = os.path.join(models_f_hat_path, f'f_hat_{member_key}.sav')
    save_model(f_hat, saving_model_path) # todo: return it
    f_hat_data_dict[member_key]['saving model path'] = saving_model_path
    f_hat_data_dict[member_key]['acc'] = evaluate_model_on_test_set(orig_df, f_hat, feature_to_learn_list, orig_df_f_loan_status)
    data_to_return['acc_list'].append(f_hat_data_dict[member_key]['acc'])
    # data_to_return['sum_acc'] += f_hat_data_dict[member_key]['acc']
    f_hat_vec = np.append(f_hat.coef_[0], f_hat.intercept_)
    f_hat_data_dict[member_key]['||f_hat_vec - f_vec||_2'] = np.linalg.norm(f_hat_vec - f_vec).item()
    data_to_return['l2_norm_list'].append(f_hat_data_dict[member_key]['||f_hat_vec - f_vec||_2'])
    # data_to_return['sum_l2_norm'] += f_hat_data_dict[member_key]['||f_hat_vec - f_vec||_2']
    f_hat_data_dict[member_key]['angle_f_hat_f'] = get_angle_between_two_vectors(f_hat_vec, f_vec)
    data_to_return['angle_f_hat_f_list'].append(f_hat_data_dict[member_key]['angle_f_hat_f'])
    # data_to_return['sum_angle_f_hat_f'] += f_hat_data_dict[member_key]['angle_f_hat_f']
    f_hat_data_dict[member_key]['prediction on f_hat'] = prediction_f_hat_on_x
    f_hat_data_dict[member_key]['did changed'] = did_changed
    if did_changed:
        data_to_return['number_moved'] += 1


def finish_data_dicts_updates(data_to_return, f_hat_data_dict, f_hat_data_json_path, data_to_return_path):
    def add_mean_and_var_for_key(base_key_name: str):
        key_list_name = base_key_name + '_list'
        mean = sum(data_to_return[key_list_name]) / len(data_to_return[key_list_name])
        data_to_return[base_key_name + '_avg'] = mean
        data_to_return[base_key_name + '_var'] = sum((val - mean) ** 2 for val in data_to_return[key_list_name]) / len(data_to_return[key_list_name])
        del data_to_return[key_list_name]

    for key in ['acc', 'l2_norm', 'angle_f_hat_f']:
        add_mean_and_var_for_key(key)
    with open(f_hat_data_json_path, 'w+') as json_file:
        json.dump(f_hat_data_dict, json_file, indent=4)
    with open(data_to_return_path, 'w+') as json_file:
        json.dump(data_to_return, json_file, indent=4)


def get_paths_for_running(dir_name_for_result, num_friends):
    models_f_hat_path = safe_create_folder(safe_create_folder(dir_name_for_result, 'f_hat_models'),
                                           f'{num_friends} friends')
    data_modified_path = os.path.join(safe_create_folder(dir_name_for_result, 'modified_data'),
                                      f'modified_test_{num_friends}.csv')
    f_hat_data_json_path = os.path.join(safe_create_folder(dir_name_for_result, 'f_hat_result'),
                                        f'f_hat_results_{num_friends}.json')
    data_to_return_path = os.path.join(safe_create_folder(dir_name_for_result, 'run_summary_result'),
                                        f'fsummary_results_{num_friends}_friends.json')
    return models_f_hat_path, data_modified_path, f_hat_data_json_path, data_to_return_path


def get_player_movements_and_update_modify_data_df(cost_func, modify_data, index, f_hat, x):
    z = cost_func.maximize_features_against_binary_model(x, f_hat, use_spare_cost=True)
    modify_data.loc[index] = z
    if f_hat.predict(z.reshape(1, -1))[0] == 1:
        did_move = True
        modify_data.loc[index] = z if len(z) > 1 else z[0]
    else:
        did_move = False
        modify_data.loc[index] = x if len(x) > 1 else x[0]
    return did_move


def train_f_hat(sample_friends_from_df, sample_friends_f_loan_status, member_dict, member_key, feature_to_learn_list):
    friends_df = sample_friends_from_df.iloc[member_dict[member_key]["friends with credit data"], :]
    friends_labels = sample_friends_f_loan_status[member_dict[member_key]["friends with credit data"]]
    if len(set(friends_labels)) < 2:
        return None
    f_hat = LinearSVC(C=1000, penalty='l2', random_state=42, max_iter=100000000).fit(
        friends_df[feature_to_learn_list], friends_labels)
    return f_hat


def write_modify_data_with_all_columns(modify_data, orig_df, data_modified_path):
    for col_name in filter(lambda c: c not in modify_data.columns, orig_df.columns):
        modify_data.insert(len(modify_data.columns), col_name, orig_df[col_name], True)
    modify_data.to_csv(data_modified_path)


def strategic_modify_learn_from_friends(clf_name, orig_df_f_loan_status, orig_df: pd.DataFrame, sample_friends_from_df: pd.DataFrame, sample_friends_f_loan_status,
                                        feature_to_learn_list, cost_func: CostFunction, member_dict: dict, f_vec, dir_name_for_result: str = None,
                                        title_for_visualization: str = None, visualization=True, num_friends=0):
    # counter = 0  # only for debugging.
    modify_data = orig_df[feature_to_learn_list].copy()
    should_save = False
    if dir_name_for_result is not None:
        should_save = True
        models_f_hat_path, data_modified_path, f_hat_data_json_path, data_to_return_path = get_paths_for_running(dir_name_for_result, num_friends)
    f_hat_data_dict = dict()
    data_to_return = init_data_to_return_dict()
    with tqdm(total=len(orig_df)) as t:
        for (index, ex), member_key in zip(orig_df[feature_to_learn_list].iterrows(), orig_df['MemberKey']):
            f_hat_data_dict[member_key] = dict()
            f_hat = train_f_hat(sample_friends_from_df, sample_friends_f_loan_status, member_dict, member_key, feature_to_learn_list)
            # f model is already trained we can load it
            # f_hat = load_model(os.path.join(models_f_hat_path, f'f_hat_{member_key}.sav'))
            x = np.array(ex)
            did_changed = False
            if f_hat is not None:
                prediction_f_hat_on_x = f_hat.predict(x.reshape(1, -1))[0]
                if prediction_f_hat_on_x == -1:
                    # counter += 1  # only for statistics and debugging we can delete it later
                    did_changed = get_player_movements_and_update_modify_data_df(cost_func, modify_data, index, f_hat, x)
            if should_save and f_hat is not None:
                update_dicts_for_f_hat_result(f_hat_data_dict, data_to_return, member_key, orig_df, f_hat,
                                              feature_to_learn_list, orig_df_f_loan_status,
                                              models_f_hat_path, f_vec, did_changed, prediction_f_hat_on_x.item())
            t.update(1)
    if should_save:
        finish_data_dicts_updates(data_to_return, f_hat_data_dict, f_hat_data_json_path, data_to_return_path)
        write_modify_data_with_all_columns(modify_data, orig_df, data_modified_path)
    # print(f'number that changed is: {counter}')

    # cost_func.get_statistic_on_num_change()
    if visualization:
        visualize_projected_changed_df(clf_name, orig_df, modify_data, feature_to_learn_list, title_for_visualization,
                                       dir_name_for_saving_visualize=dir_name_for_result, f_weights=f_vec[:-1],
                                       f_inter=f_vec[-1])
    # num_example_moved = cost_func.get_number_example_that_moved()
    return modify_data, data_to_return
