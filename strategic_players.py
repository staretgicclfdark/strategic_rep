from cost_functions import *
from tqdm import tqdm
from sklearn.svm import LinearSVC
from utills_and_consts import *
import json
from model import HardtAlgo
from projected_visualization import visualize_projected_changed_df


def get_hardt_model(cost_factor, train_path, force_train_hardt=False,
                    feature_list_to_use=six_most_significant_features):
    hardt_model_path = os.path.join(models_folder_path, f'Hardt_cost_factor={cost_factor}')
    if force_train_hardt or os.path.exists(hardt_model_path) is False:
        print(f'training Hardt model')
        train_df = pd.read_csv(train_path)
        hardt_algo = HardtAlgo(WeightedLinearCostFunction(a[:len(feature_list_to_use)], cost_factor))

        hardt_algo.fit(train_df[feature_list_to_use], train_df['LoanStatus'])
        save_model(hardt_algo, hardt_model_path)
    else:
        hardt_algo = load_model(hardt_model_path)
    return hardt_algo


def get_angle_between_two_vectors(vec1: np.array, vec2: np.array, result_in_degree: bool = True):
    '''

    :param vec1: Vector1
    :param vec2: Vector2
    :param result_in_degree: Whatever angle should be in degree or radiant
    :return: The angle between the two vectors
    '''
    angle = np.arccos(vec1 @ vec2 / (np.linalg.norm(vec1) * np.linalg.norm(vec2))).item()
    if result_in_degree:
        angle *= 180 / np.pi
    return angle


def strategic_modify_using_known_clf(orig_df: pd.DataFrame, f, feature_list: list, cost_func: CostFunction):
    '''

    :param orig_df: Data frame before any change.
    :param f: The model that players try to achieve positive score.
    :param feature_list: Features that are used f to predict loan
    :param cost_func: cost function to determine player payments
    :return: The data after each player changed his features
    '''
    modify_data = orig_df[feature_list].copy()
    with tqdm(total=len(orig_df)) as t:
        for index, ex in orig_df[feature_list].iterrows():
            x = np.array(ex)
            if f.predict(x.reshape(1, -1))[0] == -1:
                z = cost_func.maximize_features_against_binary_model(x, f)
                modify_data.loc[index] = z if len(z) > 1 else z[0]
            else:
                modify_data.loc[index] = x if len(x) > 1 else x[0]
            t.update(1)
    # insert other features that are not used for prediction but yet important:
    for col_name in filter(lambda c: c not in modify_data.columns, orig_df.columns):
        modify_data.insert(len(modify_data.columns), col_name, orig_df[col_name], True)

    return modify_data


def init_data_to_return_dict():
    data_to_return = dict()
    data_to_return['number_moved'] = 0
    data_to_return['err_list'] = list()
    data_to_return['l2_norm_list'] = list()
    data_to_return['angle_f_hat_f_list'] = list()
    return data_to_return


def update_dicts_for_f_hat_result(f_hat_data_dict, data_to_return, member_key, orig_df, f_hat, feature_to_learn_list, orig_df_f_loan_status,
                                  models_f_hat_path, f_vec, did_changed, prediction_f_hat_on_x):
    f_hat_data_dict[member_key] = dict()
    saving_model_path = os.path.join(models_f_hat_path, f'f_hat_{member_key}.sav')
    save_model(f_hat, saving_model_path)
    f_hat_data_dict[member_key]['saving model path'] = saving_model_path
    f_hat_data_dict[member_key]['err'] = evaluate_model_on_test_set(orig_df, f_hat, feature_to_learn_list, orig_df_f_loan_status)
    data_to_return['err_list'].append(f_hat_data_dict[member_key]['err'])
    f_hat_vec = np.append(f_hat.coef_[0], f_hat.intercept_)
    f_hat_data_dict[member_key]['||f_hat_vec - f_vec||_2'] = np.linalg.norm(f_hat_vec - f_vec).item()
    data_to_return['l2_norm_list'].append(f_hat_data_dict[member_key]['||f_hat_vec - f_vec||_2'])
    f_hat_data_dict[member_key]['angle_f_hat_f'] = get_angle_between_two_vectors(f_hat_vec, f_vec)
    data_to_return['angle_f_hat_f_list'].append(f_hat_data_dict[member_key]['angle_f_hat_f'])
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

    for key in ['err', 'l2_norm', 'angle_f_hat_f']:
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


def get_player_movements_and_update_modify_data_df(cost_func: MixWeightedLinearSumSquareCostFunction,
                                                   modify_data: pd.DataFrame, index: int, f_hat, x: np.array):
    '''

    :param cost_func: Cost function that detriments the payment of player move.
    :param modify_data: The data frame to update x's movements
    :param index: Index of the player after changed his features.
    :param f_hat: Model trained on the friends of member_key
    :param x: Vector features of player before he moved.
    :return: bool whatever player changed his features and update modify_data in the new features of the player.
    '''
    z = cost_func.maximize_features_against_binary_model(x, f_hat, use_spare_cost=True)
    modify_data.loc[index] = z
    if f_hat.predict(z.reshape(1, -1))[0] == 1:
        did_move = True
        modify_data.loc[index] = z if len(z) > 1 else z[0]
    else:
        did_move = False
        modify_data.loc[index] = x if len(x) > 1 else x[0]
    return did_move


def train_f_hat(sample_friends_from_df:pd.DataFrame, sample_friends_f_loan_status: pd.Series, member_dict: dict,
                member_key: str, feature_to_learn_list: list):
    '''

    :param sample_friends_from_df: Data frame to sample example to learn from.
    :param sample_friends_f_loan_status: Loan status according f model
    :param member_dict: The friends that each sample in the test set has. It means
    :param member_key: The key of the player who wants to change his features.
    :param feature_to_learn_list:
    :return: Model trained on the friends of member_key
    '''
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


def strategic_modify_learn_from_friends(clf_name, orig_df: pd.DataFrame, sample_friends_from_df: pd.DataFrame, clf ,
                                        feature_to_learn_list, cost_func: CostFunction, member_dict: dict, f_vec, dir_name_for_result: str = None,
                                        title_for_visualization: str = None, visualization=True, num_friends=0):
    '''

    :param clf_name: The Name of the classifier
    :param orig_df: Data frame that contains samples that strategically change their features
    :param sample_friends_from_df: The data frame to sample friends from
    :param clf: The model for predictions.
    :param feature_to_learn_list: list of features that are used for predictions
    :param cost_func: Cost function that detriments the payment of player move.
    :param member_dict: Friends dictionary that player can learn from.
    :param f_vec: The vector weights of the model.
    :param dir_name_for_result: In this folder the result will be saved
    :param title_for_visualization: Title of the player movement.
    :param visualization: Whatever we plot the visualization
    :param num_friends: Number of friends player learn from
    :return:
    modify_data: Data after player moved
    data_to_return: data represents some statistic to display in graph (we don't display all statistic in graphs).
    '''

    orig_df_f_loan_status = clf.predict(orig_df[feature_to_learn_list])
    sample_friends_f_loan_status = clf.predict(sample_friends_from_df[feature_to_learn_list])
    modify_data = orig_df[feature_to_learn_list].copy()
    should_save = False
    if dir_name_for_result is not None:
        should_save = True
        models_f_hat_path, data_modified_path, f_hat_data_json_path, data_to_return_path = get_paths_for_running(dir_name_for_result, num_friends)
    f_hat_data_dict = dict()
    data_to_return = init_data_to_return_dict()
    with tqdm(total=len(orig_df)) as t:
        for (index, ex), member_key in zip(orig_df[feature_to_learn_list].iterrows(), orig_df['MemberKey']):
            if os.path.exists(os.path.join(models_f_hat_path, f'f_hat_{member_key}.sav')):
                f_hat = load_model(os.path.join(models_f_hat_path, f'f_hat_{member_key}.sav'))
            else:
                f_hat = train_f_hat(sample_friends_from_df, sample_friends_f_loan_status, member_dict, member_key, feature_to_learn_list)
            f_hat_data_dict[member_key] = dict()

            x = np.array(ex)
            did_changed = False
            if f_hat is not None:
                prediction_f_hat_on_x = f_hat.predict(x.reshape(1, -1))[0]
                if prediction_f_hat_on_x == -1:
                    did_changed = get_player_movements_and_update_modify_data_df(cost_func, modify_data, index, f_hat, x)
            if should_save and f_hat is not None:
                update_dicts_for_f_hat_result(f_hat_data_dict, data_to_return, member_key, orig_df, f_hat,
                                              feature_to_learn_list, orig_df_f_loan_status,
                                              models_f_hat_path, f_vec, did_changed, prediction_f_hat_on_x.item())
            t.update(1)
    if should_save:
        finish_data_dicts_updates(data_to_return, f_hat_data_dict, f_hat_data_json_path, data_to_return_path)
        write_modify_data_with_all_columns(modify_data, orig_df, data_modified_path)
    if visualization:
        visualize_projected_changed_df(clf_name, orig_df, modify_data, feature_to_learn_list, title_for_visualization,
                                       dir_name_for_saving_visualize=dir_name_for_result, f_weights=f_vec[:-1],
                                       f_inter=f_vec[-1], clf=clf)
    return modify_data, data_to_return
