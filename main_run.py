import json
from strategic_players import *
from create_synthetic_data import create_member_friends_dict
from model import *


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


def run_strategic_full_info(train_hardt=False, cost_factor=5, epsilon=0.2,
                            feature_list_to_use=six_most_significant_features):
    dir_name_for_saving_visualize = os.path.join(result_folder_path, 'full_information_strategic')
    os.makedirs(dir_name_for_saving_visualize, exist_ok=True)
    svm_clf = get_svm_loan_return_model(svm_model_loan_returned_path, feature_list_to_use)
    modify_svm_test_df = create_strategic_data_sets_using_known_clf(dir_name_for_saving_visualize,
                                                                    cost_factor=cost_factor, epsilon=epsilon,
                                                                    f=svm_clf, clf_name='SVM')
    modify_svm_test_df.to_csv(os.path.join(dir_name_for_saving_visualize, 'modify_on_svm_test_df.csv'))

    hardt_algo = get_hardt_model(cost_factor, train_path=real_train_val_f_star_loan_status_path,
                                 force_train_hardt=train_hardt)
    modify_hardt_test_df = create_strategic_data_sets_using_known_clf(dir_name_for_saving_visualize,
                                                                      cost_factor=cost_factor, epsilon=epsilon,
                                                                      f=hardt_algo, clf_name='Hardt')
    modify_hardt_test_df.to_csv(os.path.join(dir_name_for_saving_visualize, 'modify_on_hardt_test_df.csv'))


def get_datasets_and_f_grade(f_svm, f_hardt, train_path, test_path, feature_list_to_use):
    test_f_star = pd.read_csv(test_path)
    train_f_star = pd.read_csv(train_path)
    train_svm_loan_status = f_svm.predict(train_f_star[feature_list_to_use])
    test_svm_loan_status = f_svm.predict(test_f_star[feature_list_to_use])
    train_hardt_loan_status = f_hardt.predict(train_f_star[feature_list_to_use])
    test_hardt_loan_status = f_hardt.predict(test_f_star[feature_list_to_use])

    return train_f_star, test_f_star, train_svm_loan_status, test_svm_loan_status, train_hardt_loan_status, test_hardt_loan_status


def strategic_random_friends_info(train_hadart=True, cost_factor=5, epsilon=0.2,
                                  feature_list_to_use=six_most_significant_features, spare_cost=0.2, use_bouth_classes=True):
    def init_dict_result():
        dict_result = dict()
        # dict_result['number_of_friends_to_learn_list'] = [4, 6, 10, 50, 100, 200, 500, 1000, 2000, 4000, 7000, 10000,
        #                                                   15000]
        dict_result['number_of_friends_to_learn_list'] = [4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]
        # dict_result['number_of_friends_to_learn_list'] = [8192]
        dict_result['hardt_friends_acc_list'] = []
        dict_result['svm_model_friends_acc_list'] = []
        dict_result['num_improved_list_and_y_pos_svm'] = []
        dict_result['num_degrade_list_and_y_pos_svm'] = []
        dict_result['num_improved_list_and_y_pos_hardt'] = []
        dict_result['num_degrade_list_and_y_pos_hardt'] = []
        dict_result['num_degrade_list_and_y_pos_hardt'] = []

        dict_result['num_improved_list_and_y_neg_svm'] = []
        dict_result['num_degrade_list_and_y_neg_svm'] = []
        dict_result['num_improved_list_and_y_neg_hardt'] = []
        dict_result['num_degrade_list_and_y_neg_hardt'] = []




        # dict_result['num_improved_hardt_full_info_game_list'] = []
        # dict_result['num_degrade_hardt_full_info_game_list'] = []
        dict_result['avg acc f_hat svm'] = []
        dict_result['avg acc f_hat hardt'] = []
        dict_result['var acc f_hat svm'] = []
        dict_result['var acc f_hat hardt'] = []
        dict_result['l2 f_hat svm dist'] = []
        dict_result['l2 f_hat hardt dist'] = []
        dict_result['var l2 f_hat svm dist'] = []
        dict_result['var l2 f_hat hardt dist'] = []
        dict_result['avg_angel_svm_f_hat'] = []
        dict_result['avg_angle_hardt_f_hat'] = []
        dict_result['var_angel_svm_f_hat'] = []
        dict_result['var_angel_hardt_f_hat'] = []
        dict_result['random_model_acc_list'] = []
        dict_result['number_that_moved_svm_list'] = []
        dict_result['number_that_moved_hardt_list'] = []
        return dict_result

    def create_paths_and_dirs_for_random_friends_experiment(
            experiment='changed_samples_by_gaming_random_friends_losns_status'):
        os.makedirs(result_folder_path, exist_ok=True)
        path_to_parent_folder = safe_create_folder(result_folder_path, experiment)
        path_to_base_output = safe_create_folder(path_to_parent_folder, f'cost_factor={cost_factor}_epsilon={epsilon}')
        path_to_friends_dict_dir = safe_create_folder(path_to_base_output, 'friends_dict')
        svm_folder = safe_create_folder(path_to_base_output, 'svm_results')
        hardt_folder = safe_create_folder(path_to_base_output, 'hardt_results')
        return path_to_parent_folder, path_to_base_output, path_to_friends_dict_dir, svm_folder, hardt_folder

    def update_dict_result():
        dict_result['avg_angel_svm_f_hat'].append(data_svm_res_dict['angle_f_hat_f_avg'])
        dict_result['avg_angle_hardt_f_hat'].append(data_hardt_res_dict['angle_f_hat_f_avg'])
        dict_result['l2 f_hat svm dist'].append(data_svm_res_dict['l2_norm_avg'])
        dict_result['l2 f_hat hardt dist'].append(data_hardt_res_dict['l2_norm_avg'])
        dict_result['avg acc f_hat svm'].append(data_svm_res_dict['acc_avg'])
        dict_result['avg acc f_hat hardt'].append(data_hardt_res_dict['acc_avg'])
        dict_result['var acc f_hat svm'].append(data_svm_res_dict['acc_var'])
        dict_result['var acc f_hat hardt'].append(data_hardt_res_dict['acc_var'])
        dict_result['var_angel_svm_f_hat'].append(data_svm_res_dict['angle_f_hat_f_var'])
        dict_result['var_angel_hardt_f_hat'].append(data_hardt_res_dict['angle_f_hat_f_var'])
        dict_result['var l2 f_hat svm dist'].append(data_svm_res_dict['l2_norm_var'])
        dict_result['var l2 f_hat hardt dist'].append(data_hardt_res_dict['l2_norm_var'])


        test_size = len(real_test_f_star_df)
        f_svm_pred_on_svm_modify = f_svm.predict(friends_modify_on_svm_strategic_data[feature_list_to_use])
        dict_result['num_improved_list_and_y_pos_svm'].append(
            100 * 1/ test_size * np.sum((f_svm_pred_on_svm_modify > test_pred_loans_status_svm) &
                                        (real_test_f_star_df['LoanStatus'] == 1)).item())
        dict_result['num_improved_list_and_y_neg_svm'].append(
            100 * 1/ test_size * np.sum((f_svm_pred_on_svm_modify > test_pred_loans_status_svm) &
                                        (real_test_f_star_df['LoanStatus'] == -1)).item())

        dict_result['num_degrade_list_and_y_pos_svm'].append(
            100 * 1 / test_size * np.sum((f_svm_pred_on_svm_modify < test_pred_loans_status_svm) &
                                        (real_test_f_star_df['LoanStatus'] == 1)).item())
        dict_result['num_degrade_list_and_y_neg_svm'].append(
            100 * 1 / test_size * np.sum((f_svm_pred_on_svm_modify < test_pred_loans_status_svm) &
                                         (real_test_f_star_df['LoanStatus'] == -1)).item())

        f_acc = np.sum(f_svm_pred_on_svm_modify == friends_modify_on_svm_strategic_data['LoanStatus']).item() / len(
            friends_modify_on_svm_strategic_data)
        dict_result['svm_model_friends_acc_list'].append(f_acc)
        print(f'svm acc: {f_acc}')

        random_model = RandomModel()
        dict_result['random_model_acc_list'].append(
            evaluate_model_on_test_set(real_test_f_star_df, random_model, feature_list_to_use))

        f_hardt_pred_on_hardt_modify = hardt_algo(friends_modify_on_hardt_strategic_data[feature_list_to_use])

        dict_result['num_improved_list_and_y_pos_hardt'].append(
            100 * 1 / test_size * np.sum((f_hardt_pred_on_hardt_modify > test_pred_loans_status_hardt) & (real_test_f_star_df['LoanStatus'] == 1)).item())
        dict_result['num_improved_list_and_y_neg_hardt'].append(
            100 * 1 / test_size * np.sum((f_hardt_pred_on_hardt_modify > test_pred_loans_status_hardt) & (
                        real_test_f_star_df['LoanStatus'] == -1)).item())

        dict_result['num_degrade_list_and_y_pos_hardt'].append(
            100 * 1 / test_size * np.sum((f_hardt_pred_on_hardt_modify < test_pred_loans_status_hardt) & (
                        real_test_f_star_df['LoanStatus'] == 1)).item())
        dict_result['num_degrade_list_and_y_neg_hardt'].append(
            100 * 1 / test_size * np.sum((f_hardt_pred_on_hardt_modify < test_pred_loans_status_hardt) & (
                    real_test_f_star_df['LoanStatus'] == -1)).item())
        hardt_acc = np.sum(
            f_hardt_pred_on_hardt_modify == friends_modify_on_hardt_strategic_data['LoanStatus']).item() / len(
            friends_modify_on_svm_strategic_data)
        print(hardt_acc)
        dict_result['hardt_friends_acc_list'].append(hardt_acc)
        dict_result['number_that_moved_svm_list'].append(100 * 1 / test_size * data_svm_res_dict['number_moved'])
        dict_result['number_that_moved_hardt_list'].append(100 * 1 / test_size * data_hardt_res_dict['number_moved'])

    def plot_dict_result_graph():
        def copy_value_to_list(value, length):
            return [value for _ in range(length)]

        full_info_modify_on_svm_strategic_data = pd.read_csv(svm_modify_full_information_real_test_path)
        full_info_modify_on_hardt_strategic_data = pd.read_csv(hardt_modify_full_information_real_test_path)
        svm_acc_test_modify = evaluate_model_on_test_set(full_info_modify_on_svm_strategic_data, f_svm, feature_list_to_use)

        hardt_acc_test_modify = evaluate_model_on_test_set(full_info_modify_on_hardt_strategic_data, hardt_algo, feature_list_to_use)

        svm_acc_test_not_modify = evaluate_model_on_test_set(real_test_f_star_df, f_svm, feature_list_to_use)
        hardt_acc_test_not_modify = evaluate_model_on_test_set(real_test_f_star_df, hardt_algo, feature_list_to_use)
        num_friends_exp = len(dict_result['number_of_friends_to_learn_list'])
        x_data_list = [dict_result['number_of_friends_to_learn_list'] for _ in range(6)]
        y_data_list = [dict_result['svm_model_friends_acc_list'], dict_result['hardt_friends_acc_list']]
        y_data_list.append(copy_value_to_list(svm_acc_test_modify, num_friends_exp))
        y_data_list.append(copy_value_to_list(hardt_acc_test_modify, num_friends_exp))
        y_data_list.append(copy_value_to_list(svm_acc_test_not_modify, num_friends_exp))
        y_data_list.append(copy_value_to_list(hardt_acc_test_not_modify, num_friends_exp))
        graph_label_list = ['svm learn from friends', 'Hardt learn from friends', 'svm full information',
                            'hardt full information', 'svm no change', 'hardt no change']
        saving_path = os.path.join(base_output_path, 'accuracy_vs_num_friends.png')
        plot_graph(title='accuracy vs number of random friends to learn',
                   x_label='number of random friend to learn',
                   y_label='accuracy', x_data_list=x_data_list, y_data_list=y_data_list,
                   graph_label_list=graph_label_list, saving_path=saving_path)

        x_data_list = [dict_result['number_of_friends_to_learn_list'] for _ in range(2)]
        y_data_list = [dict_result['num_improved_list_and_y_pos_svm'], dict_result['num_improved_list_and_y_pos_hardt']]
        saving_path = os.path.join(base_output_path, 'num_pos_label_improved_vs_num_friends.png')
        plot_graph(title='percent players that with positive label and improved on model vs number of random friends to learn',
                   x_label='number of friends', y_label='percent improved', x_data_list=x_data_list,
                   y_data_list=y_data_list,
                   graph_label_list=['improved on svm model', 'improved on Hardt model'], saving_path=saving_path)

        x_data_list = [dict_result['number_of_friends_to_learn_list'] for _ in range(2)]
        y_data_list = [dict_result['num_improved_list_and_y_neg_svm'], dict_result['num_improved_list_and_y_neg_hardt']]
        saving_path = os.path.join(base_output_path, 'num_neg_label_improved_vs_num_friends.png')
        plot_graph(
            title='percent players that with negative label and improved on model vs number of random friends to learn',
            x_label='number of friends', y_label='percent improved', x_data_list=x_data_list,
            y_data_list=y_data_list,
            graph_label_list=['improved on svm model', 'improved on Hardt model'], saving_path=saving_path)

        y_data_list = [dict_result['num_degrade_list_and_y_pos_svm'], dict_result['num_degrade_list_and_y_pos_hardt']]
        saving_path = os.path.join(base_output_path, 'num_pos_label_degrade_vs_num_friends.png')
        plot_graph(title='percent players that with positive label and degrade on model vs number of random friends to learn',
                   x_label='number of friends', y_label='percent degrade', x_data_list=x_data_list, y_data_list=y_data_list,
                   graph_label_list=['degrade on svm model', 'degrade on Hardt model'], saving_path=saving_path)

        y_data_list = [dict_result['num_degrade_list_and_y_neg_svm'], dict_result['num_degrade_list_and_y_neg_hardt']]
        saving_path = os.path.join(base_output_path, 'num_neg_label_degrade_vs_num_friends.png')
        plot_graph(
            title='percent players that with negative label and degrade on model vs number of random friends to learn',
            x_label='number of friends', y_label='percent degrade', x_data_list=x_data_list, y_data_list=y_data_list,
            graph_label_list=['degrade on svm model', 'degrade on Hardt model'], saving_path=saving_path)

        y_data_list = [dict_result['avg acc f_hat svm'], dict_result['avg acc f_hat hardt']]
        saving_path = os.path.join(base_output_path, 'f_hat_avg_acc_vs_num_friends.png')
        var_lists = [dict_result['var acc f_hat svm'], dict_result['var acc f_hat hardt']]
        plot_graph(title=r'$\^{f}$ avg acc vs number of random friends to learn',
                   x_label='number of friends',
                   y_label='avg acc', x_data_list=x_data_list, y_data_list=y_data_list,
                   graph_label_list=['svm model', 'Hardt model'],
                   saving_path=saving_path, var_lists=var_lists)

        y_data_list = [dict_result['l2 f_hat svm dist'], dict_result['l2 f_hat hardt dist']]
        saving_path = os.path.join(base_output_path, 'f_hat_dist_f_vs_num_friends.png')
        var_lists = [dict_result['var l2 f_hat svm dist'], dict_result['var l2 f_hat hardt dist']]
        plot_graph(title=r'$\^{f}$ dist from f vs number of random friends to learn',
                   x_label='number of friends',
                   y_label='dist l2', x_data_list=x_data_list, y_data_list=y_data_list,
                   graph_label_list=['svm model', 'Hardt model'],
                   saving_path=saving_path, var_lists=var_lists)

        y_data_list = [dict_result['avg_angel_svm_f_hat'], dict_result['avg_angle_hardt_f_hat']]
        saving_path = os.path.join(base_output_path, 'avg angle_between_f_hat_and_f_vs_num_friends.png')
        var_lists = [dict_result['var_angel_svm_f_hat'], dict_result[r'var_angel_hardt_f_hat']]
        plot_graph(title=r'angle between $\^{f}$ and f vs number of random friends to learn',
                   x_label='number of friends',
                   y_label='angle', x_data_list=x_data_list, y_data_list=y_data_list,
                   graph_label_list=['svm model', 'Hardt model'],
                   saving_path=saving_path, var_lists=var_lists)

        y_data_list = [dict_result['number_that_moved_svm_list'], dict_result['number_that_moved_hardt_list']]
        saving_path = os.path.join(base_output_path, 'number_that_moved_vs_num_friends.png')
        plot_graph(title='number that moved vs number of random friends to learn',
                   x_label='number of friends',
                   y_label='number that moved', x_data_list=x_data_list, y_data_list=y_data_list, graph_label_list=['svm model', 'Hardt model'],
                   saving_path=saving_path)

    hardt_algo = get_hardt_model(cost_factor, real_train_f_star_loan_status_path, train_hadart)
    f_svm = load_model(svm_model_loan_returned_path)
    f_svm_vec = np.append(f_svm.coef_[0], f_svm.intercept_)
    f_hardt_vec = np.append(hardt_algo.coef_[0], hardt_algo.intercept_)
    dict_result = init_dict_result()
    data_tuple = get_datasets_and_f_grade(f_svm, hardt_algo, real_train_val_f_star_loan_status_path,
                                          real_test_f_star_loan_status_path, feature_list_to_use)

    real_train_val_f_star_df, real_test_f_star_df, real_train_val_svm_loan_status, test_pred_loans_status_svm, \
    real_train_val_hardt_loan_status, test_pred_loans_status_hardt = data_tuple

    tuple_path_folders = create_paths_and_dirs_for_random_friends_experiment()
    parent_folder_path, base_output_path, friends_dict_dir_path, svm_folder, hardt_folder = tuple_path_folders

    for num_friend in dict_result['number_of_friends_to_learn_list']:
        print(num_friend)
        member_friend_dict_path = os.path.join(friends_dict_dir_path, f'random_{num_friend}friends_for_svm.json')
        member_dict = create_member_friends_dict(num_friend, real_train_val_svm_loan_status,
                                                 real_test_f_star_df, member_friend_dict_path,
                                                 force_to_crate=False, use_both_classes=use_bouth_classes)

        cost_func_for_gaming = MixWeightedLinearSumSquareCostFunction(a, epsilon=epsilon, cost_factor=cost_factor, spare_cost=spare_cost)

        friends_modify_on_svm_strategic_data, data_svm_res_dict = strategic_modify_learn_from_friends(
                                                                                             'SVM', test_pred_loans_status_svm,
                                                                                              real_test_f_star_df,
                                                                                              real_train_val_f_star_df,
                                                                                              real_train_val_svm_loan_status,
                                                                                              feature_list_to_use,
                                                                                              cost_func_for_gaming,
                                                                                              member_dict=member_dict,
                                                                                              f_vec=f_svm_vec,
                                                                                              dir_name_for_result=svm_folder,
                                                                                              title_for_visualization=f'real test learned on svm{num_friend}',
                                                                                              num_friends=num_friend
                                                                                              )


        cost_func_for_gaming = MixWeightedLinearSumSquareCostFunction(a, epsilon=epsilon, cost_factor=cost_factor, spare_cost=spare_cost)
        member_friend_dict_path = os.path.join(friends_dict_dir_path, f'random_{num_friend}friends_for_hardt.json')
        member_dict = create_member_friends_dict(num_friend, real_train_val_hardt_loan_status,
                                                 real_test_f_star_df, member_friend_dict_path,
                                                 force_to_crate=False, use_both_classes=use_bouth_classes)
        friends_modify_on_hardt_strategic_data, data_hardt_res_dict = strategic_modify_learn_from_friends(
                                                                                        'Hardt', test_pred_loans_status_hardt,
                                                                                        real_test_f_star_df,
                                                                                        real_train_val_f_star_df,
                                                                                        real_train_val_hardt_loan_status,
                                                                                        feature_list_to_use, cost_func_for_gaming,
                                                                                        member_dict=member_dict,
                                                                                        f_vec=f_hardt_vec,
                                                                                        dir_name_for_result=hardt_folder,
                                                                                        title_for_visualization=f'real test learned on hardt{num_friend}',
                                                                                        num_friends=num_friend
                                                                                    )


        update_dict_result()

    with open(os.path.join(base_output_path, 'final_random_friends_dict_result.json'), 'w') as json_file:
        json.dump(dict_result, json_file, indent=4)
    plot_dict_result_graph()



def create_main_folders():
    os.makedirs(result_folder_path, exist_ok=True)
    os.makedirs(models_folder_path, exist_ok=True)




if __name__ == '__main__':
    cost_factor = 5
    epsilon = 0.2
    spare_cost = 0
    run_strategic_full_info(train_hardt=False, cost_factor=cost_factor, epsilon=epsilon)
    strategic_random_friends_info(train_hadart=False, cost_factor=cost_factor, epsilon=epsilon, spare_cost=spare_cost, use_bouth_classes=True)

