from utills_and_consts import *
from model import *
from collections import defaultdict
import json


class ResultExpContainer:
    def __init__(self, number_of_friends_to_learn_list: list, feature_list_to_use: list, f_svm, hardt_f, base_output_path):
        self.svm_f = f_svm
        self.hardt_f = hardt_f
        dict_result = defaultdict(list)
        self.m_list = number_of_friends_to_learn_list
        self.dict_result = dict_result
        self.feature_list_to_use = feature_list_to_use
        self.base_output_path = base_output_path

    def dump_dict_result(self):
        with open(os.path.join(self.base_output_path, 'final_random_friends_dict_result.json'), 'w') as json_file:
            json.dump(self.dict_result, json_file, indent=4)

    def update_dict_result(self, data_svm_res_dict, data_hardt_res_dict, friends_modify_on_svm_strategic_data, friends_modify_on_hardt_strategic_data):
        self.dict_result['avg_angel_svm_f_hat'].append(data_svm_res_dict['angle_f_hat_f_avg'])
        self.dict_result['avg_angle_hardt_f_hat'].append(data_hardt_res_dict['angle_f_hat_f_avg'])
        self.dict_result['l2 f_hat svm dist'].append(data_svm_res_dict['l2_norm_avg'])
        self.dict_result['l2 f_hat hardt dist'].append(data_hardt_res_dict['l2_norm_avg'])

        self.dict_result['avg err f_hat svm'].append(data_svm_res_dict['err_avg'])
        self.dict_result['avg err f_hat hardt'].append(data_hardt_res_dict['err_avg'])

        self.dict_result['var err f_hat svm'].append(data_svm_res_dict['err_var'])
        self.dict_result['var err f_hat hardt'].append(data_hardt_res_dict['err_var'])
        self.dict_result['var_angel_svm_f_hat'].append(data_svm_res_dict['angle_f_hat_f_var'])
        self.dict_result['var_angel_hardt_f_hat'].append(data_hardt_res_dict['angle_f_hat_f_var'])
        self.dict_result['var l2 f_hat svm dist'].append(data_svm_res_dict['l2_norm_var'])
        self.dict_result['var l2 f_hat hardt dist'].append(data_hardt_res_dict['l2_norm_var'])

        f_err = evaluate_model_on_test_set(friends_modify_on_svm_strategic_data, self.svm_f, self.feature_list_to_use)
        self.dict_result['svm_model_friends_err_list'].append(f_err)
        print(f'svm err: {f_err}')

        hardt_err = evaluate_model_on_test_set(friends_modify_on_hardt_strategic_data, self.hardt_f, self.feature_list_to_use)

        print(f'hardt err: {hardt_err}')
        self.dict_result['hardt_friends_err_list'].append(hardt_err)

    def get_err_list_for_pop_graph(self, test_f_star: pd.DataFrame):
        '''

        :param test_f_star: Test set with loan status
        :return: List of list errors graph for pop ploting
        '''
        svm_err_random_sample_list = [err for err in self.dict_result['svm_model_friends_err_list']]
        hardt_err_random_sample_list = [err for err in self.dict_result['hardt_friends_err_list']]
        full_info_modify_svm_df = pd.read_csv(svm_modify_full_information_real_test_path)
        full_info_modify_hardt_df = pd.read_csv(hardt_modify_full_information_real_test_path)
        svm_full_info_err = evaluate_model_on_test_set(full_info_modify_svm_df, self.svm_f, self.feature_list_to_use)
        hardt_full_info_err = evaluate_model_on_test_set(full_info_modify_hardt_df, self.hardt_f, self.feature_list_to_use)
        svm_full_info_err_list = [svm_full_info_err for _ in self.m_list]
        hardt_full_info_err_list = [hardt_full_info_err for _ in self.m_list]
        svm_no_change_err = evaluate_model_on_test_set(test_f_star, self.svm_f, self.feature_list_to_use)
        svm_no_change_err_list = [svm_no_change_err for _ in self.m_list]
        hardt_no_change_err = evaluate_model_on_test_set(test_f_star, self.hardt_f, self.feature_list_to_use)
        hardt_no_change_err_list = [hardt_no_change_err for _ in self.m_list]
        y_data_list = [svm_err_random_sample_list,
                       hardt_err_random_sample_list,
                       svm_full_info_err_list,
                       hardt_full_info_err_list,
                       svm_no_change_err_list,
                       hardt_no_change_err_list]
        return y_data_list

    def plot_pop_graph(self, test_f_star, spare_cost):
        '''

        :param test_f_star: Test set with loan status
        :return: Plot the pop graph according the experiment result
        '''
        figsize_movements_plot = (8.5, 4.8)
        saving_path = os.path.join(self.base_output_path, 'number_that_moved_vs_num_friends.png')
        err_f_hat_hardt_list = [err for err in self.dict_result['avg err f_hat hardt']]
        err_f_hat_svm_list = [err for err in self.dict_result['avg err f_hat svm']]
        var_err_f_fhat_svm_list = self.dict_result['var err f_hat svm']
        var_err_f_fhat_hardt_list = self.dict_result['var err f_hat hardt']
        y_data_list = self.get_err_list_for_pop_graph(test_f_star)
        graph_label_list = ['SVM(in the dark)', 'HMPW(in the dark)', 'SVM(fully-informed)',
                            'HMPW(fully-informed)', 'SVM(non-strategic)', 'HMPW(non-strategic)']
        safety_str = ''
        if spare_cost != 0:
            safety_str = f' (safety={spare_cost})'
        title = 'POP in Prosper.com loans data' + safety_str
        color_list = ['orange', 'blue', 'orange', 'blue', 'orange', 'blue', 'm', 'c']
        alpha_list = [1, 1, 0.4, 0.4, 0.4, 0.4]
        ls_list = ['solid', 'solid', '--', '--', 'dotted', 'dotted']

        fig, ax1 = plt.subplots(figsize=figsize_movements_plot)
        plt.xlim([4, 8192])
        plt.title(title, fontsize=16)
        plt.xlabel('m', fontsize=15)

        plt.xscale('log')
        plt.ylabel('error', fontsize=15)
        for i in range(len(y_data_list)):
            ax1.plot(self.m_list, y_data_list[i], color_list[i], label=graph_label_list[i], ls=ls_list[i],
                     alpha=alpha_list[i])
            plt.legend(loc="lower center", bbox_to_anchor=(0.5, 0.8), ncol=3, prop={'size': 10})

        ax2 = fig.add_axes([0.5, 0.55, 0.4, 0.18])
        ax2.plot(self.m_list, err_f_hat_svm_list,'darkorange', label=r'SVM')
        var_below = [y - np.sqrt(var).item() for y, var in zip(err_f_hat_svm_list, var_err_f_fhat_svm_list)]
        var_above = [y + np.sqrt(var).item() for y, var in zip(err_f_hat_svm_list, var_err_f_fhat_svm_list)]
        ax2.fill_between(self.m_list, var_below, var_above, edgecolor='orange', alpha=0.2)
        ax2.plot(self.m_list, err_f_hat_hardt_list,'b', label=r'HARDT')
        var_below = [y - np.sqrt(var).item() for y, var in zip(err_f_hat_hardt_list, var_err_f_fhat_hardt_list)]
        var_above = [y + np.sqrt(var).item() for y, var in zip(err_f_hat_hardt_list, var_err_f_fhat_hardt_list)]
        ax2.fill_between(self.m_list, var_below, var_above, edgecolor='b', alpha=0.2)
        ax2.set_xscale('log')
        ax2.set_ylabel(r'$\varepsilon_2$', fontsize=15)
        ax2.set_xlabel('m', fontsize=15)

        ax2.legend(loc="upper right")

        #### finish
        hardt_full_info_err_list, hardt_err_random_sample_list = y_data_list[3], y_data_list[1]
        ax1.fill_between(self.m_list, hardt_full_info_err_list, hardt_err_random_sample_list, alpha=0.2,
                         color='lightgray', hatch="\\\\\\\\")
        if len(self.m_list) > 2:
            m_to_write_arrow = self.m_list[2]
            ax1.arrow(m_to_write_arrow, hardt_full_info_err_list[2], 0,
                      hardt_err_random_sample_list[2] - hardt_full_info_err_list[2],
                      shape='full', color='black', length_includes_head=True,
                      zorder=0, head_length=0.01, head_width=2)
            ax1.arrow(m_to_write_arrow, hardt_err_random_sample_list[2], 0,
                      hardt_full_info_err_list[2] - hardt_err_random_sample_list[2],
                      shape='full', color='black', length_includes_head=True,
                      zorder=0, head_length=0.01, head_width=2)

            ax1.text(m_to_write_arrow - 10, (hardt_full_info_err_list[2] + hardt_err_random_sample_list[2]) / 2, r'POP',
                     fontsize=15)
        plt.savefig(saving_path, dpi=300)
        plt.show()

        for (m, err) in zip(self.m_list, hardt_err_random_sample_list):
            pop = err - hardt_full_info_err_list[0] # note that hardt_full_info_err_list is the same value for all list.
            print(f'number of friends: {m} and pop:{pop} err: {err}')

    def get_percent_f_on_delta_fhat_minus_1_list(self, m_list, modified_data_folder, test_changed_on_hardt, y_1_and_f_delta_f_1):
        percent_f_on_delta_fhat_minus_1_list = list()
        for m in m_list:
            modified_by_f_hat_path = os.path.join(modified_data_folder, f'modified_test_{m}.csv')
            modify_data_f_hat = pd.read_csv(modified_by_f_hat_path)
            num_f_on_delta_fhat_minus_1 = len(modify_data_f_hat[
                                        (self.hardt_f.predict(modify_data_f_hat[self.feature_list_to_use]) == -1)
                                        & (modify_data_f_hat['LoanStatus'] == 1) & (self.hardt_f.predict(test_changed_on_hardt[self.feature_list_to_use]) == 1)])
            percent_f_on_delta_fhat_minus_1_list.append(100 * num_f_on_delta_fhat_minus_1 / len(y_1_and_f_delta_f_1))
        return percent_f_on_delta_fhat_minus_1_list

    def plot_social_inequality_graph(self, modified_data_folder: str, spare_cost: float, test_size: int):
        test_changed_on_hardt = pd.read_csv(hardt_modify_full_information_real_test_path)[:test_size]
        y_1_data_sets = test_changed_on_hardt[test_changed_on_hardt['LoanStatus'] == 1]
        y_1_and_f_delta_f_1 = y_1_data_sets[self.hardt_f.predict(y_1_data_sets[self.feature_list_to_use]) == 1]

        percent_f_on_delta_fhat_minus_1_list = self.get_percent_f_on_delta_fhat_minus_1_list(self.m_list, modified_data_folder,
                                                                                             test_changed_on_hardt, y_1_and_f_delta_f_1)
        path_to_save = os.path.join(self.base_output_path, 'fairness_graph.png')

        plt.figure(figsize=(4, 5))
        plt.ylim([0, 55])
        plt.xlim([4, 4096])
        plt.grid()
        safety_str = ''
        if spare_cost != 0:
            safety_str = f'safety={spare_cost}'
        plot_graph(title='Social inequality ' + r'($y=1$ and $f(\Delta_f(x))=1$)' + '\n' + safety_str, x_label='m',
                   y_label=r'% classified correctly', x_data_list=[self.m_list],
                   y_data_list=[percent_f_on_delta_fhat_minus_1_list], saving_path=path_to_save,
                   graph_label_list=[r'$f(\Delta_{\^{f}}(x))=-1$'], symlog_scale=True, title_size=None, y_fontsize=None,
                   x_fontsize=None)
