from utills_and_consts import *
from strategic_players import get_hardt_model
from loan_status_model_trainer import get_svm_loan_return_model
from cost_functions import MixWeightedLinearSumSquareCostFunction
from friends_dict_creator import get_member_friends_dict
from result_exp_container import ResultExpContainer
from strategic_players import strategic_modify_learn_from_friends


class StrategicRandomFriendsRunner:
    def get_datasets_and_f_grade(self, test_size: int):
        '''

        :Sets in class field that contains datasets of train+validation and test with the loan status label
        '''
        self.test_f_star = get_data_with_right_size(real_test_f_star_loan_status_path, test_size)
        self.train_val_f_star = pd.read_csv(real_train_val_f_star_loan_status_path)
        self.train_val_svm_loan_status = self.f_svm.predict(self.train_val_f_star[self.feature_list_to_use])
        self.test_svm_loan_status = self.f_svm.predict(self.test_f_star[self.feature_list_to_use])
        self.train_val_hardt_loan_status = self.f_hardt.predict(self.train_val_f_star[self.feature_list_to_use])
        self.test_hardt_loan_status = self.f_hardt.predict(self.test_f_star[self.feature_list_to_use])

    def create_paths_and_dirs_for_random_friends_experiment(self):
        os.makedirs(result_folder_path, exist_ok=True)
        path_to_parent_folder = safe_create_folder(result_folder_path, self.experiment)
        self.path_to_base_output = safe_create_folder(path_to_parent_folder, f'cost_factor={self.cost_factor}_epsilon={self.epsilon}')
        self.friends_dict_dir_path = safe_create_folder(self.path_to_base_output, 'friends_dict')
        self.svm_folder = safe_create_folder(self.path_to_base_output, 'svm_results')
        self.hardt_folder = safe_create_folder(self.path_to_base_output, 'hardt_results')

    def get_models(self, force_train_hadart, force_train_loan_model, feature_list_to_use, use_cv, train_size):
        self.feature_list_to_use = feature_list_to_use
        self.f_hardt = get_hardt_model(self.cost_factor, real_train_val_f_star_loan_status_path, force_train_hadart, train_size=train_size)
        self.f_svm = get_svm_loan_return_model(svm_model_loan_returned_path, feature_list_to_use,
                                               force_train_loan_model, use_cv=use_cv)
        self.f_svm_vec = np.append(self.f_svm.coef_[0], self.f_svm.intercept_)
        self.f_hardt_vec = np.append(self.f_hardt.coef_[0], self.f_hardt.intercept_)

    def __init__(self, number_of_friends_to_learn_list: list, cost_factor: float, epsilon: float, feature_list_to_use: list,
                 force_train_hadart: bool, force_train_svm_loan_model: bool, spare_cost: float, use_cv: bool,
                 train_size: int, test_size: int):
        self.m_list = number_of_friends_to_learn_list
        self.cost_factor = cost_factor
        self.epsilon = epsilon
        self.spare_cost = spare_cost
        self.experiment = 'dark_exp'
        self.create_paths_and_dirs_for_random_friends_experiment()
        self.get_models(force_train_hadart, force_train_svm_loan_model, feature_list_to_use, use_cv=use_cv, train_size=train_size)
        self.get_datasets_and_f_grade(test_size)

    def run_strategic_random_exp_for_clf(self, num_friend: int, clf_name: str, train_val_model_loan_status,
                                         clf, clf_folder: str, show_flag: bool, save_flag: bool, test_size: int):
        '''

        :param num_friend: Number of samples that each instance can learn from in this experiment.
        :param clf_name: Name of the model
        :param train_val_model_loan_status: Data frame column for the loan status
        :param clf: The classifier for prediction that the experiment run on.
        :param clf_folder: Folder where results are stored
        :param show_flag: If it 1 the visualization is plotted.
        :param save_flag: If it 1 Extra data about the experiment is saved
        :param test_size: Number of example in the test set that tries to achieve positive score on the model
        :return:
        '''
        member_friend_dict_path = os.path.join(self.friends_dict_dir_path, f'random_{num_friend}friends_for_{clf_name}.json')
        member_dict = get_member_friends_dict(num_friend, test_size, train_val_model_loan_status,
                                              list(self.test_f_star['MemberKey']), member_friend_dict_path, test_size)

        cost_func_for_gaming = MixWeightedLinearSumSquareCostFunction(a, epsilon=self.epsilon,
                                                                      cost_factor=self.cost_factor,
                                                                      spare_cost=self.spare_cost)

        friends_modify_on_svm_strategic_data, data_svm_res_dict = strategic_modify_learn_from_friends(
                                                clf_name, self.test_f_star, self.train_val_f_star,
                                                clf, self.feature_list_to_use, cost_func_for_gaming,
                                                member_dict=member_dict, f_vec=self.f_svm_vec, dir_name_for_result=clf_folder,
                                                title_for_visualization=f'Movement in the dark m = {num_friend}',
                                                num_friends=num_friend, show_flag=show_flag, save_flag=save_flag
                                            )
        return friends_modify_on_svm_strategic_data, data_svm_res_dict

    def execute_strategic_random_friends_exp(self, show_flag, save_flag, test_size):
        result_exp = ResultExpContainer(self.m_list, self.feature_list_to_use, self.f_svm, self.f_hardt, self.path_to_base_output)
        for num_friend in self.m_list:
            print(num_friend)
            friends_modify_on_svm_strategic_data, data_svm_res_dict = self.run_strategic_random_exp_for_clf(num_friend, 'SVM', self.train_val_svm_loan_status, self.f_svm, self.svm_folder, show_flag, save_flag, test_size)
            friends_modify_on_hardt_strategic_data, data_hardt_res_dict = self.run_strategic_random_exp_for_clf(num_friend, 'Hardt', self.train_val_hardt_loan_status, self.f_hardt, self.hardt_folder, show_flag, save_flag, test_size)
            result_exp.update_dict_result(data_svm_res_dict, data_hardt_res_dict,
                           friends_modify_on_svm_strategic_data, friends_modify_on_hardt_strategic_data)
        result_exp.plot_pop_graph(self.test_f_star, self.spare_cost)
        modified_data_folder = os.path.join(self.hardt_folder, 'modified_data')
        result_exp.plot_social_inequality_graph(modified_data_folder, self.spare_cost, test_size)
        return result_exp
