from utills_and_consts import *
from loan_status_model_trainer import get_svm_loan_return_model
from cost_functions import MixWeightedLinearSumSquareCostFunction
from strategic_players import strategic_modify_using_known_clf
from strategic_players import visualize_projected_changed_df
from strategic_players import get_hardt_model


def print_evaluation_on_modify_test(modify_test: pd.DataFrame, clf, feature_list_to_use: list, clf_name: str):
    '''

    :param modify_test: The test to evaluate the model on.
    :param clf: Trained model
    :param feature_list_to_use: List of features that the clf model uses for prediction
    :param clf_name: The name of the classifier
    :return: Prints the model error on the modify test.
    '''
    err_modify_full_information_test_modify = evaluate_model_on_test_set(modify_test, clf,
                                                                         feature_list_to_use)
    print(
        f'The error on the test set when it clf is {clf_name} trained on not modify train but the test did strategic'
                                                                     f'modify {err_modify_full_information_test_modify}')


def create_strategic_data_sets_using_known_clf(dir_name_for_saving_visualization: str, cost_factor: float, epsilon: float,
                                               f, save_visualization_projected_changed: bool = True, clf_name: str = '',
                                               features_to_use: list = six_most_significant_features, test_size=-1,
                                               show_flag: bool = True):
    '''

    :param dir_name_for_saving_visualization: Path to save the projection visualization
    :param cost_factor: Parameter that determines the scale of the cost function.
    :param epsilon: The weight of the l2 cost function.
    :param f: The model that player wants to achieve positive score.
    :param save_visualization_projected_changed: Whatever to save the projected visualization image
    :param clf_name: The name of the classifier. This will be printed on the projected visualization image
    :param features_to_use: list of features to use for predictions
    :param test_size: The number of Constants.
    :param show_flag: If it 1 the visualization is plotted.
    :return: Strategic modify dataset of the test set according to classifier f.
    '''
    real_test_f_star_loan_status = get_data_with_right_size(real_test_f_star_loan_status_path, test_size)
    f_weights, f_inter = f.coef_[0], f.intercept_

    weighted_linear_cost = MixWeightedLinearSumSquareCostFunction(a, cost_factor=cost_factor, epsilon=epsilon)
    modify_full_information_test = strategic_modify_using_known_clf(real_test_f_star_loan_status, f, features_to_use,
                                                                    weighted_linear_cost)

    visualize_projected_changed_df(clf_name, real_test_f_star_loan_status, modify_full_information_test, features_to_use,
                                   f'real test learned known on {clf_name}',
                                   to_save=save_visualization_projected_changed,
                                   dir_name_for_saving_visualize=dir_name_for_saving_visualization, f_weights=f_weights,
                                   f_inter=f_inter, show_flag=show_flag)
    return modify_full_information_test


def run_strategic_full_info(train_hardt=False, cost_factor=5, epsilon=0.2,
                            feature_list_to_use=six_most_significant_features, force_train_loan_model=False, use_cv=False,
                            train_size=-1, test_size=-1, show_flag=True):
    '''

    :param train_hardt: Whatever retrain Hardt model (If it exists)
    :param cost_factor: Parameter that determines the scale of the cost function.
    :param epsilon: The weight of the l2 cost function in the total cost constant has to pay for movement.
    :param feature_list_to_use: List of features to use for predictions
    :param force_train_loan_model: Whatever retrain loan model (If it exists)
    :param use_cv: Whatever we use cross validation
    :param train_size: The size of the training set to train Hardt model (svm always trains on all train set)
    :param test_size: The number of Constants.
    :param show_flag: If it 1 the visualization is plotted.
    :return:
    '''
    dir_name_for_saving_visualize = os.path.join(result_folder_path, 'full_information_strategic')
    os.makedirs(dir_name_for_saving_visualize, exist_ok=True)

    svm_clf = get_svm_loan_return_model(svm_model_loan_returned_path, feature_list_to_use, force_train_loan_model, use_cv)
    modify_svm_full_information_test = create_strategic_data_sets_using_known_clf(dir_name_for_saving_visualize,
                                                                    cost_factor=cost_factor, epsilon=epsilon,
                                                                    f=svm_clf, clf_name='SVM', test_size=test_size,
                                                                                  show_flag=show_flag)

    modify_svm_full_information_test.to_csv(os.path.join(dir_name_for_saving_visualize, 'modify_on_svm_test_df.csv'))
    print_evaluation_on_modify_test(modify_svm_full_information_test, svm_clf, feature_list_to_use, 'SVM')

    hardt_model = get_hardt_model(cost_factor, train_path=real_train_val_f_star_loan_status_path,
                                 force_train_hardt=train_hardt, train_size=train_size)
    modify_hardt_test_df = create_strategic_data_sets_using_known_clf(dir_name_for_saving_visualize,
                                                                      cost_factor=cost_factor, epsilon=epsilon,
                                                                      f=hardt_model, clf_name='Hardt',
                                                                      test_size=test_size, show_flag=show_flag)
    print_evaluation_on_modify_test(modify_hardt_test_df, hardt_model, feature_list_to_use, 'Hardt')
    modify_hardt_test_df.to_csv(os.path.join(dir_name_for_saving_visualize, 'modify_on_hardt_test_df.csv'))
