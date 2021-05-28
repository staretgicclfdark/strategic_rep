import numpy as np
import pickle
import matplotlib.pyplot as plt
import os
import pandas as pd


def safe_create_folder(parent_folder_path, folder_name):
    path_folder = os.path.join(parent_folder_path, folder_name)
    os.makedirs(path_folder, exist_ok=True)
    return path_folder


def plot_variance_line(x_data, y_data, var_list, color, SE, num_samples):
    if SE:
        var_below = [y - np.sqrt(var).item() / np.sqrt(num_samples) for y, var in zip(y_data, var_list)]
        var_above = [y + np.sqrt(var).item() / np.sqrt(num_samples) for y, var in zip(y_data, var_list)]
    else:
        var_below = [y - np.sqrt(var).item() for y, var in zip(y_data, var_list)]
        var_above = [y + np.sqrt(var).item() for y, var in zip(y_data, var_list)]
    plt.fill_between(x_data, var_below, var_above, edgecolor=color, alpha=0.2)


def plot_graph(title: str, x_label: str, y_label: str, x_data_list: list, y_data_list: list, saving_path: str,
               graph_label_list=None, symlog_scale=True, var_lists=None, title_size=9, SE=False, num_samples=0,
               x_fontsize=18, y_fontsize=18):

    plt.title(title, fontsize=title_size)
    plt.xlabel(x_label, fontsize=x_fontsize)
    if symlog_scale:
        plt.xscale('log')
    plt.ylabel(y_label, fontsize=y_fontsize)
    color_list = ['b', 'r', 'g', 'y', 'gray', 'k', 'm', 'c']
    if graph_label_list is not None:
        for i in range(len(x_data_list)):
            plt.plot(x_data_list[i], y_data_list[i], color_list[i], label=graph_label_list[i])
            if var_lists is not None:
                plot_variance_line(x_data_list[i], y_data_list[i], var_lists[i], color_list[i], SE, num_samples)
        plt.legend(loc="upper right")
    else:
        for i in range(len(x_data_list)):
            plt.plot(x_data_list[i], y_data_list[i], color_list[i])
            if var_lists is not None:
                plot_variance_line(x_data_list[i], y_data_list[i], var_lists[i], color_list[i], SE, num_samples)

    plt.savefig(saving_path)
    plt.show()


def evaluate_model_on_test_set(test_set, model, feature_list_to_predict, orig_df_f_loan_status=None, target_label='LoanStatus'):
    test_labels = test_set[target_label] if orig_df_f_loan_status is None else orig_df_f_loan_status
    will_loan_returned_pred = model.predict(pd.DataFrame(test_set[feature_list_to_predict]))
    return 1-np.sum(will_loan_returned_pred == test_labels) / len(will_loan_returned_pred)


def load_model(path: str):
    return pickle.load(open(path, 'rb'))


def save_model(model, model_path):
    pickle.dump(model, open(model_path, 'wb'))


def get_data_with_right_size(data_path: str, data_size: int):
    data_df = pd.read_csv(data_path)
    if data_size > 0:
        data_df = data_df[:min(data_size, len(data_df))]
    return data_df


real_train_path = 'data/train_pre2009.csv'
real_val_path = 'data/val_pre2009.csv'
real_test_path = 'data/test_pre2009.csv'
real_train_val_path = 'data/train_val_pre2009.csv'

real_train_f_star_loan_status_path = 'data/train_pre2009_f_star_loan_status.csv'
real_val_f_star_loan_status_path = 'data/val_pre2009_f_star_loan_status.csv'
real_test_f_star_loan_status_path = 'data/test_pre2009_f_star_loan_status.csv'
real_train_val_f_star_loan_status_path = 'data/train_val_pre2009_f_star_loan_status.csv'



svm_model_loan_returned_path = 'models/loan_returned_svm_model.sav'
models_folder_path = 'models'
result_folder_path = 'result'
data_folder_path = 'data'

svm_modify_full_information_real_test_path = os.path.join(result_folder_path, 'full_information_strategic', 'modify_on_svm_test_df.csv')
hardt_modify_full_information_real_test_path = os.path.join(result_folder_path, 'full_information_strategic', 'modify_on_hardt_test_df.csv')


a = 0.5 * np.array([0.5, 0.5, 1.5, -2.5, -0.5, 0.5])

feature_list_for_pred = ['TotalTrades', 'TotalInquiries',
                        'AvailableBankcardCredit', 'BankcardUtilization', 'AmountDelinquent',
                        'IncomeRange', 'LoanOriginalAmount',
                        'MonthlyLoanPayment', 'StatedMonthlyIncome', 'DebtToIncomeRatio',
                        'TradesNeverDelinquent(percentage)', 'TradesOpenedLast6Months',
                        'RevolvingCreditBalance', 'CurrentlyInGroup',
                        'IsBorrowerHomeowner',
                        'PublicRecordsLast10Years', 'PublicRecordsLast12Months',
                        'CurrentCreditLines',
                        'OpenCreditLines',
                        'OpenRevolvingAccounts',
                        'CreditHistoryLength'
                        ]




six_most_significant_features = ['AvailableBankcardCredit', 'LoanOriginalAmount', 'TradesNeverDelinquent(percentage)',
                                'BankcardUtilization', 'TotalInquiries', 'CreditHistoryLength']


