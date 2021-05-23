import pandas as pd
import os
from utills_and_consts import *


def apply_transform_creditgrade_loan_returned(credit_grade):
    loan_tresh = 4
    return 1 if credit_grade >= loan_tresh else -1


def get_real_f_star_loan_status_real_df(force_create, orig_df_path, orig_df_f_star_loan_status):
    if not os.path.exists(orig_df_f_star_loan_status) or force_create:
        orig_real_df = pd.read_csv(orig_df_path)
        orig_real_df['LoanStatus'] = orig_real_df['CreditGrade'].apply(apply_transform_creditgrade_loan_returned)
        orig_real_df.to_csv(orig_df_f_star_loan_status)
    else:
        orig_real_df = pd.read_csv(orig_df_f_star_loan_status)
    return orig_real_df


def get_f_star_loan_status_real_train_val_test_df(force_create_train=True, force_create_val=True,
                                                  force_create_test=True):
    '''

    :param force_create_train: Whatever we should create new train set or load the old set
    :param force_create_val: Whatever we should create new train set or load the old set
    :param force_create_test: Whatever we should create new train set or load the old set
    :return: The three datasets with the loan status.
    '''
    train_df = get_real_f_star_loan_status_real_df(force_create_train, real_train_path,
                                                   real_train_f_star_loan_status_path)
    val_df = get_real_f_star_loan_status_real_df(force_create_val, real_val_path,
                                                 real_val_f_star_loan_status_path)
    test_df = get_real_f_star_loan_status_real_df(force_create_test, real_test_path,
                                                  real_test_f_star_loan_status_path)
    train_val_df = pd.concat([train_df, val_df], ignore_index=True)
    train_val_df.to_csv(real_train_val_f_star_loan_status_path)
    return train_df, val_df, test_df, train_val_df
