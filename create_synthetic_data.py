import pandas as pd
import os
from collections import Counter
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
import lightgbm as lgb
import random
from model import *
from utills_and_consts import *
import json


def learn_lgb_model(out_path, target_label='CreditGrade'):
    def evaluate(model, training_df, testing_df, label, list_for_pred):
        model.fit(training_df[list_for_pred], training_df[label])
        y_true = testing_df[label]
        y_pred = model.predict(testing_df[list_for_pred])
        r2 = 1 - ((y_pred - y_true) ** 2).sum() / ((y_true.mean() - y_true) ** 2).sum()
        return r2
    # These are all the combination I tried in order to fit lgb model. running this parameters search might take days..
    # lgb_params = [{'metric': [['l2', 'auc']], 'learning_rate': [0.005, 0.01, 0.05], 'feature_fraction': [0.6, 0.9],
    #                'bagging_fraction': [0.5, 0.7, 0.9], 'bagging_freq': [10, 100], 'max_depth': [4, 8, 16],
    #                'num_leaves': [64, 128, 256], 'max_bin': [128, 256, 512, 1024], "num_iterations": [1000, 10000],
    #                'n_estimators': [100, 1000]}]

    lgb_params = [{'metric': [['l2', 'auc']], 'learning_rate': [0.01], 'feature_fraction': [0.6],
                   'bagging_fraction': [0.9], 'bagging_freq': [10], 'max_depth': [4],
                   'num_leaves': [64], 'max_bin': [512], "num_iterations": [10000],
                   'n_estimators': [100]}]

    train_df = pd.read_csv(real_train_path)
    val_df = pd.read_csv(real_val_path)
    test_df = pd.read_csv(real_test_path)

    y_train = train_df[target_label]
    X_train = train_df[feature_list_for_pred]

    clf = GridSearchCV(lgb.LGBMRegressor(), lgb_params, cv=5, scoring='r2')
    clf = clf.fit(X_train, y_train)
    best_params = clf.best_params_
    print(f'best params: {best_params}')
    lgb_model = lgb.LGBMRegressor(**best_params)

    r2 = evaluate(lgb_model, train_df, val_df, target_label, feature_list_for_pred)
    print(f'r2 on validation: {r2}')
    train_val_df = pd.concat([train_df, val_df])
    r2 = evaluate(lgb_model, train_val_df, test_df, target_label, feature_list_for_pred)
    print(f'r2 on test: {r2}')
    all_data = pd.concat([train_df, val_df, test_df])
    lgb_model.fit(all_data[feature_list_for_pred], all_data[target_label])
    pickle.dump(lgb_model, open(out_path, 'wb'))


def generate_dataset(feature_list, trained_model, fake_data_set_size=10000,
                     create_key_features=True, train_gmm=False, split_synthetic_set=False, seed=88, gmm_path='models/gmm_model.sav'):
    def get_dist_index(weights_array: np.array):
        return np.random.choice(len(weights_array), fake_data_set_size, p=weights_array)

    def split_df(df: pd.DataFrame):
        train_df, test_val = train_test_split(df, test_size=0.3, random_state=42)
        test_df, val_df = train_test_split(test_val, test_size=0.5, random_state=42)
        return train_df, val_df, test_df

    def check_gmm_model(feature_list, orig_train_path, orig_val_path):
        """
        :param feature_list: the features we use when we create the gmm model
        :prints: the function prints aic and bic measure
        """
        train_df = pd.read_csv(orig_train_path)
        val_df = pd.read_csv(orig_val_path)
        gmm_model_to_check = GaussianMixture(n_components=16)
        gmm_model_to_check.fit(train_df[feature_list])
        print(f'aic on train: {gmm_model_to_check.aic(train_df[feature_list])} and bic on train: '
              f'{gmm_model_to_check.bic(train_df[feature_list])}')
        print(f'aic on val: {gmm_model_to_check.aic(val_df[feature_list])} and bic on val: '
              f'{gmm_model_to_check.bic(val_df[feature_list])}')

    def get_gmm_model():
        if train_gmm:
            check_gmm_model(feature_list, real_train_path, real_val_path)
            np.random.seed(8)
            gmm_model = GaussianMixture(n_components=16)
            gmm_model.fit(real_data[feature_list])
            save_model(gmm_model, gmm_path)
        else:
            gmm_model = load_model(gmm_path)
        return gmm_model

    def create_synthetic():
        new_data_sets_array_list = list()
        gaussian_index_arr = get_dist_index(gmm_model.weights_)
        np.random.seed(seed=seed)
        for index, rep in Counter(gaussian_index_arr).items():
            new_data_sets_array_list.append(
                np.random.multivariate_normal(mean=gmm_model.means_[index], cov=gmm_model.covariances_[index],
                                              size=rep))
        synthetic_data_set = np.vstack(new_data_sets_array_list)
        synthetic_data_set = pd.DataFrame(data=synthetic_data_set, columns=real_data.columns)

        credit_grade_list = trained_model.predict(synthetic_data_set[feature_list_for_pred])
        synthetic_data_set.insert(len(synthetic_data_set.columns), 'CreditGrade', credit_grade_list, True)
        loan_return_status_list = synthetic_data_set['CreditGrade'].apply(apply_transform_creditgrade_loan_returned)
        synthetic_data_set.insert(len(synthetic_data_set.columns), 'LoanStatus', loan_return_status_list, True)

        if create_key_features:
            prefix_key_string = 'synthetic_sample' if split_synthetic_set else 'synthetic'
            synthetic_data_set.insert(len(synthetic_data_set.columns), 'MemberKey',
                                      [prefix_key_string + 'Mem' + str(i) for i in range(fake_data_set_size)], True)
            synthetic_data_set.insert(len(synthetic_data_set.columns), 'LoanKey',
                                      [prefix_key_string + 'Loan' + str(i) for i in range(fake_data_set_size)], True)
        return synthetic_data_set

    # The code of function generate_dataset
    path_list_to_data = [real_train_path, real_val_path, real_test_path]
    real_data = pd.concat([pd.read_csv(path)[feature_list] for path in path_list_to_data], ignore_index=True)
    gmm_model = get_gmm_model()
    synthetic_data = create_synthetic()
    # todo: if we create synthertic data create their path..
    # if split_synthetic_set:
    #     synthetic_train_df, synthetic_val_df, synthetic_test_df = split_df(synthetic_data)
    #     synthetic_train_df.to_csv(synthetic_train_path)
    #     synthetic_val_df.to_csv(synthetic_val_path)
    #     synthetic_test_df.to_csv(synthetic_test_path)
    # else:
    #     synthetic_data.to_csv(synthetic_set_to_sample_from_path)


def apply_transform_creditgrade_loan_returned(credit_grade):
    loan_tresh = 4  
    return 1 if credit_grade >= loan_tresh else -1



def sample_all_classes_in_list(labels_friends, num_friends, num_class=2, use_bouth_classes=True):
    index_friends_list = list()
    classes_set = set()
    while len(classes_set) < num_class:
        index_friends_list = random.sample(range(len(labels_friends)), num_friends)
        classes_set = set(labels_friends[index_friends_list])
        if not use_bouth_classes:
            break
    return index_friends_list


def create_member_friends_dict(num_friends, label_friends_df, df_to_create_list_friend_for, member_dict_path=None, force_to_crate=False, use_both_classes=True):
    if force_to_crate or os.path.exists(member_dict_path) is False:
        member_dict = dict()
        random.seed(8)
        for mem_key, data in df_to_create_list_friend_for.groupby('MemberKey'):
            index_friends_list = sample_all_classes_in_list(label_friends_df, num_friends)
            member_dict[mem_key] = {"friends with credit data": index_friends_list}
        if member_dict_path is not None:
            with open(member_dict_path, 'w') as f:
                json.dump(member_dict, f, indent=4)
    else:
        with open(member_dict_path, 'r') as f:
            member_dict = json.load(f)
    return member_dict


def main_create_synthetic_data(split_synthetic_set=False, train_gmm=True, seed=88, fake_data_set_size=10000):
    path_to_CG_model = 'models/lgb_model_CG.sav'
    train_model = False
    gmm_feature_list = feature_list_for_pred + ['ListingCreationDate'] # add ListingCreationDate because we might want to use only loans from the past.
    if train_model:
        learn_lgb_model(path_to_CG_model)
    cg_trained_model = load_model(path_to_CG_model)
    generate_dataset(gmm_feature_list, cg_trained_model, fake_data_set_size=fake_data_set_size, create_key_features=True,
                     train_gmm=train_gmm, split_synthetic_set=split_synthetic_set, seed=seed)



if __name__ == '__main__':
    main_create_synthetic_data()



