from sklearn.svm import LinearSVC
from sklearn.model_selection import LeaveOneOut, KFold
from create_synthetic_loan_status_data import get_f_star_loan_status_real_train_val_test_df, get_real_f_star_loan_status_real_df
from utills_and_consts import *


def train_loan_status_model_with_cv(training_set: pd.DataFrame, training_labels: pd.Series):
    '''

    :param training_set: Data frame without the target label that the model is going to train about.
    :param training_labels: The label column of the training set.
    :return: The trained model with the best c param which found in cross validation.
    '''

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


def train_loan_return_svm_model(list_features_for_pred: list, trained_model_path: str, target_label: str = 'LoanStatus',
                                use_cv: bool = False):
    '''

    :param list_features_for_pred: List of features that are used for predictions.
    :param trained_model_path: The path to save the trained model.
    :param target_label: The label to predict.
    :param use_cv: Whatever we use cross validation
    :return: The trained model.
    '''

    train_df, val_df, test_df, train_val_df = get_f_star_loan_status_real_train_val_test_df()
    if use_cv:
        print('training loan return model using cross validation')
        linear_model = train_loan_status_model_with_cv(train_val_df[list_features_for_pred], train_val_df[target_label])
    else:
        linear_model = LinearSVC(C=0.01, penalty='l2', random_state=42)
        linear_model.fit(train_val_df[list_features_for_pred], train_val_df[target_label])
    err = evaluate_model_on_test_set(test_df, linear_model, list_features_for_pred, target_label=target_label)
    print(f'err on not modify real test: {err}')
    pickle.dump(linear_model, open(trained_model_path, 'wb'))
    return linear_model


def get_svm_loan_return_model(model_loan_returned_path: str, features_to_use: list, force_train_loan_model: bool = False,
                              use_cv: bool = False):
    '''

    :param model_loan_returned_path: Path to load or save the traine loan status model
    :param features_to_use: list of features to use for predictions.
    :param force_train_loan_model: Whatever train new model or load old one (if exist)
    :param use_cv: Whatever we use cross validation
    :return: Trained loan status model.
    '''
    if force_train_loan_model or os.path.exists(model_loan_returned_path) is False:
        f = train_loan_return_svm_model(features_to_use, model_loan_returned_path, use_cv=use_cv)
    else:
        f = load_model(model_loan_returned_path)
        real_test_f_star_loan_status = get_real_f_star_loan_status_real_df(force_create=False, orig_df_path=real_test_path,
                                                                           orig_df_f_star_loan_status=real_test_f_star_loan_status_path)

        err_on_not_modify = evaluate_model_on_test_set(real_test_f_star_loan_status, f, features_to_use)
        print(f'err on not modify real test:{err_on_not_modify}')
    return f
