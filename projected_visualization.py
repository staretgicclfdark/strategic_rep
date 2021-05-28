import numpy as np
import os
import matplotlib.pyplot as plt
import pandas as pd


def apply_transform_for_2D(df: pd.DataFrame, f_weights):
    '''

    :param df: Data frame to project
    :param f_weights: The weights of the model to project according to it.
    :return: The projected data frame
    '''
    transform_matrix = np.array([[f_weights[0], 0], [f_weights[1], 0], [f_weights[2], 0],
                                 [0.5 * f_weights[3], 0.5 * f_weights[3]], [0, f_weights[4]], [0, f_weights[5]]])
    return df @ transform_matrix


def get_plot_figure_bounds_and_arrows_param(clf_name: str):
    '''

    :param clf_name: Name of the classifier to project according to it.
    :return: Bounds of the figure and arrows size.
    '''
    if clf_name == 'SVM':
        left_bound, right_bound = 1, 3
        bottom_bound, up_bound = 0.2, 1.4
    elif clf_name == 'Hardt':
        left_bound, right_bound = 0.5, 2.5
        bottom_bound, up_bound = 0, 1.2
    else:
        return None
    head_length = 0.03
    head_width = 0.02
    return left_bound, right_bound, bottom_bound, up_bound, head_length, head_width


def get_head_arrow_size(after_row, before_row):
    '''

    :param after_row: The row player changed his features
    :param before_row: The row player after he changed his features
    :return: head_length and head_width
    '''
    if np.abs(after_row[0] - before_row[0]) > 0.05 or np.abs(after_row[1] - before_row[1]) > 0.05:
        head_length = 0.03
        head_width = 0.02
    else:
        head_length = 0.01
        head_width = 0.01
    return head_length, head_width


def plot_dot_and_arrows(projected_df_before: pd.DataFrame, projected_df_after: pd.DataFrame, after_change_df: pd.DataFrame
                        , df_before: pd.DataFrame, df_after: pd.DataFrame, features_to_project: list, clf, ax, num_point_to_plot: int):
    '''

    :param projected_df_before: The data frame before movement projected to 2D
    :param projected_df_after:  The data frame after movement projected to 2D
    :param after_change_df: The data frame after the change all features
    :param df_before: Data frame before change only the relevant features for projection here.
    :param df_after: Data frame after change only the relevant features for projection here
    :param features_to_project: Features for projection
    :param clf: Model for prediction
    :param ax: ax to plot on
    :param num_point_to_plot: Number of points that can be in the plot.
    '''
    done_orange, done_green, done_red = False, False, False
    for i, (before_row_tup, after_row_tup, after_full) in enumerate(
            zip(projected_df_before.iterrows(), projected_df_after.iterrows(), after_change_df.iterrows())):

        if (np.abs(df_before.iloc[i] - df_after.iloc[i]) > 0.00001).any():
            if clf is None or 1 == clf.predict(np.array(after_full[1][features_to_project]).reshape(1, -1))[0]:
                color = 'green'
                if not done_green:
                    label = r'$\Delta_{\hat{f}}(x)$' + '(approved)'
                    done_green = True
                else:
                    label = None
            else:
                color = 'red'
                if not done_red:
                    label = r'$\Delta_{\hat{f}}(x)$' + '(denied)'
                    done_red = True
                else:
                    label = None
            if not done_orange:
                ax.scatter(projected_df_before.iloc[i][0], projected_df_before.iloc[i][1], s=10, color='orange',
                           zorder=1, label=r'$x$')
                done_orange = True
            else:
                ax.scatter(projected_df_before.iloc[i][0], projected_df_before.iloc[i][1], s=10, color='orange',
                           zorder=1)

            ax.scatter(projected_df_after.iloc[i][0], projected_df_after.iloc[i][1], s=10, color=color, zorder=1,
                       label=label)
            before_row, after_row = before_row_tup[1], after_row_tup[1]
            head_length, head_width = get_head_arrow_size(after_row, before_row)
            plt.arrow(before_row[0], before_row[1], after_row[0] - before_row[0], after_row[1] - before_row[1],
                      shape='full', color='black', length_includes_head=True,
                      zorder=0, head_length=head_length, head_width=head_width)
        else:
            ax.scatter(projected_df_before.iloc[i][0], projected_df_before.iloc[i][1], s=10, color='gray', zorder=1)
        if i > num_point_to_plot:
            break


def visualize_projected_changed_df(clf_name: str, before_change_df: pd.DataFrame, after_change_df: pd.DataFrame,
                                   features_to_project: list, title: str, f_weights, f_inter,
                                   label: str = 'LoanStatus', num_point_to_plot: int = 100,
                                   dir_for_projection_images: str = '2D_projection_images', to_save: bool = True,
                                   dir_name_for_saving_visualize: str = None, clf=None, show_flag=True):
    '''

    :param clf_name: Name of the classifier to project according to it.
    :param before_change_df: Data frame before some samples changed their features.
    :param after_change_df:  Data frame after some samples changed their features.
    :param features_to_project: Features list that are used for predictions.
    :param title: Title for the plot
    :param f_weights: Weights of the classifier.
    :param f_inter: Intercept of the classifier.
    :param label: The target column for prediction
    :param num_point_to_plot: Number of points that can be in the plot.
    :param dir_for_projection_images:
    :param to_save: Whatever to save the plot.
    :param dir_name_for_saving_visualize: The name of directory to save the plot.
    :param show_flag: If it 1 the visualization is plotted.
    '''


    figsize_movements_plot = (8.5, 4.8)
    plt.rcParams['figure.figsize'] = figsize_movements_plot
    os.makedirs(dir_name_for_saving_visualize, exist_ok=True)
    dir_for_projection_images = os.path.join(dir_name_for_saving_visualize, dir_for_projection_images)
    df_before_loan_status, df_before = before_change_df[label], before_change_df[features_to_project]
    df_after = after_change_df[features_to_project]
    projected_df_before, projected_df_after = apply_transform_for_2D(df_before, f_weights), apply_transform_for_2D(df_after, f_weights)

    fig, ax = plt.subplots(1, 1)
    ax.scatter(projected_df_before[0][:num_point_to_plot], projected_df_before[1][:num_point_to_plot], s=10)
    ax.scatter(projected_df_after[0][:num_point_to_plot], projected_df_after[1][:num_point_to_plot], s=10)

    plot_figure_data = get_plot_figure_bounds_and_arrows_param(clf_name)
    if plot_figure_data is None:
        print('clf_name should be Hardt or SVM returning without plot')
        return
    left_bound, right_bound, bottom_bound, up_bound, head_length, head_width = plot_figure_data
    plot_dot_and_arrows(projected_df_before, projected_df_after, after_change_df, df_before, df_after, features_to_project, clf, ax, num_point_to_plot)
    t = np.arange(left_bound, right_bound, 0.2)
    plt.plot(t, -t - f_inter, color='blue', zorder=0, label=r'$f(x)$')
    plt.xlim([left_bound, right_bound])
    plt.ylim([bottom_bound, up_bound])
    plt.title(title)
    plt.legend(loc="upper right", prop={'size': 8})
    if to_save:
        saving_path = os.path.join(dir_for_projection_images, title + '.png')
        os.makedirs(dir_for_projection_images, exist_ok=True)
        # plt.rcParams['figure.figsize'] = figsize_movements_plot
        plt.savefig(saving_path, format='png', dpi=300)
    if show_flag:
        plt.show()
    plt.close()
