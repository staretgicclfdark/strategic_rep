import argparse
from strategic_players import *
from strategic_full_info_experiment import run_strategic_full_info
from strategic_random_friends_info_experiment import StrategicRandomFriendsRunner
from synthetic_hardt_experiment import m_exp


def create_main_folders():
    os.makedirs(result_folder_path, exist_ok=True)
    os.makedirs(models_folder_path, exist_ok=True)

def create_args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--full_info_exp", help="run full info experiment", action="store_true")
    parser.add_argument("--dark_exp", help="run dark experiment", action="store_true")
    parser.add_argument("--synthetic_exp", help="run synthetic 1D gaussian experiment", action="store_true")
    parser.add_argument("-c", help="cost_factor", default=5)
    parser.add_argument("-e", help="epsilon", default=0.2, type=float)
    parser.add_argument("-s", help="spare cost used only id dark experiment", default=0, type=float)
    parser.add_argument("-th", help="if set hardt model will train again", action="store_true")
    parser.add_argument("-ts", help="svm loan return train again", action="store_true")
    parser.add_argument("-cv", help="only if train svm loan", action="store_true")

    return parser.parse_args()


if __name__ == '__main__':
    create_main_folders()
    args = create_args_parser()
    m_list = [4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]
    if args.full_info_exp:
        run_strategic_full_info(train_hardt=args.th, cost_factor=args.c, epsilon=args.e, use_cv=args.cv)
    elif args.dark_exp:
        st = StrategicRandomFriendsRunner(m_list, args.c, args.e, six_most_significant_features,
                                          force_train_hadart=args.th, force_train_svm_loan_model=args.ts,
                                          spare_cost=args.s, use_cv=args.cv)
        st.execute_strategic_random_friends_exp()
    elif args.synthetic_exp:
        m_exp()
