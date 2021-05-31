import argparse
from strategic_players import *
from strategic_full_info_experiment import run_strategic_full_info
from strategic_random_friends_info_experiment import StrategicRandomFriendsRunner
from synthetic_hardt_experiment import run_synthetic_hardt_exp


def create_main_folders():
    os.makedirs(result_folder_path, exist_ok=True)
    os.makedirs(models_folder_path, exist_ok=True)

def create_args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--full_info_exp", help="Run full info experiment", action="store_true")
    parser.add_argument("--dark_exp", help="Run dark experiment", action="store_true")
    parser.add_argument("--synthetic_exp", help="Run synthetic 1D gaussian experiment", action="store_true")
    parser.add_argument("-c", help="This is the scale of the cost Contestant pays for movement", default=5)
    parser.add_argument("-e", help="This is epsilon the weight of the l2 cost function in the total cost constant has to pay for movement."
                                   " default value is 0.2. Only relevant in the full information and dark experiment.",
                        default=0.2, type=float)
    parser.add_argument("-s", help="The safety that player takes in order to ensure that it gets positive classification."
                                   " Used only in dark experiment default value is 0", default=0, type=float)
    parser.add_argument("-th", help="If set hardt model will train again", action="store_true")
    parser.add_argument("-ts", help="Train svm loan return model. Only relevant in the full information and dark "
                                    "experiment.", action="store_true")
    parser.add_argument("-cv", help="Only if train svm loan is set", action="store_true")
    parser.add_argument("--list", nargs='+', default=None, help="List of the the numbers examples the Contestants can "
                                                                "learn from. Only relevant in experiments dark and "
                                                                "synthetic 1 dimension gaussian. In the dark experimant"
                                                                "the list must contain at least 3 numbers.")
    parser.add_argument("-ns", help="Number of Hardt model to use in the synthetic experiment", default=10, type=int)
    parser.add_argument("-rp", help="Number to repeat the synthetic experiment", default=200, type=int)
    parser.add_argument("-trs", help="Number of example to train the Hardt model. If this number is greater than the "
                                     "train set we use all the train set (in the synthetic experiment there is no "
                                     "limitation on the number of examples to train)."
                                     " train).", default=-1, type=int) # -1 means no limits
    parser.add_argument("-tes", help="Number of example in the test set that tries to achieve positive score on the model.", default=-1, type=int) # -1 means no limits
    parser.add_argument("-mp", help="Shows the plot of Contestant movements", action="store_true")
    parser.add_argument("--save", help="save some information about Contestant trained model. The data of this experiment is saved in:"
                                   "result/dark_exp/cost_factor={cost_factor}_epsilon={epsilon}. please note that it might take a lot of space but"
                                   "some of the data might speed up the next dark experiment. also note that if you change epsilon or cost"
                                   "factor you must delete two folders one is svm_result/f_hat_result folder and the other is hardt_results/f_hat_result_folder.",
                                    action="store_true"
                                   )

    return parser.parse_args()


if __name__ == '__main__':
    # todo! the friends dict!! migh break!
    create_main_folders()
    m_list = [4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]
    args = create_args_parser()
    if args.list is not None:
        m_list = [int(m) for m in args.list]
    if args.full_info_exp:
        run_strategic_full_info(train_hardt=args.th, cost_factor=args.c, epsilon=args.e, use_cv=args.cv,
                                train_size=args.trs, test_size=args.tes, show_flag=args.mp, force_train_loan_model=args.ts)
    elif args.dark_exp:
        st = StrategicRandomFriendsRunner(m_list, args.c, args.e, six_most_significant_features,
                                          force_train_hadart=args.th, force_train_svm_loan_model=args.ts,
                                          spare_cost=args.s, use_cv=args.cv, train_size=args.trs, test_size=args.tes)
        st.execute_strategic_random_friends_exp(args.mp, args.save, args.tes)
    elif args.synthetic_exp:
        run_synthetic_hardt_exp(m_list, args.ns, args.rp, args.trs, args.tes, args.th)
