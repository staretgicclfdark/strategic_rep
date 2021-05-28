import random
from utills_and_consts import *
import json


def sample_all_classes_in_list(labels_friends: pd.Series, num_friends: int, num_class: int = 2,
                               use_all_classes: bool = True):
    '''

    :param labels_friends: The labels of the the examples in the df we sample friends from.
    :param num_friends: The number of friends that every instance has. Here we consider friend as single instance.
    It means every sample can learn from num_friends examples. We sample examples not friend key. If two
    examples shares the same mem_key they are not sampled together.
    :param num_class: Number of different classes that the instances in the returned list have. 
    :param use_all_classes: Whatever to ensure all class in the list.
    :return: List of indexes that sampled from labels_friends.
    '''
    index_friends_list = list()
    classes_set = set()
    while len(classes_set) < num_class:
        index_friends_list = random.sample(range(len(labels_friends)), num_friends)
        classes_set = set(labels_friends[index_friends_list])
        if not use_all_classes:
            break
    return index_friends_list


def get_member_friends_dict(num_friends: int, test_size: int, label_friends: pd.Series, mem_keys_to_create_list_friend_for: list,
                            member_dict_path: str = None, force_to_crate: bool = False, use_all_classes: bool = True):
    '''

    :param num_friends: The number of friends that every instance has. Here we consider friend as single instance.
    It means every sample can learn from num_friends examples. We sample examples not friend key. If two
    examples shares the same mem_key they are not sampled together.
    :param test_size:
    :param label_friends: The labels of the the examples in the df we sample friends from.
    :param mem_keys_to_create_list_friend_for: List of member keys of dataset we create friends list for.
    :param member_dict_path: Path to load or save the member dict. If this is none create the dictionary and
    dont save it.
    :param force_to_crate: Whatever we have to create new dictionary.
    :param use_all_classes: If this is False the returned dictionary may consist only one class.
    :return: Dictionary that his keys are member_key and value is list of sample indexes in the friends df.
    '''
    if not force_to_crate and os.path.exists(member_dict_path):
        with open(member_dict_path, 'r') as f:
            member_dict = json.load(f)
        if test_size <= len(member_dict.keys):
            return member_dict

    member_dict = dict()
    random.seed(8)
    for mem_key in mem_keys_to_create_list_friend_for:
        index_friends_list = sample_all_classes_in_list(label_friends, num_friends, use_all_classes=use_all_classes)
        member_dict[mem_key] = {"friends with credit data": index_friends_list}
    if member_dict_path is not None:
        with open(member_dict_path, 'w') as f:
            json.dump(member_dict, f, indent=4)

    return member_dict
