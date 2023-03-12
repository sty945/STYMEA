import pickle
import os
from collections import Counter
import debugpy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


def read_pickle(path):
    with open(path, 'rb') as file:
        data = pickle.load(file)
    return data


def save_pickle(path, data):
    with open(path, 'wb') as file:
        pickle.dump(data, file, protocol=pickle.HIGHEST_PROTOCOL)


def read_file_to_tuple_list(path):
    """
    file name example: ill_ent_ids, triples_1, triples_2

    ill_ent_ids format example:
    6047    21213
    14526   23245
    3428    21344
    13847   15205

    triples_1, triples_2 format example:
    0   0   1
    2   1   3
    4   2   5
    """
    tuple_list = []
    with open(path, 'r', encoding='utf-8') as file:
        for line in file:
            if line.strip():
                params = line.strip('\n').split('\t')
                x = []
                for param in params:
                    if param.isdigit():
                        x.append(int(param))
                    else:
                        x.append(param)
                tuple_list.append(tuple(x))
    return tuple_list


def read_file_to_dict(path):
    """
    file name example: ent_id_1, ent_id_2

    ent_id_1, ent_id_2 format example:
    0   /m/027rn
    1   /m/06cx9
    2   /m/017dcd
    3   /m/06v8s0
    """
    result_dict = {}
    with open(path, 'r', encoding='utf-8') as file:
        for line in file:
            if line.strip():
                params = line.strip('\n').split('\t')
                entity_id = int(params[0])
                entity_name = params[1]
                result_dict[entity_name] = entity_id
    return result_dict


def get_realtion_entity(triples_tuple_list):
    relation_head_dict = {}
    relation_tail_dict = {}
    for (head, relation, tail) in triples_tuple_list:
        if relation not in relation_head_dict:
            relation_head_dict[relation] = set()
        if relation not in relation_tail_dict:
            relation_tail_dict[relation] = set()
        
        relation_head_dict[relation].add(head)
        relation_tail_dict[relation].add(tail)
    return relation_head_dict, relation_tail_dict


def load_img(entity_num, path):
    img_dict = pickle.load(open(path, "rb"))
    # init unknown img vector with mean and std deviation of the known's
    imgs_np = np.array(list(img_dict.values()))
    mean = np.mean(imgs_np, axis=0)
    std = np.std(imgs_np, axis=0)
    img_embd = np.array(
        [img_dict[i] if i in img_dict else np.random.normal(mean, std, mean.shape[0]) for i in range(entity_num)])
    print("\n%.2f%% entities have images" % (100 * len(img_dict) / entity_num))
    return img_embd


def load_img_features(entity_num, file_dir):
    # load images features
    if "V1" in file_dir:
        split = "norm"
        img_vec_path = "data/pkls/dbpedia_wikidata_15k_norm_GA_id_img_feature_dict.pkl"
    elif "V2" in file_dir:
        split = "dense"
        img_vec_path = "data/pkls/dbpedia_wikidata_15k_dense_GA_id_img_feature_dict.pkl"
    elif "FB15K" in file_dir:
        filename = os.path.split(file_dir)[-1].upper()
        img_vec_path = "data/mmkb-datasets/" + filename + "/" + filename + "_id_img_feature_dict.pkl"
    else:
        split = file_dir.split("/")[-1]
        img_vec_path = "data/pkls/" + split + "_GA_id_img_feature_dict.pkl"

    img_features = load_img(entity_num, img_vec_path)
    return img_features


def load_relation(entity_num, triple_list, top_relation_num=1000):
    relation_mat = np.zeros((entity_num, top_relation_num), dtype=np.float32)
    # get all relation
    relation_list = np.array(triple_list)[:, 1]
    top_relation = Counter(relation_list).most_common(top_relation_num)

    relation_index_dict  = {}
    for index, (relation, cnt) in enumerate(top_relation):
        relation_index_dict[relation] = index

    for triple in triple_list:
        head = triple[0]
        relation = triple[1]
        tail = triple[2]

        if relation in relation_index_dict:
            # Each element in the matrix represents the strength of the relationship 
            relation_mat[head][relation_index_dict[relation]] += 1
            relation_mat[tail][relation_index_dict[relation]] += 1
    return np.array(relation_mat)


def get_attr_counts(path_list, ent2id_dict):
    attr_counts = {}

    for path in path_list:
        with open(path, 'r', encoding='utf-8') as file:
            for line in file:
                # attribute format: “entity_name attr1 attr2 ……”
                # user line[:-1] to remove last character('\n')
                items = line[-1].split('\t')
                entity_name = items[0]
                if entity_name not in ent2id_dict:
                    continue

                for i in range(1, len(items)):
                    attr_name = items[i]
                    if attr_name not in attr_counts:
                        attr_counts[attr_name] = 1
                    else:
                        attr_counts[attr_name] += 1
    return attr_counts


def get_attr_matrix(path_list, ent2id_dict, attr2id_dict, entity_num, top_attr_num):
    # attribute matrix: the attributes owned by each entity 
    attr_matrix = np.zeros((entity_num, top_attr_num), dtype=np.float32)
    for path in path_list:
        with open(path, 'r', encoding='utf-8') as file:
            for line in file:
                # attribute format: “entity_name attr1 attr2 ……”
                # user line[:-1] to remove last character('\n')
                items = line[-1].split('\t')
                entity_name = items[0]

                if entity_name not in ent2id_dict:
                    continue

                for i in range(1, len(items)):
                    attr_name = items[i]
                    if attr_name in ent2id_dict:
                        attr_matrix[ent2id_dict[entity_name]][attr2id_dict[attr_name]] = 1.0
    return attr_matrix


def load_attr(entity_num, path_list, ent2id_dict, top_attr_num=1000):
    attr_counts = get_attr_counts(path_list, ent2id_dict)

    sorted_attr_counts = sorted(attr_counts.items(), key=lambda x:x[1], reverse=True)
    
    # map attributes to indices
    attr2id_dict = {}
    for i in range(min(top_attr_num, len(sorted_attr_counts))):
        attr_name = sorted_attr_counts[i][0]
        attr2id_dict[attr_name] = i

    attr_matrix = get_attr_matrix(path_list, ent2id_dict, attr2id_dict, entity_num, top_attr_num)
    return attr_matrix
                

def load_graph(file_dir, ratio, device):
    print('loading raw data...')
    # 0-source graph, 1-target graph
    FILE_END_NUM = [1, 2] 
    TOP_REL_NUM = 1000
    TOP_ATTR_NUM = 1000
    ent_ill_path = file_dir + '/ill_ent_ids'
    ent_ids_path = [file_dir + '/ent_ids_{0}'.format(str(i)) for i in FILE_END_NUM]
    triples_path = [file_dir + '/triples_{0}'.format(str(i)) for i in FILE_END_NUM]

    # all entity name-id dict
    ent2id_dict = {}
    # entity id
    ids_list = []
    for path in ent_ids_path:
        dict_res = read_file_to_dict(path)
        ids_list.append(dict_res.values())
        ent2id_dict.update(dict_res)

    left_entity_ids, right_entity_ids = set(ids_list[0]), set(ids_list[1])
    
    ills_tuple_list = read_file_to_tuple_list(ent_ill_path)

    all_triples_tuple_list = []
    for path in triples_path:
        all_triples_tuple_list.extend(read_file_to_tuple_list(path))
    
    relation_head_dict, relation_tail_dict = get_realtion_entity(all_triples_tuple_list)


    # shuffle ill data
    np.random.shuffle(ills_tuple_list)
    # split ill data
    ill_split_index = int(len(ills_tuple_list) // 1 * ratio)
    train_ill = np.array(ills_tuple_list[:ill_split_index], dtype=np.int32)
    test_ill = np.array(ills_tuple_list[ill_split_index:], dtype=np.int32)

    left_not_in_train = left_entity_ids - set(train_ill[:, 0].tolist())
    right_not_in_train = right_entity_ids - set(train_ill[:, 1].tolist())

    print ("#left entity : %d, #right entity: %d" % (len(left_entity_ids), len(right_entity_ids)))
    print ("#left entity not in train set: %d, #right entity not in train set: %d" 
            % (len(left_not_in_train), len(right_not_in_train)))
    
        
    ENT_NUM = len(ent2id_dict)
    REL_NUM = len(relation_head_dict)

    # load image
    origin_img_features = load_img_features(ENT_NUM, file_dir)
    img_features = F.normalize(torch.Tensor(origin_img_features).to(device))
    print("\nimage feature shape:", img_features.shape)
    
    # load relation (matrix)
    origin_rel_features = load_relation(ENT_NUM, all_triples_tuple_list, TOP_REL_NUM)
    rel_features = torch.Tensor(origin_rel_features).to(device)
    print("\nrelation feature shape:", rel_features.shape)

    # load attribute (matrix)
    ent_attr_path_list = [file_dir + '/training_attrs_{0}'.format(str(i)) for i in FILE_END_NUM]
    origin_attr_features = load_attr(ENT_NUM, ent_attr_path_list, ent2id_dict, TOP_ATTR_NUM)
    attr_features = torch.Tensor(origin_attr_features).to(device)
    print("\nattribute feature shape:", attr_features.shape)



