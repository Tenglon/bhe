import json
import torch

def flatten_hierarchy(hierarchy, parent_id=None):

    result = []

    for node in hierarchy['children']:

        if 'index' not in node:
            node['index'] = None

        result.append({'id': node['id'], 'name': node['name'], 'index': node['index'], 'parent_id': parent_id})
        if 'children' in node:
            children_result_list = flatten_hierarchy(node, parent_id = node['id'])
            result.extend(children_result_list)
    return result


def get_id2idx(imnet_json):
    root_id = 'Root'
    # Step 1: load the hierarchy from json file and set the root node
    with open(imnet_json, 'r') as f:
        hierarchy = json.load(f)
        hierarchy['name'] = 'Root'
        hierarchy['id'] = root_id
        hierarchy['index'] = None
        hierarchy['parent_id'] = None

        sub_root ={'name': 'SubRoot', 'id': 'SubRoot', 'index': None, 'parent_id': root_id}
        hierarchy['children'].append(sub_root)

    # Step 2: flatten the hierarchy and fix the parent_id of the root node
    flattened_hierarchy = flatten_hierarchy(hierarchy)
    root_node = {'name': 'Root', 'id': root_id, 'index': None, 'parent_id': None}
    flattened_hierarchy.insert(0, root_node)

    # count how many nodes has a parent_id of None
    no_parent_list =[node for node in flattened_hierarchy if node['parent_id'] is None]

    print('There are', len(no_parent_list), 'nodes with parent_id of None')

    # for all nodes with parent_id of None, set their parent_id to be the root_id, except for the root node
    for node in flattened_hierarchy:
        if node['parent_id'] is None and node['id'] != root_id:
            node['parent_id'] = root_id

    no_parent_list =[node for node in flattened_hierarchy if node['parent_id'] is None]
    print('There are', len(no_parent_list), 'nodes with parent_id of None')

    # Step 3: create a dictionary mapping from id to name, and a dictionary mapping from id to parent_id
    # create a dictionary mapping from id to name
    id2name = {node['id']: node['name'] for node in flattened_hierarchy}

    # create a dictionary mapping from id to index
    id2idx = {node['id']: node['index'] for node in flattened_hierarchy if node['index'] is not None}

    return id2idx


def get_mim_data():
    data_train = torch.load('feat/mim_3dswin_train_feat.pth')
    data_val = torch.load('feat/mim_3dswin_valid_feat.pth')

    label_set = list(set(data_train['label']))
    label_set.sort() # 有Sort很重要

    import pdb
    pdb.set_trace()
    
    Xtr, Xte = data_train['feat'], data_val['feat']
    ytr = torch.tensor([label_set.index(item )for item in data_train['label']])
    yte = torch.tensor([label_set.index(item )for item in data_val['label']])
    
    name2clsid = {v: k for k, v in enumerate(label_set)}
    hierarchy_csv = 'moments_depth_v4.csv'
    return Xtr,Xte,ytr,yte,hierarchy_csv,label_set, name2clsid

def get_cifar_data():
    save_dict = torch.load("feat/cifar100-clip-features.pt")
    Xtr, Xte = save_dict['train_features'], save_dict['test_features']
    ytr, yte = save_dict['train_labels'], save_dict['test_labels']
    train_image_idices, test_image_idices = save_dict['train_image_idices'], save_dict['test_image_idices']
    clsid2name = {0: 'apple', 1: 'aquarium_fish', 2: 'baby', 3: 'bear', 4: 'beaver', 5: 'bed', 6: 'bee', 7: 'beetle', 
    8: 'bicycle', 9: 'bottle', 10: 'bowl', 11: 'boy', 12: 'bridge', 13: 'bus', 14: 'butterfly', 15: 'camel', 
    16: 'can', 17: 'castle', 18: 'caterpillar', 19: 'cattle', 20: 'chair', 21: 'chimpanzee', 22: 'clock', 
    23: 'cloud', 24: 'cockroach', 25: 'couch', 26: 'crab', 27: 'crocodile', 28: 'cup', 29: 'dinosaur', 
    30: 'dolphin', 31: 'elephant', 32: 'flatfish', 33: 'forest', 34: 'fox', 35: 'girl', 36: 'hamster', 
    37: 'house', 38: 'kangaroo', 39: 'keyboard', 40: 'lamp', 41: 'lawn_mower', 42: 'leopard', 43: 'lion', 
    44: 'lizard', 45: 'lobster', 46: 'man', 47: 'maple_tree', 48: 'motorcycle', 49: 'mountain', 50: 'mouse', 
    51: 'mushroom', 52: 'oak_tree', 53: 'orange', 54: 'orchid', 55: 'otter', 56: 'palm_tree', 57: 'pear', 
    58: 'pickup_truck', 59: 'pine_tree', 60: 'plain', 61: 'plate', 62: 'poppy', 63: 'porcupine', 64: 'possum', 
    65: 'rabbit', 66: 'raccoon', 67: 'ray', 68: 'road', 69: 'rocket', 70: 'rose', 71: 'sea', 72: 'seal', 
    73: 'shark', 74: 'shrew', 75: 'skunk', 76: 'skyscraper', 77: 'snail', 78: 'snake', 79: 'spider', 
    80: 'squirrel', 81: 'streetcar', 82: 'sunflower', 83: 'sweet_pepper', 84: 'table', 85: 'tank', 
    86: 'telephone', 87: 'television', 88: 'tiger', 89: 'tractor', 90: 'train', 91: 'trout', 92: 'tulip', 
    93: 'turtle', 94: 'wardrobe', 95: 'whale', 96: 'willow_tree', 97: 'wolf', 98: 'woman', 99: 'worm'}
    name2clsid = {v: k for k, v in clsid2name.items()}
    # sorted list of class names
    hierarchy_csv = 'cifar100_hierarchy.csv'
    label_set = [clsid2name[i] for i in range(len(clsid2name))]
    return Xtr,Xte,ytr,yte,hierarchy_csv,label_set, name2clsid

def get_imagenet_data():
    train_save_dict = torch.load("feat/imagenet_train_clip_features.pt")
    test_save_dict = torch.load("feat/imagenet_val_clip_features.pt")
    Xtr, ytr = train_save_dict['features'], train_save_dict['labels']
    Xte, yte = test_save_dict['features'], test_save_dict['labels']

    imnet_json = 'imagenet_hierarchy.json'
    hierarchy_csv = 'imagenet_hierarchy.csv'
    name2clsid = get_id2idx(imnet_json)
    clsid2name = {v: k for k, v in name2clsid.items()}
    label_set = [clsid2name[i] for i in range(len(clsid2name))]
    return Xtr,Xte,ytr,yte,hierarchy_csv,label_set, name2clsid