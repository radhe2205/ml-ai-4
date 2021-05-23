import json

import numpy as np
import pandas as pd
import sys

normalize_method = "standard"
kmeans_k_val = 20

def standard_scale(img_vector):
    avg = np.mean(img_vector, axis=1).reshape((img_vector.shape[0], 1))
    std = np.std(img_vector, axis=1).reshape((img_vector.shape[0], 1))
    img_vector = np.divide(np.subtract(img_vector, avg), std)
    return img_vector

def min_max_transform(img_vector):
    range = np.max(img_vector, axis = 0) - np.min(img_vector, axis = 0) + 1
    img_vector = np.divide(img_vector, range)
    return img_vector

def transform_vector(img_vector):
    if normalize_method == "minmax":
        return min_max_transform(img_vector)
    if normalize_method == "standard":
        return standard_scale(img_vector)
    return img_vector

def read_data(file_name):
    df = pd.read_csv(file_name, sep=" ", header=None)
    data = df.drop(df.columns[0], axis=1).to_numpy().astype(np.float32)
    data[:,1:] = transform_vector(data[:,1:])
    return data

def read_kmean_model(filename):
    return np.loadtxt(filename)

def store_knn_model(file_name, data):
    np.savetxt(file_name, data, fmt = "%1.5f")

def store_decision_tree(file_name, clf):
    f = open(file_name, "w")
    f.write(json.dumps(clf))
    f.close()

def read_decision_tree(file_name):
    with open(file_name) as f:
        data = f.read()
    return json.loads(data)

def train_kmeans(train_file, model_file):
    data = read_data(train_file)
    store_knn_model(model_file, data)

def get_entropy(val_counts1, val_counts2):
    total_vals = np.sum(val_counts1)
    probs = val_counts1 / total_vals
    first_entropy = np.sum(np.multiply(-np.log(probs), probs))

    total_vals = np.sum(val_counts2)
    probs = val_counts2 / total_vals
    second_entropy = np.sum(np.multiply(-np.log(probs), probs))

    total_vals = np.sum(val_counts1) + np.sum(val_counts2)
    return first_entropy * np.sum(val_counts1) / total_vals + second_entropy * np.sum(val_counts2) / total_vals

def get_splits(data, i, j):
    lowers = data[data[:, 1+i] > data[:, 1+j], :]
    uppers = data[data[:, 1+i] <= data[:, 1+j], :]
    return lowers, uppers

def get_split_entropy(data, i, j):
    lowers, uppers = get_splits(data, i, j)
    lower_angles, lower_ang_cnt = np.unique(lowers[:, 0], return_counts = True)
    upper_angles, upper_ang_cnt = np.unique(uppers[:, 0], return_counts = True)

    if len(lowers) * len(uppers) == 0:
        return 1e+10, 0, 0

    return get_entropy(lower_ang_cnt, upper_ang_cnt), lower_angles[np.argmax(lower_ang_cnt)], upper_angles[np.argmax(upper_ang_cnt)]

def get_max_gain_comparison(data):
    min_entropy, lower_max_partition, upper_max_partition = get_split_entropy(data, 0, 1)
    min_entropy_comp = (0,0)
    for i in range(192):
        # for j in range(i+1, 64 * (int(i/64) + 1)):
        for j in range(i + 1, 192):
            split_entropy, l_max, u_max = get_split_entropy(data, i, j)
            if split_entropy < min_entropy:
                min_entropy = split_entropy
                min_entropy_comp = (i,j)
                lower_max_partition = l_max
                upper_max_partition = u_max
    return min_entropy_comp, lower_max_partition, upper_max_partition

def construct_decision_tree(data, max_levels = 3):
    decision_tree_vals = {}

    split_vals, l_max, u_max = get_max_gain_comparison(data)

    decision_tree_vals["split_vals"] = list(split_vals)
    decision_tree_vals["leaf"] = max_levels == 1

    if max_levels > 1:
        lowers, uppers = get_splits(data, *split_vals)
        decision_tree_vals["l_max"] = construct_decision_tree(lowers, max_levels-1)
        decision_tree_vals["u_max"] = construct_decision_tree(uppers, max_levels-1)
        return decision_tree_vals

    decision_tree_vals["l_max"] = int(l_max)
    decision_tree_vals["u_max"] = int(u_max)
    return decision_tree_vals

def classify_with_decision_tree(decision_tree_clf, data):
    results = []
    for i, row in enumerate(data):
        clf = decision_tree_clf
        while True:
            lowers, uppers = get_splits(data[i:i + 1, :], *clf["split_vals"])
            if clf["leaf"]:
                results.append((clf["l_max"], clf["u_max"])[len(lowers) == 0])
                break
            clf = (clf["l_max"], clf["u_max"])[len(lowers) == 0]
    return np.array(results)

def train_decision_tree(train_file, model_file):
    data = read_data(train_file)

    decision_tree_clf = construct_decision_tree(data[:2000], max_levels=3)
    print("Decision tree constructed.")

    results = classify_with_decision_tree(decision_tree_clf, data)
    accuracy = sum(data[:, 0] == results) / len(data)
    print("Train Accuracy: " + str(accuracy))

    store_decision_tree(model_file, decision_tree_clf)

def test_decision_tree(test_file, model_file):
    decision_tree_clf = read_decision_tree(model_file)
    print("Decision Tree loaded.")

    data = read_data(test_file)
    results = classify_with_decision_tree(decision_tree_clf, data)
    accuracy = sum(data[:, 0] == results) / len(data)
    print("Test Accuracy: " + str(accuracy))

def test_kmeans(test_file, model_file):
    model_data = read_kmean_model(model_file)
    model_pixels = model_data[:, 1:]
    model_out = model_data[:, 0]

    test_data = read_data(test_file)
    test_pixels = test_data[:, 1:]
    test_out = test_data[:, 0]

    total_correct = 0
    total_samples = 0

    for i, row in enumerate(test_pixels):
        total_samples += 1
        diff = model_pixels - row
        diff_sq = np.multiply(diff, diff)
        eucl_d = np.sum(diff_sq, axis=1)
        ind = np.argpartition(eucl_d, kmeans_k_val)[:kmeans_k_val]
        orients = model_out[ind]
        dists = eucl_d[ind]
        max_orient = 0
        max_orient_count = 0
        max_map = {0:0, 90:0, 180:0, 270:0}
        for j, val in enumerate(orients):
            max_map[val] += (1/(1 + dists[j]))
            if max_map[val] > max_orient_count:
                max_orient = val
                max_orient_count = max_map[val]
        if test_out[i] == max_orient:
            total_correct += 1

        if i > 10000:
            break

    print("Accuracy: " + str(total_correct/total_samples))

    return

def train_model(train_file, model_file, model_name):
    if model_name == "kmeans":
        train_kmeans(train_file, model_file)
    if model_name == "decision":
        train_decision_tree(train_file, model_file)
    return

def test_model(test_file, model_file, model_name):
    if model_name == "kmeans":
        test_kmeans(test_file, model_file)
    if model_name == "decision":
        test_decision_tree(test_file, model_file)
    return

if __name__ == '__main__':
    mode = sys.argv[1]
    train_test_file = sys.argv[2]
    model_file = sys.argv[3]
    model_name = sys.argv[4]

    if mode == "train":
        train_model(train_test_file, model_file, model_name)
    if mode == "test":
        test_model(train_test_file, model_file, model_name)
