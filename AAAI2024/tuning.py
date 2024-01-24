import csv
import pickle
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

# Utility function to report best scores
def report(results, output_file, n_top=None):
    if n_top is None:
        print(results)
        n_top = max(results["rank_test_balanced_accuracy"])

    with open(output_file, 'w', newline='') as csvfile:
        fieldnames = ['rank_test_balanced_accuracy', 'mean_test_balanced_accuracy', 'mean_test_f1', 'mean_test_f1_weighted', 'mean_test_accuracy', 'mean_fit_time'] + params
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for i in range(1, n_top + 1):
            candidates = np.flatnonzero(results["rank_test_balanced_accuracy"] == i)
            for candidate in candidates:
                # Printing
                print("Model with rank: {0}".format(i))
                print(
                    "Mean validation score: {0:.3f} (std: {1:.3f})".format(
                        results["mean_test_balanced_accuracy"][candidate],
                        results["std_test_balanced_accuracy"][candidate],
                    )
                )
                print("Parameters: {0}".format(results["params"][candidate]))
                print("")

                # Writing to file
                result_to_write = {
                    'rank_test_balanced_accuracy': results["rank_test_balanced_accuracy"][candidate],
                    'mean_test_balanced_accuracy': results["mean_test_balanced_accuracy"][candidate],
                    'mean_test_f1': results["mean_test_f1"][candidate],
                    'mean_test_f1_weighted': results["mean_test_f1_weighted"][candidate],
                    'mean_test_accuracy': results["mean_test_accuracy"][candidate],
                    'mean_fit_time': results["mean_fit_time"][candidate],
                }
                result_to_write.update({param: results["params"][candidate][param] for param in params})
                writer.writerow(result_to_write)

# Tuning ----------------------------------------

# We store classifiers and the associated parameter distributions to tune over
classifiers_and_param_distrs = dict()

# MLP
clf = MLPClassifier(activation='relu', solver='adam', random_state=1, max_iter=100000)
param_distr = {"learning_rate_init": [0.001, 0.01, 0.1, 1],
                "hidden_layer_sizes": [8, 16, 32, 64, (8, 8), (16, 16), (32, 32), (64, 64)]}
classifiers_and_param_distrs["MLP"] = (clf, param_distr)
params = list(param_distr.keys())

# SVC
#clf = SVC(kernel='rbf')
#param_distr = {"C": [0.01, 0.1, 1, 10, 100, 1000, 10000],
#               "gamma": ['scale', 'auto']}
#classifiers_and_param_distrs["SVC"] = (clf, param_distr)
#params = list(param_distr.keys())


benchmarks = ["jsudoku",
              "9sudoku",
              "new_random",
              "exam_timetabling",
              "nurse_rostering_adv"]


for benchmark in benchmarks:
    with open(f'full_training_sets/{benchmark}_data/dataset_X.pickle', 'rb') as handle:
        dataset_X = pickle.load(handle)

    with open(f'full_training_sets/{benchmark}_data/dataset_Y.pickle', 'rb') as handle:
        dataset_Y = pickle.load(handle)

    print(f"For benchmark {benchmark}:")
    for key in classifiers_and_param_distrs:
        clf, param_distr = classifiers_and_param_distrs[key]

        print(type(clf))
        search = GridSearchCV(clf, param_grid=param_distr, scoring=['f1', 'f1_weighted','balanced_accuracy','accuracy'], refit=False, cv=5, n_jobs=-1)
        search.fit(dataset_X, dataset_Y)
        print(f"For model {key}:")
        output_file = f"tuning_results/{benchmark}.csv"
        report(search.cv_results_, output_file)
    print("\n\n")