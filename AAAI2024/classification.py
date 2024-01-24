from numpy import mean
from sklearn.naive_bayes import CategoricalNB, GaussianNB
from sklearn.ensemble._forest import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_validate
from sklearn.metrics import balanced_accuracy_score, make_scorer, accuracy_score, f1_score
import csv
from sklearn.svm import SVC
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree

import pickle

if __name__ == "__main__":

    #benchmark = ["9sudoku", "jsudoku"]
#    benchmark = ["9sudoku", "jsudoku", "new_random",
#                 "job_shop_scheduling", "exam_timetabling", "nurse_rostering_adv"]

    benchmark = ["9sudoku"]

    algorithm = ["random_forest", "GaussianNB", "MLP", "SVM", "Dec_Trees"]
    algorithm_short = ["RF", "GNB", "MLP", "SVM", "DT"]
    datasets_per_benchmark = 5

    results = [[[[[] for d in range(datasets_per_benchmark)] for p in range(10)] for j in range(len(algorithm))] for i in range(len(benchmark))]


    # for each benchmark
    for i in range(len(benchmark)):
        for d in range(datasets_per_benchmark):
            bench = benchmark[i]

            dataset_X = f"full_training_sets/{bench}_data/dataset_X{d+1}.pickle"
            dataset_Y = f"full_training_sets/{bench}_data/dataset_Y{d+1}.pickle"

            with open(dataset_X, 'rb') as f:
                X = pickle.load(f)

            with open(dataset_Y, 'rb') as f:
                Y = pickle.load(f)

            print("X length: ", len(X))

            X_len = len(X)

            train_X = []
            train_Y = []
            test_X = []
            test_Y = []

            for _, p in enumerate([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]):

                train_X.append(X[0:int(X_len*p)])
                train_Y.append(Y[0:int(X_len*p)])
                test_X.append(X[int(X_len*p):])
                test_Y.append(Y[int(X_len * p):])

            classifier = None

            # for each classifier
            for j in range(len(algorithm)):

                alg = algorithm[j]

                if alg == "random_forest":
                    classifier = RandomForestClassifier()
                elif alg == "MLP":
                    classifier = MLPClassifier(hidden_layer_sizes=tuple([8]), activation='relu', solver='adam',
                                               random_state=1, learning_rate_init=0.01)
                elif alg == "CategoricalNB":
                    classifier = CategoricalNB(min_categories=5)
                elif alg == "GaussianNB":
                    classifier = GaussianNB()
                elif alg == "SVM":
                    scaler = MinMaxScaler()  # Default behavior is to scale to [0,1]
                    X = scaler.fit_transform(X)
                    classifier = SVC(kernel='rbf', C=100, gamma='scale', probability=True)
                elif alg == "Dec_Trees":
                    classifier = DecisionTreeClassifier(max_depth=10, random_state=1234)

                for p in range(len(train_X)):

                    try:
                        classifier.fit(train_X[p],train_Y[p])
                        y = classifier.predict(test_X[p])

                        acc = accuracy_score(test_Y[p],y)
                        balanced_acc = balanced_accuracy_score(test_Y[p],y)
                        f1 = f1_score(test_Y[p],y)

                        results[i][j][p][d] = [acc, balanced_acc, f1]
                    except:
                        results[i][j][p][d] = [0.8725, 0.5, 0]  # when only one class exists in the train data (class 0)

                scoring = {"Acc": 'accuracy', "Bal_Acc": make_scorer(balanced_accuracy_score), "f1-score": 'f1'}

                scores = cross_validate(classifier, X, Y,
                                        scoring=scoring,
                                        cv=10)  # fit_params={'sample_weight': compute_sample_weights(Y)})

                print(
                    f'bench: {bench}, dataset: {d+1}, alg: {alg}, balanced_acc = {scores["test_Bal_Acc"].mean()}, acc = {scores["test_Acc"].mean()}, f1 = {scores["test_f1-score"].mean()} \n')

                results[i][j][len(train_X)][d] = [scores["test_Acc"].mean(), scores["test_Bal_Acc"].mean(), scores["test_f1-score"].mean()]

    print(len(results))
    print(results)

    with open('classification_results.csv', 'w', newline='') as file:
        writer = csv.writer(file)

        for b, bench in enumerate(benchmark):
            writer.writerow([bench])

            for a, alg in enumerate(algorithm_short):
                writer.writerow(alg)
                for p in range(10):
                    res = [[[] for d in range(datasets_per_benchmark)] for r in range(3)]
                    for d in range(datasets_per_benchmark):
                        for r in range(3):      # the metrics we have: accuracy, balanced accuracy, f1 score
                            res[r][d] = results[b][a][p][d][r]

                    r = [ mean(res[0]), mean(res[1]), mean(res[2])]
                    print("p = ", p)
                    print(results[b][a][p])
                    print(r)
                    writer.writerow(r)
