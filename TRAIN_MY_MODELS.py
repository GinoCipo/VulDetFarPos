from msilib import sequence
import sys
import numpy
import pandas as pd
from datetime import datetime

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

# default mode / type of vulnerability
mode = "xss"

# get the vulnerability from the command line argument
if len(sys.argv) > 1:
    mode = sys.argv[1]

cpu = 1

TrainX = pd.read_pickle(f"data/{mode}_dataset-train-X")
TrainY = pd.read_pickle(f"data/{mode}_dataset-train-Y")
ValidateX = pd.read_pickle(f"data/{mode}_dataset-validate-X")
ValidateY = pd.read_pickle(f"data/{mode}_dataset-validate-Y")

X_train = numpy.array(TrainX, dtype="object")
y_train = numpy.array(TrainY, dtype="object")
X_test = numpy.array(ValidateX, dtype="object")
y_test = numpy.array(ValidateY, dtype="object")

# in the original collection of data, the 0 and 1 were used the other way round, so now they are switched so that "1" means vulnerable and "0" means clean.

for i in range(len(y_train)):
    if y_train[i] == 0:
        y_train[i] = 1
    else:
        y_train[i] = 0

for i in range(len(y_test)):
    if y_test[i] == 0:
        y_test[i] = 1
    else:
        y_test[i] = 0

max_length = 200

X_train = sequence.pad_sequences(X_train, maxlen=max_length)
X_test = sequence.pad_sequences(X_test, maxlen=max_length)
X_train = numpy.asarray(X_train).astype(numpy.float32)
y_train = numpy.asarray(y_train).astype(numpy.float32)
X_test = numpy.asarray(X_test).astype(numpy.float32)
y_test = numpy.asarray(y_test).astype(numpy.float32)

nsamples, nx, ny = X_train.shape
X_train = X_train.reshape((nsamples, nx * ny))

nsamples, nx, ny = X_test.shape
X_test = X_test.reshape((nsamples, nx * ny))

# sample = pd.read_pickle(f"data/{mode}_dataset_finaltest")
# sample["tile"] = sample["id"].apply(lambda i: "b" + str(i)[1:4])
# 
# no_features = ["id", "vs_catalog", "vs_type", "ra_k", "dec_k", "tile", "cls"] 
# X_columns = [c for c in sample.columns if c not in no_features]
# 
# grouped = sample.groupby("tile")
# data = Container({k: grouped.get_group(k).copy() for k in grouped.groups.keys()})
# 
# del grouped, sample
# 
# df = pd.concat([data.b278, data.b261])
# 
# cls = {name: idx for idx, name in enumerate(df.tile.unique())}
# df["cls"] = df.tile.apply(cls.get)
# 
# print(cls) 
# 
# X = df[X_columns].values
# y = df.cls.values

now = datetime.now()  # current date and time
nowformat = now.strftime("%H:%M")
print(f"Finished loading data, starting SVC for {mode}", nowformat)
# Split the dataset in two equal parts
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42069)

# Set the parameters by cross-validation
tuned_parameters = [
    {'kernel': ['rbf'], 'gamma': [1e-3, 1e-4], 'C': [1, 10, 100, 1000]},
    {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}
]

scores = ['precision', 'recall']

for score in scores:
    print("# Tuning hyper-parameters for %s" % score)
    print()

    clf = GridSearchCV(SVC(), tuned_parameters, cv=5, n_jobs=cpu,
                       scoring='%s_macro' % score)
    clf.fit(X_train, y_train)

    print("Best parameters set found on development set:")
    print()
    print(clf.best_params_)
    print()
    print("Grid scores on development set:")
    print()
    means = clf.cv_results_['mean_test_score']
    stds = clf.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, clf.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r"
              % (mean, std * 2, params))
    print()

    print("Detailed classification report:")
    print()
    print("The model is trained on the full development set.")
    print("The scores are computed on the full evaluation set.")
    print()
    y_true, y_pred = y_test, clf.predict(X_test)
    print(classification_report(y_true, y_pred))
    print()

now = datetime.now()  # current date and time
nowformat = now.strftime("%H:%M")
print(f"Starting Random Forest for {mode}.", nowformat)
# Split the dataset in two equal parts
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42069)

# Set the parameters by cross-validation
tuned_parameters = [
    {'max_features': ['auto', 'sqrt', "log2", None, 0.2, 0.5], 
     "min_samples_split": [2, 5, 10],
     "n_estimators": [500], 
     "criterion": ["entropy"], 
     "n_jobs": [10]}]

scores = ['precision', 'recall']

for score in scores:
    print("# Tuning hyper-parameters for %s" % score)
    print()

    clf = GridSearchCV(RandomForestClassifier(), tuned_parameters, cv=5, n_jobs=cpu,
                       scoring='%s_macro' % score)
    clf.fit(X_train, y_train)

    print("Best parameters set found on development set:")
    print()
    print(clf.best_params_)
    print()
    print("Grid scores on development set:")
    print()
    means = clf.cv_results_['mean_test_score']
    stds = clf.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, clf.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r"
              % (mean, std * 2, params))
    print()

    print("Detailed classification report:")
    print()
    print("The model is trained on the full development set.")
    print("The scores are computed on the full evaluation set.")
    print()
    y_true, y_pred = y_test, clf.predict(X_test)
    print(classification_report(y_true, y_pred))
    print()

now = datetime.now()  # current date and time
nowformat = now.strftime("%H:%M")
print(f"Starting K-Nearest Neighbors for {mode}.", nowformat)
# Split the dataset in two equal parts
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42069)

# Set the parameters by cross-validation
tuned_parameters = [
    {'n_neighbors': [3, 4, 5, 6, 7, 8, 9, 10]}]

scores = ['precision', 'recall']

for score in scores:
    print("# Tuning hyper-parameters for %s" % score)
    print()

    clf = GridSearchCV(KNeighborsClassifier(), tuned_parameters, cv=5, n_jobs=cpu,
                       scoring='%s_macro' % score)
    clf.fit(X_train, y_train)

    print("Best parameters set found on development set:")
    print()
    print(clf.best_params_)
    print()
    print("Grid scores on development set:")
    print()
    means = clf.cv_results_['mean_test_score']
    stds = clf.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, clf.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r"
              % (mean, std * 2, params))
    print()

    print("Detailed classification report:")
    print()
    print("The model is trained on the full development set.")
    print("The scores are computed on the full evaluation set.")
    print()
    y_true, y_pred = y_test, clf.predict(X_test)
    print(classification_report(y_true, y_pred))
    print()
