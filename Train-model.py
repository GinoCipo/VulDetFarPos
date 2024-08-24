
import sys
import os.path
import json
from datetime import datetime
import random
import pickle

print("")
print(f"Python {'.'.join(map(str, sys.version_info[:2]))}", "<" * 80)
print("")
if sys.version_info[:2] != (3, 8):
  raise Exception("corre esto solo en python 3.8")

import myutils

import numpy
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.preprocessing import sequence
from keras import backend as K

from keras.callbacks import EarlyStopping
from keras.layers import Dropout
from keras.layers import Bidirectional

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.utils import class_weight
# import tensorflow as tf
from gensim.models import Word2Vec, KeyedVectors


import pickle
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_curve

from sklearn.metrics import auc




# default mode / type of vulnerability
mode = "xss"

# get the vulnerability from the command line argument
if len(sys.argv) > 1:
    mode = sys.argv[1]

progress = 0
count = 0


### paramters for the filtering and creation of samples
restriction = [20000, 5, 6, 10]  # which samples to filter out
step = 5  # step lenght n in the description
fulllength = 200  # context length m in the description

mode2 = str(step) + "_" + str(fulllength)

### hyperparameters for the w2v model
mincount = (
    10  # minimum times a word has to appear in the corpus to be in the word2vec model
)
iterationen = 200  # training iterations for the word2vec model
s = 300  # dimensions of the word2vec model
w = "withString"  # word2vec model is not replacing strings but keeping them

# get word2vec model
w2v = "word2vec_" + w + str(mincount) + "-" + str(iterationen) + "-" + str(s)
w2vmodel = os.path.join("wordtovec_models", f"{w2v}.model")
# w2vmodel = f"./wordtovec_models/{w2v}.model"
# load word2vec model
if not (os.path.isfile(w2vmodel)):
    print(f"word2vec model ({w2vmodel!r})is still being created...")
    sys.exit()

w2v_model = Word2Vec.load(w2vmodel)
word_vectors = w2v_model.wv

# load data
import pathlib

DATA_PATH = pathlib.Path("data")

with open(DATA_PATH / f"plain_{mode}", "r") as infile:
    data = json.load(infile)

now = datetime.now()  # current date and time
nowformat = now.strftime("%H:%M")
print("finished loading. ", nowformat)

allblocks = []

for r in data:
    progress = progress + 1

    for c in data[r]:

        if "files" in data[r][c]:
            #  if len(data[r][c]["files"]) > restriction[3]:
            # too many files
            #    continue

            for f in data[r][c]["files"]:

                #      if len(data[r][c]["files"][f]["changes"]) >= restriction[2]:
                # too many changes in a single file
                #       continue

                if not "source" in data[r][c]["files"][f]:
                    # no sourcecode
                    continue

                if "source" in data[r][c]["files"][f]:
                    sourcecode = data[r][c]["files"][f]["source"]
                    #     if len(sourcecode) > restriction[0]:
                    # sourcecode is too long
                    #       continue

                    allbadparts = []

                    for change in data[r][c]["files"][f]["changes"]:

                        # get the modified or removed parts from each change that happened in the commit
                        badparts = change["badparts"]
                        count = count + len(badparts)

                        #     if len(badparts) > restriction[1]:
                        # too many modifications in one change
                        #       break

                        for bad in badparts:
                            # check if they can be found within the file
                            pos = myutils.findposition(bad, sourcecode)
                            if not -1 in pos:
                                allbadparts.append(bad)

                    #   if (len(allbadparts) > restriction[2]):
                    # too many bad positions in the file
                    #     break

                    if len(allbadparts) > 0:
                        #   if len(allbadparts) < restriction[2]:
                        # find the positions of all modified parts
                        positions = myutils.findpositions(allbadparts, sourcecode)

                        # get the file split up in samples
                        blocks = myutils.getblocks(
                            sourcecode, positions, step, fulllength
                        )

                        for b in blocks:
                            # each is a tuple of code and label
                            allblocks.append(b)


keys = []

# randomize the sample and split into train, validate and final test set
for i in range(len(allblocks)):
    keys.append(i)
random.shuffle(keys)

cutoff = round(0.7 * len(keys))  #     70% for the training set
cutoff2 = round(
    0.85 * len(keys)
)  #   15% for the validation set and 15% for the final test set

keystrain = keys[:cutoff]
keystest = keys[cutoff:cutoff2]
keysfinaltest = keys[cutoff2:]

print("cutoff " + str(cutoff))
print("cutoff2 " + str(cutoff2))


#with open(DATA_PATH / f"{mode}_dataset_keystrain", "wb") as fp:
#    pickle.dump(keystrain, fp)
#with open(DATA_PATH / f"{mode}_dataset_keystest", "wb") as fp:
#    pickle.dump(keystest, fp)
#with open(DATA_PATH / f"{mode}_dataset_keysfinaltest", "wb") as fp:
#    pickle.dump(keysfinaltest, fp)

TrainX = []
TrainY = []
ValidateX = []
ValidateY = []
FinaltestX = []
FinaltestY = []


print("Creating training dataset... (" + mode + ")")
for k in keystrain:
    block = allblocks[k]
    code = block[0]
    token = myutils.getTokens(code)  # get all single tokens from the snippet of code
    vectorlist = []
    for t in token:  # convert all tokens into their word2vec vector representation
        if t in word_vectors.vocab and t != " ":
            vector = w2v_model[t]
            vectorlist.append(vector.tolist())
    TrainX.append(
        vectorlist
    )  # append the list of vectors to the X (independent variable)
    TrainY.append(block[1])  # append the label to the Y (dependent variable)

print("Creating validation dataset...")
for k in keystest:
    block = allblocks[k]
    code = block[0]
    token = myutils.getTokens(code)  # get all single tokens from the snippet of code
    vectorlist = []
    for t in token:  # convert all tokens into their word2vec vector representation
        if t in word_vectors.vocab and t != " ":
            vector = w2v_model[t]
            vectorlist.append(vector.tolist())
    ValidateX.append(
        vectorlist
    )  # append the list of vectors to the X (independent variable)
    ValidateY.append(block[1])  # append the label to the Y (dependent variable)

print("Creating finaltest dataset...")
for k in keysfinaltest:
    block = allblocks[k]
    code = block[0]
    token = myutils.getTokens(code)  # get all single tokens from the snippet of code
    vectorlist = []
    for t in token:  # convert all tokens into their word2vec vector representation
        if t in word_vectors.vocab and t != " ":
            vector = w2v_model[t]
            vectorlist.append(vector.tolist())
    FinaltestX.append(
        vectorlist
    )  # append the list of vectors to the X (independent variable)
    FinaltestY.append(block[1])  # append the label to the Y (dependent variable)

print("Train length: " + str(len(TrainX)))
print("Test length: " + str(len(ValidateX)))
print("Finaltesting length: " + str(len(FinaltestX)))
now = datetime.now()  # current date and time
nowformat = now.strftime("%H:%M")
print("time: ", nowformat)


# saving samples
# In case this doesn't work add _{w2v}__{mode2} to train and validate.

with open(DATA_PATH / f"{mode}_dataset-train-X", 'wb') as fp:
    pickle.dump(TrainX, fp)
with open(DATA_PATH / f"{mode}_dataset-train-Y", 'wb') as fp:
    pickle.dump(TrainY, fp)
with open(DATA_PATH / f"{mode}_dataset-validate-X", 'wb') as fp:
    pickle.dump(ValidateX, fp)
with open(DATA_PATH / f"{mode}_dataset-validate-Y", 'wb') as fp:
    pickle.dump(ValidateY, fp)
with open(DATA_PATH / f"{mode}_dataset_finaltest_X", "wb") as fp:
    pickle.dump(FinaltestX, fp)
with open(DATA_PATH / f"{mode}_dataset_finaltest_Y", "wb") as fp:
    pickle.dump(FinaltestY, fp)
# print("saved finaltest.")


# Prepare the data for the LSTM model

X_train = numpy.array(TrainX, dtype="object")
y_train = numpy.array(TrainY, dtype="object")
X_test = numpy.array(ValidateX, dtype="object")
y_test = numpy.array(ValidateY, dtype="object")
X_finaltest = numpy.array(FinaltestX, dtype="object")
y_finaltest = numpy.array(FinaltestY, dtype="object")

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

for i in range(len(y_finaltest)):
    if y_finaltest[i] == 0:
        y_finaltest[i] = 1
    else:
        y_finaltest[i] = 0


now = datetime.now()  # current date and time
nowformat = now.strftime("%H:%M")
print("numpy array done. ", nowformat)

print(str(len(X_train)) + " samples in the training set.")
print(str(len(X_test)) + " samples in the validation set.")
print(str(len(X_finaltest)) + " samples in the final test set.")

csum = 0
for a in y_train:
    csum = csum + a
print(
    "percentage of vulnerable samples: "
    + str(int((csum / len(X_train)) * 10000) / 100)
    + "%"
)

testvul = 0
for y in y_test:
    if y == 1:
        testvul = testvul + 1
print("absolute amount of vulnerable samples in test set: " + str(testvul))


max_length = fulllength

print("=" * 80, "\n"," No se entrena nada!!!!!!!!! (Esto esta en la linea 315)", "\n", "=" * 80); sys.exit()

# hyperparameters for the LSTM model


now = datetime.now()  # current date and time
nowformat = now.strftime("%H:%M")
print("Starting LSTM: ", nowformat)


# padding sequences on the same length
X_train = sequence.pad_sequences(X_train, maxlen=max_length)
X_test = sequence.pad_sequences(X_test, maxlen=max_length)
X_finaltest = sequence.pad_sequences(X_finaltest, maxlen=max_length)
X_train = numpy.asarray(X_train).astype(numpy.float32)
y_train = numpy.asarray(y_train).astype(numpy.float32)
X_test = numpy.asarray(X_test).astype(numpy.float32)
y_test = numpy.asarray(y_test).astype(numpy.float32)
X_finaltest = numpy.asarray(X_finaltest).astype(numpy.float32)
y_finaltest = numpy.asarray(y_finaltest).astype(numpy.float32)


bilstm_model = Sequential()

bilstm_model.add(
    Bidirectional(LSTM(units=50, return_sequences=True, input_shape=(200, 200)))
)
bilstm_model.add(Dropout(0.2))

bilstm_model.add(LSTM(units=50, return_sequences=True))
bilstm_model.add(Dropout(0.2))

bilstm_model.add(LSTM(units=50, return_sequences=True))
bilstm_model.add(Dropout(0.2))

bilstm_model.add(LSTM(units=50))
bilstm_model.add(Dropout(0.2))

bilstm_model.add(Dense(units=1))

bilstm_model.compile(optimizer="adam", loss="mean_squared_error", metrics=["accuracy"])

bilstm_model.fit(
    X_train,
    y_train,
    epochs=50,
    batch_size=128,
    callbacks=[EarlyStopping(monitor="val_loss", min_delta=0.0001)],
)


bilstm_model.summary()


accr = bilstm_model.evaluate(X_test, y_test)
print("Test set\n  Loss: {:0.3f}\n  Accuracy: {:0.3f}".format(accr[0], accr[1]))
yht_probs = bilstm_model.predict(X_test).ravel()


now = datetime.now()  # current date and time
nowformat = now.strftime("%H:%M")
print("saving LSTM model " + mode + ". ", nowformat)
bilstm_model.save(
    "Model-BiLSTM/Bidirectional_LSTM_model_" + mode + ".h5"
)  # creates a HDF5 file 'my_model.h5'
print("\n\n")

for dataset in ["train", "test", "finaltest"]:

    if dataset == "train":
        yhat_classes = (bilstm_model.predict(X_train) > 0.5).astype("int32")
        accuracy = accuracy_score(y_train, yhat_classes)
        precision = precision_score(y_train, yhat_classes)
        recall = recall_score(y_train, yhat_classes)
        F1Score = f1_score(y_train, yhat_classes)

    if dataset == "test":
        yhat_classes = (bilstm_model.predict(X_test) > 0.5).astype("int32")
        accuracy = accuracy_score(y_test, yhat_classes)
        precision = precision_score(y_test, yhat_classes)
        recall = recall_score(y_test, yhat_classes)
        F1Score = f1_score(y_test, yhat_classes)

    if dataset == "finaltest":
        yhat_classes = (bilstm_model.predict(X_finaltest) > 0.5).astype("int32")
        accuracy = accuracy_score(y_finaltest, yhat_classes)
        precision = precision_score(y_finaltest, yhat_classes)
        recall = recall_score(y_finaltest, yhat_classes)
        F1Score = f1_score(y_finaltest, yhat_classes)

    print("Accuracy-Bi-LSTM: " + str(accuracy))
    print("Precision-Bi-LSTM: " + str(precision))
    print("Recall-Bi-LSTM: " + str(recall))
    print("F1 score-Bi-LSTM: %f" % F1Score)
    print("\n")

# "error", "ignore", "always", "default", "module" or "once"


# auc-lstm
y_pred_lstm = bilstm_model.predict(X_finaltest).ravel()
fpr_lstm, tpr_lstm, thresholds_lstm = roc_curve(y_finaltest, y_pred_lstm)
auc_lstm = auc(fpr_lstm, tpr_lstm)
print(auc_lstm)


####---------------other ML models----------------------------###


# set shape
nsamples, nx, ny = X_train.shape
X_train = X_train.reshape((nsamples, nx * ny))

nsamples, nx, ny = X_test.shape
X_test = X_test.reshape((nsamples, nx * ny))

nsamples, nx, ny = X_finaltest.shape
X_finaltest = X_finaltest.reshape((nsamples, nx * ny))


# lr
from sklearn.linear_model import LogisticRegression

lr = LogisticRegression(solver="liblinear")
lr1 = lr.fit(X_train, y_train)

pkl_filename = "xss-lR.pkl"
with open(pkl_filename, "wb") as file:
    pickle.dump(lr, file)

lr_probs = lr.predict(X_finaltest)

accuracylr = accuracy_score(y_finaltest, lr_probs)
print("Accuracy-lR: %f" % accuracylr)
# precision tp / (tp + fp)
precisionlr = precision_score(y_finaltest, lr_probs)
print("Precision-lR: %f" % precisionlr)
# recall: tp / (tp + fn)
recalllr = recall_score(y_finaltest, lr_probs)
print("Recall-lR: %f" % recalllr)
# f1: 2 tp / (2 tp + fp + fn)
f1lr = f1_score(y_finaltest, lr_probs)
print("F1 score-lR: %f" % f1lr)


# mlp
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier

mlp = MLPClassifier(random_state=1, max_iter=300)

mlp1 = mlp.fit(X_train, y_train)


pkl_filename = "xss-MLP.pkl"
with open(pkl_filename, "wb") as file:
    pickle.dump(mlp, file)

m_probs = mlp.predict(X_finaltest)
accuracymlp = accuracy_score(y_finaltest, m_probs)
print("Accuracyml[]: %f" % accuracymlp)
# precision tp / (tp + fp)
precisionmlp = precision_score(y_finaltest, m_probs)
print("Precision-MLP: %f" % precisionmlp)
# recall: tp / (tp + fn)
recallmlp = recall_score(y_finaltest, m_probs)
print("Recall-MLP: %f" % recallmlp)
# f1: 2 tp / (2 tp + fp + fn)
f1mlp = f1_score(y_finaltest, m_probs)
print("F1 score-MLP: %f" % f1mlp)


# Gausian
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

gnb = GaussianNB()
pred = gnb.fit(X_train, y_train)


pkl_filename = "xss-GNB.pkl"

with open(pkl_filename, "wb") as file:
    pickle.dump(gnb, file)

G_probs = gnb.predict(X_finaltest)
accuracyGNB = accuracy_score(y_finaltest, G_probs)
print("AccuracyGNB: %f" % accuracyGNB)
# precision tp / (tp + fp)
precisionGNB = precision_score(y_finaltest, G_probs)
print("PrecisionGNB: %f" % precisionGNB)
# recall: tp / (tp + fn)
recallGNB = recall_score(y_finaltest, G_probs)
print("RecallGNB: %f" % recallGNB)
# f1: 2 tp / (2 tp + fp + fn)
f1GNB = f1_score(y_finaltest, G_probs)
print("F1 scoreGNB: %f" % f1GNB)

# decision tree
from sklearn.tree import DecisionTreeClassifier

tree = DecisionTreeClassifier(criterion="entropy", random_state=0, max_depth=2)
m = tree.fit(X_train, y_train)


pkl_filename = "xss-TREE.pkl"
with open(pkl_filename, "wb") as file:
    pickle.dump(tree, file)

hat_probs = tree.predict(X_finaltest)
accuracy = accuracy_score(y_finaltest, hat_probs)
print("Accuracy-tree: %f" % accuracy)
# precision tp / (tp + fp)
precision = precision_score(y_finaltest, hat_probs)
print("Precision-tree: %f" % precision)
# recall: tp / (tp + fn)
recall = recall_score(y_finaltest, hat_probs)
print("Recall-tree: %f" % recall)
# f1: 2 tp / (2 tp + fp + fn)
f1 = f1_score(y_finaltest, hat_probs)
print("F1 score-tree: %f" % f1)


# auctree

probs = tree.predict_proba(X_finaltest)
probs = probs[:, 1]
fpr, tpr, thresholds = roc_curve(y_finaltest, probs)
auc_tree = auc(fpr, tpr)


# aucGNB

y_pred_GNB = gnb.predict_proba(X_finaltest)[:, 1]
fpr_gnb, tpr_gnb, thresholds_gnb = roc_curve(y_finaltest, y_pred_GNB)
auc_gnb = auc(fpr_gnb, tpr_gnb)
print(fpr_gnb)


# aucMLP

y_pred_mlp = mlp.predict_proba(X_finaltest)[:, 1]
fpr_mlp, tpr_mlp, thresholds_mlp = roc_curve(y_finaltest, y_pred_mlp)
auc_mlp = auc(fpr_mlp, tpr_mlp)
print(fpr_mlp)


# aucLR

y_pred_lr = lr.predict_proba(X_finaltest)[:, 1]
fpr_lr, tpr_lr, thresholds_lr = roc_curve(y_finaltest, y_pred_lr)
auc_lr = auc(fpr_lr, tpr_lr)
print(fpr_lr)


import matplotlib.pyplot as plt

plt.figure(1)
plt.plot([0, 1], [0, 1], "k--", linewidth=3.0)
plt.plot(
    fpr_lstm, tpr_lstm, label="BiLSTM (area = {:.3f})".format(auc_lstm), linewidth=3.5
)
plt.plot(
    fpr, tpr, label="Decision Tree (area = {:.3f})".format(auc_tree), linewidth=3.5
)

plt.plot(fpr_gnb, tpr_gnb, label="GNB (area = {:.3f})".format(auc_gnb), linewidth=2.7)
plt.plot(fpr_mlp, tpr_mlp, label="MLP (area = {:.3f})".format(auc_mlp), linewidth=3.5)
plt.plot(fpr_lr, tpr_lr, label="LR (area = {:.3f})".format(auc_lr), linewidth=3.5)

# plt.plot(fpr_QDA, tpr_QDA, label='QDA (area = {:.3f})'.format(auc_QDA))
# plt.plot(fpr_keras, tpr_keras, label='Keras Sequential (area = {:.3f})'.format(auc_keras))
plt.xlabel("False positive rate", fontsize=18, weight="bold")
plt.ylabel("True positive rate", fontsize=20, weight="bold")
plt.title("ROC curve", fontsize=18, weight="bold")
plt.legend(loc="best")
plt.xticks(weight="bold")
plt.yticks(weight="bold")
plt.show()
