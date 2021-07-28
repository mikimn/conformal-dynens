
from __future__ import print_function
import argparse
import os
import math

from nonconformist.icp import IcpClassifier
from nonconformist.nc import ClassifierNc, InverseProbabilityErrFunc
import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
import tensorflow.keras.preprocessing.sequence
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import GlobalAveragePooling1D
from tensorflow.keras.datasets import imdb
from tensorflow.keras.callbacks import CSVLogger

devices = tf.config.list_physical_devices('GPU')
print(devices)

# from tensorflow.contrib import learn

"""
text processing
and split text/train
"""


def prepare_text(x, num_words, max_len):

    print("to tokenize")
    # split test and training
    tokenizer = Tokenizer(num_words=num_words)
    tokenizer.fit_on_texts(x)

    print("unique tokens:", len(tokenizer.word_index))
    x = tokenizer.texts_to_sequences(x)
    x = pad_sequences(x, maxlen=max_len, padding='post', truncating='pre')

    return x


def read_yahoo_files(file_train, file_test, file_val):

    names = ["class", "questionlabel", "questionContent", "answer"]

    df_train = pd.read_csv(file_train, names=names)
    df_test = pd.read_csv(file_test, names=names)
    df_val = pd.read_csv(file_val, names=names)

    train_len = len(df_train)
    test_len = len(df_test)
    val_len = len(df_val)

    print("train len =", train_len)
    print("test len  =", test_len)
    print("val len   =", val_len)

    df = pd.concat([df_train, df_test, df_val])

    x_text = df["questionlabel"].astype(
        str) + " "+df["questionContent"].astype(str) + " "+df["answer"].astype(str)
    #x_text = df["answer"].astype(str)
    x = x_text.tolist()  # np.array(x_text)#x_text.tolist(
    df_y = df["class"]
    y = pd.get_dummies(df_y, columns=['class']).values
    return x, y, train_len, test_len, val_len


def load_data_yahoo_ans(src_path, max_words=20000, max_len=1000):
    fileTrain = src_path + "train.csv"
    fileTest = src_path + "test.csv"
    fileVal = src_path + "val.csv"

    print("to read")
    x, y, train_len, test_len, val_len = read_yahoo_files(
        fileTrain, fileTest, fileVal)

    print("to process")
    x = prepare_text(x, max_words, max_len)

    print("to split")
    x_train = x[0:train_len, :]
    y_train = y[0:train_len, :]

    x_test = x[train_len:(train_len+test_len), :]
    y_test = y[train_len:(train_len+test_len), :]

    x_val = x[(train_len+test_len):(train_len+test_len+val_len), :]
    y_val = y[(train_len+test_len):(train_len+test_len+val_len), :]

    print("New train SHAPE")
    print(x_train.shape)
    print(y_train.shape)

    print("New test SHAPE")
    print(x_test.shape)
    print(y_test.shape)

    print("New val SHAPE")
    print(x_val.shape)
    print(y_val.shape)

    print("Sample: ")
    print("X:", x_train[0])
    print("Y:", y_train[0])

    return (x_train, y_train), (x_test, y_test), (x_val, y_val)


###############################################
# user params
###############################################

parser = argparse.ArgumentParser()
parser.add_argument("-s", "--seed", help="random seed")
parser.add_argument("-sd", "--savedir", help="saving directory")
parser.add_argument("-i", "--run", help="run number", type=int)
parser.add_argument("-t", "--do_train", help="train or only evaluate", action='store_true')
parser.add_argument("-f", "--datafile", help="data file", required=True)
parser.add_argument("-k", "--topk", help="top k models to save", default=10)
parser.add_argument("-g", "--gpu", default=0)


class Bunch:
    def __init__(self, **entries):
        self.__dict__.update(entries)


args = parser.parse_args()
datafile = args.datafile
run = f'run_{args.run}'
seed = 10 * args.run
do_train = args.do_train
args = Bunch(
    seed=seed,
    savedir=f'./output/yahoo_answers_csv_imbalance3/{datafile}/baseline/{run}',
    datafile=f'./data/yahoo_answers_csv_imbalance3/{datafile}',
    do_train=do_train,
    topK=10,
    gpu=0
)
save_dir = args.savedir
datafile = args.datafile
seed = int(args.seed)
gpuId = args.gpu
top_k = args.topK

# add gpu target
# os.environ["CUDA_VISIBLE_DEVICES"] = ""
# tf.config.set_visible_devices([], 'GPU')

# add destination dir:
print("output dir", save_dir)
if not os.path.exists(save_dir):
    print("adding saving directory")
    os.makedirs(save_dir)
# Set random seed
if seed is not None:
    print('Setting seed.')
    tf.random.set_seed(seed)
    np.random.seed(seed)

###############################################
# MODEL params
###############################################
# Set parameters:
# ngram_range = 2 will add bi-grams features - we don't use this feature
ngram_range = 1
max_features = 10000
maxlen = 1014
batch_size = 16
embedding_dims = 50
epochs = 50
# params
num_classes = 10


snapshot_window_size = int(math.ceil(epochs/top_k))
print('Loading data...')
#(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)
(x_train, y_train), (x_test, y_test), (x_val, y_val) = load_data_yahoo_ans(datafile,
                                                                           max_words=max_features,
                                                                           max_len=maxlen)

print(type(x_train))
print(type(y_train))
print(len(x_train), 'train sequences')
print(len(x_test), 'test sequences')
print('Average train sequence length: {}'.format(
    np.mean(list(map(len, x_train)), dtype=int)))
print('Average test sequence length: {}'.format(
    np.mean(list(map(len, x_test)), dtype=int)))

###############################################
# MODEL definitions
###############################################


def create_ngram_set(input_list, ngram_value=2):
    """
    Extract a set of n-grams from a list of integers.

    >>> create_ngram_set([1, 4, 9, 4, 1, 4], ngram_value=2)
    {(4, 9), (4, 1), (1, 4), (9, 4)}

    >>> create_ngram_set([1, 4, 9, 4, 1, 4], ngram_value=3)
    [(1, 4, 9), (4, 9, 4), (9, 4, 1), (4, 1, 4)]
    """
    return set(zip(*[input_list[i:] for i in range(ngram_value)]))


def add_ngram(sequences, token_indice, ngram_range=2):
    """
    Augment the input list of list (sequences) by appending n-grams values.

    Example: adding bi-gram
    >>> sequences = [[1, 3, 4, 5], [1, 3, 7, 9, 2]]
    >>> token_indice = {(1, 3): 1337, (9, 2): 42, (4, 5): 2017}
    >>> add_ngram(sequences, token_indice, ngram_range=2)
    [[1, 3, 4, 5, 1337, 2017], [1, 3, 7, 9, 2, 1337, 42]]

    Example: adding tri-gram
    >>> sequences = [[1, 3, 4, 5], [1, 3, 7, 9, 2]]
    >>> token_indice = {(1, 3): 1337, (9, 2): 42, (4, 5): 2017, (7, 9, 2): 2018}
    >>> add_ngram(sequences, token_indice, ngram_range=3)
    [[1, 3, 4, 5, 1337, 2017], [1, 3, 7, 9, 2, 1337, 42, 2018]]
    """
    new_sequences = []
    for input_list in sequences:
        new_list = input_list[:]
        for ngram_value in range(2, ngram_range + 1):
            for i in range(len(new_list) - ngram_value + 1):
                ngram = tuple(new_list[i:i + ngram_value])
                if ngram in token_indice:
                    new_list.append(token_indice[ngram])
        new_sequences.append(new_list)

    return new_sequences


if ngram_range > 1:
    print('Adding {}-gram features'.format(ngram_range))
    # Create set of unique n-gram from the training set.
    ngram_set = set()
    for input_list in x_train:
        for i in range(2, ngram_range + 1):
            set_of_ngram = create_ngram_set(input_list, ngram_value=i)
            ngram_set.update(set_of_ngram)

    # Dictionary mapping n-gram token to a unique integer.
    # Integer values are greater than max_features in order
    # to avoid collision with existing features.
    start_index = max_features + 1
    token_indice = {v: k + start_index for k, v in enumerate(ngram_set)}
    indice_token = {token_indice[k]: k for k in token_indice}

    # max_features is the highest integer that could be found in the dataset.
    max_features = np.max(list(indice_token.keys())) + 1

    # Augmenting x_train and x_test with n-grams features
    x_train = add_ngram(x_train, token_indice, ngram_range)
    x_test = add_ngram(x_test, token_indice, ngram_range)
    print('Average train sequence length: {}'.format(
        np.mean(list(map(len, x_train)), dtype=int)))
    print('Average test sequence length: {}'.format(
        np.mean(list(map(len, x_test)), dtype=int)))

print('Pad sequences (samples x time)')
x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
x_test = sequence.pad_sequences(x_test, maxlen=maxlen)
print('x_train shape:', x_train.shape)
print('x_test shape:', x_test.shape)


with tf.device('/device:GPU:0'), tf.compat.v1.Session(config=tf.compat.v1.ConfigProto()):
    print('Build model...')
    model = Sequential()
    model.add(Embedding(max_features,
                        embedding_dims,
                        input_length=maxlen))
    model.add(GlobalAveragePooling1D())
    model.add(Dense(y_train.shape[1], activation='softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    cp = IcpClassifier(
        ClassifierNc(
            model,
            err_func=InverseProbabilityErrFunc()
        )
    )

    ###############################################
    # PREDICTION AND OUPUT
    ###############################################
    # add logs
    # Training log writer
    logfile = f'{save_dir}/callback_training_log.csv'
    csvlog = CSVLogger(logfile, separator=',', append=False)
    callbacks = [csvlog]

    if do_train:
        history = cp.fit(x_train, y_train,
                        batch_size=batch_size,
                        epochs=epochs,
                        validation_data=(x_val, y_val),
                        callbacks=callbacks
                        )
        train_error = history.history['loss']
        valid_accuracy = history.history['val_accuracy']
    else:
        training_log = pd.read_csv(logfile)
        valid_accuracy = list(training_log.val_accuracy)
        print('Skipping training...')



    # Turn labels one-hot => indexed
    cp.calibrate(x_val, np.argmax(y_val, axis=1))
    # Save the weights
    cp.model.save_weights(f'{save_dir}_model_weights.h5')

    # Save the model architecture
    with open(f'{save_dir}_model_architecture.json', 'w') as f:
        f.write(cp.model.to_json())

    # Save training log
    print('Saving training log...')

    # Save index for combination
    c = [str(i) for i in range(num_classes)]
    header = ','.join(c) + '\n'
    print('Writing index file and predict files...')
    f = open(f'{save_dir}/index.csv', 'w')
    top_x = []

    # we have a single file for prediciton
    # to check later.
    x = 0
    name = 'prediction_{:04d}.csv'.format(x+1)

    # use the last model to predict
    weight = valid_accuracy[epochs-1]
    f.write('{},{}\n'.format(name, weight))
    predicts = model.predict(x_test)

    # Save predicts
    f1 = open('{}/prediction_{:04d}.csv'.format(save_dir, x+1), 'w')
    f1.write(header)
    np.savetxt(f1, predicts, delimiter=",")
    f1.close()

    # Save targets
    print('Saving target file...')
    targetfile = '{}/target.csv'.format(save_dir)
    f2 = open(targetfile, 'w')
    f2.write(header)
    np.savetxt(f2, y_test, delimiter=",")
    f2.close()

    print('Saving prediction sets...')
    p = cp.predict(x_test, significance=0.1)
    f3 = open('{}/prediction_set_{:04d}.csv'.format(save_dir, x+1), 'w')
    f3.write(header)
    np.savetxt(f3, p, delimiter=",")
    f3.close()
