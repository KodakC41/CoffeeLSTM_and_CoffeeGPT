#v From: https://www.thepythoncode.com/article/text-generation-keras-python

import tensorflow as tf
import numpy as np
import os
import pickle

import tqdm
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from string import punctuation


sequence_length = 100
BATCH_SIZE = 128
EPOCHS = 10
# dataset file path
FILE_PATH = "/Users/chris/Documents/Coffee_Language_Processor/James_Clean.txt"
BASENAME = os.path.basename(FILE_PATH)
# read the data
text = open(FILE_PATH, encoding="utf-8").read()
# remove caps, comment this code if you want uppercase characters as well
text = text.lower()
# remove punctuation
text = text.translate(str.maketrans("", "", punctuation))


# print some stats
n_chars = len(text)
vocab = ''.join(sorted(set(text)))
print("unique_chars:", vocab)
n_unique_chars = len(vocab)
print("Number of characters:", n_chars)
print("Number of unique characters:", n_unique_chars)

# dictionary that converts characters to integers
char2int = {c: i for i, c in enumerate(vocab)}
# dictionary that converts integers to characters
int2char = {i: c for i, c in enumerate(vocab)}

# save these dictionaries for later generation
pickle.dump(char2int, open(f"{BASENAME}-char2int.pickle", "wb"))
pickle.dump(int2char, open(f"{BASENAME}-int2char.pickle", "wb"))

# convert all text into integers
encoded_text = np.array([char2int[c] for c in text])

# construct tf.data.Dataset object
char_dataset = tf.data.Dataset.from_tensor_slices(encoded_text)

# print first 5 characters
for char in char_dataset.take(8):
    print(char.numpy(), int2char[char.numpy()])

# build sequences by batching
sequences = char_dataset.batch(2*sequence_length + 1, drop_remainder=True)

# print sequences
for sequence in sequences.take(2):
    print(''.join([int2char[i] for i in sequence.numpy()]))

def split_sample(sample):
    # example :
    # sequence_length is 10
    # sample is "python is a great pro" (21 length)
    # ds will equal to ('python is ', 'a') encoded as integers
    ds = tf.data.Dataset.from_tensors((sample[:sequence_length], sample[sequence_length]))
    for i in range(1, (len(sample)-1) // 2):
        # first (input_, target) will be ('ython is a', ' ')
        # second (input_, target) will be ('thon is a ', 'g')
        # third (input_, target) will be ('hon is a g', 'r')
        # and so on
        input_ = sample[i: i+sequence_length]
        target = sample[i+sequence_length]
        # extend the dataset with these samples by concatenate() method
        other_ds = tf.data.Dataset.from_tensors((input_, target))
        ds = ds.concatenate(other_ds)
    return ds

# prepare inputs and targets
dataset = sequences.flat_map(split_sample)

def one_hot_samples(input_, target):
    # onehot encode the inputs and the targets
    # Example:
    # if character 'd' is encoded as 3 and n_unique_chars = 5
    # result should be the vector: [0, 0, 0, 1, 0], since 'd' is the 4th character
    return tf.one_hot(input_, n_unique_chars), tf.one_hot(target, n_unique_chars)

dataset = dataset.map(one_hot_samples)

# print first 2 samples
for element in dataset.take(2):
    print("Input:", ''.join([int2char[np.argmax(char_vector)] for char_vector in element[0].numpy()]))
    print("Target:", int2char[np.argmax(element[1].numpy())])
    print("Input shape:", element[0].shape)
    print("Target shape:", element[1].shape)
    print("="*50, "\n")

# repeat, shuffle and batch the dataset
ds = dataset.repeat().shuffle(1024).batch(BATCH_SIZE, drop_remainder=True)

model = Sequential([
    LSTM(256, input_shape=(sequence_length, n_unique_chars), return_sequences=True),
    Dropout(0.3),
    LSTM(256),
    Dense(n_unique_chars, activation="softmax"),
])

# define the model path
model_weights_path = f"results/{BASENAME}-{sequence_length}.h5"
model.summary()
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

# make results folder if does not exist yet
if not os.path.isdir("results"):
    os.mkdir("results")
# train the model
model.fit(ds, steps_per_epoch=(len(encoded_text) - sequence_length) // BATCH_SIZE, epochs=EPOCHS)
# save the model
model.save(model_weights_path)


sequence_length = 20
# dataset file path
FILE_PATH = "James_Clean.txt"
# FILE_PATH = "data/python_code.py"
BASENAME = os.path.basename(FILE_PATH)

seed = "espresso"

# load vocab dictionaries
char2int = pickle.load(open(f"{BASENAME}-char2int.pickle", "rb"))
int2char = pickle.load(open(f"{BASENAME}-int2char.pickle", "rb"))
vocab_size = len(char2int)

# building the model
model = Sequential([
    LSTM(256, input_shape=(sequence_length, vocab_size), return_sequences=True),
    Dropout(0.3),
    LSTM(256),
    Dense(vocab_size, activation="softmax"),
])

# load the optimal weights
model.load_weights(f"results/{BASENAME}-{sequence_length}.h5")

s = seed
n_chars = 400
# generate 400 characters
generated = ""
for i in tqdm.tqdm(range(n_chars), "Generating text"):
    # make the input sequence
    X = np.zeros((1, sequence_length, vocab_size))
    for t, char in enumerate(seed):
        X[0, (sequence_length - len(seed)) + t, char2int[char]] = 1
    # predict the next character
    predicted = model.predict(X, verbose=0)[0]
    # converting the vector to an integer
    next_index = np.argmax(predicted)
    # converting the integer to a character
    next_char = int2char[next_index]
    # add the character to results
    generated += next_char
    # shift seed and the predicted character
    seed = seed[1:] + next_char

print("Seed:", s)
print("Generated text:")
print(generated)