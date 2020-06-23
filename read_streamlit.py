#!/usr/bin/python3

"""RNN implementation example with Streamlit"""

__version__ = '0.1'


import numpy as np
import tensorflow as tf
import streamlit as st
import pandas as pd
import numpy as np
import time

st.title("Let it read...")

st.text("RNN for generating Beatles lyrics.")

st.text("Source: https://github.com/vishwanath79/let_it_read")


print(tf.version)


def build_model(vocab_size, embedding_dim, rnn_units, batch_size):
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(vocab_size, embedding_dim,
                                  batch_input_shape=[batch_size, None]),
        tf.keras.layers.GRU(rnn_units,
                            return_sequences=True,
                            stateful=True,
                            recurrent_initializer='glorot_uniform'),
        tf.keras.layers.Dense(vocab_size)
    ])
    return model


def generate_text(model, start_string):
    # evalauation step generating text using the learned model

    # Number of characters to generate
    num_generate = 1000

    # convert start string to vectors

    np.array(start_string)

    idx2char = ['\n', ' ', '!', '"', '&', '(', ')', ',', '-', '.', '3', ':', ';', '?', 'A', 'B', 'C', 'D', 'E', 'F',
                'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'R', 'S', 'T', 'U', 'V', 'W', 'Y', 'a', 'b', 'c', 'd',
                'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y',
                'z', '–', '’'
                ]

    char2idx = {'\n': 0, ' ': 1, '!': 2, '"': 3, '&': 4, '(': 5, ')': 6, ',': 7, '-': 8, '.': 9, '3': 10, ':': 11,
                ';': 12, '?': 13, 'A': 14, 'B': 15, 'C': 16, 'D': 17, 'E': 18, 'F': 19, 'G': 20, 'H': 21, 'I': 22,
                'J': 23, 'K': 24, 'L': 25, 'M': 26, 'N': 27, 'O': 28, 'P': 29, 'R': 30, 'S': 31, 'T': 32, 'U': 33,
                'V': 34, 'W': 35, 'Y': 36, 'a': 37, 'b': 38, 'c': 39, 'd': 40, 'e': 41, 'f': 42, 'g': 43, 'h': 44,
                'i': 45, 'j': 46, 'k': 47, 'l': 48, 'm': 49, 'n': 50, 'o': 51, 'p': 52, 'q': 53, 'r': 54, 's': 55,
                't': 56, 'u': 57, 'v': 58, 'w': 59, 'x': 60, 'y': 61, 'z': 62, '–': 63, '’': 64}

    input_eval = [char2idx[s] for s in start_string]
    input_eval = tf.expand_dims(input_eval, 0)

    # Empty string to store our results
    text_generated = []

    # Low temperatures results in more predictable text
    # Higher temperatures results in more surprising text
    # Experiment to find the best setting
    temperature = 1.0

    # Here batch size == 1
    model.reset_states()
    for i in range(num_generate):
        predictions = model(input_eval)
        # remove the batch dimension
        predictions = tf.squeeze(predictions, 0)

        # using a categorical distribution to predict the character returned by the model
        predictions = predictions / temperature
        predicted_id = tf.random.categorical(predictions, num_samples=1)[-1, 0].numpy()

        # We pass the predicted character as the next input to the model
        # along with the previous hidden state
        input_eval = tf.expand_dims([predicted_id], 0)

        text_generated.append(idx2char[predicted_id])

    return start_string + ''.join(text_generated)

def load_model(vocab_size, embedding_dim, rnn_units, lyrics):
    latest = tf.train.latest_checkpoint(checkpoint_dir)
    print(latest)
    model = build_model(vocab_size, embedding_dim, rnn_units, batch_size=1)
    # model.load_weights(tf.train.latest_checkpoint(checkpoint_dir))
    model.load_weights(checkpoint_dir)
    model.build(tf.TensorShape([1, None]))
    print(model.summary())
    ringo = generate_text(model, start_string=lyrics)
    return ringo

#def generate_lyrics():




if __name__ == '__main__':
    # length of vocab in chars
    vocab_size = 65  # len(vocab)

    # embedding dimension
    embedding_dim = 256

    # number of rnn units
    rnn_units = 1024
    #st.text('Loading data...')
    st.image("wallpaper.jpg", caption="beatles")
    checkpoint_dir = 'checkpoints/ckpt_371'
    lyrics = st.text_input("Enter the words you want to generate lyrics for ...(example - 'let it be')")

    if lyrics != "":
        with st.spinner('Running inference...'):
            time.sleep(10)

        lyrics = load_model(vocab_size, embedding_dim, rnn_units, lyrics)


        st.text(lyrics)
    else:
        pass






