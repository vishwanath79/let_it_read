# Databricks notebook source
# /FileStore/tables/beatles.txt

import tensorflow as tf
import numpy as np
import os
import time

# COMMAND ----------

path_to_file = "/dbfs/FileStore/tables/beatles.txt"


# COMMAND ----------

text = open(path_to_file,'rb').read().decode(encoding='utf-8')
#length of the text is the number of characters in it
print(f'Length of text: {len(text)} characters')


# COMMAND ----------



#first 250 chars
print(text[:250])



#%%

#unique characters

vocab = sorted(set(text))
print(f'{len(vocab)} unique characters')


#%%

#Vectorize- characters to numbers & numbers to characters
char2idx = { u:i for i , u in enumerate(vocab)}
idx2char = np.array(vocab)
text_as_int = np.array([char2idx[c] for c in text])

#%%

print('{')
for char,_ in zip(char2idx, range(20)):
    print('  {:4s}: {:3d},'.format(repr(char), char2idx[char]))
print('  ...\n}')




# COMMAND ----------

# #Vectorize- characters to numbers & numbers to characters
# char2idx = { u:i for i , u in enumerate(vocab)}
# idx2char = np.array(vocab)

print(char2idx)
print(idx2char)

# COMMAND ----------

print (f'{repr(text[:13])} ---- characters mapped to int ---- > {(text_as_int[:13])}')



# COMMAND ----------

seq_length = 100
examples_per_epoch = len(text) //(seq_length+1)
#create training examples/targets
char_dataset = tf.data.Dataset.from_tensor_slices(text_as_int)
#use tf.data.Dataset.from_tensor_slices to convert the text vector into a stream of characters indices

for i in char_dataset.take(5):
  print(idx2char[i.numpy()])


# COMMAND ----------

sequences = char_dataset.batch(seq_length+1, drop_remainder=True)

for item in sequences.take(5):
    print(repr(''.join(idx2char[item.numpy()])))


# COMMAND ----------

def split_input_target(chunk):
    input_text = chunk[:-1]
    target_text = chunk[1:]
    return input_text, target_text

dataset = sequences.map(split_input_target)


# COMMAND ----------

for input_example, target_example in dataset.take(1):
    print('Input data: ', repr(''.join(idx2char[input_example.numpy()])))
    print('Target data:', repr(''.join(idx2char[target_example.numpy()])))

#%%

for i, (input_idx, target_idx) in enumerate(zip(input_example[:5], target_example[:5])):
    print('Step {:4d}'.format(i))
    print('Step {:4d}'.format(i))
    print(' input: {} ({:s})'.format(input_idx,repr(idx2char[input_idx])))
    print(' expected output: {} ({:s})'.format(target_idx, repr(idx2char[target_idx])))


# COMMAND ----------

#CREATE TRAINING BATCHES

BATCH_SIZE = 64
BUFFER_SIZE = 10000
dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE,drop_remainder=True)
dataset

# COMMAND ----------

#length of voab in chars

vocab_size = len(vocab)

#embedding dimension
embedding_dim = 256

#number of rnn uniets
rnn_units = 1024

# COMMAND ----------

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


# COMMAND ----------

model = build_model(
  vocab_size = len(vocab),
  embedding_dim=embedding_dim,
  rnn_units=rnn_units,
  batch_size=BATCH_SIZE)

#%%

for input_example_batch, target_example_batch in dataset.take(1):
  example_batch_predictions = model(input_example_batch)
  print(example_batch_predictions.shape, "# (batch_size, sequence_length, vocab_size)")

#%%

model.summary()

# COMMAND ----------

#sample from output distributions
sampled_indices = tf.random.categorical(example_batch_predictions[0], num_samples=1)
sampled_indices = tf.squeeze(sampled_indices,axis=-1).numpy()
sampled_indices

#%%

print("Input: \n", repr("".join(idx2char[input_example_batch[0]])))
print()
print("Next Char Predictions: \n", repr("".join(idx2char[sampled_indices ])))



# COMMAND ----------

def loss(labels, logits):
  return tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)

example_batch_loss  = loss(target_example_batch, example_batch_predictions)
print("Prediction shape: ", example_batch_predictions.shape, " # (batch_size, sequence_length, vocab_size)")
print("scalar_loss:      ", example_batch_loss.numpy().mean())



#%%



# COMMAND ----------

import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping

model.compile(optimizer='adam', loss=loss,metrics=['acc'])


# Directory where the checkpoints will be saved
checkpoint_dir = '/dbfs/ml/data/stream_training_checkpoints/'
# Name of the checkpoint files
test_checkpoint_dir = '/dbfs/ml/data/stream_training_checkpoints/'
checkpoint_prefix = os.path.join(test_checkpoint_dir, 'ckpt_{epoch}')
metric = 'val_loss'
checkpoint_callback=tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_prefix,
    save_weights_only=True,monitor='loss')

EPOCHS =5000
save_early_callback = EarlyStopping(monitor='loss', patience=20)

# COMMAND ----------

# checkpoint_dir = '/dbfs/ml/data/stream_training_checkpoints'
# with open(checkpoint_dir +"/"+ "hello.h5", "w") as f:
#   f.write("HelloWorld5") 

# COMMAND ----------

history = model.fit(dataset, epochs=EPOCHS, callbacks=[checkpoint_callback,save_early_callback])
tf.train.latest_checkpoint(checkpoint_dir)

# COMMAND ----------

def generate_text(model, start_string):
    #evalauation step generating text using the learned model

    #Number of characters to generate
    num_generate = 1000
    

    #convert start string to vectors
    input_eval = [char2idx[s] for s in start_string]
    input_eval = tf.expand_dims(input_eval,0)

    #Empty string to store our results
    text_generated = []

    #Low temperatures results in more predictable text
    #Higher temperatures results in more surprising text
    #Experiment to find the best setting
    temperature = 1.0

    #Here batch size == 1
    model.reset_states()
    for i in range(num_generate):
      predictions = model(input_eval)
      # remove the batch dimension
      predictions = tf.squeeze(predictions, 0)

      # using a categorical distribution to predict the character returned by the model
      predictions = predictions / temperature
      predicted_id = tf.random.categorical(predictions, num_samples=1)[-1,0].numpy()

      # We pass the predicted character as the next input to the model
      # along with the previous hidden state
      input_eval = tf.expand_dims([predicted_id], 0)

      text_generated.append(idx2char[predicted_id])

    return (start_string + ''.join(text_generated))


# COMMAND ----------

tf.train.latest_checkpoint(checkpoint_dir)
model = build_model(vocab_size, embedding_dim, rnn_units, batch_size=1)

model.load_weights(tf.train.latest_checkpoint(checkpoint_dir))

model.build(tf.TensorShape([1, None]))

#%%

model.summary()




#%%

print(generate_text(model, start_string="hey jude  "))



