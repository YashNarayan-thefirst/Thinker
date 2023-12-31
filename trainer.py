import json
import numpy as np
import tensorflow as tf
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

# Function to preprocess and train on a chunk of data
def preprocess_and_train(data, model, tokenizer, max_sequence_length, batch_size):
    X_train, y_train = preprocess_data(data, tokenizer, max_sequence_length)
    history = model.fit(X_train, y_train, epochs=50, batch_size=batch_size, validation_split=0.2, verbose=2)
    return model

# Function to preprocess a batch of data
def preprocess_data(data, tokenizer, max_sequence_length):
    inputs = [item['input'] for item in data]
    outputs = [item['output'] for item in data]

    input_sequences = tokenizer.texts_to_sequences(inputs)
    output_sequences = tokenizer.texts_to_sequences(outputs)

    input_sequences = pad_sequences(input_sequences, maxlen=max_sequence_length, padding='post')
    output_sequences = pad_sequences(output_sequences, maxlen=max_sequence_length, padding='post')

    return input_sequences, output_sequences

# Main training loop
chunk_size = 2000  # Set an appropriate chunk size
batch_size = 32
data = []

with open('dataset.jsonl', 'r') as file:
    tokenizer = None  # Initialize tokenizer once
    model = None  # Initialize model once
    max_sequence_length = None  # Initialize max_sequence_length once

    for line in file:
        item = json.loads(line)
        data.append(item)

        if len(data) >= chunk_size:
            if tokenizer is None:
                # Build the tokenizer and get vocab size and max sequence length
                tokenizer = Tokenizer()
                tokenizer.fit_on_texts([item['input'] for item in data] + [item['output'] for item in data])
                vocab_size = len(tokenizer.word_index) + 1
                max_sequence_length = max(len(seq) for seq in data)

            if model is None:
                # Build the model on the first chunk
                model = tf.keras.Sequential([
                    tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=128, input_length=max_sequence_length),
                    tf.keras.layers.LSTM(128, return_sequences=True),
                    tf.keras.layers.Dense(vocab_size, activation='softmax')
                ])

                model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

            # Train the model on the current chunk
            model = preprocess_and_train(data, model, tokenizer, max_sequence_length, batch_size)

            # Clear the batch to release memory
            data = []

# Save the final model
model.save('model.h5')
def generate_output(input_text, model, tokenizer):
    input_seq = tokenizer.texts_to_sequences([input_text])
    input_seq = pad_sequences(input_seq, maxlen=max_sequence_length, padding='post')
    predicted_seq = model.predict(input_seq)[0]
    predicted_text = tokenizer.sequences_to_texts([np.argmax(predicted_seq, axis=-1)])[0]
    return predicted_text
# Load the model
loaded_model = tf.keras.models.load_model('model.h5')

# Inference function remains the same as in the previous code

# Example of using the model for inference
user_input = "What is the capital of France?"
output_text = generate_output(user_input, loaded_model, tokenizer)
print(f"Response: {output_text}")
