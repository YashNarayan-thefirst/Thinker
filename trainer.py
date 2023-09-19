import json
import numpy as np
import tensorflow as tf
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Embedding, LSTM, Dense, Bidirectional
from keras.optimizers import Adam

# Function to preprocess and train on a chunk of data using Rust
def preprocess_and_train(data, model, tokenizer, max_sequence_length, batch_size):
    processed_input, processed_output = preprocess_data(data, tokenizer, max_sequence_length)
    model.fit(processed_input, processed_output, batch_size=batch_size, epochs=1, verbose=1)
    return model


def preprocess_data(data, tokenizer, max_sequence_length):
    # Perform preprocessing here
    # Replace this with your actual preprocessing logic

    # Initialize empty lists to hold processed input and output data
    processed_input_data = []
    processed_output_data = []

    for item in data:
        # Process and convert input data (string) into suitable format
        input_sequence = tokenizer.texts_to_sequences([item['input']])
        input_sequence = pad_sequences(input_sequence, maxlen=max_sequence_length, padding='post')
        processed_input_data.append(input_sequence)

        # Process and convert output data (string) into suitable format
        output_sequence = tokenizer.texts_to_sequences([item['output']])
        output_sequence = pad_sequences(output_sequence, maxlen=max_sequence_length, padding='post')
        processed_output_data.append(output_sequence)

    # Convert processed data into NumPy arrays
    processed_input_data = np.array(processed_input_data)
    processed_output_data = np.array(processed_output_data)

    return processed_input_data, processed_output_data

# Main training loop (Python part)
chunk_size = 1000  # Set an appropriate chunk size
batch_size = 32
data = []
max_size = 50*2**20
with open('dataset.jsonl', 'r') as file:
    tokenizer = Tokenizer()  # Initialize tokenizer once
    for line in file:
        item = json.loads(line)
        max_size += len(line)  # Approximate JSON string length as data size

        # Append the item to the data list if it doesn't exceed the memory limit
        if max_size <= max_size:
            data.append(item)
        else:
            # If the memory limit is reached, preprocess and train the model
            if tokenizer.word_index == {}:
                tokenizer.fit_on_texts([item['input'] for item in data] + [item['output'] for item in data])
                vocab_size = len(tokenizer.word_index) + 1
                max_sequence_length = max(len(seq) for seq in data)

            if model is None:
                # Build the model on the first chunk
                model = tf.keras.Sequential([
                    Embedding(input_dim=vocab_size, output_dim=128, input_length=max_sequence_length),
                    Bidirectional(LSTM(128, return_sequences=True)),
                    Dense(vocab_size, activation='softmax')
                ])

                optimizer = Adam(learning_rate=0.001)
                model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

            # Perform preprocessing on the chunk
            processed_input, processed_output = preprocess_data(data, tokenizer, max_sequence_length)

            # Train the model on the preprocessed data
            model.fit(processed_input, processed_output, batch_size=batch_size, epochs=1, verbose=1)
            
            # Clear the batch to release memory
            data = []

if data:
    print(f"Processing remaining {len(data)} items.")
    model = preprocess_and_train(data, model, tokenizer, max_sequence_length, batch_size)

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

# Example of using the model for inference
user_input = "What is the capital of France?"
output_text = generate_output(user_input, loaded_model, tokenizer)
print(f"Response: {output_text}")
