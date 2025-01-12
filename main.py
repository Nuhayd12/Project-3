import tensorflow as tf
from tensorflow import keras
from keras import layers
import numpy as np
import pandas as pd
import string
import tkinter as tk
from tkinter import ttk
import pickle
import os

class Config:
    vocab_size_french = 5000
    vocab_size_tamil = 5000
    max_length = 10
    embedding_dim = 128
    hidden_dim = 256
    batch_size = 32
    epochs = 50
    learning_rate = 0.001

config = Config()

def create_tokenizer(num_words):
    return keras.preprocessing.text.Tokenizer(
        num_words=num_words,
        filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',
        lower=True,
        split=' ',
        oov_token='<UNK>'
    )

def prepare_data(french_texts, tamil_texts):
    # Add special tokens to Tamil texts
    tamil_texts = ['<start> ' + text + ' <end>' for text in tamil_texts]
    
    # Create and fit tokenizers
    french_tokenizer = create_tokenizer(config.vocab_size_french)
    tamil_tokenizer = create_tokenizer(config.vocab_size_tamil)
    
    french_tokenizer.fit_on_texts(french_texts)
    tamil_tokenizer.fit_on_texts(tamil_texts)
    
    # Convert to sequences
    encoder_input_data = french_tokenizer.texts_to_sequences(french_texts)
    decoder_input_data = tamil_tokenizer.texts_to_sequences(tamil_texts)
    
    # Pad sequences
    encoder_input_data = keras.preprocessing.sequence.pad_sequences(
        encoder_input_data,
        maxlen=config.max_length,
        padding='post'
    )
    
    decoder_input_data = keras.preprocessing.sequence.pad_sequences(
        decoder_input_data,
        maxlen=config.max_length,
        padding='post'
    )
    
    # Create target data (shifted by one position)
    decoder_target_data = np.zeros_like(decoder_input_data)
    decoder_target_data[:, :-1] = decoder_input_data[:, 1:]
    
    return (encoder_input_data, decoder_input_data, decoder_target_data,
            french_tokenizer, tamil_tokenizer)

def create_model():
    # Encoder
    encoder_inputs = keras.Input(shape=(config.max_length,), name="encoder_inputs")
    
    encoder_embedding = layers.Embedding(
        config.vocab_size_french,
        config.embedding_dim,
        name="encoder_embedding"
    )(encoder_inputs)
    
    encoder = layers.LSTM(
        config.hidden_dim,
        return_sequences=True,
        return_state=True,
        name="encoder_lstm"
    )
    encoder_outputs, state_h, state_c = encoder(encoder_embedding)
    
    # Decoder
    decoder_inputs = keras.Input(shape=(config.max_length,), name="decoder_inputs")
    
    decoder_embedding = layers.Embedding(
        config.vocab_size_tamil,
        config.embedding_dim,
        name="decoder_embedding"
    )(decoder_inputs)
    
    decoder_lstm = layers.LSTM(
        config.hidden_dim,
        return_sequences=True,
        name="decoder_lstm"
    )
    decoder_outputs = decoder_lstm(decoder_embedding, initial_state=[state_h, state_c])
    
    # Attention
    attention = layers.Attention(name="attention_layer")
    context_vector = attention([decoder_outputs, encoder_outputs])
    
    decoder_concat = layers.Concatenate(name="concat_layer")
    decoder_combined_context = decoder_concat([decoder_outputs, context_vector])
    
    # Output
    output = layers.Dense(
        config.hidden_dim, 
        activation="relu",
        name="dense_1"
    )(decoder_combined_context)
    
    output = layers.Dropout(0.5, name="dropout_1")(output)
    
    output = layers.Dense(
        config.vocab_size_tamil,
        activation="softmax",
        name="dense_output"
    )(output)
    
    # Create model
    model = keras.Model(
        inputs=[encoder_inputs, decoder_inputs],
        outputs=output,
        name="translation_model"
    )
    
    # Compile
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=config.learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    
    return model

class TranslatorGUI:
    def __init__(self, model, french_tokenizer, tamil_tokenizer):
        self.window = tk.Tk()
        self.window.title("French to Tamil Translator (5-letter words)")
        self.window.geometry("600x400")
        
        self.model = model
        self.french_tokenizer = french_tokenizer
        self.tamil_tokenizer = tamil_tokenizer
        
        self.setup_gui()
    
    def setup_gui(self):
        # Input frame
        input_frame = ttk.LabelFrame(self.window, text="Enter French Text", padding="10")
        input_frame.pack(fill="x", padx=10, pady=5)
        
        self.input_text = tk.Text(input_frame, height=3)
        self.input_text.pack(fill="x")
        
        # Translate button
        self.translate_button = ttk.Button(
            self.window,
            text="Translate",
            command=self.translate
        )
        self.translate_button.pack(pady=10)
        
        # Output frame
        output_frame = ttk.LabelFrame(self.window, text="Tamil Translation", padding="10")
        output_frame.pack(fill="x", padx=10, pady=5)
        
        self.output_text = tk.Text(output_frame, height=3)
        self.output_text.pack(fill="x")
    
    def translate(self):
        input_text = self.input_text.get("1.0", "end-1c").strip()
        words = input_text.split()
        translated_words = []

        for word in words:
            if len(word) == 5:
                # Prepare input
                sequence = self.french_tokenizer.texts_to_sequences([word])
                encoder_input = keras.preprocessing.sequence.pad_sequences(
                    sequence,
                maxlen=config.max_length,
                padding='post'
            )
            
                # Create decoder input
                decoder_input = np.zeros((1, config.max_length))
                decoder_input[0, 0] = self.tamil_tokenizer.word_index['start']  # Start token for decoder

                # Predict
                output = self.model.predict(
                {
                    'encoder_inputs': encoder_input,
                    'decoder_inputs': decoder_input
                },
                verbose=0
            )
            
                # Get predicted sequence
                predicted_seq = np.argmax(output[0], axis=-1)

                # Process the predicted sequence to get the translated word
                predicted_word = self.tamil_tokenizer.sequences_to_texts([predicted_seq])[0]
                predicted_word = predicted_word.replace('start', '').replace('end', '').strip()

                translated_words.append(predicted_word)
            else:
                translated_words.append(word)

        self.output_text.delete("1.0", tk.END)
        self.output_text.insert("1.0", " ".join(translated_words))
    
    def run(self):
        self.window.mainloop() 
        
def main():
    # Check for data file
    if not os.path.exists("french_tamil_data.csv"):
        print("Error: french_tamil_data.csv not found!")
        return
    
    # Load data
    try:
        data = pd.read_csv("french_tamil_data.csv")
        french_texts = data['french'].astype(str).tolist()
        tamil_texts = data['tamil'].astype(str).tolist()
    except Exception as e:
        print(f"Error loading data: {e}")
        return
    
    # Prepare data
    encoder_input_data, decoder_input_data, decoder_target_data, \
    french_tokenizer, tamil_tokenizer = prepare_data(french_texts, tamil_texts)
    
    # Create and train model
    model = create_model()
    
    # Split data
    split_idx = int(len(encoder_input_data) * 0.8)
    
    # Training data
    train_encoder_input = encoder_input_data[:split_idx]
    train_decoder_input = decoder_input_data[:split_idx]
    train_target = decoder_target_data[:split_idx]
    
    # Validation data
    val_encoder_input = encoder_input_data[split_idx:]
    val_decoder_input = decoder_input_data[split_idx:]
    val_target = decoder_target_data[split_idx:]
    
    # Training callbacks
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=5,
            restore_best_weights=True
        ),
        keras.callbacks.ModelCheckpoint(
            'best_model.h5',
            monitor='val_accuracy',
            save_best_only=True
        )
    ]
    
    # Train the model
    try:
        history = model.fit(
            {
                'encoder_inputs': train_encoder_input,
                'decoder_inputs': train_decoder_input
            },
            train_target,
            validation_data=(
                {
                    'encoder_inputs': val_encoder_input,
                    'decoder_inputs': val_decoder_input
                },
                val_target
            ),
            epochs=config.epochs,
            batch_size=config.batch_size,
            callbacks=callbacks
        )
        # Save training history
        with open('training_history.pkl', 'wb') as f:
            pickle.dump(history.history, f)  # Save the history
        # Save tokenizers
        with open('french_tokenizer.pkl', 'wb') as f:
            pickle.dump(french_tokenizer, f)
        with open('tamil_tokenizer.pkl', 'wb') as f:
            pickle.dump(tamil_tokenizer, f)
        
        # Start GUI
        gui = TranslatorGUI(model, french_tokenizer, tamil_tokenizer)
        gui.run()
        
    except Exception as e:
        print(f"Error during training: {e}")

if __name__ == "__main__":
    main()