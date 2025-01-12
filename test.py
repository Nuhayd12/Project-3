import pickle
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.models import load_model
import numpy as np

# Load the model and training history
model = load_model('best_model.h5')
with open('training_history.pkl', 'rb') as f:
    history = pickle.load(f)

# Plot training history
plt.figure(figsize=(12, 4))

# Plot accuracy
plt.subplot(1, 2, 1)
plt.plot(history['accuracy'], label='Training Accuracy')
plt.plot(history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

# Plot loss
plt.subplot(1, 2, 2)
plt.plot(history['loss'], label='Training Loss')
plt.plot(history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.savefig('training_metrics.png')
plt.show()

# Load tokenizers and test the model
with open('french_tokenizer.pkl', 'rb') as f:
    french_tokenizer = pickle.load(f)
with open('tamil_tokenizer.pkl', 'rb') as f:
    tamil_tokenizer = pickle.load(f)

def translate_word(word):
    if len(word) != 5:
        return "Word must be exactly 5 letters"
    
    # Tokenize and pad the input word
    sequence = french_tokenizer.texts_to_sequences([word])
    padded = tf.keras.preprocessing.sequence.pad_sequences(sequence, maxlen=5, padding='post')
    
    # Get prediction
    prediction = model.predict(padded)
    
    # Convert prediction to Tamil characters
    predicted_sequence = np.argmax(prediction, axis=-1)[0]
    
    # Convert indices back to characters
    reverse_word_map = dict(map(reversed, tamil_tokenizer.word_index.items()))
    tamil_chars = [reverse_word_map.get(i, '') for i in predicted_sequence if i != 0]
    
    return ''.join(tamil_chars)

# Test the model with some example words
test_words = ['livre', 'monde', 'table']
print("\nTesting the model with example words:")
for word in test_words:
    print(f"{word} → {translate_word(word)}")

# Calculate and print final metrics
final_accuracy = history['accuracy'][-1]
final_val_accuracy = history['val_accuracy'][-1]
final_loss = history['loss'][-1]
final_val_loss = history['val_loss'][-1]

print("\nFinal Training Metrics:")
print(f"Training Accuracy: {final_accuracy:.4f}")
print(f"Validation Accuracy: {final_val_accuracy:.4f}")
print(f"Training Loss: {final_loss:.4f}")
print(f"Validation Loss: {final_val_loss:.4f}")