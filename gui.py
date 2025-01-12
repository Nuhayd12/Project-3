import tkinter as tk
from tkinter import ttk, messagebox
import tensorflow as tf
from keras.models import load_model
from keras.utils import pad_sequences
import pickle
import numpy as np

class TranslatorGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("French to Tamil Translator")
        self.root.geometry("600x400")
        self.root.configure(bg='#f0f0f0')

        # Load the model and tokenizers
        try:
            self.model = load_model('best_model.keras')
            with open('french_tokenizer.pkl', 'rb') as f:
                self.french_tokenizer = pickle.load(f)
            with open('tamil_tokenizer.pkl', 'rb') as f:
                self.tamil_tokenizer = pickle.load(f)
        except Exception as e:
            messagebox.showerror("Error", "Could not load model or tokenizers. Please ensure all files are present.")
            root.destroy()
            return

        # Create main frame
        self.main_frame = ttk.Frame(root, padding="20")
        self.main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # Style configuration
        style = ttk.Style()
        style.configure('Title.TLabel', font=('Helvetica', 16, 'bold'))
        style.configure('Result.TLabel', font=('Helvetica', 14))

        # Title
        self.title_label = ttk.Label(
            self.main_frame, 
            text="French to Tamil Translator\n(5-letter words only)", 
            style='Title.TLabel'
        )
        self.title_label.grid(row=0, column=0, columnspan=2, pady=20)

        # Input section
        self.input_label = ttk.Label(
            self.main_frame, 
            text="Enter French Word:", 
            font=('Helvetica', 12)
        )
        self.input_label.grid(row=1, column=0, sticky=tk.W, pady=5)

        self.input_entry = ttk.Entry(
            self.main_frame, 
            width=30, 
            font=('Helvetica', 12)
        )
        self.input_entry.grid(row=2, column=0, columnspan=2, pady=5)

        # Translate button
        self.translate_button = ttk.Button(
            self.main_frame, 
            text="Translate", 
            command=self.translate
        )
        self.translate_button.grid(row=3, column=0, columnspan=2, pady=20)

        # Output section
        self.output_frame = ttk.LabelFrame(
            self.main_frame, 
            text="Translation", 
            padding="10"
        )
        self.output_frame.grid(row=4, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=10)

        self.output_label = ttk.Label(
            self.output_frame, 
            text="", 
            style='Result.TLabel'
        )
        self.output_label.grid(row=0, column=0, pady=10)

        # Status bar
        self.status_var = tk.StringVar()
        self.status_bar = ttk.Label(
            root, 
            textvariable=self.status_var, 
            relief=tk.SUNKEN, 
            anchor=tk.W
        )
        self.status_bar.grid(row=1, column=0, sticky=(tk.W, tk.E))

        # Bind Enter key to translate function
        self.input_entry.bind('<Return>', lambda e: self.translate())

    def translate(self):
        # Get input word
        french_word = self.input_entry.get().strip().lower()

        # Validate input
        if not french_word:
            self.status_var.set("Please enter a word")
            return
        
        if len(french_word) != 5:
            self.status_var.set("Please enter a 5-letter word only")
            self.output_label.configure(text="---")
            return

        try:
            # Tokenize and pad the input word
            sequence = self.french_tokenizer.texts_to_sequences([french_word])
            padded = pad_sequences(sequence, maxlen=5, padding='post')
            
            # Get prediction
            prediction = self.model.predict(padded, verbose=0)
            
            # Convert prediction to Tamil characters
            predicted_sequence = np.argmax(prediction, axis=-1)[0]
            
            # Convert indices back to characters
            reverse_word_map = dict(map(reversed, self.tamil_tokenizer.word_index.items()))
            tamil_chars = [reverse_word_map.get(i, '') for i in predicted_sequence if i != 0]
            
            # Display result
            tamil_word = ''.join(tamil_chars)
            self.output_label.configure(text=tamil_word)
            self.status_var.set("Translation completed")

        except Exception as e:
            self.status_var.set("Error during translation")
            messagebox.showerror("Error", str(e))

def main():
    root = tk.Tk()
    app = TranslatorGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()