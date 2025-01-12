# French to Tamil Translator (5-letter words)

This project is a deep learning-based application that translates French words (limited to 5 letters) into Tamil. It utilizes an encoder-decoder architecture with LSTM networks and incorporates attention mechanisms for improved translation accuracy.

## Features
- Translates 5-letter French words to their corresponding Tamil equivalents.
- User-friendly GUI built with Tkinter.
- Adjustable training configuration parameters.

## Prerequisites

Make sure you have the following packages installed:

- Python 3.6 or higher
- TensorFlow 2.x
- Keras
- NumPy
- pandas
- Tkinter (usually included with Python installs)

## Installation Instructions

1. Clone this repository to your local machine:
   ```bash
   git clone https://github.com/NuhaydShaik12/french-tamil-translator.git

2. Navigate to the project directory:

   ```bash
   cd french-tamil-translator

3. Create a virtual environment (optional but recommended):

   ```bash
   python -m venv venv
   source venv/bin/activate  # For Windows use `venv\Scripts\activate`

4. Install the required packages:

   ```bash
   pip install -r requirements.txt

## Usage

1. Start the application:

   ```bash
   python main.py

2. Enter a 5-letter French word in the input field.

3. Click the "Translate" button to get the Tamil translation.

4. The translated word will appear in the output field.

## Datasets

Ensure that the CSV file containing the mappings between French and Tamil words is in the project directory. The expected column headers are french and tamil.
## Note: This project uses a large dataset hence the dataset could not be pushed into the repository. Used a dummy dataset for this one.

## License

This project is licensed under the MIT License. See the LICENSE file for details.

## Acknowledgments
TensorFlow and Keras for making deep learning accessible.
Improve the Model using BLUE scores.
The contributors of the datasets used for training the model.
The Tkinter community for facilitating GUI development in Python.
