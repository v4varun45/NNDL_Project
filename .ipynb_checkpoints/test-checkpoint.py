import sys
import numpy as np
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from tensorflow.keras.optimizers import Adam
from keras.layers import Dense, Embedding, LSTM, Flatten,Bidirectional
from keras.models import Sequential
from tensorflow import keras
import pickle
import warnings

warnings.filterwarnings("ignore")

model_path = r'D:\Masters\mscs\CS5720-Neural Network and Deep Learning\NN_Final_Project\checkpoints\model_lstm_final.h5'
tokenizer_path = r'D:\Masters\mscs\CS5720-Neural Network and Deep Learning\NN_Final_Project\checkpoints\tokenizer.pkl'

def load_keras_model(model_path):
    """Load the Keras model from the specified path."""
    try:
        model = load_model(model_path)
        print("Model loaded successfully.")
        return model
    except Exception as e:
        print(f"An error occurred while loading the model: {e}")
        sys.exit(1)

def load_tokenizer(tokenizer_path):
    """Load the tokenizer used during model training."""
    try:
        with open(tokenizer_path, 'rb') as handle:
            tokenizer = pickle.load(handle)
        #print("Tokenizer loaded successfully.")
        return tokenizer
    except Exception as e:
        print(f"An error occurred while loading the tokenizer: {e}")
        sys.exit(1)

def prepare_text(text, tokenizer, max_len=100):
    """Convert text to padded sequence."""
    sequences = tokenizer.texts_to_sequences([text])
    padded_sequence = pad_sequences(sequences, maxlen=max_len)
    return padded_sequence
def build_model():
    # Define LSTM model
    lstm_model = Sequential()
    lstm_model.add(Embedding(max_features, 256,input_length = max_len))
    lstm_model.add(LSTM(units=196, dropout=0.2,return_sequences=True))
    lstm_model.add(Bidirectional(LSTM(units=128,recurrent_dropout=0.2)))
    lstm_model.add(Flatten())
    lstm_model.add(Dense(64, activation='relu'))
    lstm_model.add(Dense(1, activation='sigmoid'))
    
    return lstm_model

def predict_sentiment(model, text, tokenizer):
    """Predict sentiment from text using the loaded model and tokenizer."""
    processed_text = prepare_text(text, tokenizer)
    prediction = model.predict(processed_text)
    if prediction > 0.5:
      return 'Positive'
    else:
      return 'Negative'
      
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python test_model.py <model_path> <tokenizer_path> <text_to_evaluate>")
        sys.exit(1)

    #model_path = sys.argv[1]
    #tokenizer_path = sys.argv[2]
    text_to_evaluate = sys.argv[1]

    # Load the model and tokenizer
    # model = load_keras_model(model_path)
    model = build_model()
    model.load_weights(model_path)
    tokenizer = load_tokenizer(tokenizer_path)

    # Perform prediction
    sentiment = predict_sentiment(model, text_to_evaluate, tokenizer)
    print(f"Predicted sentiment: {sentiment}")
