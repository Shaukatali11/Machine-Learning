import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))


# app.py
from flask import Flask, render_template, request, jsonify
from src.data_loader import DataLoader
from src.preprocessing import Preprocessor
from src.model import Model

app = Flask(__name__)

# Load the data and train the model (or load a pre-trained model)
data_loader = DataLoader()
data = data_loader.load_data()

preprocessor = Preprocessor(data)
X_train, X_test, y_train, y_test = preprocessor.split_data()
X_train_scaled, X_test_scaled = preprocessor.scale_data(X_train, X_test)

model = Model()
trained_model = model.train(X_train_scaled, y_train)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extract input values from form
        input1 = float(request.form['input1'])
        input2 = float(request.form['input2'])
        # Add more inputs if necessary

        # Assuming a simple 2-feature input for demonstration
        input_data = [[input1, input2]]
        
        # Preprocess input data (e.g., scaling)
        # Perform prediction
        prediction = trained_model.predict(input_data)

        # Return result to the front-end
        return render_template('index.html', result=prediction[0])

    except Exception as e:
        return str(e)

if __name__ == "__main__":
    app.run(debug=True)
