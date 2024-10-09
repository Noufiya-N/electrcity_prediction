from flask import Flask, request, render_template
import pickle
import numpy as np
import os

app = Flask(__name__)

# Load the trained model
model_path = 'model/model.pkl'  # Ensure this path is correct
try:
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
except FileNotFoundError:
    raise Exception(f"Model file not found at {model_path}. Please ensure it is included in your project.")

# Home route to display the input form
@app.route('/')
def home():
    return render_template('index.html')

# Prediction route to handle user input and return prediction
@app.route('/predict', methods=['POST'])
def predict():
    # Get the data from the form
    try:
        system_load = float(request.form['system_load'])
    except ValueError:
        return "Invalid input. Please enter a valid number for system load."

    # Prepare the input for the model
    input_data = np.array([[system_load]])

    # Make prediction
    predicted_price = model.predict(input_data)[0]

    # Return the prediction
    return render_template('result.html', prediction=predicted_price)

# Set the host and port from environment variables
port = int(os.environ.get("PORT", 5000))  # Default to 5000 if PORT is not set

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=port)  # Bind to 0.0.0.0 for external access
