# In app.py, create a flask endpoint /score that receives a text as a POST request and gives a response in the json format consisting of prediction and propensity
# In test.py, write an integration test function test_flask(...) that does the following:
# -  launches the flask app using command line (e.g. use os.system)

from flask import Flask, request, jsonify
import joblib
from score import score
import mlflow

app = Flask(__name__)

# Load the trained model (replace 'model.pkl' with the actual path to your model)
model = joblib.load('Assignment-4/logistic_regression_model.joblib')  # Update with your model file name

@app.route('/score', methods=['POST'])
def score_endpoint():
    data = request.get_json()
    text = data.get('text', '')

    # Set a default threshold (you can modify this as needed)
    threshold = 0.5

    # Get the prediction and propensity score
    prediction, propensity = score(text, model, threshold)

    # Return the response in JSON format
    response = {
        'prediction': str(prediction),
        'propensity': propensity
    }
    return jsonify(response)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)