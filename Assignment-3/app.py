# In app.py, create a flask endpoint /score that receives a text as a POST request and gives a response in the json format consisting of prediction and propensity
# In test.py, write an integration test function test_flask(...) that does the following:
# -  launches the flask app using command line (e.g. use os.system)

from flask import Flask, request, jsonify
import joblib
from score import score
import mlflow

app = Flask(__name__)

# Load the trained model (replace 'model.pkl' with the actual path to your model)

mlflow.set_tracking_uri("sqlite:////home/hegemon/AppliedMachineLearning/Assignment-2/mlflow.db")
id = "82a3a8bc05b34b48862a1d952279580a"
uri = f"runs:/{id}/Logistic Regression_model"
model = mlflow.sklearn.load_model(uri)

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
    app.run(debug=True)
    