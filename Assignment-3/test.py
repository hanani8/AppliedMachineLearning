# In test.py, write a unit test function test_score(...) to test the score function.
# You may reload and use the best model saved during experiments in train.ipynb (in joblib/pkl format) for testing the score function.
# You may consider the following points to construct your test cases:
# does the function produce some output without crashing (smoke test)
# are the input/output formats/types as expected (format test)
# is prediction value 0 or 1 (sanity check)
# is propensity score between 0 and 1 (sanity check)
# if you put the threshold to 0 does the prediction always become 1 (edge case input)
# if you put the threshold to 1 does the prediction always become 0 (edge case input)
# on an obvious spam input text is the prediction 1 (typical input)
# on an obvious non-spam input text is the prediction 0 (typical input)

import joblib
import numpy
from score import score
import mlflow
import pytest

def test_score():
    # Load the trained model 
    mlflow.set_tracking_uri("sqlite:////home/hegemon/AppliedMachineLearning/Assignment-2/mlflow.db")
    id = "82a3a8bc05b34b48862a1d952279580a"
    uri = f"runs:/{id}/Logistic Regression_model"
    model = mlflow.sklearn.load_model(uri)

    # Test cases
    test_cases = [
        ("This is a spam message!", 0, True),  # Threshold 0 should always predict True
        ("This is a spam message!", 1, False), # Threshold 1 should always predict False

        ("Congratulations! You've won a free iPhone. Click here to claim your prize.", 0.5, True),  # Typical spam input
        ("Hello, how are you?", 0.5, False),     # Typical non-spam input

        ("", 0.5, False),                         # Edge case: empty string

    ]

    for text, threshold, expected_prediction in test_cases:
        prediction, propensity = score(text, model, threshold)

        # Smoke test: Check if the function produces output without crashing
        assert isinstance(prediction, numpy.bool_), "Prediction should be a boolean"
        assert isinstance(propensity, float), "Propensity should be a float"

        # Sanity checks
        assert prediction in [True, False], "Prediction should be either True or False"
        assert 0 <= propensity <= 1, "Propensity should be between 0 and 1"

        # Edge case tests
        if threshold == 0:
            assert prediction == True, "With threshold 0, prediction should always be True"
        if threshold == 1:
            assert prediction == False, "With threshold 1, prediction should always be False"

        # Typical input tests
        if threshold == 0 or threshold == 1:
            continue  # Skip typical input tests for edge case thresholds
        assert prediction == expected_prediction, f"Expected {expected_prediction} but got {prediction} for text: '{text}'"

    print("All tests passed!")

    mlflow.end_run()


#     In test.py, write an integration test function test_flask(...) that does the following:
# -  launches the flask app using command line (e.g. use os.system)
# -  test the response from the localhost endpoint
# -  closes the flask app using command line

#        In coverage.txt produce the coverage report output of the unit test and integration test using pytest

import os
import requests
import time

def test_flask():
    # Launch the Flask app using command line
    os.system("python app.py &")  # Run the app in the background
    time.sleep(5)  # Wait for the server to start

    # Test the response from the localhost endpoint
    url = "http://127.0.0.1:5000/score"
    test_data = {"text": "This is a spam message!"}
    response = requests.post(url, json=test_data)
    assert response.status_code == 200, "Expected status code 200"
    response_data = response.json()
    assert 'prediction' in response_data, "Response should contain 'prediction'"
    assert 'propensity' in response_data, "Response should contain 'propensity'"

    # Close the Flask app using command line
    os.system("pkill -f app.py")  # Kill the Flask app process
    print("Integration test passed!")


# In coverage.txt produce the coverage report output of the unit test and integration test using pytest

# To produce the coverage report output using pytest, you can run the following command in your terminal:
# pytest --cov=./ --cov-report=term-missing test.py