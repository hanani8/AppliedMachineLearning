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

import os
import shutil
import subprocess
import time

import joblib
import mlflow
import numpy
import pytest
import requests

from score import score

def test_score():
    # Load the trained model 
    model = joblib.load('Assignment-4/logistic_regression_model.joblib')  # Update with your model file name

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

    mlflow.end_run()


def test_flask():
    # Launch the Flask app using command line
    os.system("python Assignment-4/app.py &")  # Run the app in the background
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
    os.system("pkill -f Assignment-4/app.py")  # Kill the Flask app process


def test_docker():
    if shutil.which("docker") is None:
        pytest.skip("Docker CLI is not available on this machine")

    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    dockerfile_path = os.path.join(project_root, "Assignment-4", "Dockerfile")
    image_tag = "assignment3-flask:test"
    container_name = "assignment3-flask-test-container"

    build_cmd = [
        "docker",
        "build",
        "-f",
        dockerfile_path,
        "-t",
        image_tag,
        project_root,
    ]
    subprocess.run(build_cmd, check=True)

    run_cmd = [
        "docker",
        "run",
        "--rm",
        "-d",
        "--name",
        container_name,
        "-p",
        "5000:5000",
        image_tag,
    ]
    subprocess.run(run_cmd, check=True)

    try:
        url = "http://127.0.0.1:5000/score"
        payload = {"text": "Congratulations! You have won a free vacation. Click now!"}

        response = None
        for _ in range(20):
            try:
                response = requests.post(url, json=payload, timeout=3)
                if response.status_code == 200:
                    break
            except requests.RequestException:
                pass
            time.sleep(1)

        assert response is not None, "No response received from the Dockerized app"
        assert response.status_code == 200, "Expected status code 200 from /score"

        body = response.json()
        assert "prediction" in body
        assert "propensity" in body
        assert body["prediction"] in ["True", "False"]
        assert isinstance(body["propensity"], float)
        assert 0.0 <= body["propensity"] <= 1.0
    finally:
        subprocess.run(
            ["docker", "stop", container_name],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )


# In coverage.txt produce the coverage report output of the unit test and integration test using pytest

# To produce the coverage report output using pytest, you can run the following command in your terminal:
# pytest --cov=./ --cov-report=term-missing test.py