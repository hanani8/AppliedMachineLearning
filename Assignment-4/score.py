# In score.py, write a function with the following signature that scores a trained model on a text:

#             def score(text:str, 
#                             model:sklearn.estimator, 
#                             threshold:float) -> prediction:bool, 
#                                                          propensity:float

from sklearn.base import BaseEstimator
import joblib

def score(text:str, model:BaseEstimator, threshold:float) -> (bool, float):

    # Load Vectorizer from "../Assignment-2/tfidf_vectorizer.joblib"

    vectorizer = joblib.load('Assignment-4/tfidf_vectorizer.joblib')

    # Transform the input text using the loaded vectorizer
    text_vectorized = vectorizer.transform([text])

    
    probabilities = model.predict_proba(text_vectorized)[:, 1]  # Get probability of the positive class


    propensity = probabilities[0]  # Get the propensity score for the input text

    # Determine the prediction based on the threshold
    prediction = propensity >= threshold

    return (prediction, propensity)


# if __name__ == "__main__":
#     # Example usage
#     import mlflow
#     mlflow.set_tracking_uri("sqlite:////home/hegemon/AppliedMachineLearning/Assignment-2/mlflow.db")
#     id = "82a3a8bc05b34b48862a1d952279580a"
#     uri = f"runs:/{id}/Logistic Regression_model"
#     model = mlflow.sklearn.load_model(uri)

#     # SPAM Message
#     text = "Click here to win a free iPhone & 100000 dollars!"
#     threshold = 0.5
#     prediction, propensity = score(text, model, threshold)
#     print(f"Prediction: {prediction}, Propensity: {propensity}")
