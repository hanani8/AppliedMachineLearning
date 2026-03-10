FROM python:3.11-slim

# Keep Python logs unbuffered and avoid bytecode files inside the container.
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install only the packages required by Assignment-3 app/score code.
RUN pip install --no-cache-dir \
	flask \
	joblib \
	scikit-learn \
	mlflow \
	numpy

# Copy the Flask app files requested in the assignment.
COPY Assignment-3/app.py /app/Assignment-3/app.py
COPY Assignment-3/score.py /app/Assignment-3/score.py

# Copy model/vectorizer assets used by app.py and score.py.
COPY Assignment-2 /app/Assignment-2

WORKDIR /app/Assignment-3

EXPOSE 5000

CMD ["python", "app.py"]
