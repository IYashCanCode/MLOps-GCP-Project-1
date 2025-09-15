FROM python:3.10

WORKDIR /app



RUN pip install flask lightgbm joblib pandas numpy scikit-learn


COPY bank_churn_model.pkl .
COPY app.py .


EXPOSE 8080


CMD ["python", "app.py"]
