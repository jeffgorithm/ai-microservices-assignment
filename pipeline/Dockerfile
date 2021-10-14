FROM python:3.8

COPY requirements.txt requirements.txt
COPY data.csv data.csv
COPY train_model.py train_model.py

RUN pip install -r requirements.txt

ENTRYPOINT ["python", "train_model.py"]