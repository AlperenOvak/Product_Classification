FROM python:3.10-slim
WORKDIR /app
COPY . /app
RUN pip install -r requirements.txt
EXPOSE 5002
CMD gunicorn --workers=4 --bind 0.0.0.0:5002 app:app