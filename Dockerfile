FROM python:3.10-slim
WORKDIR /app
COPY . /app
RUN pip install -r requirements.txt
RUN pip install gunicorn
EXPOSE 5002
CMD gunicorn --bind 0.0.0.0:5002 app:app