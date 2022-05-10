# syntax=docker/dockerfile:1
FROM python:3.7-slim
WORKDIR "/project"
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt
EXPOSE 8000
COPY . .
CMD ["python", "project/manage.py", "runserver", "--noreload"]