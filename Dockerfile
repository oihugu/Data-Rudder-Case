# syntax=docker/dockerfile:1
FROM python:3.7-apline
WORKDIR /project
RUN apk add --no-cache gcc musl-dev linux-headers
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt
EXPOSE 8000
COPY . .
CMD ["python", "manage.py", "runserver"]