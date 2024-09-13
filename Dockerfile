FROM --platform=linux/arm64 python:3.12-slim

#Set working directory to /app inside container
WORKDIR /app

#Copy code and input directors into container
COPY code/ /app/code
COPY input/ /app/input

#Copy requirements.txt and install dependences
COPY code/requirements.txt /app/
RUN pip install --upgrade pip setuptools wheel
RUN pip install -r requirements.txt

#Set default command to run Python script
CMD ["python", "code/main.py"]
