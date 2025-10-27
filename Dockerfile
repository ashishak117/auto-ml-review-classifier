FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY src ./src
COPY models ./models
ENV PYTHONUNBUFFERED=1

# Default: just show help; later we’ll add entrypoints to train/predict
CMD ["python", "-c", "print('Container ready. Use docker run with appropriate commands.')"]
