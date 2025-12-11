FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Expose the port Render sets
ENV PORT 5000

# Run the app using Gunicorn
CMD ["gunicorn", "server:app", "--bind", "0.0.0.0:$PORT"]
