# Use official Python 3.11 slim image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy requirements and install
COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Copy all files
COPY . .

# Expose the port Render sets (optional, mostly for documentation)
ENV PORT 5000

# Run the app with Gunicorn, expanding $PORT via shell
CMD ["sh", "-c", "gunicorn server:app --bind 0.0.0.0:$PORT"]
