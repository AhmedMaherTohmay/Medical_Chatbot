# Use the official Python image from the Docker Hub
FROM python:3.10-slim
# Set the working directory in the container
WORKDIR /app

# Install the required system dependencies for building Python packages
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libatlas-base-dev \
    build-essential \
    git-lfs

# Initialize Git LFS
RUN git lfs install
# Copy the requirements.txt into the container at /app
COPY requirments.txt .

# Install any Python dependencies specified in requirements.txt
RUN pip install --no-cache-dir -r requirments.txt

# Copy the rest of the application code into the container
COPY . .

# Pull LFS files
RUN git lfs pull

# Expose port 5000 for Flask
EXPOSE 5000

# Set environment variables for Flask
ENV FLASK_APP=app.py
ENV FLASK_RUN_HOST=0.0.0.0

# Command to run the Flask app
#CMD ["flask", "run"]
CMD ["python", "app.py"]
