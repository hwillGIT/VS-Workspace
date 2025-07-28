# This Dockerfile builds the Docker image for the Webtester backend.

# Use a lightweight Python image as the base image.
# This image includes Python and pip, which are needed to run the Flask application.
FROM python:3.9-alpine

# Set the working directory inside the container.
# All subsequent commands will be executed in this directory.
WORKDIR /app

# Copy the requirements.txt file to the working directory.
# This file lists the Python dependencies.
COPY requirements.txt ./

# Install the Python dependencies.
# Installing dependencies in a separate layer helps with caching.
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the backend application code to the working directory.
COPY . .

# Expose port 5000, the default Flask development server port, so the container can receive incoming requests.
EXPOSE 5000

# The default command to run when the container starts.
# This command starts the Flask development server.
# Using '0.0.0.0' makes the server accessible from outside the container.
CMD ["flask", "run", "--host=0.0.0.0"]

# Note: This Dockerfile sets up the environment to run the Flask backend.
# Ensure that the .dockerignore file in the backend directory excludes unnecessary files
# like virtual environments and build artifacts to keep the image clean.
# The Flask application will be run using the 'flask run' command.