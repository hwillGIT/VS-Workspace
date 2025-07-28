# This Dockerfile builds the Docker image for the Webtester frontend.

# Use a lightweight Node.js image as the base image.
# This image includes Node.js and npm, which are needed to build the React application.
FROM node:18-alpine as builder

# Set the working directory inside the container.
# All subsequent commands will be executed in this directory.
WORKDIR /app

# Copy the package.json and package-lock.json files to the working directory.
# These files contain the project's dependencies.
COPY package*.json ./

# Install the project dependencies.
# The --force flag is used here to potentially resolve dependency conflicts,
# but it's generally better to fix conflicts in package.json if possible.
# Running npm install in a separate layer helps with caching.
RUN npm install --force

# Copy the rest of the frontend application code to the working directory.
COPY . .

# Build the React application for production.
# This command creates optimized static assets in the 'build' directory.
RUN npm run build

# Use a lightweight web server image to serve the built frontend application.
# This keeps the final image size small.
FROM nginx:alpine

# Copy the built React application files from the builder stage to the Nginx web server directory.
# These are the static files that Nginx will serve.
COPY --from=builder /app/build /usr/share/nginx/html

# Expose port 80, the default HTTP port, so the container can receive incoming requests.
EXPOSE 80

# The default command to run when the container starts.
# This command starts the Nginx web server in the foreground.
CMD ["nginx", "-g", "daemon off;"]

# Note: This Dockerfile sets up a multi-stage build.
# The first stage ('builder') builds the React application, and the second stage
# uses a minimal Nginx image to serve the static files, resulting in a smaller final image.
# Ensure that the .dockerignore file in the frontend directory excludes unnecessary files
# like node_modules and build artifacts to keep the image clean.