# Dockerfile

# Use a slim, modern Python base image
FROM python:3.12-slim

# Set the working directory inside the container
WORKDIR /app

# Set environment variables to prevent Python from writing .pyc files
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Install uv for dependency management
RUN pip install uv

# Copy only the dependency definition file first to leverage Docker layer caching
COPY pyproject.toml .

# Install dependencies, including development tools for verification
RUN uv sync --system --extra dev

# Copy the rest of the application source code
COPY . .

# Download NLTK and spaCy models during the build
RUN python -m nltk.downloader wordnet punkt
RUN python -m spacy download en_core_web_sm

# Expose the port for the web UI/visualization
EXPOSE 5000

# Define the default command to run when the container starts
CMD ["axiom-train"]