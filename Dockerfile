# Use an official Python runtime as a parent image
FROM python:3.11-slim-buster

# Set the working directory to /app
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Make port 8501 available to the world outside this container
EXPOSE 8000

# Define environment variable
ENV STREAMLIT_SERVER_PORT 8000
ENV STREAMLIT_SERVER_ADDRESS 0.0.0.0
ENV STREAMLIT_SERVER_ENABLE_CORS false
ENV STREAMLIT_SERVER_HEADLESS true

# Run app.py when the container launches
CMD ["streamlit", "run", "main.py", "--server.port=${STREAMLIT_SERVER_PORT}", "--server.address=${STREAMLIT_SERVER_ADDRESS}", "--server.enableCORS=${STREAMLIT_SERVER_ENABLE_CORS}", "--server.headless=${STREAMLIT_SERVER_HEADLESS}"]
