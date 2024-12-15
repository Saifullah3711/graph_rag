# Use Python 3.11 slim image
FROM python:3.11-slim

# Set the working directory
WORKDIR /app

# Copy all project files into the container
COPY . /app

# Upgrade pip and install dependencies
RUN python -m pip install --upgrade pip && \
    pip install -r requirements.txt

# Expose the port for Streamlit
EXPOSE 8501

# Command to run the Streamlit app
CMD ["streamlit", "run", "st_chatbot.py", "--server.port=8501", "--server.address=0.0.0.0"]
