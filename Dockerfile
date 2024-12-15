# Use Python 3.11 slim image
FROM python:3.11-slim

# Set the working directory
WORKDIR /app

# Copy all project files into the container
COPY . /app

# Install dependencies with no cache
RUN pip install --no-cache-dir -r requirements.txt

# Expose the port for Streamlit
EXPOSE 8501

# Command to run the Streamlit app
CMD ["streamlit", "run", "st_chatbot.py", "--server.port=8501", "--server.address=0.0.0.0"]
