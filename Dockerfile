# Use a more recent Python slim base image
FROM python:3.11-slim

# Set working directory in the container
WORKDIR /app

# Set environment variable to avoid buffering issues with Python logs
ENV PYTHONUNBUFFERED=1

# Install system dependencies required for Chrome, ChromeDriver, and other libraries
RUN apt-get update && apt-get install -y \
    gnupg \
    wget \
    unzip \
    curl \
    libglib2.0-0 \
    libnss3 \
    libgconf-2-4 \
    libfontconfig1 \
    libxrender1 \
    libxtst6 \
    libxi6 \
    libgtk-3-0 \
    && rm -rf /var/lib/apt/lists/*

# Install Google Chrome
RUN wget -q -O /tmp/chrome.deb https://dl.google.com/linux/direct/google-chrome-stable_current_amd64.deb \
    && apt-get update \
    && apt-get install -y --no-install-recommends /tmp/chrome.deb \
    && rm /tmp/chrome.deb \
    && rm -rf /var/lib/apt/lists/*

# Télécharger et installer ChromeDriver
RUN wget -q "https://storage.googleapis.com/chrome-for-testing-public/138.0.7204.92/linux64/chromedriver-linux64.zip"  -O /tmp/chromedriver.zip && \
    unzip /tmp/chromedriver.zip -d /tmp/ && \
    mv /tmp/chromedriver-linux64/chromedriver /usr/local/bin/chromedriver && \
    chmod +x /usr/local/bin/chromedriver && \
    rm -rf /tmp/chromedriver.zip /tmp/chromedriver-linux64

# Copy the requirements file
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire project directory
COPY . .

# Create directories for scraping outputs (if not already present)
# Create directories for scraping outputs
RUN mkdir -p /app/data /app/excels_anstat_new /app/telechargements_temp

# Expose the Streamlit default port
EXPOSE 8501

# Healthcheck to ensure Streamlit server is running
HEALTHCHECK --interval=30s --timeout=3s CMD curl -f http://localhost:8501/healthz || exit 1

# Command to run the Streamlit app
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]