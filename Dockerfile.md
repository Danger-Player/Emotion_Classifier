# 1. Start from a base image — a pre-built Python environment
FROM python:3.12-slim

# 2. Set the working directory inside the container
WORKDIR /app

# 3. Copy dependency list first (explained below), then install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 4. Copy the rest of your project files
COPY . .

# 5. Tell Docker which port the app listens on
EXPOSE 8000

# 6. The command that runs when the container starts
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]