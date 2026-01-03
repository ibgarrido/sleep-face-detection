# Base ligera
FROM python:3.11-slim

# Evitar archivos basura
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# --- CORRECCIÓN AQUÍ ---
# Cambiamos 'libgl1-mesa-glx' por 'libgl1'
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Instalar dependencias Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiar TODO el código y el MODELO (.task)
COPY . .

# Comando por defecto
CMD ["python", "inference_logistic.py"]
CMD ["python", "inference_bayesian.py"]
