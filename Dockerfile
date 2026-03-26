FROM python:3.10-slim

# 1. Crear el usuario "user" con ID 1000 (Requisito de Hugging Face)
RUN useradd -m -u 1000 user

# 2. Configurar variables de entorno
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1
ENV PATH="/home/user/.local/bin:$PATH"

# 3. Instalar dependencias del sistema necesarias para compilar
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    python3-dev \
    git \
    wget \
    unzip \
    && rm -rf /var/lib/apt/lists/*

# 4. Cambiar al usuario no-root y establecer el directorio de trabajo
USER user
WORKDIR /home/user/app

# 5. Instalar Torch CPU explícitamente PRIMERO
RUN pip install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cpu

# 6. Copiar requirements e instalarlos
COPY --chown=user:user requirements.txt .
RUN pip install -r requirements.txt

# 7. Instalar Detectron2
RUN pip install 'git+https://github.com/facebookresearch/detectron2.git'

# 8. Descargar y descomprimir el modelo desde Drive
# REEMPLAZA ESTO CON TU ID REAL DE DRIVE (el que está en la URL de tu archivo .zip)
ARG MODEL_ID="1yBPzBLgI2tK2_WOSTXq74R8OT_dB6hGw"
RUN gdown ${MODEL_ID} -O model_bundle.zip && \
    unzip model_bundle.zip -d model_data && \
    rm model_bundle.zip

# 9. Copiar el resto del código de la aplicación
COPY --chown=user:user app ./app

# 10. Indicar a Uvicorn y Hugging Face el puerto correcto
EXPOSE 7860

# 11. Comando de inicio (1 solo worker para no saturar la RAM)
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "1"]