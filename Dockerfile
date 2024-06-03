# Gunakan image dasar yang sesuai
FROM python

# Setel direktori kerja
WORKDIR /app

# Salin requirements.txt dan instal dependensi
COPY requirements.txt  requirements.txt
RUN pip install -r requirements.txt

# Salin semua file aplikasi ke dalam direktori kerja
COPY . .


CMD ["python", "predict.py"]
