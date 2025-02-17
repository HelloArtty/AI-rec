# ใช้ Python เวอร์ชันล่าสุดที่รองรับ
FROM python:3.10

# กำหนดโฟลเดอร์ทำงานใน Container
WORKDIR /app

# คัดลอกไฟล์ทั้งหมดจากโปรเจกต์ไปยัง Container
COPY . /app

# ติดตั้ง dependencies
RUN pip install --no-cache-dir -r requirements.txt

# เปิดพอร์ต 8080 (Cloud Run ใช้พอร์ตนี้)
EXPOSE 8080

# คำสั่งรันเซิร์ฟเวอร์ FastAPI ด้วย Uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080", "--reload"]
