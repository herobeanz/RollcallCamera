"""
Cấu hình hệ thống phân tích video nhà ăn
"""
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# ==================== Đường dẫn ====================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
EMPLOYEES_DIR = os.path.join(DATA_DIR, "employees")
REGISTERED_DIR = os.path.join(DATA_DIR, "registered")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
RESULTS_DIR = os.path.join(OUTPUT_DIR, "results")
REPORTS_DIR = os.path.join(OUTPUT_DIR, "reports")
LOGS_DIR = os.path.join(BASE_DIR, "logs")

# Tạo thư mục nếu chưa tồn tại
for dir_path in [DATA_DIR, EMPLOYEES_DIR, REGISTERED_DIR, 
                 OUTPUT_DIR, RESULTS_DIR, REPORTS_DIR, LOGS_DIR]:
    os.makedirs(dir_path, exist_ok=True)

# ==================== Cấu hình IP Camera (ONVIF/RTSP) ====================
# Địa chỉ IP hoặc hostname của camera
CAMERA_HOST = os.getenv("CAMERA_HOST", "172.16.20.41")

# Port ONVIF service (thường là 80, 8080, hoặc 554)
CAMERA_PORT = int(os.getenv("CAMERA_PORT", "80"))

# Username để đăng nhập vào camera
CAMERA_USERNAME = os.getenv("CAMERA_USERNAME", "admin")

# Password để đăng nhập vào camera (từ environment variable)
CAMERA_PASSWORD = os.getenv("CAMERA_PASSWORD", "")