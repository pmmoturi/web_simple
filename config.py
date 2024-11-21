import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Database configuration
DB_CONFIG = {
    'dbname': os.getenv('DB_NAME', 'finment'),
    'user': os.getenv('DB_USER', 'postgres'),
    'password': os.getenv('DB_PASSWORD', ''),
    'host': os.getenv('DB_HOST', 'localhost'),
    'port': os.getenv('DB_PORT', '5432')
}

# OpenAI configuration
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

# Flask configuration
FLASK_SECRET_KEY = os.getenv('FLASK_SECRET_KEY', 'your-secret-key-here')
UPLOAD_FOLDER = os.getenv('UPLOAD_FOLDER', 'uploads')
