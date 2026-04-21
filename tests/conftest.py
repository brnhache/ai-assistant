import os

# Load before any service code imports Settings.
os.environ.setdefault("INBOUND_SERVICE_TOKEN", "test-inbound-token")
os.environ.setdefault("DESERT_API_BASE_URL", "http://example.invalid/api")
os.environ.setdefault("DESERT_SERVICE_TOKEN", "")
os.environ.setdefault("OPENAI_API_KEY", "")
os.environ.setdefault("CORS_ORIGINS", "*")

from config.settings import get_settings

get_settings.cache_clear()
