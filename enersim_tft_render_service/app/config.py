import os

SUPABASE_URL = os.environ["SUPABASE_URL"]
SUPABASE_SERVICE_ROLE_KEY = os.environ["SUPABASE_SERVICE_ROLE_KEY"]
MODEL_BUCKET = os.getenv("MODEL_BUCKET", "forecast-models")
DEFAULT_TIMEZONE = os.getenv("DEFAULT_TIMEZONE", "Europe/Amsterdam")

ENCODER_LENGTH = int(os.getenv("ENCODER_LENGTH", "672"))      # 7 dagen kwartierdata
FORECAST_HORIZON = int(os.getenv("FORECAST_HORIZON", "96"))   # 24 uur vooruit
TRAINING_LOOKBACK_DAYS = int(os.getenv("TRAINING_LOOKBACK_DAYS", "365"))
MAX_EPOCHS = int(os.getenv("MAX_EPOCHS", "40"))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "128"))
LEARNING_RATE = float(os.getenv("LEARNING_RATE", "0.001"))
