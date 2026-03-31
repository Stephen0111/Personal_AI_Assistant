import json
import os
import datetime
from google.cloud import storage
from dotenv import load_dotenv

load_dotenv()

BUCKET = os.getenv("GCS_BUCKET_NAME")


def log_event(event_type: str, data: dict):
    """Append an event JSON line to GCS. Cheap and serverless."""
    if not BUCKET:
        return  # Skip if not configured
    try:
        client = storage.Client()
        bucket = client.bucket(BUCKET)
        date = datetime.date.today().isoformat()
        blob = bucket.blob(f"events/{date}.jsonl")
        
        event = {"timestamp": datetime.datetime.utcnow().isoformat(),
                 "type": event_type, **data}
        
        # Append mode via download → append → upload
        try:
            existing = blob.download_as_text()
        except Exception:
            existing = ""
        blob.upload_from_string(existing + json.dumps(event) + "\n")
    except Exception as e:
        print(f"Analytics error (non-fatal): {e}")
