
import functions_framework
import json, os, hmac, hashlib
from google.cloud import storage, secretmanager
from github import Github

def get_secret(name):
    client = secretmanager.SecretManagerServiceClient()
    project = os.environ["GCP_PROJECT"]
    path = f"projects/{project}/secrets/{name}/versions/latest"
    res = client.access_secret_version(name=path)
    return res.payload.data.decode("UTF-8")

def verify_signature(payload, sig_header, secret):
    if not sig_header: return False
    expected = "sha256=" + hmac.new(
        secret.encode(), payload, hashlib.sha256).hexdigest()
    return hmac.compare_digest(expected, sig_header)

@functions_framework.http
def ingest_github(request):
    # 1. Auth & Validation
    try:
        secret = get_secret("github-webhook-secret")
        sig = request.headers.get("X-Hub-Signature-256", "")
        if not verify_signature(request.data, sig, secret):
            return ("Unauthorized", 401)
            
        if request.headers.get("X-GitHub-Event") == "ping":
            return ("Pong", 200)

        payload = request.get_json()
        repo_name = payload.get("repository", {}).get("full_name", "unknown")
        
        return (f"Successfully received event for {repo_name}", 200)
    except Exception as e:
        print(f"Error: {str(e)}")
        return (str(e), 500)
