import json
from cryptography.fernet import Fernet
from app.config import get_settings

def get_fernet() -> Fernet:
    settings = get_settings()
    return Fernet(settings.FERNET_KEY.encode() if isinstance(settings.FERNET_KEY, str) else settings.FERNET_KEY)

def encrypt_hla_data(data: dict) -> bytes:
    """Encrypt HLA data dict using Fernet symmetric encryption."""
    f = get_fernet()
    json_bytes = json.dumps(data).encode("utf-8")
    return f.encrypt(json_bytes)

def decrypt_hla_data(encrypted: bytes) -> dict:
    """Decrypt Fernet-encrypted HLA data back to dict."""
    f = get_fernet()
    decrypted = f.decrypt(encrypted)
    return json.loads(decrypted.decode("utf-8"))
