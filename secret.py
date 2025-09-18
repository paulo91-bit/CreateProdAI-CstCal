import secrets
import string

def generate_secure_string(length):
    """Generates a secure, random string of a given length."""
    # Define the characters to choose from (letters, digits, punctuation)
    alphabet = string.ascii_letters + string.digits + string.punctuation
    return ''.join(secrets.choice(alphabet) for i in range(length))

# Generate the 32-character key for AES-256
encryption_key = generate_secure_string(32)

# Generate the 16-character IV
encryption_iv = generate_secure_string(16)

print(f"MAISY3G_ENCRYPTION_KEY: \"{encryption_key}\"")
print(f"MAISY3G_ENCRYPTION_IV:  \"{encryption_iv}\"")
