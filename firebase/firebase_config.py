import firebase_admin
from firebase_admin import credentials

import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

KEY_PATH = os.path.join(BASE_DIR, "firebase_key.json")

if not firebase_admin._apps:

    cred = credentials.Certificate(KEY_PATH)

    firebase_admin.initialize_app(cred)

print("Firebase Initialized Successfully")