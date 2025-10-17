import os
from sentence_transformers import SentenceTransformer

token = input("Entrez votre HF_TOKEN : ").strip()
try:
    model = SentenceTransformer("paraphrase-MiniLM-L3-v2", use_auth_token=token)
    print("✅ Succès : accès au modèle Hugging Face avec ce token !")
except Exception as e:
    print(f"❌ Erreur d'accès Hugging Face : {e}")