import faiss
import pickle
import numpy as np
from pymongo import MongoClient
from sentence_transformers import SentenceTransformer
from bson import ObjectId

# === CONFIGURATION ===
MONGO_URI = "mongodb://localhost:27017"
DB_NAME = "CVExtraction"
COLLECTION_NAME = "CVExtractionCollection"
FAISS_INDEX_FILE = "cv_index.faiss"
ID_MAPPING_FILE = "id_mapping.pkl"

# === CHARGEMENT DU MODELE D'EMBEDDING ===
model = SentenceTransformer("all-MiniLM-L6-v2")

# === CONNEXION À MONGODB ===
client = MongoClient(MONGO_URI)
collection = client[DB_NAME][COLLECTION_NAME]
cvs = list(collection.find({}))

# === PRÉPARATION DES EMBEDDINGS ===
vectors = []
id_mapping = []

for cv in cvs:
    comp = " ".join(cv.get("competences", []))
    bio = cv.get("biographie", "")
    exp = " ".join([e.get("description", "") for e in cv.get("experiences", [])])

    # Embedding individuel
    emb_comp = model.encode(comp) if comp else np.zeros(384)
    emb_bio = model.encode(bio) if bio else np.zeros(384)
    emb_exp = model.encode(exp) if exp else np.zeros(384)

    # PONDÉRATION NUMÉRIQUE
    weighted_vector = (
        0.2 * emb_comp +
        0.4 * emb_bio +
        0.6 * emb_exp
    )

    vectors.append(weighted_vector)

# === CONSTRUCTION DE L'INDEX FAISS ===
dimension = len(vectors[0])
index = faiss.IndexFlatL2(dimension)
index.add(np.array(vectors).astype("float32"))

# === SAUVEGARDE ===
faiss.write_index(index, FAISS_INDEX_FILE)
with open(ID_MAPPING_FILE, "wb") as f:
    pickle.dump(id_mapping, f)

print(f"✅ Index FAISS sauvegardé avec {len(vectors)} vecteurs pondérés.")
