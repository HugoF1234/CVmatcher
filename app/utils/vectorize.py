import faiss
import pickle
import numpy as np
from pymongo import MongoClient
from sentence_transformers import SentenceTransformer
from bson import ObjectId
from app.config import MONGO_URI, DB_NAME, COLLECTION_NAME, FAISS_INDEX_FILE, ID_MAPPING_FILE

model = SentenceTransformer("all-MiniLM-L6-v2")

def update_faiss_index():
    client = MongoClient(MONGO_URI)
    collection = client[DB_NAME][COLLECTION_NAME]
    cvs = list(collection.find({}))

    vectors = []
    id_mapping = []

    for cv in cvs:
        comp = " ".join(cv.get("competences", []))
        bio = cv.get("biographie", "")
        exp = " ".join([e.get("description", "") for e in cv.get("experiences", [])])

        emb_comp = model.encode(comp) if comp else np.zeros(384)
        emb_bio = model.encode(bio) if bio else np.zeros(384)
        emb_exp = model.encode(exp) if exp else np.zeros(384)

        weighted_vector = (
            0.2 * emb_comp +
            0.4 * emb_bio +
            0.6 * emb_exp
        )

        vectors.append(weighted_vector)
        id_mapping.append(str(cv["_id"]))

    if not vectors:
        print("⚠️ Aucun vecteur à indexer.")
        return

    dimension = len(vectors[0])
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(vectors).astype("float32"))

    faiss.write_index(index, FAISS_INDEX_FILE)
    with open(ID_MAPPING_FILE, "wb") as f:
        pickle.dump(id_mapping, f)

    print(f"✅ Index FAISS mis à jour avec {len(vectors)} profils.")
