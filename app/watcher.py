import os
import logging
from pymongo import MongoClient
from config import DB_NAME

logger = logging.getLogger(__name__)

def run_watch():
    """Met à jour les CVs depuis Google Drive - Version améliorée"""
    try:
        logger.info("🔍 Début de la mise à jour des CVs")
        
        # Import des modules nécessaires
        from app.utils.drive_utils import connect_to_drive, list_pdfs, download_file
        from app.utils.enrich_db import process_and_insert_cv, get_mongo_collection
        from app.utils.vectorize import update_faiss_index
        
        # Connexion MongoDB pour tracker les CVs vus
        from config import get_mongo_client
        client = get_mongo_client()
        
        if not client:
            logger.error("❌ Impossible de se connecter à MongoDB")
            return False
            
        db = client[DB_NAME]
        seen_collection = db["seen_cvs"]

        def get_seen():
            """Récupère les IDs déjà traités depuis MongoDB"""
            try:
                seen_doc = seen_collection.find_one({"_id": "seen_files"})
                return set(seen_doc.get("file_ids", [])) if seen_doc else set()
            except Exception as e:
                logger.error(f"Erreur récupération seen files: {e}")
                return set()

        def save_seen(seen_ids):
            """Sauvegarde les IDs traités dans MongoDB"""
            try:
                seen_collection.replace_one(
                    {"_id": "seen_files"}, 
                    {"_id": "seen_files", "file_ids": list(seen_ids)}, 
                    upsert=True
                )
                logger.info(f"📝 Sauvegarde: {len(seen_ids)} fichiers trackés")
            except Exception as e:
                logger.error(f"Erreur sauvegarde seen files: {e}")

        # Vérifier la collection MongoDB
        collection = get_mongo_collection()
        if collection is None:
            logger.error("❌ Collection MongoDB non disponible")
            return False
            
        initial_count = collection.count_documents({})
        logger.info(f"📊 CVs en base au départ: {initial_count}")
        
        # Connexion Google Drive
        try:
            service = connect_to_drive()
            logger.info("✅ Connexion Google Drive réussie")
        except Exception as e:
            logger.error(f"❌ Erreur connexion Google Drive: {e}")
            return False
        
        # Listage des PDFs
        folder_id = "16CpxlBPbm8ZMRBH-B7tj5cmn4h3bXCbt"
        
        try:
            pdfs = list_pdfs(service, folder_id)
            logger.info(f"📁 {len(pdfs)} fichiers trouvés dans Google Drive")
        except Exception as e:
            logger.error(f"❌ Erreur listage PDFs: {e}")
            return False

        if not pdfs:
            logger.warning("⚠️ Aucun PDF trouvé dans le dossier")
            return True  # Pas d'erreur, juste rien à traiter

        # Récupération des fichiers déjà vus
        seen = get_seen()
        new_seen = set(seen)
        processed_count = 0
        error_count = 0

        logger.info(f"🔄 Traitement: {len(pdfs)} PDFs total, {len(seen)} déjà vus")

        for i, pdf in enumerate(pdfs, 1):
            pdf_id = pdf["id"]
            filename = pdf["name"]
            
            logger.info(f"📄 [{i}/{len(pdfs)}] Traitement de {filename}")
            
            if pdf_id in seen:
                logger.info(f"⏭️ {filename} déjà traité, ignoré")
                continue

            try:
                # Téléchargement temporaire
                logger.info(f"⬇️ Téléchargement de {filename}...")
                download_success = download_file(service, pdf_id, filename)
                
                if not download_success:
                    logger.warning(f"⚠️ Échec téléchargement {filename}")
                    error_count += 1
                    continue
                
                # Traitement et insertion en base
                logger.info(f"🔄 Traitement CV {filename}...")
                success = process_and_insert_cv(filename)
                
                if success:
                    logger.info(f"✅ CV {filename} traité avec succès")
                    new_seen.add(pdf_id)
                    processed_count += 1
                else:
                    logger.warning(f"⚠️ Échec traitement {filename}")
                    error_count += 1
                
                # Nettoyage du fichier temporaire
                try:
                    if os.path.exists(filename):
                        os.remove(filename)
                        logger.debug(f"🗑️ Fichier temporaire {filename} supprimé")
                except Exception as cleanup_error:
                    logger.warning(f"⚠️ Impossible de supprimer {filename}: {cleanup_error}")
                        
            except Exception as e:
                logger.error(f"❌ Erreur traitement {filename}: {e}")
                error_count += 1
                
                # Nettoyer en cas d'erreur
                try:
                    if os.path.exists(filename):
                        os.remove(filename)
                except:
                    pass

        # Sauvegarder la liste des fichiers traités
        save_seen(new_seen)
        
        # Statistiques finales
        final_count = collection.count_documents({})
        added_count = final_count - initial_count
        
        logger.info(f"📊 Statistiques:")
        logger.info(f"   - PDFs traités: {processed_count}")
        logger.info(f"   - Erreurs: {error_count}")
        logger.info(f"   - CVs en base: {initial_count} → {final_count} (+{added_count})")
        
        # Mettre à jour l'index FAISS si de nouveaux CVs ont été ajoutés
        if processed_count > 0:
            logger.info("🔄 Mise à jour de l'index FAISS...")
            try:
                faiss_success = update_faiss_index()
                if faiss_success:
                    logger.info("✅ Index FAISS mis à jour et stocké en base")
                else:
                    logger.warning("⚠️ Erreur lors de la mise à jour FAISS")
            except Exception as e:
                logger.error(f"❌ Erreur FAISS: {e}")
        else:
            logger.info("ℹ️ Aucun nouveau CV à indexer")
        
        logger.info("✅ Mise à jour terminée avec succès")
        return True
        
    except Exception as e:
        logger.error(f"❌ Erreur globale dans run_watch: {e}")
        return False

if __name__ == "__main__":
    # Configuration du logging pour les tests
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    print("🚀 Test de mise à jour des CVs")
    success = run_watch()
    
    if success:
        print("✅ Test réussi")
    else:
        print("❌ Test échoué")
