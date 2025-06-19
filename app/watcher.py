import os
import logging
from pymongo import MongoClient
from config import MONGO_URI, DB_NAME

logger = logging.getLogger(__name__)

def run_watch():
    """Met à jour les CVs depuis Google Drive - Version cloud"""
    try:
        from app.utils.drive_utils import connect_to_drive, list_pdfs, download_file
        from app.utils.enrich_db import process_and_insert_cv
        from app.utils.vectorize import update_faiss_index
        
        # Utiliser MongoDB pour tracker les CVs vus
        from config import MONGO_CLIENT_OPTIONS
        try:
            client = MongoClient(MONGO_URI, **MONGO_CLIENT_OPTIONS)
            # Test de connexion
            client.admin.command('ping')
            logger.info("✅ Connexion MongoDB watcher réussie")
        except Exception as e:
            logger.warning(f"⚠️ Erreur SSL MongoDB, tentative fallback: {e}")
            # Fallback sans options SSL
            client = MongoClient(MONGO_URI)
            client.admin.command('ping')
            logger.info("✅ Connexion MongoDB fallback réussie")
            
        db = client[DB_NAME]
        seen_collection = db["seen_cvs"]

        def get_seen():
            """Récupère les IDs déjà traités depuis MongoDB"""
            seen_doc = seen_collection.find_one({"_id": "seen_files"})
            return set(seen_doc.get("file_ids", [])) if seen_doc else set()

        def save_seen(seen_ids):
            """Sauvegarde les IDs traités dans MongoDB"""
            seen_collection.replace_one(
                {"_id": "seen_files"}, 
                {"_id": "seen_files", "file_ids": list(seen_ids)}, 
                upsert=True
            )

        logger.info("🔍 Début de la mise à jour des CVs (version cloud)")
        
        seen = get_seen()
        service = connect_to_drive()
        folder_id = "16CpxlBPbm8ZMRBH-B7tj5cmn4h3bXCbt"
        
        try:
            pdfs = list_pdfs(service, folder_id)
            logger.info(f"📁 {len(pdfs)} fichiers trouvés dans Google Drive")
        except Exception as e:
            logger.error(f"❌ Erreur connexion Google Drive: {e}")
            return False

        new_seen = set(seen)
        processed_count = 0

        for pdf in pdfs:
            if pdf["id"] not in seen:
                try:
                    # Télécharger temporairement (sera supprimé au redémarrage)
                    filename = pdf["name"]
                    download_file(service, pdf["id"], filename)
                    
                    # Traiter et insérer en base
                    success = process_and_insert_cv(filename)
                    
                    if success:
                        logger.info(f"✅ Nouveau CV traité : {filename}")
                        new_seen.add(pdf["id"])
                        processed_count += 1
                        
                        # Nettoyer le fichier temporaire
                        try:
                            os.remove(filename)
                            logger.debug(f"🗑️ Fichier temporaire supprimé : {filename}")
                        except Exception as cleanup_error:
                            logger.warning(f"⚠️ Impossible de supprimer {filename}: {cleanup_error}")
                    else:
                        logger.warning(f"⚠️ Échec traitement : {filename}")
                        # Supprimer le fichier même en cas d'échec
                        try:
                            os.remove(filename)
                        except:
                            pass
                        
                except Exception as e:
                    logger.error(f"❌ Erreur traitement {pdf['name']}: {e}")
                    # Nettoyer en cas d'erreur
                    try:
                        if 'filename' in locals():
                            os.remove(filename)
                    except:
                        pass

        # Sauvegarder la liste des fichiers traités
        save_seen(new_seen)
        logger.info(f"📊 {processed_count} nouveaux CVs traités")
        
        # Mettre à jour l'index FAISS si de nouveaux CVs ont été ajoutés
        if processed_count > 0:
            logger.info("🔄 Mise à jour de l'index FAISS...")
            faiss_success = update_faiss_index()
            if faiss_success:
                logger.info("✅ Index FAISS mis à jour et stocké en base")
            else:
                logger.warning("⚠️ Erreur lors de la mise à jour FAISS")
        else:
            logger.info("ℹ️ Aucun nouveau CV à traiter")
        
        logger.info("✅ Mise à jour terminée")
        return True
        
    except Exception as e:
        logger.error(f"❌ Erreur dans run_watch: {e}")
        return False

if __name__ == "__main__":
    run_watch()
