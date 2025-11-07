import os
import logging
import time
import random
from pymongo import MongoClient
from config import DB_NAME, COLLECTION_NAME

logger = logging.getLogger(__name__)

def run_watch(batch_size=3, max_time_minutes=8):
    """Met à jour les CVs par petits lots pour éviter les timeouts"""
    start_time = time.time()
    max_time_seconds = max_time_minutes * 60
    
    try:
        logger.info(f"🔍 Début mise à jour CVs (lots de {batch_size}, max {max_time_minutes}min)")
        
        from app.utils.drive_utils import connect_to_drive, list_pdfs, download_file
        from app.utils.enrich_db import process_and_insert_cv, get_mongo_collection
        from app.utils.vectorize import update_faiss_index, sync_faiss_with_db
        
        from config import get_mongo_client
        client = get_mongo_client()
        
        if not client:
            logger.error("❌ Impossible de se connecter à MongoDB")
            return False
            
        db = client[DB_NAME]
        collection = db[COLLECTION_NAME]
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

        if collection is None:
            logger.error("❌ Collection MongoDB non disponible")
            return False
        
        initial_count = collection.count_documents({})
        logger.info(f"📊 CVs en base au départ: {initial_count}")
        
        try:
            service = connect_to_drive()
            logger.info("✅ Connexion Google Drive réussie")
        except Exception as e:
            logger.error(f"❌ Erreur connexion Google Drive: {e}")
            return False
        
        from config import GOOGLE_DRIVE_FOLDER_ID
        folder_id = GOOGLE_DRIVE_FOLDER_ID
        
        try:
            pdfs = list_pdfs(service, folder_id)
            logger.info(f"📁 {len(pdfs)} fichiers trouvés dans Google Drive")
        except Exception as e:
            logger.error(f"❌ Erreur listage PDFs: {e}")
            return False

        if not pdfs:
            logger.warning("⚠️ Aucun PDF trouvé dans le dossier")
            return True

        seen = get_seen()
        new_seen = set(seen)
        processed_count = 0
        error_count = 0

        pdfs_to_process = [pdf for pdf in pdfs if pdf["id"] not in seen]
        random.shuffle(pdfs_to_process)
        logger.info(f"🔄 {len(pdfs_to_process)} PDFs à traiter sur {len(pdfs)} total (ordre aléatoire)")

        if not pdfs_to_process:
            logger.info("✅ Tous les PDFs sont déjà traités")
            return True

        for batch_start in range(0, len(pdfs_to_process), batch_size):
            elapsed_time = time.time() - start_time
            if elapsed_time > max_time_seconds:
                logger.warning(f"⏰ Timeout atteint ({max_time_minutes}min), arrêt du traitement")
                break
            
            batch_end = min(batch_start + batch_size, len(pdfs_to_process))
            batch = pdfs_to_process[batch_start:batch_end]
            
            logger.info(f"📦 Lot {batch_start//batch_size + 1}: traitement de {len(batch)} PDFs")
            
            for i, pdf in enumerate(batch):
                pdf_id = pdf["id"]
                filename = pdf["name"]
                
                elapsed_time = time.time() - start_time
                if elapsed_time > max_time_seconds:
                    logger.warning(f"⏰ Timeout pendant traitement de {filename}")
                    break
                
                logger.info(f"📄 [{batch_start + i + 1}/{len(pdfs_to_process)}] {filename}")
                
                try:
                    logger.info(f"⬇️ Téléchargement...")
                    download_success = download_file(service, pdf_id, filename)
                    
                    if not download_success:
                        logger.warning(f"⚠️ Échec téléchargement {filename}")
                        error_count += 1
                        continue
                    
                    processing_start = time.time()
                    logger.info(f"🔄 Traitement...")
                    success = process_and_insert_cv(filename, collection=collection)
                    processing_time = time.time() - processing_start
                    
                    if success:
                        logger.info(f"✅ {filename} traité en {processing_time:.1f}s")
                        new_seen.add(pdf_id)
                        processed_count += 1
                    else:
                        logger.warning(f"⚠️ Échec traitement {filename} (non ajouté à seen_cvs)")
                        error_count += 1
                    
                    try:
                        if os.path.exists(filename):
                            os.remove(filename)
                    except Exception as cleanup_error:
                        logger.warning(f"⚠️ Nettoyage {filename}: {cleanup_error}")
                    
                    time.sleep(1)
                        
                except Exception as e:
                    logger.error(f"❌ Erreur traitement {filename}: {e}")
                    error_count += 1
                    
                    try:
                        if os.path.exists(filename):
                            os.remove(filename)
                    except:
                        pass
            
            save_seen(new_seen)
            logger.info(f"💾 Progression sauvegardée: {processed_count} traités")
            
            if batch_end < len(pdfs_to_process):
                logger.info("⏸️ Pause 3s entre les lots...")
                time.sleep(3)

        final_count = collection.count_documents({})
        added_count = final_count - initial_count
        total_time = time.time() - start_time
        
        logger.info(f"📊 Statistiques finales:")
        logger.info(f"   - Temps total: {total_time:.1f}s")
        logger.info(f"   - PDFs traités: {processed_count}")
        logger.info(f"   - Erreurs: {error_count}")
        logger.info(f"   - CVs en base: {initial_count} → {final_count} (+{added_count})")
        
        
        remaining = len(pdfs_to_process) - processed_count - error_count
        if remaining > 0:
            logger.info(f"ℹ️ {remaining} PDFs restants à traiter (relancez la mise à jour)")
        else:
            logger.info("✅ Tous les PDFs ont été traités")
        
        logger.info("✅ Mise à jour terminée")
        return True
        
    except Exception as e:
        logger.error(f"❌ Erreur globale dans run_watch: {e}")
        return False

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    print("🚀 Test de mise à jour des CVs par lots")
    success = run_watch(batch_size=2, max_time_minutes=5)
    
    if success:
        print("✅ Test réussi")
    else:
        print("❌ Test échoué")
