import os
import logging
import time
from pymongo import MongoClient
from config import DB_NAME

logger = logging.getLogger(__name__)

def run_watch(batch_size=3, max_time_minutes=8):
    """Met à jour les CVs par petits lots pour éviter les timeouts"""
    start_time = time.time()
    max_time_seconds = max_time_minutes * 60
    
    try:
        logger.info(f"🔍 Début mise à jour CVs (lots de {batch_size}, max {max_time_minutes}min)")
        
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
            return True

        # Récupération des fichiers déjà vus
        seen = get_seen()
        new_seen = set(seen)
        processed_count = 0
        error_count = 0

        # Filtrer les PDFs non traités
        pdfs_to_process = [pdf for pdf in pdfs if pdf["id"] not in seen]
        logger.info(f"🔄 {len(pdfs_to_process)} PDFs à traiter sur {len(pdfs)} total")

        if not pdfs_to_process:
            logger.info("✅ Tous les PDFs sont déjà traités")
            return True

        # Traitement par lots
        for batch_start in range(0, len(pdfs_to_process), batch_size):
            # Vérifier le temps écoulé
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
                
                # Vérifier le temps pour chaque fichier
                elapsed_time = time.time() - start_time
                if elapsed_time > max_time_seconds:
                    logger.warning(f"⏰ Timeout pendant traitement de {filename}")
                    break
                
                logger.info(f"📄 [{batch_start + i + 1}/{len(pdfs_to_process)}] {filename}")
                
                try:
                    # Téléchargement avec timeout court
                    logger.info(f"⬇️ Téléchargement...")
                    download_success = download_file(service, pdf_id, filename)
                    
                    if not download_success:
                        logger.warning(f"⚠️ Échec téléchargement {filename}")
                        error_count += 1
                        continue
                    
                    # Traitement avec monitoring du temps
                    processing_start = time.time()
                    logger.info(f"🔄 Traitement...")
                    success = process_and_insert_cv(filename)
                    processing_time = time.time() - processing_start
                    
                    if success:
                        logger.info(f"✅ {filename} traité en {processing_time:.1f}s")
                        new_seen.add(pdf_id)  # Ajout à seen uniquement si insertion réussie
                        processed_count += 1
                    else:
                        logger.warning(f"⚠️ Échec traitement {filename} (non ajouté à seen_cvs)")
                        error_count += 1
                    
                    # Nettoyage immédiat
                    try:
                        if os.path.exists(filename):
                            os.remove(filename)
                    except Exception as cleanup_error:
                        logger.warning(f"⚠️ Nettoyage {filename}: {cleanup_error}")
                    
                    # Petite pause entre les fichiers pour éviter la surcharge
                    time.sleep(1)
                        
                except Exception as e:
                    logger.error(f"❌ Erreur traitement {filename}: {e}")
                    error_count += 1
                    
                    # Nettoyer en cas d'erreur
                    try:
                        if os.path.exists(filename):
                            os.remove(filename)
                    except:
                        pass
            
            # Sauvegarder après chaque lot
            save_seen(new_seen)
            logger.info(f"💾 Progression sauvegardée: {processed_count} traités")
            
            # Pause entre les lots
            if batch_end < len(pdfs_to_process):
                logger.info("⏸️ Pause 3s entre les lots...")
                time.sleep(3)

        # Statistiques finales
        final_count = collection.count_documents({})
        added_count = final_count - initial_count
        total_time = time.time() - start_time
        
        logger.info(f"📊 Statistiques finales:")
        logger.info(f"   - Temps total: {total_time:.1f}s")
        logger.info(f"   - PDFs traités: {processed_count}")
        logger.info(f"   - Erreurs: {error_count}")
        logger.info(f"   - CVs en base: {initial_count} → {final_count} (+{added_count})")
        
        # Mettre à jour l'index FAISS si de nouveaux CVs ont été ajoutés
        # if processed_count > 0:
        #     logger.info("🔄 Mise à jour index FAISS...")
        #     try:
        #         faiss_success = update_faiss_index()
        #         if faiss_success:
        #             logger.info("✅ Index FAISS mis à jour")
        #         else:
        #             logger.warning("⚠️ Erreur FAISS")
        #     except Exception as e:
        #         logger.error(f"❌ Erreur FAISS: {e}")
        # L'ajout incrémental FAISS est déjà fait dans enrich_db, inutile de revectoriser toute la base ici.
        
        # Indiquer s'il reste des fichiers à traiter
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
    # Configuration du logging pour les tests
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
