from app.utils.drive_utils import connect_to_drive, list_pdfs, download_file
from app.utils.enrich_db import process_and_insert_cv
from app.utils.vectorize import update_faiss_index
import os

ALREADY_SEEN_FILE = "faiss_index/seen_cvs.txt"
def run_watch():
    from app.utils.drive_utils import connect_to_drive, list_pdfs, download_file
    from app.utils.enrich_db import process_and_insert_cv
    from app.utils.vectorize import update_faiss_index
    import os

    ALREADY_SEEN_FILE = "faiss_index/seen_cvs.txt"

    def get_seen():
        if not os.path.exists(ALREADY_SEEN_FILE):
            return set()
        with open(ALREADY_SEEN_FILE, "r") as f:
            return set(f.read().splitlines())

    def save_seen(seen_ids):
        with open(ALREADY_SEEN_FILE, "w") as f:
            for sid in seen_ids:
                f.write(f"{sid}\n")

    seen = get_seen()
    service = connect_to_drive()
    folder_id = "16CpxlBPbm8ZMRBH-B7tj5cmn4h3bXCbt"
    pdfs = list_pdfs(service, folder_id)
    new_seen = set(seen)

    for pdf in pdfs:
        if pdf["id"] not in seen:
            download_file(service, pdf["id"], pdf["name"])
            success = process_and_insert_cv(pdf["name"])
            if success:
                print(f"✅ Nouveau CV traité : {pdf['name']}")
                new_seen.add(pdf["id"])

    save_seen(new_seen)
    update_faiss_index()


if __name__ == "__main__":
    run_watch()
