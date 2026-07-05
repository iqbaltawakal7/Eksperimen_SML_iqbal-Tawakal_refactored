"""
Skrip untuk mengirim request prediksi ke serving server (mock_serving.py)
dan mencatat hasil serta latensinya.

Jalankan 1.mock_serving.py terlebih dahulu di terminal terpisah, baru
jalankan skrip ini untuk mensimulasikan traffic inference.
"""

import json
import logging
import time
import urllib.request

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

SERVING_URL = "http://127.0.0.1:5001/invocations"


def send_inference_request(payload: dict, url: str = SERVING_URL) -> dict:
    """Mengirim satu request prediksi ke serving server dan mengembalikan hasilnya."""
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}, method="POST"
    )

    start = time.time()
    with urllib.request.urlopen(req, timeout=5) as response:
        elapsed = time.time() - start
        result = json.loads(response.read().decode("utf-8"))

    logger.info("Prediksi diterima dalam %.3f detik: %s", elapsed, result)
    return result


def run_sample_requests(n_requests: int = 5) -> None:
    """Mengirim beberapa request contoh secara berturut-turut untuk simulasi traffic."""
    sample_payload = {"inputs": [[0.0] * 5]}  # sesuaikan bentuk input dengan model Anda

    for i in range(1, n_requests + 1):
        try:
            send_inference_request(sample_payload)
        except Exception as exc:
            logger.error("Request #%d gagal: %s", i, exc)
        time.sleep(0.5)


if __name__ == "__main__":
    logger.info("=== TAHAP 7: MENJALANKAN INFERENCE KE SERVING SERVER ===")
    run_sample_requests()
