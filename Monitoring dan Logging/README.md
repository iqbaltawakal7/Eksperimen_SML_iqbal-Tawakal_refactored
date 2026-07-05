# Monitoring dan logging

Urutan menjalankan:

1. `1.mock_serving.py` — serving server tiruan di `http://127.0.0.1:5001/invocations`.
2. `7.inference.py` — mengirim beberapa request contoh ke serving server.
3. `3.prometheus_exporter.py` — expose metrik custom di `http://localhost:8000/metrics` (masih simulasi, lihat catatan di file).
4. `2.prometheus.yml` — konfigurasi Prometheus untuk scrape exporter di atas.
5. Dashboard Grafana — bukti screenshot ada di folder `4.bukti monitoring Prometheus/`, `5.bukti monitoring Grafana/`, `6.bukti alerting Grafana/`.
