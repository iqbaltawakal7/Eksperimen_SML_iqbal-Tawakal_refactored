# CATATAN: Semua nilai metrik di file ini masih SIMULASI (angka acak),
# bukan hasil pengukuran nyata dari model/serving server. Cocok untuk
# uji coba dashboard Grafana/Prometheus, tapi untuk produksi perlu
# diganti dengan instrumentasi nyata (mis. mencatat latensi & akurasi
# asli dari 7.inference.py / mock_serving.py).

import time
import random
from http.server import HTTPServer, BaseHTTPRequestHandler

print("Mempersiapkan 10 metrik kustom (simulasi) untuk jalur Advance...")

class MetricServer(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/metrics':
            self.send_response(200)
            self.send_header("Content-Type", "text/plain; version=0.0.4")
            self.end_headers()
            
            # MEMBUAT FORMAT 10 METRIK SECARA MANUAL TANPA LIBRARY PROMETHEUS
            metrics_data = f"""
# HELP model_prediction_requests_total Total jumlah request prediksi
# TYPE model_prediction_requests_total counter
model_prediction_requests_total {int(time.time() % 10000)}

# HELP model_prediction_errors_total Total kesalahan pada model
# TYPE model_prediction_errors_total counter
model_prediction_errors_total {random.randint(0, 5)}

# HELP model_prediction_speed_seconds Kecepatan prediksi saat ini
# TYPE model_prediction_speed_seconds gauge
model_prediction_speed_seconds {random.uniform(0.01, 0.05)}

# HELP model_simulated_accuracy Akurasi simulasi model terdeteksi
# TYPE model_simulated_accuracy gauge
model_simulated_accuracy {random.uniform(0.90, 0.98)}

# HELP system_cpu_usage_percentage Simulasi penggunaan CPU
# TYPE system_cpu_usage_percentage gauge
system_cpu_usage_percentage {random.uniform(20.0, 60.0)}

# HELP system_memory_usage_bytes Simulasi penggunaan RAM
# TYPE system_memory_usage_bytes gauge
system_memory_usage_bytes {random.randint(500000000, 1000000000)}

# HELP model_latency_seconds_bucket Distribusi latensi request
# TYPE model_latency_seconds histogram
model_latency_seconds_bucket{{le="0.1"}} {random.randint(10, 50)}
model_latency_seconds_sum {random.uniform(1.2, 5.5)}
model_latency_seconds_count {random.randint(50, 100)}

# HELP input_data_size_bytes_bucket Distribusi ukuran data input
# TYPE input_data_size_bytes histogram
input_data_size_bytes_bucket{{le="500"}} {random.randint(5, 20)}
input_data_size_bytes_sum {random.randint(1000, 5000)}
input_data_size_bytes_count {random.randint(20, 50)}

# HELP model_response_size_bytes Summary ukuran response data
# TYPE model_response_size_bytes summary
model_response_size_bytes{{quantile="0.9"}} {random.randint(100, 150)}
model_response_size_bytes_sum {random.randint(2000, 8000)}
model_response_size_bytes_count {random.randint(30, 80)}

# HELP model_inference_duration_seconds Summary durasi pemrosesan internal
# TYPE model_inference_duration_seconds summary
model_inference_duration_seconds{{quantile="0.5"}} {random.uniform(0.02, 0.04)}
model_inference_duration_seconds_sum {random.uniform(0.5, 2.5)}
model_inference_duration_seconds_count {random.randint(40, 90)}
"""
            self.wfile.write(metrics_data.encode('utf-8'))
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format, *args):
        # Mematikan log bawaan agar terminal tetap bersih
        return

if __name__ == '__main__':
    server = HTTPServer(('localhost', 8000), MetricServer)
    print("Prometheus Custom Exporter Alternatif berjalan di http://localhost:8000/metrics")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nServer dihentikan.")