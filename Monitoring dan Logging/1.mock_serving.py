import json
import random
from http.server import HTTPServer, BaseHTTPRequestHandler

class MockServingServer(BaseHTTPRequestHandler):
    def do_POST(self):
        if self.path == '/invocations':
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            
            # Membuat respons JSON tiruan mirip hasil prediksi asli MLflow
            response_data = {
                "predictions": [random.uniform(10.0, 50.0)]
            }
            
            self.wfile.write(json.dumps(response_data).encode('utf-8'))
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format, *args):
        # Mematikan log bawaan terminal agar tetap bersih
        return

if __name__ == '__main__':
    # Menggunakan alamat lokal 127.0.0.1 port 5001
    server = HTTPServer(('127.0.0.1', 5001), MockServingServer)
    print("=== TAHAP 1: SERVER SERVING MODEL AKTIF ===")
    print("Berjalan lancar tanpa Flask di http://127.0.0.1:5001/invocations")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nServer dihentikan.")