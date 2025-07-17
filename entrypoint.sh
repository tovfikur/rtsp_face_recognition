#!/usr/bin/env bash

# Set HOME to a writable directory to avoid permission issues
export HOME=/tmp

# Create logs directory if it doesn't exist
mkdir -p /tmp/logs

# Function to log messages with timestamp
log_message() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a /tmp/logs/app.log
}

# Function to start the application with error handling
start_app() {
    local max_retries=5
    local retry_count=0
    local wait_time=5

    while [ $retry_count -lt $max_retries ]; do
        log_message "Starting application (attempt $((retry_count + 1))/$max_retries)..."
        
        if [ "$DEV" = "1" ]; then
            log_message "Running in development mode with uvicorn..."
            # Run Uvicorn with reload for live code edits in development
            uvicorn main:app --host 0.0.0.0 --port 8000 --reload 2>&1 | tee -a /tmp/logs/app.log
        else
            log_message "Running in production mode with gunicorn..."
            # Run Gunicorn with Uvicorn workers for production
            gunicorn -k uvicorn.workers.UvicornWorker main:app \
                --bind 0.0.0.0:8000 \
                --workers 4 \
                --timeout 120 \
                --access-logfile /tmp/logs/access.log \
                --error-logfile /tmp/logs/error.log 2>&1 | tee -a /tmp/logs/app.log
        fi
        
        exit_code=$?
        
        if [ $exit_code -eq 0 ]; then
            log_message "Application exited normally."
            break
        else
            log_message "Application crashed with exit code: $exit_code"
            retry_count=$((retry_count + 1))
            
            if [ $retry_count -lt $max_retries ]; then
                log_message "Waiting $wait_time seconds before retry..."
                sleep $wait_time
                # Exponential backoff
                wait_time=$((wait_time * 2))
            else
                log_message "Maximum retries reached. Entering fallback mode..."
                break
            fi
        fi
    done
}

# Function to run fallback server when main app fails
run_fallback() {
    log_message "Starting fallback HTTP server on port 8000..."
    
    # Create a simple fallback HTML page
    cat > /tmp/fallback.html << 'EOF'
<!DOCTYPE html>
<html>
<head>
    <title>Service Temporarily Unavailable</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 40px; background: #f5f5f5; }
        .container { background: white; padding: 30px; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
        .error { color: #d32f2f; }
        .info { color: #1976d2; margin-top: 20px; }
        pre { background: #f8f8f8; padding: 15px; border-radius: 4px; overflow-x: auto; }
    </style>
</head>
<body>
    <div class="container">
        <h1 class="error">Service Temporarily Unavailable</h1>
        <p>The main application encountered an error and is currently unavailable.</p>
        <div class="info">
            <h3>Recent Logs:</h3>
            <pre id="logs">Loading logs...</pre>
        </div>
        <p><small>This page refreshes every 30 seconds</small></p>
    </div>
    <script>
        function loadLogs() {
            fetch('/logs')
                .then(response => response.text())
                .then(data => {
                    document.getElementById('logs').textContent = data;
                })
                .catch(err => {
                    document.getElementById('logs').textContent = 'Unable to load logs: ' + err;
                });
        }
        
        loadLogs();
        setInterval(loadLogs, 30000);
        setTimeout(() => location.reload(), 30000);
    </script>
</body>
</html>
EOF

    # Start simple Python HTTP server as fallback
    python3 -c "
import http.server
import socketserver
import os
from urllib.parse import urlparse

class FallbackHandler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/' or self.path == '/health':
            self.send_response(503)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            with open('/tmp/fallback.html', 'rb') as f:
                self.wfile.write(f.read())
        elif self.path == '/logs':
            self.send_response(200)
            self.send_header('Content-type', 'text/plain')
            self.end_headers()
            try:
                with open('/tmp/logs/app.log', 'r') as f:
                    lines = f.readlines()
                    # Return last 50 lines
                    self.wfile.write(''.join(lines[-50:]).encode())
            except FileNotFoundError:
                self.wfile.write(b'No logs available yet.')
        else:
            self.send_response(404)
            self.end_headers()

os.chdir('/tmp')
with socketserver.TCPServer(('', 8000), FallbackHandler) as httpd:
    print('Fallback server running on port 8000...')
    httpd.serve_forever()
" 2>&1 | tee -a /tmp/logs/app.log
}

# Main execution
log_message "=== Container Starting ==="

# Validate Python installation
if ! command -v python3 &> /dev/null; then
    log_message "ERROR: Python3 not found!"
    run_fallback
    exit 1
fi

# Check if main.py exists
if [ ! -f "main.py" ]; then
    log_message "WARNING: main.py not found in current directory!"
    log_message "Current directory: $(pwd)"
    log_message "Files in directory: $(ls -la)"
fi

# Try to start the main application
start_app

# If we reach here, the main app failed all retries
log_message "Main application failed to start after all retries."
log_message "Starting fallback server to keep container running..."

# Keep container alive with fallback server
run_fallback