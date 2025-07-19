# RTSP Face Recognition System

A comprehensive real-time face recognition system with RTSP stream support, WebSocket communication, and secure API endpoints. Built with FastAPI, PyTorch, and FaceNet for high-accuracy facial recognition in live video streams.

## 🎯 Features

### Core Functionality
- **Real-time Face Recognition**: Process live video streams from RTSP cameras
- **Multi-face Detection**: Detect and identify multiple faces simultaneously
- **Face Registration**: Register new faces with multiple reference images
- **Live Streaming**: View processed video streams with bounding boxes
- **Callback Integration**: Send detection events to external systems
- **WebSocket Support**: Real-time detection via WebSocket connections

### Security & Authentication
- **JWT Authentication**: Secure API access with token-based authentication
- **User Management**: Admin-controlled user creation and management
- **Session Management**: Cookie-based authentication for web interface
- **CORS Support**: Configurable cross-origin resource sharing

### Technical Features
- **GPU Acceleration**: CUDA support for faster processing
- **Database Storage**: SQLite with SQLAlchemy ORM
- **Configuration Management**: Dynamic parameter adjustment
- **Thread Pool Processing**: Concurrent RTSP stream handling
- **Memory Optimization**: Efficient frame processing and cleanup

## 🏗️ Architecture

### System Components

```
┌─────────────────────────────────────────────────────────────┐
│                    RTSP Face Recognition System              │
├─────────────────────────────────────────────────────────────┤
│  Frontend Layer                                             │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐         │
│  │   Login     │  │   Config    │  │ RTSP Mgmt   │         │
│  │   Page      │  │   Panel     │  │   Panel     │         │
│  └─────────────┘  └─────────────┘  └─────────────┘         │
├─────────────────────────────────────────────────────────────┤
│  API Layer (FastAPI)                                       │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐         │
│  │ Auth        │  │ Face Ops    │  │ RTSP        │         │
│  │ Endpoints   │  │ Endpoints   │  │ Endpoints   │         │
│  └─────────────┘  └─────────────┘  └─────────────┘         │
├─────────────────────────────────────────────────────────────┤
│  Processing Layer                                           │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐         │
│  │ Face        │  │ RTSP        │  │ WebSocket   │         │
│  │ Detection   │  │ Processing  │  │ Handler     │         │
│  └─────────────┘  └─────────────┘  └─────────────┘         │
├─────────────────────────────────────────────────────────────┤
│  AI/ML Layer                                               │
│  ┌─────────────┐  ┌─────────────┐                         │
│  │ MTCNN       │  │ Inception   │                         │
│  │ Detection   │  │ ResNet V1   │                         │
│  └─────────────┘  └─────────────┘                         │
├─────────────────────────────────────────────────────────────┤
│  Data Layer                                                │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐         │
│  │ SQLite DB   │  │ File System │  │ Memory      │         │
│  │ (Users,     │  │ (Templates, │  │ Cache       │         │
│  │ Faces, etc) │  │ Static)     │  │             │         │
│  └─────────────┘  └─────────────┘  └─────────────┘         │
└─────────────────────────────────────────────────────────────┘
```

### Data Flow

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│ RTSP Camera │───▶│   Frame     │───▶│ Face        │
│   Stream    │    │ Capture     │    │ Detection   │
└─────────────┘    └─────────────┘    └─────────────┘
                                              │
                                              ▼
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│ External    │◀───│  Callback   │◀───│ Face        │
│  System     │    │  Handler    │    │ Recognition │
└─────────────┘    └─────────────┘    └─────────────┘
                                              │
                                              ▼
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│ Web Client  │◀───│ WebSocket/  │◀───│ Bounding    │
│  Display    │    │ HTTP Stream │    │ Box Draw    │
└─────────────┘    └─────────────┘    └─────────────┘
```

### Database Schema

```sql
-- Users table for authentication
CREATE TABLE users (
    id INTEGER PRIMARY KEY,
    username STRING UNIQUE,
    hashed_password STRING
);

-- Configuration parameters
CREATE TABLE config (
    key STRING PRIMARY KEY,
    value STRING
);

-- Registered face embeddings
CREATE TABLE registered_faces (
    id INTEGER PRIMARY KEY,
    person_id STRING UNIQUE,
    embedding BLOB,  -- Pickled numpy array
    updated_at DATETIME
);

-- RTSP stream configurations
CREATE TABLE rtsp_streams (
    id INTEGER PRIMARY KEY,
    name STRING,
    rtsp_url STRING,
    is_active BOOLEAN,
    created_at DATETIME,
    last_detection DATETIME
);
```

## 🚀 Installation

### Prerequisites

- Python 3.8+
- CUDA-compatible GPU (optional, for acceleration)
- FFmpeg (for RTSP stream processing)

### Dependencies

The project includes a comprehensive `requirements.txt` file with all necessary dependencies:

```bash
# Install all dependencies
pip install -r requirements.txt

# Key libraries included:
# - torch, torchvision (PyTorch ML framework)
# - facenet-pytorch (Face recognition models)
# - opencv-python (Computer vision)
# - fastapi, uvicorn (Web framework)
# - sqlalchemy (Database ORM)
# - pyjwt, passlib[bcrypt] (Authentication)
# - jinja2 (Template engine)
# - httpx (HTTP client for callbacks)
```

**Note**: The Docker setup automatically handles all dependency installation and model file requirements.

### Setup Instructions

1. **Clone the repository**
```bash
git clone https://github.com/tovfikur/rtsp_face_recognition.git
cd rtsp_face_recognition
```

2. **Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Environment configuration**
```bash
# Create .env file (optional)
SECRET_KEY=your-super-secure-secret-key-here
```

5. **Initialize database**
```bash
# Database tables are created automatically on first run
python main.py
```

## 🎮 Usage

### Starting the Server

```bash
# Development mode
python main.py

# Production mode with Uvicorn
uvicorn main:app --host 0.0.0.0 --port 8000 --workers 4
```

### Default Credentials

- **Username**: `admin`
- **Password**: `admin`

**⚠️ Important**: Change the default admin password immediately after installation.

### Web Interface Access

- **Main Dashboard**: http://localhost:8000/
- **API Documentation**: http://localhost:8000/docs
- **RTSP Management**: http://localhost:8000/rtsp/manage
- **WebSocket Test**: http://localhost:8000/ws/test

## 📡 API Endpoints

### Authentication

#### POST `/token`
Obtain JWT access token
```bash
curl -X POST "http://localhost:8000/token" \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "username=admin&password=admin"
```

### Face Operations

#### POST `/register`
Register a new person with face images
```bash
curl -X POST "http://localhost:8000/register" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -F "person_id=john_doe" \
  -F "images=@photo1.jpg" \
  -F "images=@photo2.jpg"
```

#### POST `/authenticate`
Authenticate person from uploaded image
```bash
curl -X POST "http://localhost:8000/authenticate" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -F "file=@test_image.jpg"
```

#### POST `/compare`
Compare two face images
```bash
curl -X POST "http://localhost:8000/compare" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -F "reference=@person1.jpg" \
  -F "check=@person2.jpg"
```

### RTSP Stream Management

#### POST `/rtsp/streams`
Add new RTSP stream
```bash
curl -X POST "http://localhost:8000/rtsp/streams" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Main Entrance",
    "rtsp_url": "rtsp://admin:password@192.168.1.100:554/stream1"
  }'
```

#### POST `/rtsp/streams/{stream_id}/start`
Start monitoring RTSP stream
```bash
curl -X POST "http://localhost:8000/rtsp/streams/1/start" \
  -H "Authorization: Bearer YOUR_TOKEN"
```

#### GET `/rtsp/streams/{stream_id}/video`
View live processed video stream
```
http://localhost:8000/rtsp/streams/1/video
```

### Configuration

#### GET `/config`
Get current system configuration
```bash
curl -X GET "http://localhost:8000/config" \
  -H "Authorization: Bearer YOUR_TOKEN"
```

#### PUT `/config`
Update system configuration
```bash
curl -X PUT "http://localhost:8000/config" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "tolerance": 0.8,
    "detection_threshold": 0.95
  }'
```

## 🔌 WebSocket Integration

### Real-time Detection

Connect to WebSocket endpoint for live face detection:

```javascript
const ws = new WebSocket('ws://localhost:8000/ws/detection?token=YOUR_JWT_TOKEN');

// Send base64-encoded image frame
ws.send(base64ImageData);

// Receive detection results
ws.onmessage = function(event) {
    const results = JSON.parse(event.data);
    console.log('Detected persons:', results);
};
```

### Frame Processing Flow

1. **Client sends** base64-encoded image frame
2. **Server processes** frame through MTCNN → InceptionResnetV1
3. **Face matching** against registered embeddings
4. **Results returned** as JSON array of person IDs

## ⚙️ Configuration Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `TOLERANCE` | 0.9 | Face matching threshold (lower = stricter) |
| `DETECTION_THRESHOLD` | 0.95 | Face detection confidence threshold |
| `CALLBACK_URL` | "" | External system notification endpoint |
| `CALLBACK_TOKEN` | "" | Authentication token for callbacks |

### Callback Integration

Configure external system notifications:

```bash
curl -X PUT "http://localhost:8000/callback-config" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "callback_url": "https://your-system.com/api/face-detected",
    "callback_token": "your-api-key"
  }'
```

The system sends POST requests with this payload:
```json
{
  "employee_id": 123,
  "check_in": "2025-01-20 14:30:00",
  "check_out": "2025-01-20 22:30:00"
}
```

## 🐳 Docker Deployment

The project includes a sophisticated multi-stage Docker build optimized for both development and production environments.

### Multi-Stage Dockerfile

The Dockerfile uses a two-stage build process:

**Stage 1 - Builder**: Compiles dependencies and builds wheels
**Stage 2 - Runtime**: Creates slim production image

Key features:
- **Multi-stage build** for optimized image size
- **Non-root user** security implementation
- **Volume mounting** for development
- **Required model files** (shape_predictor_68_face_landmarks.dat, nn4.small2.v1.t7)
- **FFmpeg integration** for RTSP stream processing
- **Flexible entrypoint** supporting dev/prod modes

### Required Model Files

Before building, ensure these OpenFace model files are in your project root:

```bash
# Download required model files
wget http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
bunzip2 shape_predictor_68_face_landmarks.dat.bz2

wget https://storage.cmusatyalab.org/openface-models/nn4.small2.v1.t7
```

### Development Setup

```bash
# Clone and build
git clone https://github.com/tovfikur/rtsp_face_recognition.git
cd rtsp_face_recognition

# Start development environment
docker-compose up --build
```

The compose file includes:
- **Development mode**: `DEV=1` environment variable
- **Live reload**: Volume mounting for code changes
- **Log persistence**: Dedicated volume for application logs
- **Auto-restart**: Container restart policy

### Docker Compose Configuration

```yaml
version: "3.8"
services:
  app:
    build: .
    ports:
      - "8000:8000"
    environment:
      - DEV=1  # Enables development mode with hot reload
    volumes:
      - .:/app  # Live code mounting for development
      - logs:/tmp/logs  # Persistent log storage
    restart: unless-stopped

volumes:
  logs:  # Named volume for log persistence
```

### Production Deployment

For production, modify the environment:

```yaml
version: "3.8"
services:
  app:
    build: .
    ports:
      - "8000:8000"
    environment:
      - DEV=0  # Production mode
      - SECRET_KEY=your-super-secure-production-key
    volumes:
      - ./db:/app/db  # Persist database
      - logs:/tmp/logs
    restart: unless-stopped
    deploy:
      resources:
        limits:
          memory: 2G
        reservations:
          memory: 1G

volumes:
  logs:
```

### Container Features

#### Security
- **Non-root execution**: Application runs as `appuser`
- **Sudo access**: Available for system-level operations
- **Secure defaults**: Proper file permissions and user isolation

#### Performance
- **Optimized libraries**: Pre-compiled OpenCV and ML dependencies
- **Efficient caching**: Multi-stage build reduces final image size
- **Resource management**: Configurable memory limits

#### Development Support
- **Live reload**: Code changes reflect immediately
- **Volume mounting**: Full project directory accessible
- **Log access**: Persistent logging across container restarts

### Running the Container

```bash
# Development mode (with live reload)
docker-compose up

# Production mode
DEV=0 docker-compose up -d

# View logs
docker-compose logs -f app

# Access container shell
docker-compose exec app bash

# Stop services
docker-compose down
```

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `DEV` | 1 | Development mode (0 for production) |
| `SECRET_KEY` | auto | JWT signing key |
| `PYTHONDONTWRITEBYTECODE` | 1 | Prevent .pyc files |
| `PYTHONUNBUFFERED` | 1 | Real-time logging |

### Entrypoint Script

The `entrypoint.sh` script automatically chooses the appropriate server mode:

- **Development**: Uvicorn with auto-reload
- **Production**: Gunicorn with multiple workers

This provides optimal performance for each environment while maintaining simplicity in deployment.

## 🔧 Advanced Configuration

### GPU Acceleration

The system automatically detects and uses CUDA-compatible GPUs:

```python
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
```

### Memory Optimization

For production deployments:

```python
# Adjust frame processing settings
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimize buffering
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  # Optimize resolution
cap.set(cv2.CAP_PROP_FPS, 10)  # Control frame rate
```

### Performance Tuning

1. **CPU**: Use ThreadPoolExecutor with max_workers=5
2. **Memory**: Explicit garbage collection after frame processing
3. **Network**: MJPEG streaming with quality optimization
4. **Database**: Connection pooling with SQLAlchemy

## 🚨 Security Considerations

### Production Security

1. **Change default credentials** immediately
2. **Set strong SECRET_KEY** environment variable
3. **Enable HTTPS** in production
4. **Configure CORS** for specific origins
5. **Implement rate limiting** for API endpoints
6. **Regular security updates** for dependencies

### Authentication Flow

```
Client Request → JWT Token Validation → User Verification → Resource Access
```

### Token Management

- **Expiry**: 30 minutes default
- **Storage**: HTTP-only cookies + Authorization headers
- **Refresh**: Manual re-authentication required

## 📊 Monitoring & Logging

### System Logs

```python
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("FaceRecognitionAPI")
```

### Performance Metrics

- Frame processing time monitoring
- Detection accuracy tracking
- System resource utilization
- RTSP connection stability

### Health Checks

Monitor these endpoints:
- `/docs` - API documentation availability
- `/config` - System configuration status
- `/rtsp/streams` - Active stream monitoring

## 🤝 Contributing

### Development Setup

1. Fork the repository
2. Create feature branch: `git checkout -b feature/amazing-feature`
3. Commit changes: `git commit -m 'Add amazing feature'`
4. Push to branch: `git push origin feature/amazing-feature`
5. Open Pull Request

### Code Standards

- Follow PEP 8 style guidelines
- Add type hints for new functions
- Include docstrings for public methods
- Write unit tests for new features

## 📄 License

Copyright © 2025 Kendroo. All rights reserved.

Developed by: **Iftekar Hossan** (tovfikur)

## 🆘 Support

### Common Issues

**Q: RTSP stream not connecting**
- Verify camera URL and credentials
- Check network connectivity
- Ensure FFmpeg is installed

**Q: Face detection not working**
- Verify image quality and lighting
- Check GPU/CUDA installation
- Adjust detection threshold

**Q: WebSocket connection fails**
- Validate JWT token format
- Check browser WebSocket support
- Verify CORS configuration

### Contact

- **GitHub**: [tovfikur](https://github.com/tovfikur)
- **Project**: [rtsp_face_recognition](https://github.com/tovfikur/rtsp_face_recognition)

### Version History

- **v1.0.0**: Initial release with core features
  - Face detection and recognition
  - RTSP stream processing
  - WebSocket real-time communication
  - JWT authentication
  - Web-based management interface

---

*Built with ❤️ using FastAPI, PyTorch, and modern web technologies.*