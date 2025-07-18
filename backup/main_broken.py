import os
import cv2
import pickle
import base64
import logging
import numpy as np
import torch
import time
from datetime import datetime, timedelta
from typing import List, Optional, Any, Dict, Tuple
import httpx
import threading
import asyncio
import json
from concurrent.futures import ThreadPoolExecutor
from fastapi import FastAPI, File, UploadFile, HTTPException, Depends, WebSocket, WebSocketDisconnect, Form, Request, Response, status, Query
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from pydantic import BaseModel
from sqlalchemy import create_engine, Column, Integer, String, LargeBinary, DateTime, Boolean, Float, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.sql import func
import jwt
from passlib.context import CryptContext
from facenet_pytorch import MTCNN, InceptionResnetV1
from scipy.spatial import distance as dist
from scipy.optimize import linear_sum_assignment
from collections import OrderedDict, defaultdict
from filterpy.kalman import KalmanFilter
import dlib
from imutils import face_utils
import skimage.transform
from sqlalchemy import inspect

# ----------------------------------------------------------------------------------
# DATABASE SETUP
# ----------------------------------------------------------------------------------
DB_FOLDER = "db"
if not os.path.exists(DB_FOLDER):
    os.makedirs(DB_FOLDER)

SQLALCHEMY_DATABASE_URL = f"sqlite:///{DB_FOLDER}/app.db"
engine = create_engine(SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

TEMPLATES_DIR = "templates"
if not os.path.exists(TEMPLATES_DIR):
    os.makedirs(TEMPLATES_DIR)

STATIC_DIR = "static"
if not os.path.exists(STATIC_DIR):
    os.makedirs(STATIC_DIR)

LOG_DIR = "/app/logs"
if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)
    try:
        os.chmod(LOG_DIR, 0o777)
    except PermissionError:
        pass  # Handle permission issues gracefully

# Use stdout if /tmp/logs is not writable
try:
    with open('/tmp/logs/app.log', 'a') as f:
        pass
    log_file = '/tmp/logs/app.log'
except (PermissionError, FileNotFoundError):
    log_file = None

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s: %(message)s",
    handlers=[
        logging.StreamHandler()  # Always log to stdout
    ] + ([logging.FileHandler(log_file)] if log_file else [])
)
logger = logging.getLogger("FaceRecognitionAPI")

templates = Jinja2Templates(directory=TEMPLATES_DIR)

# ----------------------------------------------------------------------------------
# DATABASE MODELS
# ----------------------------------------------------------------------------------
class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True)
    hashed_password = Column(String)

class ConfigEntry(Base):
    __tablename__ = "config"
    key = Column(String, primary_key=True, index=True)
    value = Column(String)

class RegisteredFace(Base):
    __tablename__ = "registered_faces"
    id = Column(Integer, primary_key=True, index=True)
    person_id = Column(String, unique=True, index=True)
    embedding = Column(LargeBinary)
    openface_embedding = Column(LargeBinary)
    hog_features = Column(LargeBinary)
    updated_at = Column(DateTime, default=datetime.utcnow)
    confidence = Column(Float)
    face_quality = Column(Float)
    landmarks = Column(LargeBinary)

class TrackingHistory(Base):
    __tablename__ = "tracking_history"
    id = Column(Integer, primary_key=True, index=True)
    person_id = Column(String, index=True)
    track_id = Column(Integer, index=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    position_x = Column(Float)
    position_y = Column(Float)
    velocity_x = Column(Float)
    velocity_y = Column(Float)
    confidence = Column(Float)
    bbox_x = Column(Float)
    bbox_y = Column(Float)
    bbox_w = Column(Float)
    bbox_h = Column(Float)

class PersonTrajectory(Base):
    __tablename__ = "person_trajectories"
    id = Column(Integer, primary_key=True, index=True)
    person_id = Column(String, index=True)
    track_id = Column(Integer, index=True)
    start_time = Column(DateTime)
    end_time = Column(DateTime)
    trajectory_data = Column(Text)
    total_distance = Column(Float)
    avg_speed = Column(Float)

class RTSPStream(Base):
    __tablename__ = "rtsp_streams"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, index=True)
    rtsp_url = Column(String)
    is_active = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    last_detection = Column(DateTime)
    detection_count = Column(Integer, default=0)
    avg_processing_time = Column(Float)

class DetectionEvent(Base):
    __tablename__ = "detection_events"
    id = Column(Integer, primary_key=True, index=True)
    person_id = Column(String, index=True)
    stream_id = Column(Integer, index=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    confidence = Column(Float)
    bbox_x = Column(Float)
    bbox_y = Column(Float)
    bbox_w = Column(Float)
    bbox_h = Column(Float)
    face_quality = Column(Float)

# ----------------------------------------------------------------------------------
# DATABASE MIGRATION
# ----------------------------------------------------------------------------------
def migrate_database():
    inspector = inspect(engine)
    
    # Check and migrate rtsp_streams table
    if inspector.has_table('rtsp_streams'):
        table_columns = inspector.get_columns("rtsp_streams")
        column_names = [col['name'] for col in table_columns]
        
        with engine.connect() as connection:
            with connection.begin():
                if 'detection_count' not in column_names:
                    logger.info("Adding detection_count column to rtsp_streams table")
                    connection.execute("ALTER TABLE rtsp_streams ADD COLUMN detection_count INTEGER DEFAULT 0")
                if 'avg_processing_time' not in column_names:
                    logger.info("Adding avg_processing_time column to rtsp_streams table")
                    connection.execute("ALTER TABLE rtsp_streams ADD COLUMN avg_processing_time FLOAT")
    
    # Check and migrate registered_faces table
    if inspector.has_table('registered_faces'):
        table_columns = inspector.get_columns("registered_faces")
        column_names = [col['name'] for col in table_columns]
        
        with engine.connect() as connection:
            with connection.begin():
                if 'openface_embedding' not in column_names:
                    logger.info("Adding openface_embedding column to registered_faces table")
                    connection.execute("ALTER TABLE registered_faces ADD COLUMN openface_embedding BLOB")
                if 'hog_features' not in column_names:
                    logger.info("Adding hog_features column to registered_faces table")
                    connection.execute("ALTER TABLE registered_faces ADD COLUMN hog_features BLOB")
                if 'confidence' not in column_names:
                    logger.info("Adding confidence column to registered_faces table")
                    connection.execute("ALTER TABLE registered_faces ADD COLUMN confidence REAL")
                if 'face_quality' not in column_names:
                    logger.info("Adding face_quality column to registered_faces table")
                    connection.execute("ALTER TABLE registered_faces ADD COLUMN face_quality REAL")
                if 'landmarks' not in column_names:
                    logger.info("Adding landmarks column to registered_faces table")
                    connection.execute("ALTER TABLE registered_faces ADD COLUMN landmarks BLOB")

# ----------------------------------------------------------------------------------
# SECURITY: OAuth2 & JWT
# ----------------------------------------------------------------------------------
SECRET_KEY = os.getenv("SECRET_KEY", "your-secret-key")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24 * 7  # 7 days

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/token")

def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password: str) -> str:
    return pwd_context.hash(password)

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
    to_encode.update({"exp": expire})
    token = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return token

def decode_token(token: str) -> Any:
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload
    except jwt.PyJWTError:
        raise HTTPException(status_code=401, detail="Could not validate credentials")

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def get_user(db: Session, username: str) -> Optional[User]:
    return db.query(User).filter(User.username == username).first()

async def get_current_user(token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)) -> User:
    payload = decode_token(token)
    username: str = payload.get("sub")
    if username is None:
        raise HTTPException(status_code=401, detail="Invalid authentication credentials")
    user = get_user(db, username)
    if user is None:
        raise HTTPException(status_code=401, detail="User not found")
    return user

async def get_current_admin(current_user: User = Depends(get_current_user)) -> User:
    if current_user.username != "admin":
        raise HTTPException(status_code=403, detail="Not enough privileges")
    return current_user

# ----------------------------------------------------------------------------------
# INITIALIZATION: DEFAULT CONFIG & ADMIN USER
# ----------------------------------------------------------------------------------
DEFAULT_CONFIG = {
    "TOLERANCE": "0.9",
    "DETECTION_THRESHOLD": "0.95",
    "CALLBACK_URL": "",
    "CALLBACK_TOKEN": ""
}

last_callback_times: Dict[str, float] = {}
COOLDOWN_SECONDS = 60

async def send_callback(person_id: str):
    if person_id == "unknown":
        return
    current_time = time.time()
    if person_id in last_callback_times:
        time_since_last_callback = current_time - last_callback_times[person_id]
        if time_since_last_callback < COOLDOWN_SECONDS:
            logger.info(f"Skipping callback for {person_id}: cooldown period active ({time_since_last_callback:.1f}s < {COOLDOWN_SECONDS}s)")
            return
    try:
        employee_id = int(person_id)
    except ValueError:
        logger.error(f"Invalid person_id: {person_id}, cannot convert to integer.")
        return
    db = SessionLocal()
    try:
        url_entry = db.query(ConfigEntry).filter(ConfigEntry.key == "CALLBACK_URL").first()
        token_entry = db.query(ConfigEntry).filter(ConfigEntry.key == "CALLBACK_TOKEN").first()
        if not url_entry or not url_entry.value:
            logger.info("No callback URL set, skipping callback.")
            return
        callback_url = url_entry.value
        headers = {"Content-Type": "application/json"}
        if token_entry and token_entry.value:
            headers["API-Key"] = token_entry.value
        current_time_str = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
        check_out_time = (datetime.utcnow() + timedelta(hours=8)).strftime("%Y-%m-%d %H:%M:%S")
        data = {
            "employee_id": employee_id,
            "check_in": current_time_str,
            "check_out": check_out_time
        }
        async with httpx.AsyncClient() as client:
            try:
                response = await client.post(callback_url, json=data, headers=headers)
                response.raise_for_status()
                last_callback_times[person_id] = current_time
                logger.info(f"Callback sent to {callback_url} successfully for {person_id}.")
            except httpx.HTTPError as e:
                logger.error(f"Failed to send callback to {callback_url}: {str(e)}")
    finally:
        db.close()

async def cleanup_old_callback_records():
    current_time = time.time()
    expired_ids = [
        pid for pid, timestamp in last_callback_times.items()
        if current_time - timestamp > COOLDOWN_SECONDS * 10
    ]
    for pid in expired_ids:
        del last_callback_times[pid]

def init_config(db: Session):
    for key, value in DEFAULT_CONFIG.items():
        if not db.query(ConfigEntry).filter(ConfigEntry.key == key).first():
            db.add(ConfigEntry(key=key, value=value))
    db.commit()

def init_admin(db: Session):
    if not db.query(User).filter(User.username == "admin").first():
        admin = User(username="admin", hashed_password=get_password_hash("admin123"))
        db.add(admin)
        db.commit()

# ----------------------------------------------------------------------------------
# FACE RECOGNITION MODELS & HELPERS
# ----------------------------------------------------------------------------------
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
logger.info(f"Using device: {device}")

try:
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    mtcnn = MTCNN(image_size=160, margin=20, device=device)
    resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
    # Skip OpenFace if not available
    openface_available = False
    try:
        import openface
        align = openface.AlignDlib("shape_predictor_68_face_landmarks.dat")
        net = openface.TorchNeuralNet("nn4.small2.v1.t7", imgDim=96, cuda=torch.cuda.is_available())
        openface_available = True
        logger.info("OpenFace models loaded successfully")
    except (AttributeError, ImportError, Exception) as e:
        logger.error(f"OpenFace model loading failed: {e}")
        logger.info("OpenFace disabled, using fallback methods (FaceNet, HOG)")
except Exception as e:
    logger.error(f"Error loading other models: {e}")
    logger.info("Some models may not be available, fallback methods will be used")

async def read_imagefile(file: UploadFile) -> np.ndarray:
    image_bytes = await file.read()
    np_arr = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(status_code=400, detail="Invalid image")
    return image

# ----------------------------------------------------------------------------------
# FACE QUALITY ASSESSMENT
# ----------------------------------------------------------------------------------
def assess_face_quality(face_image: np.ndarray, landmarks: np.ndarray = None) -> float:
    quality_score = 0.0
    h, w = face_image.shape[:2]
    if h < 60 or w < 60:
        return 0.1
    if len(face_image.shape) == 3:
        gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
    else:
        gray = face_image
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    sharpness_score = min(laplacian_var / 500.0, 1.0)
    quality_score += sharpness_score * 0.3
    brightness = np.mean(gray)
    brightness_score = 1.0 - abs(brightness - 127.5) / 127.5
    quality_score += brightness_score * 0.2
    contrast = np.std(gray)
    contrast_score = min(contrast / 50.0, 1.0)
    quality_score += contrast_score * 0.2
    size_score = min(min(h, w) / 100.0, 1.0)
    quality_score += size_score * 0.3
    return min(quality_score, 1.0)

# ----------------------------------------------------------------------------------
# ENHANCED FEATURE EXTRACTION
# ----------------------------------------------------------------------------------
def extract_hog_features(image: np.ndarray, orientations: int = 9, 
                        pixels_per_cell: Tuple[int, int] = (8, 8),
                        cells_per_block: Tuple[int, int] = (2, 2)) -> np.ndarray:
    from skimage.feature import hog
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.resize(image, (64, 128))
    features = hog(image, orientations=orientations, 
                   pixels_per_cell=pixels_per_cell,
                   cells_per_block=cells_per_block,
                   visualize=False, feature_vector=True)
    return features

def get_openface_embedding(image: np.ndarray) -> np.ndarray:
    if not openface_available:
        raise Exception("OpenFace not available")
    try:
        if len(image.shape) == 3:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            image_rgb = image
        bb = align.getLargestFaceBoundingBox(image_rgb)
        if bb is None:
            raise Exception("No face bounding box found")
        aligned_face = align.align(96, image_rgb, bb, 
                                 landmarkIndices=openface.AlignDlib.OUTER_EYES_AND_NOSE)
        if aligned_face is None:
            raise Exception("Face alignment failed")
        rep = net.forward(aligned_face)
        return rep
    except Exception as e:
        logger.warning(f"OpenFace embedding failed: {e}")
        raise

def extract_facial_landmarks(image: np.ndarray) -> np.ndarray:
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    faces = detector(gray)
    if len(faces) == 0:
        raise Exception("No face detected for landmarks")
    landmarks = predictor(gray, faces[0])
    landmarks = face_utils.shape_to_np(landmarks)
    return landmarks

def compute_enhanced_embedding(image: np.ndarray) -> Dict[str, Any]:
    results = {}
    quality_score = assess_face_quality(image)
    results['quality'] = quality_score
    
    # Quick face check using MTCNN first
    face_detected = False
    try:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        face_boxes, _ = mtcnn.detect(image_rgb)
        face_detected = face_boxes is not None and len(face_boxes) > 0
    except:
        pass
    
    # Only try landmark extraction if a face was detected
    if face_detected:
        try:
            landmarks = extract_facial_landmarks(image)
            results['landmarks'] = landmarks
        except Exception as e:
            logger.debug(f"Landmark extraction skipped: {e}")
            results['landmarks'] = None
    else:
        results['landmarks'] = None
        
    try:
        hog_features = extract_hog_features(image)
        results['hog_features'] = hog_features
    except Exception as e:
        logger.warning(f"HOG feature extraction failed: {e}")
        results['hog_features'] = None
    if openface_available:
        try:
            openface_embedding = get_openface_embedding(image)
            results['openface_embedding'] = openface_embedding
        except Exception as e:
            logger.warning(f"OpenFace embedding skipped: {e}")
            results['openface_embedding'] = None
    else:
        results['openface_embedding'] = None
    try:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        face = mtcnn(image_rgb)
        if face is not None:
            face = face.unsqueeze(0).to(device)
            with torch.no_grad():
                facenet_embedding = resnet(face).cpu().numpy().flatten()
            results['facenet_embedding'] = facenet_embedding
        else:
            results['facenet_embedding'] = None
    except Exception as e:
        logger.warning(f"FaceNet embedding failed: {e}")
        results['facenet_embedding'] = None
    return results

# ----------------------------------------------------------------------------------
# ENHANCED KALMAN TRACKER
# ----------------------------------------------------------------------------------
class EnhancedKalmanTracker:
    def __init__(self, initial_bbox: Tuple[float, float, float, float]):
        self.kf = KalmanFilter(dim_x=8, dim_z=4)
        x, y, w, h = initial_bbox
        self.kf.x = np.array([x, y, 0., 0., w, h, 0., 0.])
        self.kf.F = np.array([
            [1, 0, 1, 0, 0, 0, 0, 0],
            [0, 1, 0, 1, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 1, 0],
            [0, 0, 0, 0, 0, 1, 0, 1],
            [0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 1]
        ])
        self.kf.H = np.array([
            [1, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0]
        ])
        self.kf.R *= 10.0
        self.kf.P *= 1000.0
        self.kf.Q *= 0.01
        self.time_since_update = 0
        self.hit_streak = 0
        self.age = 0
        self.history = []
    
    def predict(self):
        self.kf.predict()
        self.age += 1
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1
        x, y, w, h = self.kf.x[0], self.kf.x[1], self.kf.x[4], self.kf.x[5]
        return (x, y, w, h)
    
    def update(self, bbox: Tuple[float, float, float, float]):
        self.time_since_update = 0
        self.history = []
        self.hit_streak += 1
        self.kf.update(np.array([bbox[0], bbox[1], bbox[2], bbox[3]]))
    
    def get_state(self) -> Tuple[float, float, float, float]:
        x, y, w, h = self.kf.x[0], self.kf.x[1], self.kf.x[4], self.kf.x[5]
        return (x, y, w, h)
    
    def get_velocity(self) -> Tuple[float, float]:
        return (self.kf.x[2], self.kf.x[3])

# ----------------------------------------------------------------------------------
# MULTI-PERSON TRACKING SYSTEM
# ----------------------------------------------------------------------------------
class PersonTracker:
    def __init__(self, max_disappeared: int = 30, max_distance: float = 100):
        self.next_id = 0
        self.trackers = OrderedDict()
        self.disappeared = OrderedDict()
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance
        self.trajectories = defaultdict(list)
    
    def register(self, bbox: Tuple[float, float, float, float]) -> int:
        tracker_id = self.next_id
        self.trackers[tracker_id] = EnhancedKalmanTracker(bbox)
        self.disappeared[tracker_id] = 0
        self.next_id += 1
        return tracker_id
    
    def deregister(self, tracker_id: int):
        if tracker_id in self.trackers:
            del self.trackers[tracker_id]
            del self.disappeared[tracker_id]
    
    def update(self, detections: List[Tuple[float, float, float, float]]) -> Dict[int, Dict]:
        if len(detections) == 0:
            for tracker_id in list(self.disappeared.keys()):
                self.disappeared[tracker_id] += 1
                if self.disappeared[tracker_id] > self.max_disappeared:
                    self.deregister(tracker_id)
            return {}
        predictions = {}
        for tracker_id, tracker in self.trackers.items():
            pred_bbox = tracker.predict()
            predictions[tracker_id] = pred_bbox
        if len(self.trackers) == 0:
            for detection in detections:
                self.register(detection)
            return self._get_current_states()
        cost_matrix = self._compute_cost_matrix(predictions, detections)
        if cost_matrix.size > 0:
            row_indices, col_indices = linear_sum_assignment(cost_matrix)
            matched_trackers = set()
            matched_detections = set()
            for row, col in zip(row_indices, col_indices):
                if cost_matrix[row, col] <= self.max_distance:
                    tracker_id = list(self.trackers.keys())[row]
                    self.trackers[tracker_id].update(detections[col])
                    self.disappeared[tracker_id] = 0
                    matched_trackers.add(tracker_id)
                    matched_detections.add(col)
            for tracker_id in self.trackers.keys():
                if tracker_id not in matched_trackers:
                    self.disappeared[tracker_id] += 1
                    if self.disappeared[tracker_id] > self.max_disappeared:
                        self.deregister(tracker_id)
            for i, detection in enumerate(detections):
                if i not in matched_detections:
                    self.register(detection)
        return self._get_current_states()
    
    def _compute_cost_matrix(self, predictions: Dict, detections: List) -> np.ndarray:
        if not predictions or not detections:
            return np.array([])
        tracker_ids = list(predictions.keys())
        cost_matrix = np.zeros((len(tracker_ids), len(detections)))
        for i, tracker_id in enumerate(tracker_ids):
            pred_bbox = predictions[tracker_id]
            pred_center = (pred_bbox[0] + pred_bbox[2]/2, pred_bbox[1] + pred_bbox[3]/2)
            for j, detection in enumerate(detections):
                det_center = (detection[0] + detection[2]/2, detection[1] + detection[3]/2)
                distance = np.sqrt((pred_center[0] - det_center[0])**2 + 
                                   (pred_center[1] - det_center[1])**2)
                cost_matrix[i, j] = distance
        return cost_matrix
    
    def _get_current_states(self) -> Dict[int, Dict]:
        states = {}
        for tracker_id, tracker in self.trackers.items():
            bbox = tracker.get_state()
            velocity = tracker.get_velocity()
            confidence = max(0, 1.0 - (tracker.time_since_update / self.max_disappeared))
            states[tracker_id] = {
                'bbox': bbox,
                'velocity': velocity,
                'confidence': confidence,
                'age': tracker.age,
                'hit_streak': tracker.hit_streak
            }
            center = (bbox[0] + bbox[2]/2, bbox[1] + bbox[3]/2)
            self.trajectories[tracker_id].append({
                'timestamp': time.time(),
                'center': center,
                'bbox': bbox
            })
            if len(self.trajectories[tracker_id]) > 100:
                self.trajectories[tracker_id] = self.trajectories[tracker_id][-100:]
        return states

# ----------------------------------------------------------------------------------
# ENHANCED PERSON DETECTION
# ----------------------------------------------------------------------------------
class EnhancedPersonDetector:
    def __init__(self):
        self.hog = cv2.HOGDescriptor()
        self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    def detect_persons(self, frame: np.ndarray) -> List[Tuple[float, float, float, float]]:
        try:
            (rects, weights) = self.hog.detectMultiScale(
                frame, 
                winStride=(4, 4), 
                padding=(8, 8), 
                scale=1.05,
                finalThreshold=2
            )
            detections = []
            for (x, y, w, h) in rects:
                detections.append((float(x), float(y), float(w), float(h)))
            detections = self._apply_nms(detections, weights)
            return detections
        except Exception as e:
            logger.warning(f"Person detection failed: {e}")
            return []
    
    def detect_faces_in_person(self, frame: np.ndarray, person_bbox: Tuple[float, float, float, float]) -> List[Tuple[float, float, float, float]]:
        try:
            x, y, w, h = person_bbox
            x, y, w, h = int(x), int(y), int(w), int(h)
            person_roi = frame[y:y+h, x:x+w]
            gray_roi = cv2.cvtColor(person_roi, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(
                gray_roi, 
                scaleFactor=1.1, 
                minNeighbors=5, 
                minSize=(30, 30)
            )
            face_detections = []
            for (fx, fy, fw, fh) in faces:
                face_detections.append((float(x + fx), float(y + fy), float(fw), float(fh)))
            return face_detections
        except Exception as e:
            logger.warning(f"Face detection in person failed: {e}")
            return []
    
    def _apply_nms(self, detections: List, weights: np.ndarray, threshold: float = 0.3) -> List:
        if len(detections) == 0:
            return []
        boxes = []
        scores = []
        for i, (x, y, w, h) in enumerate(detections):
            boxes.append([x, y, w, h])
            scores.append(float(weights[i]) if i < len(weights) else 1.0)
        indices = cv2.dnn.NMSBoxes(boxes, scores, 0.5, threshold)
        if len(indices) > 0:
            indices = indices.flatten()
            return [detections[i] for i in indices]
        return []old_callback_records():
    current_time = time.time()
    expired_ids = [
        pid for pid, timestamp in last_callback_times.items()
        if current_time - timestamp > COOLDOWN_SECONDS * 10
    ]
    for pid in expired_ids:
        del last_callback_times[pid]

def init_config(db: Session):
    for key, value in DEFAULT_CONFIG.items():
        if not db.query(ConfigEntry).filter(ConfigEntry.key == key).first():
            db.add(ConfigEntry(key=key, value=value))
    db.commit()

def init_admin(db: Session):
    if not db.query(User).filter(User.username == "admin").first():
        admin = User(username="admin", hashed_password=get_password_hash("admin123"))
        db.add(admin)
        db.commit()

# ----------------------------------------------------------------------------------
# FACE RECOGNITION MODELS & HELPERS
# ----------------------------------------------------------------------------------
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
logger.info(f"Using device: {device}")

try:
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    mtcnn = MTCNN(image_size=160, margin=20, device=device)
    resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
    # Skip OpenFace if not available
    openface_available = False
    try:
        import openface
        align = openface.AlignDlib("shape_predictor_68_face_landmarks.dat")
        net = openface.TorchNeuralNet("nn4.small2.v1.t7", imgDim=96, cuda=torch.cuda.is_available())
        openface_available = True
        logger.info("OpenFace models loaded successfully")
    except (AttributeError, ImportError, Exception) as e:
        logger.error(f"OpenFace model loading failed: {e}")
        logger.info("OpenFace disabled, using fallback methods (FaceNet, HOG)")
except Exception as e:
    logger.error(f"Error loading other models: {e}")
    logger.info("Some models may not be available, fallback methods will be used")

async def read_imagefile(file: UploadFile) -> np.ndarray:
    image_bytes = await file.read()
    np_arr = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(status_code=400, detail="Invalid image")
    return image

# ----------------------------------------------------------------------------------
# FACE QUALITY ASSESSMENT
# ----------------------------------------------------------------------------------
def assess_face_quality(face_image: np.ndarray, landmarks: np.ndarray = None) -> float:
    quality_score = 0.0
    h, w = face_image.shape[:2]
    if h < 60 or w < 60:
        return 0.1
    if len(face_image.shape) == 3:
        gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
    else:
        gray = face_image
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    sharpness_score = min(laplacian_var / 500.0, 1.0)
    quality_score += sharpness_score * 0.3
    brightness = np.mean(gray)
    brightness_score = 1.0 - abs(brightness - 127.5) / 127.5
    quality_score += brightness_score * 0.2
    contrast = np.std(gray)
    contrast_score = min(contrast / 50.0, 1.0)
    quality_score += contrast_score * 0.2
    size_score = min(min(h, w) / 100.0, 1.0)
    quality_score += size_score * 0.3
    return min(quality_score, 1.0)

# ----------------------------------------------------------------------------------
# ENHANCED FEATURE EXTRACTION
# ----------------------------------------------------------------------------------
def extract_hog_features(image: np.ndarray, orientations: int = 9, 
                        pixels_per_cell: Tuple[int, int] = (8, 8),
                        cells_per_block: Tuple[int, int] = (2, 2)) -> np.ndarray:
    from skimage.feature import hog
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.resize(image, (64, 128))
    features = hog(image, orientations=orientations, 
                   pixels_per_cell=pixels_per_cell,
                   cells_per_block=cells_per_block,
                   visualize=False, feature_vector=True)
    return features

def get_openface_embedding(image: np.ndarray) -> np.ndarray:
    if not openface_available:
        raise Exception("OpenFace not available")
    try:
        if len(image.shape) == 3:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            image_rgb = image
        bb = align.getLargestFaceBoundingBox(image_rgb)
        if bb is None:
            raise Exception("No face bounding box found")
        aligned_face = align.align(96, image_rgb, bb, 
                                 landmarkIndices=openface.AlignDlib.OUTER_EYES_AND_NOSE)
        if aligned_face is None:
            raise Exception("Face alignment failed")
        rep = net.forward(aligned_face)
        return rep
    except Exception as e:
        logger.warning(f"OpenFace embedding failed: {e}")
        raise

def extract_facial_landmarks(image: np.ndarray) -> np.ndarray:
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    faces = detector(gray)
    if len(faces) == 0:
        raise Exception("No face detected for landmarks")
    landmarks = predictor(gray, faces[0])
    landmarks = face_utils.shape_to_np(landmarks)
    return landmarks

def compute_enhanced_embedding(image: np.ndarray) -> Dict[str, Any]:
    results = {}
    quality_score = assess_face_quality(image)
    results['quality'] = quality_score
    
    # Quick face check using MTCNN first
    face_detected = False
    try:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        face_boxes, _ = mtcnn.detect(image_rgb)
        face_detected = face_boxes is not None and len(face_boxes) > 0
    except:
        pass
    
    # Only try landmark extraction if a face was detected
    if face_detected:
        try:
            landmarks = extract_facial_landmarks(image)
            results['landmarks'] = landmarks
        except Exception as e:
            logger.debug(f"Landmark extraction skipped: {e}")
            results['landmarks'] = None
    else:
        results['landmarks'] = None
        
    try:
        hog_features = extract_hog_features(image)
        results['hog_features'] = hog_features
    except Exception as e:
        logger.warning(f"HOG feature extraction failed: {e}")
        results['hog_features'] = None
    if openface_available:
        try:
            openface_embedding = get_openface_embedding(image)
            results['openface_embedding'] = openface_embedding
        except Exception as e:
            logger.warning(f"OpenFace embedding skipped: {e}")
            results['openface_embedding'] = None
    else:
        results['openface_embedding'] = None
    try:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        face = mtcnn(image_rgb)
        if face is not None:
            face = face.unsqueeze(0).to(device)
            with torch.no_grad():
                facenet_embedding = resnet(face).cpu().numpy().flatten()
            results['facenet_embedding'] = facenet_embedding
        else:
            results['facenet_embedding'] = None
    except Exception as e:
        logger.warning(f"FaceNet embedding failed: {e}")
        results['facenet_embedding'] = None
    return results

# ----------------------------------------------------------------------------------
# ENHANCED KALMAN TRACKER
# ----------------------------------------------------------------------------------
class EnhancedKalmanTracker:
    def __init__(self, initial_bbox: Tuple[float, float, float, float]):
        self.kf = KalmanFilter(dim_x=8, dim_z=4)
        x, y, w, h = initial_bbox
        self.kf.x = np.array([x, y, 0., 0., w, h, 0., 0.])
        self.kf.F = np.array([
            [1, 0, 1, 0, 0, 0, 0, 0],
            [0, 1, 0, 1, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 1, 0],
            [0, 0, 0, 0, 0, 1, 0, 1],
            [0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 1]
        ])
        self.kf.H = np.array([
            [1, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0]
        ])
        self.kf.R *= 10.0
        self.kf.P *= 1000.0
        self.kf.Q *= 0.01
        self.time_since_update = 0
        self.hit_streak = 0
        self.age = 0
        self.history = []
    
    def predict(self):
        self.kf.predict()
        self.age += 1
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1
        x, y, w, h = self.kf.x[0], self.kf.x[1], self.kf.x[4], self.kf.x[5]
        return (x, y, w, h)
    
    def update(self, bbox: Tuple[float, float, float, float]):
        self.time_since_update = 0
        self.history = []
        self.hit_streak += 1
        self.kf.update(np.array([bbox[0], bbox[1], bbox[2], bbox[3]]))
    
    def get_state(self) -> Tuple[float, float, float, float]:
        x, y, w, h = self.kf.x[0], self.kf.x[1], self.kf.x[4], self.kf.x[5]
        return (x, y, w, h)
    
    def get_velocity(self) -> Tuple[float, float]:
        return (self.kf.x[2], self.kf.x[3])

# ----------------------------------------------------------------------------------
# MULTI-PERSON TRACKING SYSTEM
# ----------------------------------------------------------------------------------
class PersonTracker:
    def __init__(self, max_disappeared: int = 30, max_distance: float = 100):
        self.next_id = 0
        self.trackers = OrderedDict()
        self.disappeared = OrderedDict()
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance
        self.trajectories = defaultdict(list)
    
    def register(self, bbox: Tuple[float, float, float, float]) -> int:
        tracker_id = self.next_id
        self.trackers[tracker_id] = EnhancedKalmanTracker(bbox)
        self.disappeared[tracker_id] = 0
        self.next_id += 1
        return tracker_id
    
    def deregister(self, tracker_id: int):
        if tracker_id in self.trackers:
            del self.trackers[tracker_id]
            del self.disappeared[tracker_id]
    
    def update(self, detections: List[Tuple[float, float, float, float]]) -> Dict[int, Dict]:
        if len(detections) == 0:
            for tracker_id in list(self.disappeared.keys()):
                self.disappeared[tracker_id] += 1
                if self.disappeared[tracker_id] > self.max_disappeared:
                    self.deregister(tracker_id)
            return {}
        predictions = {}
        for tracker_id, tracker in self.trackers.items():
            pred_bbox = tracker.predict()
            predictions[tracker_id] = pred_bbox
        if len(self.trackers) == 0:
            for detection in detections:
                self.register(detection)
            return self._get_current_states()
        cost_matrix = self._compute_cost_matrix(predictions, detections)
        if cost_matrix.size > 0:
            row_indices, col_indices = linear_sum_assignment(cost_matrix)
            matched_trackers = set()
            matched_detections = set()
            for row, col in zip(row_indices, col_indices):
                if cost_matrix[row, col] <= self.max_distance:
                    tracker_id = list(self.trackers.keys())[row]
                    self.trackers[tracker_id].update(detections[col])
                    self.disappeared[tracker_id] = 0
                    matched_trackers.add(tracker_id)
                    matched_detections.add(col)
            for tracker_id in self.trackers.keys():
                if tracker_id not in matched_trackers:
                    self.disappeared[tracker_id] += 1
                    if self.disappeared[tracker_id] > self.max_disappeared:
                        self.deregister(tracker_id)
            for i, detection in enumerate(detections):
                if i not in matched_detections:
                    self.register(detection)
        return self._get_current_states()
    
    def _compute_cost_matrix(self, predictions: Dict, detections: List) -> np.ndarray:
        if not predictions or not detections:
            return np.array([])
        tracker_ids = list(predictions.keys())
        cost_matrix = np.zeros((len(tracker_ids), len(detections)))
        for i, tracker_id in enumerate(tracker_ids):
            pred_bbox = predictions[tracker_id]
            pred_center = (pred_bbox[0] + pred_bbox[2]/2, pred_bbox[1] + pred_bbox[3]/2)
            for j, detection in enumerate(detections):
                det_center = (detection[0] + detection[2]/2, detection[1] + detection[3]/2)
                distance = np.sqrt((pred_center[0] - det_center[0])**2 + 
                                   (pred_center[1] - det_center[1])**2)
                cost_matrix[i, j] = distance
        return cost_matrix
    
    def _get_current_states(self) -> Dict[int, Dict]:
        states = {}
        for tracker_id, tracker in self.trackers.items():
            bbox = tracker.get_state()
            velocity = tracker.get_velocity()
            confidence = max(0, 1.0 - (tracker.time_since_update / self.max_disappeared))
            states[tracker_id] = {
                'bbox': bbox,
                'velocity': velocity,
                'confidence': confidence,
                'age': tracker.age,
                'hit_streak': tracker.hit_streak
            }
            center = (bbox[0] + bbox[2]/2, bbox[1] + bbox[3]/2)
            self.trajectories[tracker_id].append({
                'timestamp': time.time(),
                'center': center,
                'bbox': bbox
            })
            if len(self.trajectories[tracker_id]) > 100:
                self.trajectories[tracker_id] = self.trajectories[tracker_id][-100:]
        return states

# ----------------------------------------------------------------------------------
# ENHANCED PERSON DETECTION
# ----------------------------------------------------------------------------------
class EnhancedPersonDetector:
    def __init__(self):
        self.hog = cv2.HOGDescriptor()
        self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    def detect_persons(self, frame: np.ndarray) -> List[Tuple[float, float, float, float]]:
        try:
            (rects, weights) = self.hog.detectMultiScale(
                frame, 
                winStride=(4, 4), 
                padding=(8, 8), 
                scale=1.05,
                finalThreshold=2
            )
            detections = []
            for (x, y, w, h) in rects:
                detections.append((float(x), float(y), float(w), float(h)))
            detections = self._apply_nms(detections, weights)
            return detections
        except Exception as e:
            logger.warning(f"Person detection failed: {e}")
            return []
    
    def detect_faces_in_person(self, frame: np.ndarray, person_bbox: Tuple[float, float, float, float]) -> List[Tuple[float, float, float, float]]:
        try:
            x, y, w, h = person_bbox
            x, y, w, h = int(x), int(y), int(w), int(h)
            person_roi = frame[y:y+h, x:x+w]
            gray_roi = cv2.cvtColor(person_roi, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(
                gray_roi, 
                scaleFactor=1.1, 
                minNeighbors=5, 
                minSize=(30, 30)
            )
            face_detections = []
            for (fx, fy, fw, fh) in faces:
                face_detections.append((float(x + fx), float(y + fy), float(fw), float(fh)))
            return face_detections
        except Exception as e:
            logger.warning(f"Face detection in person failed: {e}")
            return []
    
    def _apply_nms(self, detections: List, weights: np.ndarray, threshold: float = 0.3) -> List:
        if len(detections) == 0:
            return []
        boxes = []
        scores = []
        for i, (x, y, w, h) in enumerate(detections):
            boxes.append([x, y, w, h])
            scores.append(float(weights[i]) if i < len(weights) else 1.0)
        indices = cv2.dnn.NMSBoxes(boxes, scores, 0.5, threshold)
        if len(indices) > 0:
            indices = indices.flatten()
            return [detections[i] for i in indices]
        return []

# ----------------------------------------------------------------------------------
# ENHANCED FACE RECOGNIZER
# ----------------------------------------------------------------------------------
class EnhancedFaceRecognizer:
    def __init__(self, db_session: Session):
        self.db = db_session
        self.tolerance = 0.9
        self.openface_tolerance = 0.4
        self.hog_tolerance = 0.5
        self.load_config()
    
    def load_config(self):
        try:
            config_entry = self.db.query(ConfigEntry).filter(ConfigEntry.key == "TOLERANCE").first()
            if config_entry:
                self.tolerance = float(config_entry.value)
        except Exception as e:
            logger.warning(f"Failed to load config: {e}")
    
    def recognize_face(self, face_image: np.ndarray) -> Tuple[str, float, Dict]:
        try:
            features = compute_enhanced_embedding(face_image)
            if features.get('quality', 0) < 0.3:
                return "unknown", 0.0, {"reason": "poor_quality"}
            registered_faces = self.db.query(RegisteredFace).all()
            if not registered_faces:
                return "unknown", 0.0, {"reason": "no_registered_faces"}
            best_match = None
            best_confidence = 0.0
            match_details = {}
            for registered_face in registered_faces:
                confidence_scores = []
                if features.get('facenet_embedding') is not None and registered_face.embedding:
                    try:
                        stored_embedding = pickle.loads(registered_face.embedding)
                        distance = np.linalg.norm(features['facenet_embedding'] - stored_embedding)
                        facenet_confidence = max(0, 1 - distance / self.tolerance)
                        confidence_scores.append(facenet_confidence * 0.4)
                    except Exception as e:
                        logger.warning(f"FaceNet comparison failed: {e}")
                if openface_available and features.get('openface_embedding') is not None and registered_face.openface_embedding:
                    try:
                        stored_openface = pickle.loads(registered_face.openface_embedding)
                        distance = np.linalg.norm(features['openface_embedding'] - stored_openface)
                        openface_confidence = max(0, 1 - distance / self.openface_tolerance)
                        confidence_scores.append(openface_confidence * 0.4)
                    except Exception as e:
                        logger.warning(f"OpenFace comparison failed: {e}")
                if features.get('hog_features') is not None and registered_face.hog_features:
                    try:
                        stored_hog = pickle.loads(registered_face.hog_features)
                        cos_sim = np.dot(features['hog_features'], stored_hog) / (
                            np.linalg.norm(features['hog_features']) * np.linalg.norm(stored_hog)
                        )
                        hog_confidence = max(0, cos_sim)
                        confidence_scores.append(hog_confidence * 0.2)
                    except Exception as e:
                        logger.warning(f"HOG comparison failed: {e}")
                if confidence_scores:
                    avg_confidence = sum(confidence_scores) / len(confidence_scores)
                    if avg_confidence > best_confidence:
                        best_confidence = avg_confidence
                        best_match = registered_face.person_id
                        match_details = {
                            'method': 'multi_modal',
                            'scores': confidence_scores,
                            'quality': features.get('quality', 0),
                            'face_id': registered_face.id
                        }
            if best_confidence < 0.5:
                return "unknown", best_confidence, match_details
            return best_match, best_confidence, match_details
        except Exception as e:
            logger.error(f"Face recognition failed: {e}")
            return "unknown", 0.0, {"error": str(e)}
    
    def register_face(self, face_image: np.ndarray, person_id: str) -> bool:
        try:
            features = compute_enhanced_embedding(face_image)
            if features.get('quality', 0) < 0.4:
                logger.warning(f"Face quality too low for registration: {features.get('quality')}")
                return False
            existing_face = self.db.query(RegisteredFace).filter(
                RegisteredFace.person_id == person_id
            ).first()
            if existing_face:
                face_record = existing_face
            else:
                face_record = RegisteredFace(person_id=person_id)
            if features.get('facenet_embedding') is not None:
                face_record.embedding = pickle.dumps(features['facenet_embedding'])
            if features.get('openface_embedding') is not None:
                face_record.openface_embedding = pickle.dumps(features['openface_embedding'])
            if features.get('hog_features') is not None:
                face_record.hog_features = pickle.dumps(features['hog_features'])
            if features.get('landmarks') is not None:
                face_record.landmarks = pickle.dumps(features['landmarks'])
            face_record.face_quality = features.get('quality', 0)
            face_record.confidence = 1.0
            face_record.updated_at = datetime.utcnow()
            if not existing_face:
                self.db.add(face_record)
            self.db.commit()
            logger.info(f"Face registered successfully for person: {person_id}")
            return True
        except Exception as e:
            logger.error(f"Face registration failed: {e}")
            self.db.rollback()
            return False

# ----------------------------------------------------------------------------------
# RTSP STREAM MANAGER
# ----------------------------------------------------------------------------------
class RTSPStreamManager:
    def __init__(self, db_session: Session):
        self.db = db_session
        self.active_streams = {}
        self.stream_threads = {}
        self.person_detector = EnhancedPersonDetector()
        self.face_recognizer = EnhancedFaceRecognizer(db_session)
        self.person_tracker = PersonTracker()
        self.frame_skip = 3
        self.frame_count = 0
    
    def start_stream(self, stream_id: int, rtsp_url: str, stream_name: str) -> bool:
        if stream_id in self.active_streams:
            logger.warning(f"Stream {stream_id} already active")
            return False
        stream_record = RTSPStream(
            id=stream_id,
            name=stream_name,
            rtsp_url=rtsp_url,
            is_active=True
        )
        existing_stream = self.db.query(RTSPStream).filter(RTSPStream.id == stream_id).first()
        if existing_stream:
            existing_stream.is_active = True
            existing_stream.rtsp_url = rtsp_url
            existing_stream.name = stream_name
        else:
            self.db.add(stream_record)
        self.db.commit()
        self.active_streams[stream_id] = {
            'url': rtsp_url,
            'name': stream_name,
            'active': True,
            'cap': None,
            'stats': {'frames_processed': 0, 'detections': 0, 'recognitions': 0}
        }
        thread = threading.Thread(
            target=self._process_stream,
            args=(stream_id, rtsp_url),
            daemon=True
        )
        thread.start()
        self.stream_threads[stream_id] = thread
        logger.info(f"Started stream {stream_id}: {stream_name}")
        return True
    
    def stop_stream(self, stream_id: int) -> bool:
        if stream_id not in self.active_streams:
            logger.warning(f"Stream {stream_id} not active")
            return False
        self.active_streams[stream_id]['active'] = False
        stream_record = self.db.query(RTSPStream).filter(RTSPStream.id == stream_id).first()
        if stream_record:
            stream_record.is_active = False
            self.db.commit()
        if stream_id in self.stream_threads:
            self.stream_threads[stream_id].join(timeout=5.0)
            del self.stream_threads[stream_id]
        if self.active_streams[stream_id]['cap']:
            self.active_streams[stream_id]['cap'].release()
        del self.active_streams[stream_id]
        logger.info(f"Stopped stream {stream_id}")
        return True
    
    def _process_stream(self, stream_id: int, rtsp_url: str):
        cap = cv2.VideoCapture(rtsp_url)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        cap.set(cv2.CAP_PROP_FPS, 15)
        if not cap.isOpened():
            logger.error(f"Failed to open stream {stream_id}: {rtsp_url}")
            return
        self.active_streams[stream_id]['cap'] = cap
        frame_count = 0
        while self.active_streams[stream_id]['active']:
            try:
                ret, frame = cap.read()
                if not ret:
                    logger.warning(f"Failed to read frame from stream {stream_id}")
                    time.sleep(0.1)
                    continue
                frame_count += 1
                if frame_count % self.frame_skip != 0:
                    continue
                start_time = time.time()
                results = self._process_frame(frame, stream_id)
                processing_time = time.time() - start_time
                self.active_streams[stream_id]['stats']['frames_processed'] += 1
                stream_record = self.db.query(RTSPStream).filter(RTSPStream.id == stream_id).first()
                if stream_record:
                    stream_record.last_detection = datetime.utcnow()
                    stream_record.detection_count += len(results.get('detections', []))
                    stream_record.avg_processing_time = processing_time
                    self.db.commit()
                for recognition in results.get('recognitions', []):
                    # Run callback in event loop
                    asyncio.create_task(send_callback(recognition['person_id']))
                time.sleep(0.01)
            except Exception as e:
                logger.error(f"Error processing stream {stream_id}: {e}")
                time.sleep(1)
        cap.release()
        logger.info(f"Stream {stream_id} processing ended")
    
    def _process_frame(self, frame: np.ndarray, stream_id: int) -> Dict[str, Any]:
        results = {
            'detections': [],
            'recognitions': [],
            'tracking': {},
            'timestamp': datetime.utcnow()
        }
        try:
            person_detections = self.person_detector.detect_persons(frame)
            tracking_results = self.person_tracker.update(person_detections)
            results['tracking'] = tracking_results
            for track_id, track_info in tracking_results.items():
                bbox = track_info['bbox']
                confidence = track_info['confidence']
                if confidence < 0.5:
                    continue
                face_detections = self.person_detector.detect_faces_in_person(frame, bbox)
                for face_bbox in face_detections:
                    fx, fy, fw, fh = [int(x) for x in face_bbox]
                    face_roi = frame[fy:fy+fh, fx:fx+fw]
                    if face_roi.size == 0:
                        continue
                    person_id, face_confidence, match_details = self.face_recognizer.recognize_face(face_roi)
                    detection_event = DetectionEvent(
                        person_id=person_id,
                        stream_id=stream_id,
                        confidence=face_confidence,
                        bbox_x=fx,
                        bbox_y=fy,
                        bbox_w=fw,
                        bbox_h=fh,
                        face_quality=match_details.get('quality', 0)
                    )
                    self.db.add(detection_event)
                    tracking_history = TrackingHistory(
                        person_id=person_id,
                        track_id=track_id,
                        position_x=bbox[0] + bbox[2]/2,
                        position_y=bbox[1] + bbox[3]/2,
                        velocity_x=track_info['velocity'][0],
                        velocity_y=track_info['velocity'][1],
                        confidence=confidence,
                        bbox_x=bbox[0],
                        bbox_y=bbox[1],
                        bbox_w=bbox[2],
                        bbox_h=bbox[3]
                    )
                    self.db.add(tracking_history)
                    results['recognitions'].append({
                        'person_id': person_id,
                        'confidence': face_confidence,
                        'track_id': track_id,
                        'bbox': face_bbox,
                        'details': match_details
                    })
                results['detections'].append({
                    'track_id': track_id,
                    'bbox': bbox,
                    'confidence': confidence,
                    'velocity': track_info['velocity']
                })
            self.db.commit()
        except Exception as e:
            logger.error(f"Frame processing error: {e}")
            self.db.rollback()
        return results

# ----------------------------------------------------------------------------------
# WEBSOCKET MANAGER
# ----------------------------------------------------------------------------------
class WebSocketManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []
    
    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info(f"WebSocket connected. Total connections: {len(self.active_connections)}")
    
    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        logger.info(f"WebSocket disconnected. Total connections: {len(self.active_connections)}")
    
    async def send_personal_message(self, message: str, websocket: WebSocket):
        try:
            await websocket.send_text(message)
        except Exception as e:
            logger.error(f"Error sending personal message: {e}")
    
    async def broadcast(self, message: str):
        if self.active_connections:
            disconnected = []
            for connection in self.active_connections:
                try:
                    await connection.send_text(message)
                except Exception:
                    disconnected.append(connection)
            for connection in disconnected:
                self.disconnect(connection)

# ----------------------------------------------------------------------------------
# FASTAPI APPLICATION SETUP
# ----------------------------------------------------------------------------------
app = FastAPI(title="Enhanced Face Recognition API", version="2.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
stream_manager = None
websocket_manager = WebSocketManager()
executor = ThreadPoolExecutor(max_workers=10)

# ----------------------------------------------------------------------------------
# PYDANTIC MODELS
# ----------------------------------------------------------------------------------
class Token(BaseModel):
    access_token: str
    token_type: str

class ConfigUpdate(BaseModel):
    tolerance: Optional[float] = None
    detection_threshold: Optional[float] = None
    callback_url: Optional[str] = None
    callback_token: Optional[str] = None

class UserCreate(BaseModel):
    username: str
    password: str

class PersonInfo(BaseModel):
    person_id: str
    updated_at: datetime

class RegisterFaceRequest(BaseModel):
    person_id: str
    image_data: str

class StreamRequest(BaseModel):
    name: str
    rtsp_url: str

class PersonSearchRequest(BaseModel):
    person_id: Optional[str] = None
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    stream_id: Optional[int] = None

# ----------------------------------------------------------------------------------
# STARTUP AND SHUTDOWN EVENTS
# ----------------------------------------------------------------------------------
@app.on_event("startup")
async def startup_event():
    global stream_manager
    db = SessionLocal()
    try:
        # First create all tables
        Base.metadata.create_all(bind=engine)
        # Then apply migrations
        migrate_database()
        # Initialize default data
        init_config(db)
        init_admin(db)
        stream_manager = RTSPStreamManager(db)
        logger.info("Application started successfully")
    finally:
        db.close()

@app.on_event("shutdown")
async def shutdown_event():
    global stream_manager
    if stream_manager:
        for stream_id in list(stream_manager.active_streams.keys()):
            stream_manager.stop_stream(stream_id)
    executor.shutdown(wait=True)
    logger.info("Application shutdown complete")

# ----------------------------------------------------------------------------------
# HTML INTERFACE WITH AUTHENTICATION
# ----------------------------------------------------------------------------------
@app.get("/login", response_class=HTMLResponse)
async def login_page(request: Request):
    return templates.TemplateResponse("login.html", {"request": request})

@app.get("/register.html", response_class=HTMLResponse, summary="Register Face Page")
async def register_face_page(request: Request):
    token = request.cookies.get("token")
    if not token:
        return RedirectResponse(url="/login", status_code=status.HTTP_302_FOUND)
    try:
        decode_token(token)
    except HTTPException:
        return RedirectResponse(url="/login", status_code=status.HTTP_302_FOUND)
    return templates.TemplateResponse("register.html", {"request": request})

@app.get("/register.html", response_class=HTMLResponse, summary="Register Face Page")
async def register_face_page(request: Request):
    token = request.cookies.get("token")
    if not token:
        return RedirectResponse(url="/login", status_code=status.HTTP_302_FOUND)
    try:
        decode_token(token)
    except HTTPException:
        return RedirectResponse(url="/login", status_code=status.HTTP_302_FOUND)
    return templates.TemplateResponse("register.html", {"request": request})

@app.get("/logout", response_class=RedirectResponse)
async def logout(response: Response):
    response.delete_cookie("token")
    return RedirectResponse(url="/login")

@app.get("/", response_class=HTMLResponse)
async def config_page(request: Request):
    token = request.cookies.get("token")
    if not token:
        return RedirectResponse(url="/login", status_code=status.HTTP_302_FOUND)
    try:
        decode_token(token)
    except HTTPException:
        return RedirectResponse(url="/login", status_code=status.HTTP_302_FOUND)
    return templates.TemplateResponse("config.html", {"request": request})

@app.get("/stream.html", response_class=HTMLResponse, summary="Stream Player Page")
async def stream_player_page(request: Request):
    token = request.cookies.get("token")
    if not token:
        return RedirectResponse(url="/login", status_code=status.HTTP_302_FOUND)
    try:
        decode_token(token)
    except HTTPException:
        return RedirectResponse(url="/login", status_code=status.HTTP_302_FOUND)
    return templates.TemplateResponse("stream.html", {"request": request})

@app.get("/debug_auth", response_class=HTMLResponse)
async def debug_auth_page(request: Request):
    return templates.TemplateResponse("debug_auth.html", {"request": request})

# ----------------------------------------------------------------------------------
# AUTHENTICATION ENDPOINTS
# ----------------------------------------------------------------------------------
@app.post("/token", response_model=Token)
async def login_for_access_token(response: Response, form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    user = get_user(db, form_data.username)
    if not user or not verify_password(form_data.password, user.hashed_password):
        raise HTTPException(status_code=401, detail="Incorrect username or password")
    access_token = create_access_token(data={"sub": user.username})
    response.set_cookie(
        key="token", 
        value=access_token,
        httponly=False,  # Allow JavaScript access for debugging
        max_age=ACCESS_TOKEN_EXPIRE_MINUTES * 60,
        samesite="lax",
        path="/",
        secure=False  # Set to True if using HTTPS
    )
    return {"access_token": access_token, "token_type": "bearer"}

@app.post("/users", response_model=dict)
async def create_user(user: UserCreate, current_admin: User = Depends(get_current_admin), db: Session = Depends(get_db)):
    if get_user(db, user.username):
        raise HTTPException(status_code=400, detail="Username already registered")
    new_user = User(username=user.username, hashed_password=get_password_hash(user.password))
    db.add(new_user)
    db.commit()
    return {"message": f"User '{user.username}' created successfully."}

# ----------------------------------------------------------------------------------
# FACE RECOGNITION ENDPOINTS
# ----------------------------------------------------------------------------------
@app.post("/compare")
async def compare_images(reference: UploadFile = File(...), check: UploadFile = File(...), current_user: User = Depends(get_current_user)):
    ref_image = await read_imagefile(reference)
    check_image = await read_imagefile(check)
    face_recognizer = EnhancedFaceRecognizer(SessionLocal())
    ref_features = compute_enhanced_embedding(ref_image)
    check_features = compute_enhanced_embedding(check_image)
    if ref_features.get('quality', 0) < 0.3 or check_features.get('quality', 0) < 0.3:
        raise HTTPException(status_code=400, detail="Poor face quality in one or both images")
    confidence_scores = []
    if ref_features.get('facenet_embedding') is not None and check_features.get('facenet_embedding') is not None:
        distance = np.linalg.norm(ref_features['facenet_embedding'] - check_features['facenet_embedding'])
        facenet_confidence = max(0, 1 - distance / face_recognizer.tolerance)
        confidence_scores.append(facenet_confidence * 0.4)
    if openface_available and ref_features.get('openface_embedding') is not None and check_features.get('openface_embedding') is not None:
        distance = np.linalg.norm(ref_features['openface_embedding'] - check_features['openface_embedding'])
        openface_confidence = max(0, 1 - distance / face_recognizer.openface_tolerance)
        confidence_scores.append(openface_confidence * 0.4)
    if ref_features.get('hog_features') is not None and check_features.get('hog_features') is not None:
        cos_sim = np.dot(ref_features['hog_features'], check_features['hog_features']) / (
            np.linalg.norm(ref_features['hog_features']) * np.linalg.norm(check_features['hog_features'])
        )
        hog_confidence = max(0, cos_sim)
        confidence_scores.append(hog_confidence * 0.2)
    if not confidence_scores:
        raise HTTPException(status_code=400, detail="No valid features extracted")
    avg_confidence = sum(confidence_scores) / len(confidence_scores)
    match = avg_confidence >= 0.5
    if match:
        # Create a task to run the callback asynchronously
        loop = asyncio.get_event_loop()
        loop.create_task(send_callback("matched"))
    return {"match": match, "confidence": avg_confidence}

@app.post("/authenticate")
async def authenticate_face(file: UploadFile = File(...), current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    image = await read_imagefile(file)
    face_recognizer = EnhancedFaceRecognizer(db)
    person_id, confidence, details = face_recognizer.recognize_face(image)
    # Create a task to run the callback asynchronously
    loop = asyncio.get_event_loop()
    loop.create_task(send_callback(person_id))
    return {"person_id": person_id, "confidence": confidence, "details": details}

@app.post("/register")
async def register_person(person_id: str = Form(...), images: List[UploadFile] = File(...), current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    face_recognizer = EnhancedFaceRecognizer(db)
    valid_features = []
    for file in images:
        image = await read_imagefile(file)
        try:
            features = compute_enhanced_embedding(image)
            if features.get('quality', 0) >= 0.4:
                valid_features.append(features)
        except HTTPException as e:
            logger.warning(f"Skipping an image: {e.detail}")
    if not valid_features:
        raise HTTPException(status_code=400, detail="No valid face detected in any of the images")
    avg_features = {
        'facenet_embedding': None,
        'openface_embedding': None,
        'hog_features': None,
        'landmarks': None,
        'quality': 0.0,
        'confidence': 1.0
    }
    facenet_embeddings = [f['facenet_embedding'] for f in valid_features if f.get('facenet_embedding') is not None]
    if facenet_embeddings:
        avg_features['facenet_embedding'] = np.mean(facenet_embeddings, axis=0)
    openface_embeddings = [f['openface_embedding'] for f in valid_features if f.get('openface_embedding') is not None]
    if openface_embeddings:
        avg_features['openface_embedding'] = np.mean(openface_embeddings, axis=0)
    hog_features_list = [f['hog_features'] for f in valid_features if f.get('hog_features') is not None]
    if hog_features_list:
        avg_features['hog_features'] = np.mean(hog_features_list, axis=0)
    landmarks_list = [f['landmarks'] for f in valid_features if f.get('landmarks') is not None]
    if landmarks_list:
        avg_features['landmarks'] = np.mean(landmarks_list, axis=0)
    avg_features['quality'] = np.mean([f['quality'] for f in valid_features])
    existing_face = db.query(RegisteredFace).filter(RegisteredFace.person_id == person_id).first()
    if existing_face:
        face_record = existing_face
    else:
        face_record = RegisteredFace(person_id=person_id)
    if avg_features['facenet_embedding'] is not None:
        face_record.embedding = pickle.dumps(avg_features['facenet_embedding'])
    if avg_features['openface_embedding'] is not None:
        face_record.openface_embedding = pickle.dumps(avg_features['openface_embedding'])
    if avg_features['hog_features'] is not None:
        face_record.hog_features = pickle.dumps(avg_features['hog_features'])
    if avg_features['landmarks'] is not None:
        face_record.landmarks = pickle.dumps(avg_features['landmarks'])
    face_record.face_quality = avg_features['quality']
    face_record.confidence = avg_features['confidence']
    face_record.updated_at = datetime.utcnow()
    if not existing_face:
        db.add(face_record)
    db.commit()
    return {"message": f"Person '{person_id}' registered/updated successfully."}d = existing_face
    else:
        face_record = RegisteredFace(person_id=person_id)
    if avg_features['facenet_embedding'] is not None:
        face_record.embedding = pickle.dumps(avg_features['facenet_embedding'])
    if avg_features['openface_embedding'] is not None:
        face_record.openface_embedding = pickle.dumps(avg_features['openface_embedding'])
    if avg_features['hog_features'] is not None:
        face_record.hog_features = pickle.dumps(avg_features['hog_features'])
    if avg_features['landmarks'] is not None:
        face_record.landmarks = pickle.dumps(avg_features['landmarks'])
    face_record.face_quality = avg_features['quality']
    face_record.confidence = avg_features['confidence']
    face_record.updated_at = datetime.utcnow()
    if not existing_face:
        db.add(face_record)
    db.commit()
    return {"message": f"Person '{person_id}' registered/updated successfully."}

# ----------------------------------------------------------------------------------
# CONFIGURATION ENDPOINTS
# ----------------------------------------------------------------------------------
@app.get("/config")
async def get_config(current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    entries = db.query(ConfigEntry).all()
    return {entry.key: entry.value for entry in entries}

@app.put("/config")
async def update_config(update: ConfigUpdate, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    if update.tolerance is not None:
        entry = db.query(ConfigEntry).filter(ConfigEntry.key == "TOLERANCE").first()
        if entry:
            entry.value = str(update.tolerance)
        else:
            db.add(ConfigEntry(key="TOLERANCE", value=str(update.tolerance)))
    if update.detection_threshold is not None:
        entry = db.query(ConfigEntry).filter(ConfigEntry.key == "DETECTION_THRESHOLD").first()
        if entry:
            entry.value = str(update.detection_threshold)
        else:
            db.add(ConfigEntry(key="DETECTION_THRESHOLD", value=str(update.detection_threshold)))
    if update.callback_url is not None:
        entry = db.query(ConfigEntry).filter(ConfigEntry.key == "CALLBACK_URL").first()
        if entry:
            entry.value = update.callback_url
        else:
            db.add(ConfigEntry(key="CALLBACK_URL", value=update.callback_url))
    if update.callback_token is not None:
        entry = db.query(ConfigEntry).filter(ConfigEntry.key == "CALLBACK_TOKEN").first()
        if entry:
            entry.value = update.callback_token
        else:
            db.add(ConfigEntry(key="CALLBACK_TOKEN", value=update.callback_token))
    db.commit()
    return {"message": "Configuration updated successfully."}

@app.get("/callback-config")
async def get_callback_config(current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    url_entry = db.query(ConfigEntry).filter(ConfigEntry.key == "CALLBACK_URL").first()
    token_entry = db.query(ConfigEntry).filter(ConfigEntry.key == "CALLBACK_TOKEN").first()
    return {
        "callback_url": url_entry.value if url_entry else "",
        "callback_token": token_entry.value if token_entry else ""
    }

@app.put("/callback-config")
async def update_callback_config(config: ConfigUpdate, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    if config.callback_url is not None:
        entry = db.query(ConfigEntry).filter(ConfigEntry.key == "CALLBACK_URL").first()
        if entry:
            entry.value = config.callback_url
        else:
            db.add(ConfigEntry(key="CALLBACK_URL", value=config.callback_url))
    if config.callback_token is not None:
        entry = db.query(ConfigEntry).filter(ConfigEntry.key == "CALLBACK_TOKEN").first()
        if entry:
            entry.value = config.callback_token
        else:
            db.add(ConfigEntry(key="CALLBACK_TOKEN", value=config.callback_token))
    db.commit()
    return {"message": "Callback configuration updated."}

# ----------------------------------------------------------------------------------
# PERSON LISTING
# ----------------------------------------------------------------------------------
@app.get("/persons", response_model=List[PersonInfo])
async def list_persons(current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    persons = db.query(RegisteredFace).all()
    return [{"person_id": person.person_id, "updated_at": person.updated_at} for person in persons]

# ----------------------------------------------------------------------------------
# WEBSOCKET ENDPOINTS
# ----------------------------------------------------------------------------------
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket_manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            await websocket_manager.send_personal_message(f"Echo: {data}", websocket)
    except WebSocketDisconnect:
        websocket_manager.disconnect(websocket)

@app.websocket("/ws/detection")
async def websocket_detection(websocket: WebSocket, token: str = Query(...)):
    try:
        payload = decode_token(token)
        username: str = payload.get("sub")
        if username is None:
            await websocket.close(code=1008)
            return
    except Exception:
        await websocket.close(code=1008)
        return
    
    await websocket.accept()
    db = SessionLocal()
    
    try:
        while True:
            try:
                data = await websocket.receive_text()
                img_data = base64.b64decode(data)
                np_arr = np.frombuffer(img_data, np.uint8)
                frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
                
                if frame is None:
                    await websocket.send_text("Invalid image frame")
                    continue
                
                face_recognizer = EnhancedFaceRecognizer(db)
                person_id, confidence, details = face_recognizer.recognize_face(frame)
                await send_callback(person_id)
                await websocket.send_text(person_id)
                
            except WebSocketDisconnect:
                logger.info("WebSocket disconnected")
                break
            except Exception as e:
                logger.error(f"Error processing frame: {str(e)}")
                try:
                    await websocket.send_text(f"Error processing frame: {str(e)}")
                except:
                    # Connection might be closed, just break
                    break
    finally:
        db.close()

@app.get("/ws/test", response_class=HTMLResponse)
async def websocket_test_page(request: Request):
    return templates.TemplateResponse("ws_test.html", {"request": request})

@app.get("/ws/live", response_class=HTMLResponse)
async def websocket_live_page(request: Request):
    return templates.TemplateResponse("ws_live.html", {"request": request})

# ----------------------------------------------------------------------------------
# STREAM MANAGEMENT ENDPOINTS
# ----------------------------------------------------------------------------------
@app.post("/streams")
async def create_stream(request: StreamRequest, current_user: User = Depends(get_current_admin), db: Session = Depends(get_db)):
    stream_id = (db.query(func.max(RTSPStream.id)).scalar() or 0) + 1
    stream_manager.start_stream(stream_id, request.rtsp_url, request.name)
    return {"message": f"Stream {request.name} started", "stream_id": stream_id}

@app.get("/streams")
async def get_streams(current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    streams = db.query(RTSPStream).all()
    return [
        {
            "id": stream.id,
            "name": stream.name,
            "rtsp_url": stream.rtsp_url,
            "is_active": stream.is_active,
            "created_at": stream.created_at,
            "last_detection": stream.last_detection,
            "detection_count": stream.detection_count,
            "avg_processing_time": stream.avg_processing_time
        }
        for stream in streams
    ]

@app.post("/rtsp/streams/{stream_id}/start")
async def start_rtsp_monitoring(stream_id: int, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    stream = db.query(RTSPStream).filter(RTSPStream.id == stream_id).first()
    if not stream:
        raise HTTPException(status_code=404, detail="RTSP stream not found")
    if stream_manager.start_stream(stream_id, stream.rtsp_url, stream.name):
        stream.is_active = True
        db.commit()
        return {"message": f"Started monitoring RTSP stream: {stream.name}"}
    raise HTTPException(status_code=400, detail="Stream is already running")

@app.post("/rtsp/streams/{stream_id}/stop")
async def stop_rtsp_monitoring(stream_id: int, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    stream = db.query(RTSPStream).filter(RTSPStream.id == stream_id).first()
    if not stream:
        raise HTTPException(status_code=404, detail="RTSP stream not found")
    if stream_manager.stop_stream(stream_id):
        stream.is_active = False
        db.commit()
        return {"message": f"Stopped monitoring RTSP stream: {stream.name}"}
    return {"message": "Stream was not running"}

@app.delete("/streams/{stream_id}")
async def delete_stream(stream_id: int, current_user: User = Depends(get_current_admin), db: Session = Depends(get_db)):
    stream_manager.stop_stream(stream_id)
    stream = db.query(RTSPStream).filter(RTSPStream.id == stream_id).first()
    if stream:
        db.delete(stream)
        db.commit()
    return {"message": f"Stream {stream_id} deleted"}

# ----------------------------------------------------------------------------------
# SEARCH AND ANALYTICS ENDPOINTS
# ----------------------------------------------------------------------------------
@app.post("/search-person")
async def search_person(request: PersonSearchRequest, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    query = db.query(DetectionEvent)
    if request.person_id:
        query = query.filter(DetectionEvent.person_id == request.person_id)
    if request.start_date:
        try:
            start_date = datetime.fromisoformat(request.start_date.replace("Z", "+00:00"))
            query = query.filter(DetectionEvent.timestamp >= start_date)
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid start_date format")
    if request.end_date:
        try:
            end_date = datetime.fromisoformat(request.end_date.replace("Z", "+00:00"))
            query = query.filter(DetectionEvent.timestamp <= end_date)
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid end_date format")
    if request.stream_id:
        query = query.filter(DetectionEvent.stream_id == request.stream_id)
    events = query.order_by(DetectionEvent.timestamp.desc()).limit(100).all()
    return [
        {
            "id": event.id,
            "person_id": event.person_id,
            "stream_id": event.stream_id,
            "timestamp": event.timestamp,
            "confidence": event.confidence,
            "bbox": [event.bbox_x, event.bbox_y, event.bbox_w, event.bbox_h],
            "face_quality": event.face_quality
        }
        for event in events
    ]

@app.get("/analytics/person-count")
async def get_person_count(current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    one_hour_ago = datetime.utcnow() - timedelta(hours=1)
    unique_persons = db.query(DetectionEvent.person_id).filter(
        DetectionEvent.timestamp >= one_hour_ago,
        DetectionEvent.person_id != "unknown"
    ).distinct().count()
    return {"unique_persons_last_hour": unique_persons}

@app.get("/test-auth")
async def test_auth(current_user: User = Depends(get_current_user)):
    return {"message": f"Authenticated as {current_user.username}", "user_id": current_user.id}

@app.get("/analytics/detection-stats")
async def get_detection_stats(current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    total_detections = db.query(DetectionEvent).count()
    yesterday = datetime.utcnow() - timedelta(days=1)
    recent_detections = db.query(DetectionEvent).filter(
        DetectionEvent.timestamp >= yesterday
    ).count()
    most_detected = db.query(
        DetectionEvent.person_id,
        func.count(DetectionEvent.id).label('count')
    ).filter(
        DetectionEvent.person_id != "unknown"
    ).group_by(DetectionEvent.person_id).order_by(
        func.count(DetectionEvent.id).desc()
    ).first()
    return {
        "total_detections": total_detections,
        "recent_detections": recent_detections,
        "most_detected_person": most_detected.person_id if most_detected else None,
        "most_detected_count": most_detected.count if most_detected else 0
    }

# ----------------------------------------------------------------------------------
# MAIN ENTRY POINT
# ----------------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)d = existing_face
    else:
        face_record = RegisteredFace(person_id=person_id)
    if avg_features['facenet_embedding'] is not None:
        face_record.embedding = pickle.dumps(avg_features['facenet_embedding'])
    if avg_features['openface_embedding'] is not None:
        face_record.openface_embedding = pickle.dumps(avg_features['openface_embedding'])
    if avg_features['hog_features'] is not None:
        face_record.hog_features = pickle.dumps(avg_features['hog_features'])
    if avg_features['landmarks'] is not None:
        face_record.landmarks = pickle.dumps(avg_features['landmarks'])
    face_record.face_quality = avg_features['quality']
    face_record.confidence = avg_features['confidence']
    face_record.updated_at = datetime.utcnow()
    if not existing_face:
        db.add(face_record)
    db.commit()
    return {"message": f"Person '{person_id}' registered/updated successfully."}

# ----------------------------------------------------------------------------------
# CONFIGURATION ENDPOINTS
# ----------------------------------------------------------------------------------
@app.get("/config")
async def get_config(current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    entries = db.query(ConfigEntry).all()
    return {entry.key: entry.value for entry in entries}

@app.put("/config")
async def update_config(update: ConfigUpdate, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    if update.tolerance is not None:
        entry = db.query(ConfigEntry).filter(ConfigEntry.key == "TOLERANCE").first()
        if entry:
            entry.value = str(update.tolerance)
        else:
            db.add(ConfigEntry(key="TOLERANCE", value=str(update.tolerance)))
    if update.detection_threshold is not None:
        entry = db.query(ConfigEntry).filter(ConfigEntry.key == "DETECTION_THRESHOLD").first()
        if entry:
            entry.value = str(update.detection_threshold)
        else:
            db.add(ConfigEntry(key="DETECTION_THRESHOLD", value=str(update.detection_threshold)))
    if update.callback_url is not None:
        entry = db.query(ConfigEntry).filter(ConfigEntry.key == "CALLBACK_URL").first()
        if entry:
            entry.value = update.callback_url
        else:
            db.add(ConfigEntry(key="CALLBACK_URL", value=update.callback_url))
    if update.callback_token is not None:
        entry = db.query(ConfigEntry).filter(ConfigEntry.key == "CALLBACK_TOKEN").first()
        if entry:
            entry.value = update.callback_token
        else:
            db.add(ConfigEntry(key="CALLBACK_TOKEN", value=update.callback_token))
    db.commit()
    return {"message": "Configuration updated successfully."}

@app.get("/callback-config")
async def get_callback_config(current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    url_entry = db.query(ConfigEntry).filter(ConfigEntry.key == "CALLBACK_URL").first()
    token_entry = db.query(ConfigEntry).filter(ConfigEntry.key == "CALLBACK_TOKEN").first()
    return {
        "callback_url": url_entry.value if url_entry else "",
        "callback_token": token_entry.value if token_entry else ""
    }

@app.put("/callback-config")
async def update_callback_config(config: ConfigUpdate, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    if config.callback_url is not None:
        entry = db.query(ConfigEntry).filter(ConfigEntry.key == "CALLBACK_URL").first()
        if entry:
            entry.value = config.callback_url
        else:
            db.add(ConfigEntry(key="CALLBACK_URL", value=config.callback_url))
    if config.callback_token is not None:
        entry = db.query(ConfigEntry).filter(ConfigEntry.key == "CALLBACK_TOKEN").first()
        if entry:
            entry.value = config.callback_token
        else:
            db.add(ConfigEntry(key="CALLBACK_TOKEN", value=config.callback_token))
    db.commit()
    return {"message": "Callback configuration updated."}

# ----------------------------------------------------------------------------------
# PERSON LISTING
# ----------------------------------------------------------------------------------
@app.get("/persons", response_model=List[PersonInfo])
async def list_persons(current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    persons = db.query(RegisteredFace).all()
    return [{"person_id": person.person_id, "updated_at": person.updated_at} for person in persons]

# ----------------------------------------------------------------------------------
# WEBSOCKET ENDPOINTS
# ----------------------------------------------------------------------------------
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket_manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            await websocket_manager.send_personal_message(f"Echo: {data}", websocket)
    except WebSocketDisconnect:
        websocket_manager.disconnect(websocket)

@app.websocket("/ws/detection")
async def websocket_detection(websocket: WebSocket, token: str = Query(...)):
    try:
        payload = decode_token(token)
        username: str = payload.get("sub")
        if username is None:
            await websocket.close(code=1008)
            return
    except Exception:
        await websocket.close(code=1008)
        return
    
    await websocket.accept()
    db = SessionLocal()
    
    try:
        while True:
            try:
                data = await websocket.receive_text()
                img_data = base64.b64decode(data)
                np_arr = np.frombuffer(img_data, np.uint8)
                frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
                
                if frame is None:
                    await websocket.send_text("Invalid image frame")
                    continue
                
                face_recognizer = EnhancedFaceRecognizer(db)
                person_id, confidence, details = face_recognizer.recognize_face(frame)
                await send_callback(person_id)
                await websocket.send_text(person_id)
                
            except WebSocketDisconnect:
                logger.info("WebSocket disconnected")
                break
            except Exception as e:
                logger.error(f"Error processing frame: {str(e)}")
                try:
                    await websocket.send_text(f"Error processing frame: {str(e)}")
                except:
                    # Connection might be closed, just break
                    break
    finally:
        db.close()

@app.get("/ws/test", response_class=HTMLResponse)
async def websocket_test_page(request: Request):
    return templates.TemplateResponse("ws_test.html", {"request": request})

@app.get("/ws/live", response_class=HTMLResponse)
async def websocket_live_page(request: Request):
    return templates.TemplateResponse("ws_live.html", {"request": request})

# ----------------------------------------------------------------------------------
# STREAM MANAGEMENT ENDPOINTS
# ----------------------------------------------------------------------------------
@app.post("/streams")
async def create_stream(request: StreamRequest, current_user: User = Depends(get_current_admin), db: Session = Depends(get_db)):
    stream_id = (db.query(func.max(RTSPStream.id)).scalar() or 0) + 1
    stream_manager.start_stream(stream_id, request.rtsp_url, request.name)
    return {"message": f"Stream {request.name} started", "stream_id": stream_id}

@app.get("/streams")
async def get_streams(current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    streams = db.query(RTSPStream).all()
    return [
        {
            "id": stream.id,
            "name": stream.name,
            "rtsp_url": stream.rtsp_url,
            "is_active": stream.is_active,
            "created_at": stream.created_at,
            "last_detection": stream.last_detection,
            "detection_count": stream.detection_count,
            "avg_processing_time": stream.avg_processing_time
        }
        for stream in streams
    ]

@app.post("/rtsp/streams/{stream_id}/start")
async def start_rtsp_monitoring(stream_id: int, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    stream = db.query(RTSPStream).filter(RTSPStream.id == stream_id).first()
    if not stream:
        raise HTTPException(status_code=404, detail="RTSP stream not found")
    if stream_manager.start_stream(stream_id, stream.rtsp_url, stream.name):
        stream.is_active = True
        db.commit()
        return {"message": f"Started monitoring RTSP stream: {stream.name}"}
    raise HTTPException(status_code=400, detail="Stream is already running")

@app.post("/rtsp/streams/{stream_id}/stop")
async def stop_rtsp_monitoring(stream_id: int, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    stream = db.query(RTSPStream).filter(RTSPStream.id == stream_id).first()
    if not stream:
        raise HTTPException(status_code=404, detail="RTSP stream not found")
    if stream_manager.stop_stream(stream_id):
        stream.is_active = False
        db.commit()
        return {"message": f"Stopped monitoring RTSP stream: {stream.name}"}
    return {"message": "Stream was not running"}

@app.delete("/streams/{stream_id}")
async def delete_stream(stream_id: int, current_user: User = Depends(get_current_admin), db: Session = Depends(get_db)):
    stream_manager.stop_stream(stream_id)
    stream = db.query(RTSPStream).filter(RTSPStream.id == stream_id).first()
    if stream:
        db.delete(stream)
        db.commit()
    return {"message": f"Stream {stream_id} deleted"}

# ----------------------------------------------------------------------------------
# SEARCH AND ANALYTICS ENDPOINTS
# ----------------------------------------------------------------------------------
@app.post("/search-person")
async def search_person(request: PersonSearchRequest, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    query = db.query(DetectionEvent)
    if request.person_id:
        query = query.filter(DetectionEvent.person_id == request.person_id)
    if request.start_date:
        try:
            start_date = datetime.fromisoformat(request.start_date.replace("Z", "+00:00"))
            query = query.filter(DetectionEvent.timestamp >= start_date)
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid start_date format")
    if request.end_date:
        try:
            end_date = datetime.fromisoformat(request.end_date.replace("Z", "+00:00"))
            query = query.filter(DetectionEvent.timestamp <= end_date)
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid end_date format")
    if request.stream_id:
        query = query.filter(DetectionEvent.stream_id == request.stream_id)
    events = query.order_by(DetectionEvent.timestamp.desc()).limit(100).all()
    return [
        {
            "id": event.id,
            "person_id": event.person_id,
            "stream_id": event.stream_id,
            "timestamp": event.timestamp,
            "confidence": event.confidence,
            "bbox": [event.bbox_x, event.bbox_y, event.bbox_w, event.bbox_h],
            "face_quality": event.face_quality
        }
        for event in events
    ]

@app.get("/analytics/person-count")
async def get_person_count(current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    one_hour_ago = datetime.utcnow() - timedelta(hours=1)
    unique_persons = db.query(DetectionEvent.person_id).filter(
        DetectionEvent.timestamp >= one_hour_ago,
        DetectionEvent.person_id != "unknown"
    ).distinct().count()
    return {"unique_persons_last_hour": unique_persons}

@app.get("/test-auth")
async def test_auth(current_user: User = Depends(get_current_user)):
    return {"message": f"Authenticated as {current_user.username}", "user_id": current_user.id}

@app.get("/analytics/detection-stats")
async def get_detection_stats(current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    total_detections = db.query(DetectionEvent).count()
    yesterday = datetime.utcnow() - timedelta(days=1)
    recent_detections = db.query(DetectionEvent).filter(
        DetectionEvent.timestamp >= yesterday
    ).count()
    most_detected = db.query(
        DetectionEvent.person_id,
        func.count(DetectionEvent.id).label('count')
    ).filter(
        DetectionEvent.person_id != "unknown"
    ).group_by(DetectionEvent.person_id).order_by(
        func.count(DetectionEvent.id).desc()
    ).first()
    return {
        "total_detections": total_detections,
        "recent_detections": recent_detections,
        "most_detected_person": most_detected.person_id if most_detected else None,
        "most_detected_count": most_detected.count if most_detected else 0
    }

# ----------------------------------------------------------------------------------
# MAIN ENTRY POINT
# ----------------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
