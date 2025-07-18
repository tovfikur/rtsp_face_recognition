import os
import cv2
import pickle
import base64
import logging
import numpy as np
import torch
import time
from datetime import datetime, timedelta
from typing import List, Optional, Any, Dict
import httpx
import threading
import asyncio
import json
from concurrent.futures import ThreadPoolExecutor
from fastapi import FastAPI, File, UploadFile, HTTPException, Depends, WebSocket, WebSocketDisconnect, Form, Request, Response, status, Query
from fastapi.responses import HTMLResponse,  RedirectResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from pydantic import BaseModel

from sqlalchemy import create_engine, Column, Integer, String, LargeBinary, DateTime  
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.sql import func
from sqlalchemy import Column, Boolean


import jwt  # pip install pyjwt
from passlib.context import CryptContext

from facenet_pytorch import MTCNN, InceptionResnetV1

# ----------------------------------------------------------------------------------
# DATABASE SETUP
# ----------------------------------------------------------------------------------
# Create a folder for the SQLite database if it does not exist.
DB_FOLDER = "db"
if not os.path.exists(DB_FOLDER):
    os.makedirs(DB_FOLDER)

# Define the database URL and initialize SQLAlchemy components.
SQLALCHEMY_DATABASE_URL = f"sqlite:///{DB_FOLDER}/app.db"
engine = create_engine(SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


# Create templates directory if it doesn't exist
TEMPLATES_DIR = "templates"
if not os.path.exists(TEMPLATES_DIR):
    os.makedirs(TEMPLATES_DIR)

# Create static directory if it doesn't exist
STATIC_DIR = "static"
if not os.path.exists(STATIC_DIR):
    os.makedirs(STATIC_DIR)

# Setup Jinja2 templates
templates = Jinja2Templates(directory=TEMPLATES_DIR)


class User(Base):
    """
    User table for storing user credentials.
    
    Attributes:
      - id: Primary key.
      - username: Unique username.
      - hashed_password: Hashed version of the user's password.
    """
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True)
    hashed_password = Column(String)

class ConfigEntry(Base):
    """
    Configuration table to store key-value pairs.
    
    Attributes:
      - key: Configuration parameter name.
      - value: Value associated with the key.
    """
    __tablename__ = "config"
    key = Column(String, primary_key=True, index=True)
    value = Column(String)

class RegisteredFace(Base):
    """
    RegisteredFace table stores face embeddings for registered persons.
    
    The face embedding is computed from one or more images, averaged, and then stored as a pickled numpy array.
    
    Attributes:
      - id: Primary key.
      - person_id: Unique identifier for the person.
      - embedding: Pickled numpy array representing the face embedding.
      - updated_at: Timestamp of the latest update.
    """
    __tablename__ = "registered_faces"
    id = Column(Integer, primary_key=True, index=True)
    person_id = Column(String, unique=True, index=True)
    embedding = Column(LargeBinary)
    updated_at = Column(DateTime, default=datetime.utcnow)

# Create all tables in the database.
Base.metadata.create_all(bind=engine)

# ----------------------------------------------------------------------------------
# SECURITY: OAuth2 & JWT
# ----------------------------------------------------------------------------------
# SECRET_KEY should be set in a production environment.
SECRET_KEY = os.getenv("SECRET_KEY", "your-secret-key")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/token")

def verify_password(plain_password: str, hashed_password: str) -> bool:
    """
    Verify a plaintext password against its hashed version.
    """
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password: str) -> str:
    """
    Hash a plaintext password.
    """
    return pwd_context.hash(password)

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    """
    Create a JWT access token.
    
    Steps:
      1. Copy the data to encode.
      2. Set an expiration time.
      3. Encode the payload using the secret key and algorithm.
    """
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
    to_encode.update({"exp": expire})
    token = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return token

def decode_token(token: str) -> Any:
    """
    Decode and verify a JWT token.
    
    Raises:
      - HTTPException if the token is invalid or expired.
    """
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload
    except jwt.PyJWTError:
        raise HTTPException(status_code=401, detail="Could not validate credentials")

def get_db():
    """
    Dependency that creates a new SQLAlchemy session per request.
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def get_user(db: Session, username: str) -> Optional[User]:
    """
    Retrieve a user from the database by username.
    """
    return db.query(User).filter(User.username == username).first()

async def get_current_user(token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)) -> User:
    """
    Dependency that returns the current authenticated user based on JWT token.
    
    Steps:
      1. Decode the JWT token.
      2. Extract the username from the token payload.
      3. Retrieve the user from the database.
    """
    payload = decode_token(token)
    username: str = payload.get("sub")
    if username is None:
        raise HTTPException(status_code=401, detail="Invalid authentication credentials")
    user = get_user(db, username)
    if user is None:
        raise HTTPException(status_code=401, detail="User not found")
    return user

async def get_current_user_flexible(request: Request, db: Session = Depends(get_db)) -> User:
    """
    Flexible authentication that tries multiple methods:
    1. Authorization header (Bearer token)
    2. Cookie (token)
    3. Query parameter (token) - for debugging
    """
    token = None
    
    # Method 1: Try Authorization header first
    authorization = request.headers.get("Authorization")
    if authorization and authorization.startswith("Bearer "):
        token = authorization[7:]  # Remove "Bearer " prefix
        print(f"Auth method: Bearer header, token: {token[:20]}...")
    
    # Method 2: Try cookie
    if not token:
        token = request.cookies.get("token")
        if token:
            print(f"Auth method: Cookie, token: {token[:20]}...")
    
    # Method 3: Try backup cookie
    if not token:
        token = request.cookies.get("auth_token")
        if token:
            print(f"Auth method: Backup cookie, token: {token[:20]}...")
    
    # Method 4: Try query parameter (for debugging)
    if not token:
        token = request.query_params.get("token")
        if token:
            print(f"Auth method: Query param, token: {token[:20]}...")
    
    if not token:
        print("No token found in any location")
        raise HTTPException(status_code=401, detail="No authentication token found")
    
    try:
        payload = decode_token(token)
        username: str = payload.get("sub")
        if username is None:
            raise HTTPException(status_code=401, detail="Invalid token: no username")
        
        user = get_user(db, username)
        if user is None:
            raise HTTPException(status_code=401, detail="User not found")
        
        print(f"Authentication successful for user: {username}")
        return user
    except Exception as e:
        print(f"Token validation failed: {str(e)}")
        raise HTTPException(status_code=401, detail=f"Token validation failed: {str(e)}")

async def get_current_admin(current_user: User = Depends(get_current_user)) -> User:
    """
    Dependency that ensures the current user is an administrator.
    In this sample, the admin is identified by username 'admin'.
    """
    if current_user.username != "admin":
        raise HTTPException(status_code=403, detail="Not enough privileges")
    return current_user

# ----------------------------------------------------------------------------------
# FACE RECOGNITION MODELS & HELPERS
# ----------------------------------------------------------------------------------
# Set up device (GPU if available, otherwise CPU) and logging.
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("FaceRecognitionAPI")
logger.info(f"Using device: {device}")

# Initialize face detection (MTCNN) and embedding (InceptionResnetV1) models.
mtcnn = MTCNN(image_size=160, margin=20, device=device)
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

def compute_embedding(image: np.ndarray) -> np.ndarray:
    """
    Compute a face embedding for a given image.
    
    Data Flow:
      1. Convert the input image from BGR (OpenCV format) to RGB.
      2. Use MTCNN to detect a face within the image.
      3. If a face is found, convert it into a tensor, pass it through the InceptionResnetV1 model,
         and return the flattened embedding.
    
    Raises:
      - HTTPException if no face is detected.
    """
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    face = mtcnn(image_rgb)
    if face is None:
        raise HTTPException(status_code=400, detail="No face detected in the image")
    face = face.unsqueeze(0).to(device)
    with torch.no_grad():
        embedding = resnet(face).cpu().numpy().flatten()
    return embedding

async def read_imagefile(file: UploadFile) -> np.ndarray:
    """
    Read and decode an image file (UploadFile) into a numpy array.
    
    Data Flow:
      1. Read raw bytes from the file.
      2. Convert the bytes into a numpy array.
      3. Decode the image using OpenCV.
    
    Raises:
      - HTTPException if the image is invalid.
    """
    image_bytes = await file.read()
    np_arr = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(status_code=400, detail="Invalid image")
    return image

# ----------------------------------------------------------------------------------
# INITIALIZATION: DEFAULT CONFIG & ADMIN USER
# ----------------------------------------------------------------------------------
DEFAULT_CONFIG = {
    "TOLERANCE": "0.9",           # Tolerance for face matching (stored as string).
    "DETECTION_THRESHOLD": "0.95", # Detection threshold (stored as string).
    "CALLBACK_URL": "",    # New: endpoint to send detections
    "CALLBACK_TOKEN": ""   # New: token for authenticating callback
}

# In-memory cache to store last callback times for each person_id
last_callback_times: Dict[str, float] = {}
COOLDOWN_SECONDS = 60  # 1 minute cooldown

async def send_callback(person_id: str):
    # Skip callback if person_id is "Unknown"
    if person_id == "Unknown":
        return
        
    # Check if this ID was sent within the cooldown period
    current_time = time.time()
    if person_id in last_callback_times:
        time_since_last_callback = current_time - last_callback_times[person_id]
        if time_since_last_callback < COOLDOWN_SECONDS:
            logger.info(f"Skipping callback for {person_id}: cooldown period active ({time_since_last_callback:.1f}s < {COOLDOWN_SECONDS}s)")
            return
    
    # Convert person_id to integer for employee_id
    try:
        employee_id = int(person_id)
    except ValueError:
        logger.error(f"Invalid person_id: {person_id}, cannot convert to integer.")
        return
        
    # Database session
    db: Session = SessionLocal()
    try:
        # Retrieve callback URL and token from database
        url_entry = db.query(ConfigEntry).filter(ConfigEntry.key == "CALLBACK_URL").first()
        token_entry = db.query(ConfigEntry).filter(ConfigEntry.key == "CALLBACK_TOKEN").first()
        if not url_entry or not url_entry.value:
            logger.info("No callback URL set, skipping callback.")
            return
            
        callback_url = url_entry.value
        headers = {"Content-Type": "application/json"}
        if token_entry and token_entry.value:
            headers["API-Key"] = token_entry.value
            
        # Generate timestamps
        current_time_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        check_out_time = (datetime.now() + timedelta(hours=8)).strftime("%Y-%m-%d %H:%M:%S")
        
        # Prepare JSON payload
        data = {
            "employee_id": employee_id,
            "check_in": current_time_str,
            "check_out": check_out_time
        }
        
        # Send POST request
        async with httpx.AsyncClient() as client:
            try:
                response = await client.post(callback_url, json=data, headers=headers)
                response.raise_for_status()
                # Update the last callback time for this ID
                last_callback_times[person_id] = current_time
                logger.info(f"Callback sent to {callback_url} successfully for {person_id}.")
            except httpx.HTTPError as e:
                logger.error(f"Failed to send callback to {callback_url}: {str(e)}")
    finally:
        db.close()

# Optional: You may want to add a cleanup function to prevent memory leaks
# from storing too many IDs in the last_callback_times dictionary
async def cleanup_old_callback_records():
    current_time = time.time()
    expired_ids = [
        pid for pid, timestamp in last_callback_times.items()
        if current_time - timestamp > COOLDOWN_SECONDS * 10  # Remove entries older than 10x cooldown
    ]
    for pid in expired_ids:
        del last_callback_times[pid]

def init_config(db: Session):
    """
    Initialize configuration parameters in the database if they do not exist.
    """
    for key, value in DEFAULT_CONFIG.items():
        if not db.query(ConfigEntry).filter(ConfigEntry.key == key).first():
            db.add(ConfigEntry(key=key, value=value))
    db.commit()

def init_admin(db: Session):
    """
    Initialize a default admin user if not present in the database.
    Default credentials: username 'admin', password 'admin'
    """
    if not db.query(User).filter(User.username == "admin").first():
        admin = User(username="admin", hashed_password=get_password_hash("admin"))
        db.add(admin)
        db.commit()

# ----------------------------------------------------------------------------------
# FASTAPI APPLICATION SETUP & EXTENDED OPENAPI DOCUMENTATION
# ----------------------------------------------------------------------------------
# Extended OpenAPI description that covers data flow, security, model operations,
# and detailed instructions for each endpoint.
app_description = """
# Face Recognition API

This API provides secure face recognition services with detailed documentation.  
It is designed for tasks including face comparison, authentication, and real-time detection via WebSockets.

---

## Overview

### Data Flow
1. **Authentication:**  
   - Users must obtain a JWT token via the **/token** endpoint using their credentials.
   - JWT tokens are used to authenticate all subsequent requests.

2. **Image Processing & Face Recognition:**  
   - **Image Upload:** Endpoints such as **/compare**, **/authenticate**, and **/register** accept image uploads.
   - **Processing Steps:**  
     a. Images are read and decoded using OpenCV.  
     b. The image is converted from BGR to RGB.  
     c. The MTCNN model detects a face, and the InceptionResnetV1 model computes the face embedding.
   - **Comparison:**  
     - The Euclidean distance between embeddings is computed to compare faces.
     - A configured tolerance value determines whether faces match.

3. **Database Interaction:**  
   - **Users, Configuration, and Registered Faces:**  
     Data is stored using SQLite via SQLAlchemy.
   - **Face Embeddings:**  
     Registered faces store a pickled numpy array (embedding) for future comparison.

4. **Real-Time Detection (WebSocket):**  
   - Clients send base64‑encoded frames to **/ws/detection**.
   - Each frame is processed as above, and the matching registered face (or 'Unknown') is returned.

---

## Endpoints Detail

### Security & User Management
- **/token (POST):**  
  Authenticate by submitting username and password as form data.  
  Returns a JWT token that is valid for 30 minutes.

- **/users (POST):**  
  *Admin Only.* Create a new user by providing a JSON body with `username` and `password`.

### Face Operations
- **/compare (POST):**  
  Accepts two images (keys: `reference` and `check`).  
  Computes face embeddings for both images, calculates the Euclidean distance, and returns a match result based on the tolerance.

- **/authenticate (POST):**  
  Upload an image to compute its embedding and compare it against registered faces.  
  Returns the matching person ID if the distance is below tolerance; otherwise, returns 'Unknown'.

- **/register (POST):**  
  Register or update a person's face embedding.  
  Provide a unique `person_id` and multiple images. The system computes the average embedding and stores it in the database.

### Configuration
- **/config (GET):**  
  Retrieve current configuration parameters (e.g., TOLERANCE and DETECTION_THRESHOLD).

- **/config (PUT):**  
  Update configuration parameters by providing a JSON body with new values.

### Data Retrieval
- **/persons (GET):**  
  List all registered persons along with the timestamp of their last update.

### Real-Time Detection
- **/ws/detection (WebSocket):**  
  Establish a WebSocket connection using a JWT token.  
  Clients send base64‑encoded image frames to perform real-time face detection.  
  The server processes each frame and sends back the detected person ID (or "Unknown").

### Testing Tools
- **/ws/test (GET):**  
  Serves a simple HTML page to test WebSocket connections by sending an image file.

- **/ws/live (GET):**  
  Provides an HTML client that uses your webcam to capture a live video feed and continuously send frames to the WebSocket endpoint.

---

## Internal Components

### Authentication & Security
- **JWT Token Generation:**  
  Tokens include an expiry and are signed with a secret key.

### Image Processing
- **Face Detection & Embedding:**  
  Utilizes MTCNN for detecting faces and InceptionResnetV1 to compute embeddings.
- **Error Handling:**  
  If no face is detected or if the image is invalid, appropriate HTTP errors are raised.

### Database Operations
- **SQLAlchemy ORM:**  
  Manages interactions with the SQLite database for users, configuration, and registered face data.
- **Session Management:**  
  Each API request creates a new database session that is closed after the request completes.

---

This documentation provides a step-by-step explanation of the entire data flow from user authentication to image processing and storage. Explore individual endpoint documentation in the interactive docs for more details on each operation.
Developed by: Iftekar Hossan
Copyright © 2025 Kendroo. All rights reserved.
"""

app = FastAPI(
    title="Attendance from facial recognition",
    description=app_description,
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json"
)

# Allow all CORS origins (adjust in production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Startup event: Initialize default configuration and admin user.
@app.on_event("startup")
def startup_event():
    db = SessionLocal()
    try:
        init_config(db)
        init_admin(db)
        logger.info("Startup initialization complete.")
    finally:
        db.close()

# ----------------------------------------------------------------------------------
# API MODELS
# ----------------------------------------------------------------------------------
class Token(BaseModel):
    access_token: str
    token_type: str

class ConfigUpdate(BaseModel):
    tolerance: Optional[float] = None
    detection_threshold: Optional[float] = None

class UserCreate(BaseModel):
    username: str
    password: str

class PersonInfo(BaseModel):
    person_id: str
    updated_at: datetime
    
class CallbackConfig(BaseModel):
    callback_url: Optional[str] = None
    callback_token: Optional[str] = None


# ----------------------------------------------------------------------------------
# HTML INTERFACE WITH AUTHENTICATION
# ----------------------------------------------------------------------------------

@app.get("/login", response_class=HTMLResponse)
async def login_page(request: Request):
    # Always return a TemplateResponse here
    return templates.TemplateResponse("login.html", {"request": request})


@app.get("/logout")
async def logout(response: Response):
    response.delete_cookie("token")
    return RedirectResponse(url="/login")

@app.get("/", response_class=HTMLResponse)
async def config_page(request: Request, db: Session = Depends(get_db)):
    print("Home page accessed")
    
    # Try multiple token sources
    token = None
    auth_source = None
    
    # Check Authorization header
    authorization = request.headers.get("Authorization")
    if authorization and authorization.startswith("Bearer "):
        token = authorization[7:]
        auth_source = "header"
    
    # Check cookies
    if not token:
        token = request.cookies.get("token")
        if token:
            auth_source = "cookie-token"
    
    if not token:
        token = request.cookies.get("auth_token")
        if token:
            auth_source = "cookie-auth_token"
    
    print(f"Token found in: {auth_source}, token: {token[:20] if token else 'None'}...")
    
    if not token:
        print("No token found, redirecting to login")
        return RedirectResponse(url="/login", status_code=status.HTTP_302_FOUND)

    # Validate token
    try:
        decode_token(token)
        print("Token validation successful")
    except HTTPException as e:
        print(f"Token validation failed: {e.detail}")
        return RedirectResponse(url="/login", status_code=status.HTTP_302_FOUND)

    return templates.TemplateResponse("config.html", {"request": request})


# ----------------------------------------------------------------------------------
# CALLBACK CONFIG ENDPOINTS
# ----------------------------------------------------------------------------------

@app.get("/callback-config", summary="Get callback configuration")
async def get_callback_config(current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    url = db.query(ConfigEntry).filter(ConfigEntry.key == "CALLBACK_URL").first().value
    token = db.query(ConfigEntry).filter(ConfigEntry.key == "CALLBACK_TOKEN").first().value
    return {"callback_url": url, "callback_token": token}

@app.put("/callback-config", summary="Update callback configuration")
async def update_callback_config(config: CallbackConfig, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    if config.callback_url is not None:
        entry = db.query(ConfigEntry).filter(ConfigEntry.key == "CALLBACK_URL").first()
        entry.value = config.callback_url
    if config.callback_token is not None:
        entry = db.query(ConfigEntry).filter(ConfigEntry.key == "CALLBACK_TOKEN").first()
        entry.value = config.callback_token
    db.commit()
    return {"message": "Callback configuration updated."}


# ----------------------------------------------------------------------------------
# AUTHENTICATION ENDPOINTS
# ----------------------------------------------------------------------------------
@app.post("/token", response_model=Token, summary="Generate access token",
          description="Submit your username and password (as form data) to receive a JWT token for authentication. "
                      "The token is required for accessing secure endpoints. This token expires after 30 minutes.")
async def login_for_access_token(response: Response, form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    print(f"Login attempt for user: {form_data.username}")
    
    user = get_user(db, form_data.username)
    if not user or not verify_password(form_data.password, user.hashed_password):
        print("Login failed: Invalid credentials")
        raise HTTPException(status_code=401, detail="Incorrect username or password")
    
    access_token = create_access_token(data={"sub": user.username})
    print(f"Token created for {user.username}: {access_token[:20]}...")
    
    # Set multiple cookie variations to ensure compatibility
    response.set_cookie(
        key="token", 
        value=access_token,
        httponly=False,  # Allow JavaScript access for debugging
        max_age=ACCESS_TOKEN_EXPIRE_MINUTES * 60,
        samesite="lax",
        secure=False  # Set to True in production with HTTPS
    )
    
    # Also set a backup cookie
    response.set_cookie(
        key="auth_token", 
        value=access_token,
        httponly=False,
        max_age=ACCESS_TOKEN_EXPIRE_MINUTES * 60,
        samesite="lax",
        secure=False
    )
    
    print("Cookies set successfully")
    return {"access_token": access_token, "token_type": "bearer"}

# ----------------------------------------------------------------------------------
# FACE COMPARISON ENDPOINT
# ----------------------------------------------------------------------------------
@app.post("/compare", summary="Compare two images",
          description="Upload two images (keys: 'reference' and 'check'). The API computes their face embeddings "
                      "using MTCNN and InceptionResnetV1, calculates the Euclidean distance between embeddings, "
                      "and returns a match result based on the configured tolerance.")
async def compare_images(reference: UploadFile = File(...),
                         check: UploadFile = File(...),
                         current_user: User = Depends(get_current_user)):
    ref_image = await read_imagefile(reference)
    check_image = await read_imagefile(check)
    ref_embedding = compute_embedding(ref_image)
    check_embedding = compute_embedding(check_image)
    distance = np.linalg.norm(ref_embedding - check_embedding)
    match = bool(distance < float(DEFAULT_CONFIG["TOLERANCE"]))
    if match:
        await send_callback("matched")
    return {"match": match, "distance": float(distance)}

# ----------------------------------------------------------------------------------
# FACE AUTHENTICATION ENDPOINT
# ----------------------------------------------------------------------------------
@app.post("/authenticate", summary="Authenticate via face recognition",
          description="Upload an image to compute its embedding and compare against registered faces stored in the database. "
                      "If the Euclidean distance is below the tolerance, the associated person_id is returned; otherwise, 'Unknown' is returned.")
async def authenticate_face(file: UploadFile = File(...),
                            current_user: User = Depends(get_current_user),
                            db: Session = Depends(get_db)):
    image = await read_imagefile(file)
    embedding = compute_embedding(image)
    config_entry = db.query(ConfigEntry).filter(ConfigEntry.key == "TOLERANCE").first()
    tolerance = float(config_entry.value) if config_entry else float(DEFAULT_CONFIG["TOLERANCE"])
    registered_faces = db.query(RegisteredFace).all()
    if not registered_faces:
        raise HTTPException(status_code=404, detail="No registered faces found.")
    distances = []
    for face in registered_faces:
        stored_embedding = pickle.loads(face.embedding)
        dist = np.linalg.norm(stored_embedding - embedding)
        distances.append((face.person_id, dist))
    best_match = min(distances, key=lambda x: x[1])
    person_id, min_distance = best_match
    result = person_id if min_distance < tolerance else "Unknown"
    await send_callback(result)
    return {"person_id": result, "distance": float(min_distance)}

# ----------------------------------------------------------------------------------
# CONFIGURATION ENDPOINTS
# ----------------------------------------------------------------------------------
@app.get("/config", summary="Get current configuration",
         description="Retrieve the current configuration parameters (e.g., TOLERANCE and DETECTION_THRESHOLD) from the database.")
async def get_config(current_user: User = Depends(get_current_user_flexible), db: Session = Depends(get_db)):
    entries = db.query(ConfigEntry).all()
    return {entry.key: entry.value for entry in entries}

@app.put("/config", summary="Update configuration parameters",
         description="Update configuration parameters by sending a JSON body with new values for tolerance and detection threshold. "
                     "The updated values will affect how strict face matching is performed.")
async def update_config(update: ConfigUpdate,
                        current_user: User = Depends(get_current_user_flexible),
                        db: Session = Depends(get_db)):
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
    db.commit()
    return {"message": "Configuration updated successfully."}

# ----------------------------------------------------------------------------------
# PERSON REGISTRATION ENDPOINT
# ----------------------------------------------------------------------------------
@app.post("/register", summary="Register a person with images",
          description="Register or update a person's face embedding by providing a unique person_id (as form data) "
                      "and multiple images. The system computes the average embedding from all valid images and stores it.")
async def register_person(person_id: str = Form(...),
                          images: List[UploadFile] = File(...),
                          current_user: User = Depends(get_current_user),
                          db: Session = Depends(get_db)):
    embeddings = []
    for file in images:
        image = await read_imagefile(file)
        try:
            emb = compute_embedding(image)
            embeddings.append(emb)
        except HTTPException as e:
            logger.warning(f"Skipping an image: {e.detail}")
    if not embeddings:
        raise HTTPException(status_code=400, detail="No valid face detected in any of the images.")
    avg_embedding = np.mean(np.array(embeddings), axis=0)
    serialized_embedding = pickle.dumps(avg_embedding)
    reg_face = db.query(RegisteredFace).filter(RegisteredFace.person_id == person_id).first()
    if reg_face:
        reg_face.embedding = serialized_embedding
        reg_face.updated_at = datetime.utcnow()
    else:
        reg_face = RegisteredFace(person_id=person_id, embedding=serialized_embedding, updated_at=datetime.utcnow())
        db.add(reg_face)
    db.commit()
    return {"message": f"Person '{person_id}' registered/updated successfully."}

# ----------------------------------------------------------------------------------
# ADMIN USER CREATION ENDPOINT
# ----------------------------------------------------------------------------------
@app.post("/users", summary="Create a new user",
          description="*Admin Only.* To create a new user, first authenticate as admin (default: username 'admin', password 'admin') via **/token**. "
                      "Then call this endpoint with a JSON body containing the new user's 'username' and 'password'. "
                      "Ensure your production environment sets a secure `SECRET_KEY`.",
          response_model=dict)
async def create_user(user: UserCreate,
                      current_admin: User = Depends(get_current_admin),
                      db: Session = Depends(get_db)):
    if get_user(db, user.username):
        raise HTTPException(status_code=400, detail="Username already registered")
    new_user = User(username=user.username, hashed_password=get_password_hash(user.password))
    db.add(new_user)
    db.commit()
    return {"message": f"User '{user.username}' created successfully."}

# ----------------------------------------------------------------------------------
# PERSON LISTING ENDPOINT
# ----------------------------------------------------------------------------------
@app.get("/persons", response_model=List[PersonInfo], summary="List registered persons",
         description="Retrieve a list of all registered persons with their person_id and the timestamp of the last update.")
async def list_persons(current_user: User = Depends(get_current_user),
                       db: Session = Depends(get_db)):
    persons = db.query(RegisteredFace).all()
    return [{"person_id": person.person_id, "updated_at": person.updated_at} for person in persons]

# ----------------------------------------------------------------------------------
# REAL-TIME FACE DETECTION VIA WEBSOCKET
# ----------------------------------------------------------------------------------
@app.websocket("/ws/detection")
async def websocket_detection(websocket: WebSocket, token: str):
    """
    WebSocket endpoint for real-time face detection.
    
    Data Flow:
      1. Client connects using a JWT token (passed as a query parameter).
      2. Each message is expected to be a base64‑encoded image frame.
      3. The frame is decoded, converted to an image, and processed to compute a face embedding.
      4. The embedding is compared with stored registered face embeddings.
      5. The detected person_id (if matching) or "Unknown" is sent back as text.
      
    **Usage:**  
    - Connect using: `ws://<host>:8000/ws/detection?token=<JWT_TOKEN>`  
    - Send base64‑encoded image frames to receive detection results.
    
    *Note:* Swagger UI does not support testing WebSocket endpoints.
    """
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
            except WebSocketDisconnect:
                logger.info("WebSocket disconnected")
                break
            try:
                img_data = base64.b64decode(data)
                np_arr = np.frombuffer(img_data, np.uint8)
                frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
                if frame is None:
                    await websocket.send_text("Invalid image frame")
                    continue
                embedding = compute_embedding(frame)
            except Exception as e:
                await websocket.send_text(f"Error processing frame: {str(e)}")
                continue

            registered_faces = db.query(RegisteredFace).all()
            if not registered_faces:
                result = "No registered faces"
            else:
                distances = []
                for face in registered_faces:
                    stored_embedding = pickle.loads(face.embedding)
                    dist = np.linalg.norm(stored_embedding - embedding)
                    distances.append((face.person_id, dist))
                best_match = min(distances, key=lambda x: x[1])
                person_id, min_distance = best_match
                config_entry = db.query(ConfigEntry).filter(ConfigEntry.key == "TOLERANCE").first()
                tolerance = float(config_entry.value) if config_entry else float(DEFAULT_CONFIG["TOLERANCE"])
                result = person_id if min_distance < tolerance else "Unknown"
                await send_callback(result)
            await websocket.send_text(result)
    finally:
        db.close()
        

# ----------------------------------------------------------------------------------
# DEBUG ENDPOINTS
# ----------------------------------------------------------------------------------
@app.get("/debug/auth")
async def debug_auth(request: Request):
    """Debug endpoint to check authentication status"""
    
    result = {
        "cookies": dict(request.cookies),
        "headers": {k: v for k, v in request.headers.items() if k.lower() not in ['authorization']},
        "query_params": dict(request.query_params)
    }
    
    # Try to get token
    token = None
    auth_method = None
    
    authorization = request.headers.get("Authorization")
    if authorization and authorization.startswith("Bearer "):
        token = authorization[7:]
        auth_method = "Bearer header"
    elif request.cookies.get("token"):
        token = request.cookies.get("token")
        auth_method = "Cookie (token)"
    elif request.cookies.get("auth_token"):
        token = request.cookies.get("auth_token")
        auth_method = "Cookie (auth_token)"
    
    result["token_found"] = bool(token)
    result["auth_method"] = auth_method
    result["token_preview"] = token[:20] + "..." if token else None
    
    if token:
        try:
            payload = decode_token(token)
            result["token_valid"] = True
            result["username"] = payload.get("sub")
            result["expires"] = payload.get("exp")
        except Exception as e:
            result["token_valid"] = False
            result["token_error"] = str(e)
    
    return result

# ----------------------------------------------------------------------------------
# WEBSOCKET TEST PAGES
# ----------------------------------------------------------------------------------
# Basic WebSocket test page for manual testing.

@app.get("/ws/test", response_class=HTMLResponse, summary="WebSocket Test Page",
         description="This page provides a simple interface to test the WebSocket endpoint. Enter your JWT token, connect, and send an image file to test real-time detection.")
async def websocket_test_page():
    return templates.TemplateResponse("ws_test.html", {"request": {}})

# Live Video WebSocket test page that accesses the webcam.

@app.get("/ws/live", response_class=HTMLResponse, summary="Live Video WebSocket Test Page",
         description="This page accesses your webcam, captures video frames, and sends them continuously to the WebSocket detection endpoint. "
                     "Use this tool for real-time face detection testing.")
async def websocket_live_page():
    return templates.TemplateResponse("ws_live.html", {"request": {}})


# =============================================================================
#  ENHANCED RTSP DATABASE MODEL WITH LIVE STREAMING
# =============================================================================
class RTSPStream(Base):
    """
    RTSP Stream table to store RTSP camera configurations.
    
    Attributes:
      - id: Primary key.
      - name: Human-readable name for the stream.
      - rtsp_url: RTSP URL for the camera stream.
      - is_active: Boolean flag to indicate if stream is currently active.
      - created_at: Timestamp when the stream was added.
      - last_detection: Timestamp of the last face detection.
    """
    __tablename__ = "rtsp_streams"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, index=True)
    rtsp_url = Column(String)
    is_active = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    last_detection = Column(DateTime)

    
Base.metadata.create_all(bind=engine)
# =============================================================================
#  ENHANCED PYDANTIC MODELS
# =============================================================================
class RTSPStreamCreate(BaseModel):
    name: str
    rtsp_url: str

class RTSPStreamInfo(BaseModel):
    id: int
    name: str
    rtsp_url: str
    is_active: bool
    created_at: datetime
    last_detection: Optional[datetime]
    
    class Config:
        from_attributes = True

# =============================================================================
#  ENHANCED GLOBAL VARIABLES FOR STREAM MANAGEMENT 
# =============================================================================
# Dictionary to store active RTSP stream threads
active_streams: Dict[int, Dict[str, Any]] = {}
# Thread pool executor for RTSP processing
executor = ThreadPoolExecutor(max_workers=10)

# =============================================================================
#  ENHANCED RTSP PROCESSING FUNCTIONS
# =============================================================================

def process_rtsp_stream(stream_id: int, rtsp_url: str, stream_name: str):
    """
    Process RTSP stream in a separate thread for face detection (similar to webcam).
    
    Args:
        stream_id: Database ID of the RTSP stream
        rtsp_url: RTSP URL to connect to
        stream_name: Human-readable name for logging
    """
    logger.info(f"Starting RTSP stream processing for {stream_name} (ID: {stream_id})")
    
    cap = cv2.VideoCapture(rtsp_url)
    if not cap.isOpened():
        logger.error(f"Failed to open RTSP stream: {rtsp_url}")
        return
    
    # Set buffer size to reduce latency
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    # Set frame dimensions for better performance
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 15)
    
    db = SessionLocal()
    try:
        # Get tolerance from config
        config_entry = db.query(ConfigEntry).filter(ConfigEntry.key == "TOLERANCE").first()
        tolerance = float(config_entry.value) if config_entry else float(DEFAULT_CONFIG["TOLERANCE"])
        
        # Get registered faces
        registered_faces = db.query(RegisteredFace).all()
        
        last_process_time = 0
        frame_count = 0
        
        while stream_id in active_streams and active_streams[stream_id].get("running", False):
            ret, frame = cap.read()
            if not ret:
                logger.warning(f"Failed to read frame from {stream_name}")
                time.sleep(1)
                continue
            
            current_time = time.time()
            frame_count += 1
            
            # Process every 3rd frame (similar to webcam processing frequency)
            if frame_count % 3 != 0:
                continue
                
            # Process at maximum 1 frame per second to avoid spam
            if current_time - last_process_time < 1.0:
                continue
                
            last_process_time = current_time
                
            try:
                # Compute embedding for the frame (same as webcam)
                embedding = compute_embedding(frame)
                
                if registered_faces:
                    distances = []
                    for face in registered_faces:
                        stored_embedding = pickle.loads(face.embedding)
                        dist = np.linalg.norm(stored_embedding - embedding)
                        distances.append((face.person_id, dist))
                    
                    best_match = min(distances, key=lambda x: x[1])
                    person_id, min_distance = best_match
                    
                    if min_distance < tolerance:
                        logger.info(f"Face detected in {stream_name}: {person_id} (distance: {min_distance:.3f})")
                        
                        # Update last detection time
                        stream_record = db.query(RTSPStream).filter(RTSPStream.id == stream_id).first()
                        if stream_record:
                            stream_record.last_detection = datetime.utcnow()
                            db.commit()
                        
                        # Send callback (same as webcam)
                        asyncio.run(send_callback(person_id))
                        
                        # Add delay after detection to avoid spam (same as webcam)
                        time.sleep(2)
                        
            except Exception as e:
                logger.warning(f"Error processing frame from {stream_name}: {str(e)}")
                continue
            
            # Small delay to prevent excessive CPU usage
            time.sleep(0.1)
            
            # Cleanup callback records periodically
            if frame_count % 300 == 0:  # Every 300 frames
                asyncio.run(cleanup_old_callback_records())
                
    finally:
        cap.release()
        db.close()
        logger.info(f"RTSP stream processing stopped for {stream_name}")

def start_rtsp_stream(stream_id: int, rtsp_url: str, stream_name: str):
    """Start RTSP stream processing in a separate thread."""
    if stream_id in active_streams:
        return False  # Already running
    
    active_streams[stream_id] = {
        "running": True,
        "thread": None
    }
    
    # Start processing in thread pool
    future = executor.submit(process_rtsp_stream, stream_id, rtsp_url, stream_name)
    active_streams[stream_id]["thread"] = future
    
    return True

def stop_rtsp_stream(stream_id: int):
    """Stop RTSP stream processing."""
    if stream_id in active_streams:
        active_streams[stream_id]["running"] = False
        # Wait a moment for graceful shutdown
        time.sleep(0.5)
        del active_streams[stream_id]
        return True
    return False


# =============================================================================
#  ENHANCED RTSP API ENDPOINTS 
# =============================================================================


@app.post("/rtsp/streams", response_model=RTSPStreamInfo, summary="Add RTSP stream")
async def add_rtsp_stream(
    stream: RTSPStreamCreate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Add a new RTSP stream for face detection monitoring.
    """
    new_stream = RTSPStream(
        name=stream.name,
        rtsp_url=stream.rtsp_url,
        is_active=False
    )
    db.add(new_stream)
    db.commit()
    db.refresh(new_stream)
    
    return RTSPStreamInfo(
        id=new_stream.id,
        name=new_stream.name,
        rtsp_url=new_stream.rtsp_url,
        is_active=new_stream.is_active,
        created_at=new_stream.created_at,
        last_detection=new_stream.last_detection
    )



@app.get("/rtsp/streams", response_model=List[RTSPStreamInfo], summary="List RTSP streams")
async def list_rtsp_streams(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Get list of all configured RTSP streams.
    """
    streams = db.query(RTSPStream).all()
    return [
        RTSPStreamInfo(
            id=stream.id,
            name=stream.name,
            rtsp_url=stream.rtsp_url,
            is_active=stream.is_active,
            created_at=stream.created_at,
            last_detection=stream.last_detection
        )
        for stream in streams
    ]

@app.post("/rtsp/streams/{stream_id}/start", summary="Start RTSP stream monitoring")
async def start_rtsp_monitoring(
    stream_id: int,
    # current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Start face detection monitoring for an RTSP stream.
    """
    stream = db.query(RTSPStream).filter(RTSPStream.id == stream_id).first()
    if not stream:
        raise HTTPException(status_code=404, detail="RTSP stream not found")
    
    if start_rtsp_stream(stream_id, stream.rtsp_url, stream.name):
        stream.is_active = True
        db.commit()
        return {"message": f"Started monitoring RTSP stream: {stream.name}"}
    else:
        raise HTTPException(status_code=400, detail="Stream is already running")


@app.post("/rtsp/streams/{stream_id}/stop", summary="Stop RTSP stream monitoring")
async def stop_rtsp_monitoring(
    stream_id: int,
    # current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Stop face detection monitoring for an RTSP stream.
    """
    stream = db.query(RTSPStream).filter(RTSPStream.id == stream_id).first()
    if not stream:
        raise HTTPException(status_code=404, detail="RTSP stream not found")
    
    if stop_rtsp_stream(stream_id):
        stream.is_active = False
        db.commit()
        return {"message": f"Stopped monitoring RTSP stream: {stream.name}"}
    else:
        return {"message": "Stream was not running"}


@app.delete("/rtsp/streams/{stream_id}", summary="Delete RTSP stream")
async def delete_rtsp_stream(
    stream_id: int,
    # current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Delete an RTSP stream configuration.
    """
    stream = db.query(RTSPStream).filter(RTSPStream.id == stream_id).first()
    if not stream:
        raise HTTPException(status_code=404, detail="RTSP stream not found")
    
    # Stop monitoring if active
    if stream.is_active:
        stop_rtsp_stream(stream_id)
    
    db.delete(stream)
    db.commit()
    return {"message": f"RTSP stream '{stream.name}' deleted successfully"}

# =============================================================================
#  ENHANCED RTSP MANAGEMENT HTML PAGE WITH LIVE PREVIEW
# =============================================================================

@app.get("/rtsp/manage", response_class=HTMLResponse, summary="RTSP Management Page")
async def rtsp_management_page(request: Request):
    # token = request.cookies.get("access_token")
    # if not token:
    #     return RedirectResponse(url="/login", status_code=status.HTTP_302_FOUND)
    # try:
    #     decode_token(token)
    # except HTTPException:
        # return RedirectResponse(url="/login", status_code=status.HTTP_302_FOUND)
    return templates.TemplateResponse("rtsp_management.html", {"request": request})

# =============================================================================
#  CLEANUP ON SHUTDOWN
# =============================================================================

@app.on_event("shutdown")
def shutdown_event():
    """Clean up active RTSP streams on application shutdown."""
    logger.info("Shutting down RTSP streams...")
    for stream_id in list(active_streams.keys()):
        stop_rtsp_stream(stream_id)
    executor.shutdown(wait=True)
    logger.info("Application shutdown complete.")


# ------------------------------------------------------------------------------
# Database initialization RSTP Stream Management
# ------------------------------------------------------------------------------


def migrate_rtsp_table():
    """
    Simplified migration for RTSP table.
    """
    db = SessionLocal()
    try:
        # Check and add created_at column if missing
        try:
            db.execute("SELECT created_at FROM rtsp_streams LIMIT 1")
        except Exception:
            try:
                db.execute("ALTER TABLE rtsp_streams ADD COLUMN created_at DATETIME DEFAULT CURRENT_TIMESTAMP")
                db.commit()
                logger.info("Added created_at column to rtsp_streams table")
            except Exception as e:
                logger.error(f"Failed to add created_at column: {e}")
                
        # Check and add last_detection column if missing
        try:
            db.execute("SELECT last_detection FROM rtsp_streams LIMIT 1")
        except Exception:
            try:
                db.execute("ALTER TABLE rtsp_streams ADD COLUMN last_detection DATETIME")
                db.commit()
                logger.info("Added last_detection column to rtsp_streams table")
            except Exception as e:
                logger.error(f"Failed to add last_detection column: {e}")
                
    finally:
        db.close()

# ----------------------------------------------------------------------------------
# MAIN EXECUTION
# ----------------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    app.debug = True
    migrate_rtsp_table()
    uvicorn.run(app, host="0.0.0.0", port=8000)
