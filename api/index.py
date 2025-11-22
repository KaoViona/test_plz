# index.py
import io, os
import numpy as np
from PIL import Image
from fastapi import FastAPI, UploadFile, File, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import firebase_admin
from firebase_admin import credentials, auth
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from catfaces_demo import load_model, detect_cat_faces, face_to_feature, K, UNKNOWN_THRESHOLD

# ===================== FastAPI 初始化 =====================
app = FastAPI(title="Cat Face ID API", version="1.2")

# ===================== CORS 設定 =====================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# ===================== 靜態檔案 =====================
if not os.path.exists("frontend"):
    os.makedirs("frontend")
app.mount("/static", StaticFiles(directory="frontend"), name="static")

# ===================== Firebase Admin 初始化 =====================
cred_path = os.path.join(os.path.dirname(__file__), "firebase-admin.json")
if not firebase_admin._apps:
    cred = credentials.Certificate(cred_path)
    firebase_admin.initialize_app(cred)

# ===================== 載入模型 & 資料庫 =====================
knn, id2name = load_model()
comments_db = {}  # 留言板
users_db = {}     # 使用者資料

# ===================== JWT 驗證（支援 Swagger Authorize） =====================
bearer_scheme = HTTPBearer()

def verify_token(credentials: HTTPAuthorizationCredentials = Depends(bearer_scheme)):
    token = credentials.credentials
    try:
        decoded = auth.verify_id_token(token)
        return decoded
    except Exception:
        raise HTTPException(status_code=401, detail="Invalid or expired token")

# ===================== 路由 =====================
@app.get("/")
def root():
    """首頁回傳 index.html"""
    path = os.path.join("frontend", "index.html")
    if os.path.exists(path):
        return FileResponse(path)
    return {"detail":"index.html not found"}

# ------------------ 使用者資料 ------------------
@app.post("/register_or_login")
def register_or_login(payload: dict, decoded=Depends(verify_token)):
    uid = decoded["uid"]
    email = decoded.get("email", "")
    name = payload.get("name", "")
    users_db[uid] = {"email": email, "name": name}
    return {"status":"ok", "uid": uid, "email": email, "name": name}

@app.get("/users")
def list_users(decoded=Depends(verify_token)):
    return users_db

# ------------------ 貓辨識 ------------------
@app.post("/predict")
async def predict(file: UploadFile = File(...), decoded=Depends(verify_token)):
    if knn is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    try:
        raw = await file.read()
        img = Image.open(io.BytesIO(raw)).convert("RGB")
        img = np.array(img)[:, :, ::-1]  # RGB -> BGR

        H, W = img.shape[:2]
        faces = detect_cat_faces(img)
        boxes = []

        for (x, y, w, h) in faces:
            feat = face_to_feature(img, (x, y, w, h)).reshape(1, -1)
            pred = knn.predict(feat)[0]
            distances, _ = knn.kneighbors(feat, n_neighbors=K, return_distance=True)
            proba = float(np.clip((1 - distances[0]).mean(), 0.0, 1.0))
            name = id2name.get(int(pred), "Unknown")
            if proba < UNKNOWN_THRESHOLD:
                name = "Unknown"
            boxes.append({"x": int(x),"y": int(y),"w": int(w),"h": int(h),"name": name,"proba": proba})
        return {"width": W, "height": H, "boxes": boxes}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# ------------------ 已知貓名 ------------------
@app.get("/labels")
def labels(decoded=Depends(verify_token)):
    return {"count": len(id2name), "labels": [id2name[i] for i in sorted(id2name.keys())]}

# ------------------ 留言板 ------------------
@app.get("/comments")
def get_comments(cat_name: str, decoded=Depends(verify_token)):
    return {"cat": cat_name, "comments": comments_db.get(cat_name, [])}

@app.post("/comment")
def post_comment(cat_name: str, payload: dict, decoded=Depends(verify_token)):
    text = payload.get("text", "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="Empty comment")
    user_email = decoded.get("email", "unknown")
    comments_db.setdefault(cat_name, []).append({"user": user_email, "text": text})
    return {"cat": cat_name, "comments": comments_db[cat_name]}
