# Live Emotion Classifier

A CNN-based facial emotion classifier, trained from scratch in PyTorch on the FER2013 dataset, served via a FastAPI backend, containerized with Docker, and deployed live with a real-time webcam demo.

**Live demo:** https://emotion-classifier-5qcn.onrender.com

---

## Project structure
<pre>
  Emotion_Classifier/
├── app.py                 # FastAPI server (routes: /, /predict)
├── inference.py           # Model loading, face detection, preprocessing, prediction
├── model.py               # CNN architecture (ConvBlock, EmotionCNN)
├── train.py               # Training loop (GPU, LR scheduling, checkpointing)
├── dataset.py             # FER2013 loading & PyTorch Dataset/DataLoader setup
├── webcam.html            # Live webcam demo frontend
├── emotion_cnn.pt         # Trained model weights
├── Dockerfile             # Container setup for deployment
├── .dockerignore          # Files excluded from Docker build
├── requirements.txt       # Python dependencies
└── README.md              # Project documentation
</pre>

---

## What it does

Upload a photo, or use your webcam, and the model predicts your facial expression in real time across 7 categories: **Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral**.

---

## Architecture

A CNN built from scratch (no pretrained backbone, no transfer learning):

| File | Description |
|------|-------------|
| `model.py` | CNN architecture — `ConvBlock` and `EmotionCNN` classes |
| `dataset.py` | FER2013 dataset loading with PyTorch `Dataset` and `DataLoader` |
| `train.py` | Training loop with GPU support, learning rate scheduling, and checkpointing |
| `inference.py` | Model loading, face detection, image preprocessing, prediction logic |
| `emotion_cnn.pt` | Trained model weights |
| `app.py` | FastAPI server with routes `/` (home) and `/predict` (inference) |
| `Dockerfile` | Container setup for deployment |
| `webcam.html` | Live webcam demo frontend |
| `requirements.txt` | Python dependencies |

---

## Training

- **Dataset:** [FER2013](https://www.kaggle.com/datasets/msambare/fer2013) — ~35,000 labeled 48×48 grayscale facial images
- **Framework:** PyTorch, trained on a local NVIDIA RTX 3050 (GPU)
- **Optimizer:** Adam, with a `StepLR` learning rate scheduler (halving every 5 epochs) to stabilize convergence in later epochs
- **Regularization:** Dropout + data augmentation (random horizontal flip, random rotation) to reduce overfitting on this relatively small, imbalanced dataset
- **Result:** ~60% validation accuracy — a solid result from a from-scratch CNN, given FER2013 is a notoriously hard dataset (human agreement on it is often cited around 65-70%, since many labels are genuinely ambiguous)

---

## A key lesson: train/inference distribution mismatch

Early webcam testing performed noticeably worse than validation accuracy suggested. The reason: **FER2013's images are tightly cropped, face-filling photos** — but a raw webcam frame includes background, shoulders, and room context, with the face occupying only a small portion of the frame. Feeding that whole frame into a 48×48 input meant the model was seeing something very different from anything in training.

**Fix:** added a face-detection preprocessing step (OpenCV's Haar Cascade classifier) that detects and crops to just the face *before* the existing grayscale/resize pipeline — aligning live inference input with the training distribution. This produced a substantial, immediately noticeable accuracy improvement in the live demo.

This ended up being a more impactful fix than any architecture or hyperparameter change — real-world deployment issues are often about **matching preprocessing between training and inference**, not just model quality.

---

## Serving & deployment

- **FastAPI** backend exposing a `/predict` endpoint (accepts an image upload, returns predicted emotion + confidence) and a root route serving the live webcam demo page
- **Docker** — containerized for consistent, portable deployment; CPU-only inference (fast enough for real-time use without needing a GPU in production)
- **Deployed on Render** (free tier) via direct `Dockerfile`-based deployment, with automatic redeploys on every push to `main`

### Live webcam demo

The frontend (`webcam.html`) captures a frame from the browser's webcam roughly once per second, sends it to `/predict`, and displays the result with a smooth, color-coded UI — no page reloads, no manual uploads needed.

---

## Running locally

```bash
pip install -r requirements.txt
uvicorn app:app --reload
```
Visit `http://localhost:8000/` for the live webcam demo, or `http://localhost:8000/docs` for the interactive API explorer.

## Running with Docker

```bash
docker build -t emotion-classifier .
docker run -p 8000:8000 emotion-classifier
```

---

## Known limitations

- **Render's free tier spins down after inactivity** — the first request after idling can take 30-60 seconds to "wake up"
- **Single-face assumption** — if multiple faces appear in frame, only the largest (closest) detected face is classified
- **FER2013's known label noise** — some images are genuinely ambiguous even to human raters, capping achievable accuracy regardless of model improvements

## Possible future improvements

- Transfer learning with a pretrained backbone for a further accuracy boost
- Class-weighted loss to address FER2013's class imbalance (e.g. "Disgust" is underrepresented)
- Return top-3 predictions instead of just the top-1, for more nuanced feedback
- CI/CD pipeline (GitHub Actions) to run tests before each deploy
