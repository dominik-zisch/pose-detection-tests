# Pose Detection Webcam Tests

Simple Python scripts that show **live pose detection** from your webcam. They draw a skeleton on your body and show landmark details in a side panel.

There are two scripts:

| Script | What it uses |
|--------|----------------|
| `test_yolo11_pose_webcam.py` | YOLO11 pose (Ultralytics) — 17 body keypoints, can use your NVIDIA GPU |
| `test_mediapipe_pose_webcam.py` | MediaPipe Pose Landmarker — 33 body landmarks, runs on CPU on Windows |

Both scripts:

- Open your webcam (camera index 0)
- Show the video with pose drawn on it
- Show FPS on the video
- Show a side panel with joint names, IDs, visibility, and coordinates
- Quit when you press **`q`**

---

## What you need

- **Python 3.10 or newer** — download from [python.org](https://www.python.org/downloads/)
  - On Windows, check **“Add Python to PATH”** during install
- A **webcam**
- For the YOLO script with GPU speed: an **NVIDIA GPU** and up-to-date drivers (optional — CPU works too, just slower)

---

## One-time setup

You only do this **once** per project folder.

### 1. Open a terminal in this folder

- **Windows:** open the folder in File Explorer, click the address bar, type `cmd`, press Enter  
  Or in VS Code / Cursor: **Terminal → New Terminal**

### 2. Create a virtual environment (venv)

A **venv** is a private copy of Python packages for this project. It keeps things tidy and avoids breaking other Python projects on your PC.

**You only need to create it once.** After that, you just activate it when you want to run the scripts.

```cmd
python -m venv .venv
```

This creates a folder called `.venv` in the project. You can ignore that folder — it’s just for Python packages.

### 3. Activate the venv

**Every time** you open a new terminal to run these scripts, activate the venv first:

**Windows (Command Prompt):**

```cmd
.venv\Scripts\activate
```

**Windows (PowerShell):**

```powershell
.venv\Scripts\Activate.ps1
```

**Mac / Linux:**

```bash
source .venv/bin/activate
```

When it worked, you’ll see `(.venv)` at the start of your command line, for example:

```text
(.venv) C:\Users\You\pose-detection-tests>
```

### 4. Install the packages

Still with the venv activated:

```cmd
pip install -r requirements.txt
```

#### YOLO only: install PyTorch with GPU support (recommended on NVIDIA)

If you have an NVIDIA GPU (e.g. RTX 3060), install PyTorch with CUDA **before** or **after** the step above — do this once:

```cmd
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```

You don’t need this for the MediaPipe script.

---

## Running the scripts

Each time you want to run a script:

1. Open a terminal in this folder  
   - **Windows:** open the folder in File Explorer, click the address bar, type `cmd`, press Enter  
   - Or in VS Code / Cursor: **Terminal → New Terminal**

2. Activate the venv (you only created it once — you just activate it each time):

   **Windows (Command Prompt):**

   ```cmd
   .venv\Scripts\activate
   ```

   **Windows (PowerShell):**

   ```powershell
   .venv\Scripts\Activate.ps1
   ```

   **Mac / Linux:**

   ```bash
   source .venv/bin/activate
   ```

   You should see `(.venv)` at the start of your command line.

3. Run one of the scripts:

   **YOLO11 pose:**

   ```cmd
   python test_yolo11_pose_webcam.py
   ```

   **MediaPipe pose:**

   ```cmd
   python test_mediapipe_pose_webcam.py
   ```

The first run may download a model file (YOLO or MediaPipe). That’s normal.

Press **`q`** in the video window to quit.

---

## If the venv already exists

You **do not** create it again. Just:

1. Open a terminal in this folder  
2. Activate it:

   ```cmd
   .venv\Scripts\activate
   ```

3. Run a script:

   ```cmd
   python test_yolo11_pose_webcam.py
   ```

---

## Troubleshooting

**Webcam won’t open**

- Close other apps using the camera (Zoom, Teams, etc.)
- Make sure no other script is already using the camera

**`(.venv)` doesn’t appear after activate**

- Make sure you’re in the project folder (the one that contains the `.venv` folder)
- On PowerShell, you may need: `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` (once), then try activate again

**YOLO is slow**

- Install PyTorch with CUDA (see step 4 above)
- When the script starts, it should print `Using device: cuda` and your GPU name

**MediaPipe on Windows**

- Uses CPU only (MediaPipe GPU isn’t supported on Windows in Python). That’s expected; the lite model is still fast enough for a webcam.

**Packages missing / import errors**

- Activate the venv, then run: `pip install -r requirements.txt`

---

## Project files

| File | Purpose |
|------|---------|
| `test_yolo11_pose_webcam.py` | YOLO11 webcam pose test |
| `test_mediapipe_pose_webcam.py` | MediaPipe webcam pose test |
| `requirements.txt` | Python packages to install |
| `.venv/` | Virtual environment (created by you, not in git) |
