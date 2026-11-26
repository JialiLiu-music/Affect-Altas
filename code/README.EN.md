# Music Emotion Recognition — CPU-Based Audio Upload Demo
[中文](README.CN.md) | [English](README.EN.md)

This directory contains the decoupled frontend and backend implementation of the research companion demo. **The system operates exclusively on CPUs and does not return simulated prediction scores.**

The prediction task estimates the overall **Valence** and **Arousal** of an audio clip. It is not intended to provide second-by-second emotional trajectories, emotional diagnoses, or prediction accuracy measurements.

## Features and Limitations

- **Frontend:** A Chinese-language interface supporting audio selection, local playback, file uploads, task progress tracking, overall two-dimensional emotion visualization, and exploratory temporal curves. It also supports segment-level details, playback timeline synchronization, and JSON/CSV export.
- **FastAPI Backend:** Supports WAV, MP3, FLAC, and OGG audio files up to 20 MB, with durations ranging from 1 to 90 seconds. Audio clips are processed in full without automatic truncation. M4A files must be converted to a supported format before uploading.
- **CPU-only inference:** A single CPU worker processes tasks sequentially. Once initialized, the model remains loaded within the same inference subprocess. A maximum of three tasks may wait in the queue. Each task has a timeout of 20 minutes, including audio decoding and model loading. The subprocess is terminated if this limit is exceeded.
- **Temporary storage:** Uploaded audio files are stored in a temporary directory and deleted upon successful completion, failure, or service shutdown. Prediction results are retained in memory only and expire one hour after completion. Original uploaded filenames are not retained.
- **Calibration:** Overall predictions are uncalibrated by default (`calibration: none`). The PMEmo adaptation-pool offsets reported in the paper are not applied to arbitrary uploaded music. Valid calibration requires ratings from the target dataset.
- **Exploratory temporal windows:** The system uses fixed 10-second windows with a 5-second hop size. Audio clips shorter than 10 seconds produce a single full-length window. For longer clips, a final window is added to cover the remaining audio. Valence and Arousal curves are positioned according to window center timestamps, with detailed start and end times available for inspection.
- **Feature extraction:** MERT extracts features from the entire audio clip only once. Temporal windows reuse the existing 1 Hz features, and the static prediction head operates on temporally averaged window features. This approach does not independently crop and re-encode each audio window, nor does it constitute a dynamic emotion model. Its outputs should not be interpreted as second-by-second emotional ground truth. Since these windows are shorter than the original 45-second training clips, their local prediction accuracy remains unverified.
- **Raw temporal predictions:** Curves are neither smoothed nor automatically calibrated. Original window predictions are preserved in JSON/CSV exports. Adjacent windows overlap and should not be treated as statistically independent prediction samples.
- **Visualization:** Coordinates are clipped to [-1, 1] for frontend visualization only. Raw model outputs are preserved. Prediction scores are not probabilities and do not represent universally agreed emotional responses across listeners.
- **Local deployment:** The service binds to `127.0.0.1` by default. It is currently a local research demonstration rather than a hardened public anonymous upload service.

## Current Implementation Status

As of October 12, 2026, CPU Torch 2.6.0+cpu, Transformers 4.38.2, and the required audio-processing dependencies have been installed. The pinned MERT revision and the `mean_mlp` prediction head used in the paper are available.

The SHA checksum of the prediction head matches the original experimental records, and all 211 MERT weight tensors have passed verification.

The health endpoint reports `ready`. After the first inference task completes, the model status becomes `loaded=true`, allowing real audio files to be processed.

Through the actual HTTP API, generated test audio clips of 3 seconds and 45 seconds were successfully processed using the real MERT model and the prediction head from the paper. Uploaded files were subsequently removed.

| Test Audio Duration | CPU Processing Time |
| --- | --- |
| 3 seconds | Approximately 3.473 seconds |
| 45 seconds | Approximately 8.761 seconds |

These tests demonstrate the functionality of the inference pipeline only. They do not constitute an evaluation of music emotion recognition accuracy or a performance guarantee across different audio inputs and hardware configurations.

Browser-based visual, keyboard accessibility, and playback acceptance testing has not yet been performed.

## Directory Structure

- `backend/app/`: CPU inference model, fixed feature specifications, task queue, upload constraints, and API endpoints.
- `backend/scripts/prepare_models.py`: Downloads the public MERT model at a pinned revision and verifies the existing prediction head.
- `backend/tests/`: API boundary tests and real audio decoding tests.
- `frontend/`: React/TypeScript Chinese-language interface and shared components.
- `models/`: Local model directory, excluded from Git.
- `.runtime/uploads/`: Temporary audio storage, excluded from Git.

## 1. Start the Backend

**Environment:** Windows PowerShell, Python 3.10–3.12.

```powershell
cd E:\project\test\ljl\lunwen2\code\backend
python -m venv .venv
.\.venv\Scripts\python -m pip install --index-url https://pypi.org/simple -r requirements-api.txt
.\.venv\Scripts\python -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --workers 1
```

Run only one API worker to avoid ownership conflicts involving the task queue and temporary upload directory.

- API documentation: http://127.0.0.1:8000/docs
- Health endpoint: http://127.0.0.1:8000/api/health

API dependencies are separated from inference dependencies so that service availability and model asset status can still be reported clearly when model weights are unavailable.

## 2. Prepare CPU Inference Dependencies and Model Weights

Stop the backend service before installing the following dependencies in the `backend` directory:

```powershell
.\.venv\Scripts\python -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python -m pip install --index-url https://pypi.org/simple -r requirements.txt
```

The system uses the CPU version of PyTorch and explicitly disables CUDA within the inference worker.

MERT runs through custom model code associated with a pinned revision. The preparation script records the model source and SHA checksums of individual files in a manifest. At runtime, the model is loaded offline, and all 211 weight tensors are verified.

### Prediction Head — Original Research Weights Required

Retrieve the following file from the original experimental server and save it to:

`code/models/mean_mlp-seed20261011.pt`

Do not train or generate a randomly initialized replacement prediction head.

- **Server-relative path:** `runs/static-emotion-v2-20261008/mean_mlp-seed20261011/best.pt`
- **Source project:** `/root/autodl-tmp/music-emotion-rebuild-20261008/`
- **SHA256:** `18a8f83c9ed879a6c214f8f21332a048dcf9196c2d8766f89b4cf35e6f3c469d`

The demonstration uses the predefined seed `20261011`. Seeds are not selected according to uploaded audio or prediction outcomes.

The 13.50% improvement reported in the paper was obtained by averaging across three random seeds and adaptation pools. The default source-domain predictions in this demonstration do not reproduce that calibration improvement.

### Pinned MERT Model

```powershell
$env:HF_ENDPOINT='https://hf-mirror.com'
.\.venv\Scripts\python scripts\prepare_models.py --model-dir ..\models\mert --head-path ..\models\mean_mlp-seed20261011.pt
```

**Model specifications:**

| Parameter | Value |
| --- | --- |
| Backbone | `m-a-p/MERT-v1-95M` |
| Revision | `12af15fef9d0ac838c3f475bfbbf26d2060dd4f5` |
| Execution device | CPU |
| Prediction head | `mean_mlp` |
| Random seed | `20261011` |

The preparation script verifies the prediction head before downloading the backbone. Smaller networks from alternative sources are not accepted as substitutes.

Alternatively, the environment variables `MER_MODEL_DIR` and `MER_HEAD_PATH` can be used to specify previously prepared model locations.

Existing model snapshots must be prepared and verified against their source and file manifest. Unverified weights must not be presented as the original research model.

After restarting the API, the model asset status becomes available. During the first inference task, the weights are loaded and verified again. If any tensor fails verification, the task explicitly fails rather than returning predictions from randomly initialized weights.

The `.env.example` file is provided for reference only. The backend reads variables from the process environment and does not automatically load `.env` files.

### Backbone-Only Preparation

If the original experimental server is temporarily unavailable, the public backbone can be prepared independently:

```powershell
$env:HF_ENDPOINT='https://hf-mirror.com'
.\.venv\Scripts\python scripts\prepare_models.py --model-dir ..\models\mert --backbone-only --transport http
```

The HTTP transport option is intended for mirrors that do not provide the HEAD metadata required by the Hugging Face Hub SDK.

The script verifies the pinned commit before checking the LFS SHA256 checksum of the model weights and the Git blob SHA1 checksums of the source code and configuration files. Interrupted downloads can be resumed.

The `--backbone-only` option neither generates nor replaces the trained prediction head and does not enable complete emotion prediction.

If a valid manifest already exists, downloading the backbone again is unnecessary. Only the prediction head with the matching SHA checksum needs to be supplied.

The script `backend/scripts/fetch_source_head.py` can retrieve the trained prediction head from the original experimental server.

SSH passwords are read exclusively from the `MER_SSH_PASSWORD` process environment variable. The SSH identity must match the original host-key records in the workspace.

If the server is unreachable, the original instance must be restored or valid replacement connection information must be provided.

The `verify_backbone_cpu.py` script independently verifies the 211 downloaded backbone tensors and performs a real CPU forward pass. Without the trained prediction head, however, this does not constitute a complete music emotion recognition pipeline.

## 3. Run the Frontend

**Requirement:** Node.js 22.12 or later.

Open another terminal and execute:

```powershell
cd E:\project\test\ljl\lunwen2\code\frontend
npm ci
npm run dev
```

Open the following address in a browser:

http://localhost:5173

For additional configuration details, refer to `frontend/README.md`.

The default backend API address is `http://127.0.0.1:8000`, and the frontend development server uses port `5173`.

The frontend and backend run in separate terminals.

If custom ports are used, both the frontend `VITE_API_BASE_URL` and backend `MER_CORS_ORIGINS` environment variables must be updated accordingly.

## API Specification

### `GET /api/health`

Returns information about:

- CPU execution status
- Dependency and model asset availability
- Model loading status
- Task queue
- Service constraints

The `ready` status indicates that preliminary asset verification has succeeded. Errors may still occur during the initial full model loading process.

Successful task completion should be regarded as evidence of actual inference execution.

### `POST /api/jobs`

Accepts an audio file through the multipart field `file`.

Returns a job ID and an access token.

| HTTP Status | Description |
| --- | --- |
| 503 | Required model assets are unavailable |
| 413 | Uploaded file exceeds the size limit |
| 415 | Unsupported audio format |
| 429 | Task queue is full |

### `GET /api/jobs/{job_id}`

Requires an access token through the `X-Job-Token` request header.

Returns the task stage, progress, raw Valence and Arousal predictions, and related results.

Access tokens must not be included in URLs or logs. Missing or invalid tokens both result in an HTTP 404 response.

Progress represents the proportion of completed processing windows rather than an estimated time of arrival (ETA).

CPU processing time is reported only after an actual inference task has completed.

## Testing and Verification

```powershell
cd E:\project\test\ljl\lunwen2\code\backend
.\.venv\Scripts\python -m pytest -q
```

Audio tests require NumPy, SoundFile, and librosa. API tests do not require PyTorch.

API tests use explicitly designated test workers and do not evaluate MERT prediction accuracy.

Real audio tests cover:

- Full-length audio decoding
- Channel averaging
- Audio resampling
- Rejection of audio exceeding the duration limit
- Rejection of corrupted audio
- Rejection of audio containing non-finite values

### Feature Extraction Specifications

The feature extraction implementation follows the methodology described in the paper.

| Parameter | Configuration |
| --- | --- |
| Sampling rate | 24 kHz |
| Audio channels | Mono |
| MERT layer aggregation | Average of the last four layers |
| Window duration | 10 seconds |
| Hop size | 5 seconds |
| Temporal feature resolution | 1 Hz |
| Temporal aggregation | Float32 temporal mean |

An additional temporal bin is not created when the remaining audio duration is shorter than 0.5 seconds, consistent with the original feature extractor.

Audio inputs are not automatically truncated to the first 45 seconds.

## Public Deployment Considerations

The project is not automatically deployed or pushed to a remote repository.

Before making the demonstration publicly accessible, additional security and infrastructure measures are required, including:

- HTTPS reverse proxy
- Strict upload rate limiting
- Authentication and usage quotas
- Resource monitoring
- Privacy notices and data-handling policies

The `MER_API_TOKEN` environment variable can be configured to require Bearer authentication.

However, tokens embedded in a publicly distributed frontend cannot remain confidential. Authentication should instead be handled by an authenticated backend or through user-provided credentials.

Shared secrets must not be distributed through bundled `VITE` environment variables.

The research demonstration is accessible only from the local machine by default. Do not expose the API directly to the public internet without appropriate security protections.

## Responsible Use

Users are responsible for ensuring that they have the necessary rights and permissions to upload audio content, including compliance with copyright and privacy requirements.

The server processes uploaded audio temporarily.

Model predictions are intended for music information retrieval research and exploratory analysis. They must not be used for medical purposes or individual psychological diagnosis.
