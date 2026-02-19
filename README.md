# Harmonia V3 — Complete Claude Code Build Plan

**Version:** 1.2 | **Date:** 19 February 2026  
**Scope:** Full system build — MetaFBP Visual Intelligence + PIIP Psychometric System + HLA Biological Compatibility + Matching Engine + Reporting + Evidence Traceability + Gemini Calibration Database + GCP Deployment + Comprehensive Documentation + Frontend Integration Spec  
**Stack:** Python 3.11 / FastAPI / PostgreSQL / Redis / PyTorch / Google Cloud Platform  
**Deployment Target:** Google Cloud Platform (Cloud Run + Cloud SQL + Memorystore + Cloud Storage + Cloud Build)  
**Repo:** New GitHub repo (greenfield build)

---

## Pre-Build: MetaFBP Files You Need From Your Local Training Machine

Before starting any Claude Code sessions, you need to transfer specific files from the computer where you trained MetaFBP using the MetaVisionLab/MetaFBP repository. The training process produces two checkpoint files through a two-stage pipeline, and both are required for production inference.

### What MetaFBP Training Produces

MetaFBP uses a two-stage training architecture. Stage 1 trains a ResNet-18 feature extractor (the "Universal Extractor" that captures objective beauty commonality). Stage 2 trains the meta-learning components — a Predictor (single FC layer, 512→1) and a Parameter Generator MLP (FC→ReLU→FC, 512→512) — which together enable per-user personalisation.

### Files to Copy from Your USB / Local Machine

**Required for production inference:**

| File | What It Is | Expected Size | Where It Comes From |
|------|-----------|---------------|-------------------|
| Stage 1 checkpoint (`.pth`) | ResNet-18 state_dict — the frozen feature extractor | ~44 MB | Output of `train_fea.py` (or `train_fea.sh`) |
| Stage 2 checkpoint (`.pth`) | Predictor FC weights + Meta-generator MLP weights | ~1–5 MB | Output of `train.py` (or `train.sh`) |
| `model/meta.py` | Meta-learner class definition | Small | From the MetaFBP repo |
| `model/learner.py` | Learner architecture (layer definitions) | Small | From the MetaFBP repo |
| `model/__init__.py` | Module init | Small | From the MetaFBP repo |
| `data/` directory | Data loading and preprocessing utilities | Small | From the MetaFBP repo |
| `util/` directory | Utility functions used during inference | Small | From the MetaFBP repo |
| `test.py` | Inference entry point (reference for how to load and run the model) | Small | From the MetaFBP repo |

**Not needed for production (can leave behind):**
`train.py`, `train_fea.py`, `train_fea.sh`, `train.sh`, `test_fea.py`, raw dataset folders (SCUT-FBP5500, US10K, etc.), optimizer state dicts, TensorBoard logs.

### How to Identify Your Checkpoint Files

Look in the directory where you ran training. The MetaFBP codebase (built on dragen1860/MAML-Pytorch) typically saves checkpoints as `.pth` files. The Stage 1 checkpoint will be the larger file (~44 MB) because it contains the full ResNet-18 weights. The Stage 2 checkpoint will be smaller (~1–5 MB) because it only contains the predictor and generator MLP weights.

The model stores parameters in `nn.ParameterList` objects called `self.vars` (trainable) and `self.vars_bn` (batch norm running statistics). Your checkpoint keys will follow patterns like `vars.0`, `vars.1`, ..., `vars_bn.0`, `vars_bn.1`, etc.

### Critical Architecture Detail: MetaFBP Requires a User Calibration Step

MetaFBP cannot score a single face image in isolation. The meta-learning approach requires each user to first provide a "support set" — a small collection of face images with their personal beauty ratings (on a 1–5 scale). The model then performs k=10 inner-loop gradient adaptation steps on this support set to personalise the predictor weights before it can score any new (query) faces.

This means the Harmonia onboarding flow must include a calibration phase where users swipe/rate sample face images. The adapted predictor weights are then cached per user (in Redis) and reused for all subsequent candidate scoring. This is implemented in Session 4.

### Image Preprocessing Specification

All images fed to the model must use standard ImageNet normalisation: `mean=[0.485, 0.456, 0.406]`, `std=[0.229, 0.224, 0.225]`, resized to 224×224 pixels. At inference time, use center crop (training uses random crop + horizontal flip, but inference uses only center crop).

---

## Session 0: Repository Scaffolding & Database Foundation

**Goal:** Create the GitHub repo, establish the FastAPI project structure, set up PostgreSQL schema, and configure the development environment. This session builds the skeleton that every subsequent session depends on.

**Context for Claude Code:**
You are building the Harmonia V3 dating/matching engine from scratch. The system has three phases: Phase 1 (MetaFBP visual intelligence — neural beauty prediction personalised per user), Phase 2 (PIIP psychometric system — personality profiling via scenario-based questions parsed by Gemini AI into Seven Deadly Sins scores), and Phase 3 (HLA biological compatibility — genetic compatibility via MHC allele analysis). These phases feed into a matching engine that calculates Willingness to Meet (WtM) scores and generates multi-level reports.

The backend is Python 3.11 with FastAPI. Database is PostgreSQL (Cloud SQL). Redis is used for caching (Memorystore for Redis). Deployment target is Google Cloud Platform (Cloud Run).

### Tasks

1. **Initialise the GitHub repository** with a clean `.gitignore` (Python, Node, `.env`, `__pycache__`, `.pth` files, uploaded media) and MIT license.

2. **Create the project directory structure** matching the V3 spec (Section 2.1):

```
harmonia-v3/
├── app/
│   ├── __init__.py
│   ├── main.py                    # FastAPI application entry point
│   ├── config.py                  # Environment configuration (Pydantic Settings)
│   ├── database.py                # SQLAlchemy async engine + session factory
│   ├── models/                    # SQLAlchemy ORM models
│   │   ├── __init__.py
│   │   ├── user.py                # User model
│   │   ├── profile.py             # Personality profile model (PIIP)
│   │   ├── visual.py              # Visual preference model (MetaFBP)
│   │   ├── hla.py                 # HLA genetic data model
│   │   ├── match.py               # Match model + reasoning chain
│   │   └── questionnaire.py       # Question responses model
│   ├── schemas/                   # Pydantic request/response schemas
│   │   ├── __init__.py
│   │   ├── user.py
│   │   ├── questionnaire.py
│   │   ├── profile.py
│   │   ├── match.py
│   │   └── report.py
│   ├── api/                       # FastAPI route handlers
│   │   ├── __init__.py
│   │   ├── router.py              # Main API router
│   │   ├── users.py               # User CRUD endpoints
│   │   ├── questionnaire.py       # PIIP question submission
│   │   ├── visual.py              # MetaFBP calibration + scoring
│   │   ├── hla.py                 # HLA data upload + scoring
│   │   ├── matching.py            # Match calculation + retrieval
│   │   ├── reports.py             # Report generation endpoints
│   │   └── admin/                 # Admin-only endpoints
│   │       ├── __init__.py
│   │       └── calibration.py     # Review queue, approve/correct/reject calibration examples
│   ├── services/                  # Business logic layer
│   │   ├── __init__.py
│   │   ├── gemini_service.py      # Gemini API integration for PIIP parsing
│   │   ├── profile_service.py     # Profile aggregation pipeline
│   │   ├── similarity_service.py  # Perceived similarity calculation
│   │   ├── visual_service.py      # MetaFBP inference wrapper
│   │   ├── hla_service.py         # HLA scoring + olfactory prediction
│   │   ├── matching_service.py    # Three-stage cascaded matching
│   │   ├── report_service.py      # Multi-level report generation
│   │   └── calibration_service.py # Calibration DB: few-shot retrieval + admin review
│   ├── ml/                        # Machine learning model code
│   │   ├── __init__.py
│   │   ├── metafbp/               # MetaFBP model code (from your training)
│   │   │   ├── __init__.py
│   │   │   ├── meta.py            # Meta-learner class (from MetaVisionLab repo)
│   │   │   ├── learner.py         # Learner architecture (from MetaVisionLab repo)
│   │   │   ├── inference.py       # Production inference wrapper
│   │   │   └── preprocessing.py   # Image preprocessing (ImageNet normalisation)
│   │   └── trait_extraction/      # Visual trait extraction (glasses, hair, etc.)
│   │       ├── __init__.py
│   │       └── extractor.py
│   └── utils/                     # Shared utilities
│       ├── __init__.py
│       ├── encryption.py          # Fernet encryption for HLA data
│       ├── logging.py             # Structured JSON logging
│       └── storage.py             # GCS integration: upload, download, signed URLs, integrity checks
├── scripts/                       # Standalone scripts
│   ├── cluster_generator.py       # PIIP cluster generation agent (Claude Haiku)
│   ├── calibration_manager.py     # Calibration DB management (coverage, stats, batch ops)
│   └── seed_questions.py          # Seed the 6 Felix questions into DB
├── models/                        # PyTorch model weights (gitignored)
│   ├── universal_extractor.pth    # Stage 1 checkpoint (~44MB)
│   └── meta_generator.pth         # Stage 2 checkpoint (~1-5MB)
├── alembic/                       # Database migrations
│   ├── env.py
│   └── versions/
├── infra/                         # Google Cloud infrastructure
│   ├── cloudbuild.yaml            # Cloud Build CI/CD pipeline
│   ├── cloud-run-service.yaml     # Cloud Run service definition (knative)
│   ├── setup-gcp.sh               # One-shot GCP project bootstrap script
│   └── terraform/                 # Optional: Terraform IaC for all GCP resources
│       ├── main.tf
│       ├── variables.tf
│       └── outputs.tf
├── alembic.ini
├── requirements.txt
├── Dockerfile
├── .dockerignore
├── .gcloudignore
├── .env.example
├── .gitignore
└── README.md
```

3. **Create the database models** (SQLAlchemy with async support). The key tables are:

**users** — `id` (UUID, PK), `email`, `display_name`, `age`, `gender`, `location`, `photos` (JSON array of URLs), `created_at`, `updated_at`, `is_active`.

**questionnaire_responses** — `id` (UUID, PK), `user_id` (FK→users), `question_number` (1-6), `question_text`, `response_text`, `word_count`, `created_at`. Unique constraint on (user_id, question_number).

**personality_profiles** — `id` (UUID, PK), `user_id` (FK→users, unique), `version` (int), `sins` (JSONB — the 7 aggregated sin scores with confidence, variance, evidence), `quality_score` (float 0-100), `quality_tier` (enum: high/moderate/low/rejected), `response_styles` (JSONB), `flags` (JSON array), `metadata` (JSONB), `source` (enum: real_user/claude_agent), `created_at`, `updated_at`.

**visual_preferences** — `id` (UUID, PK), `user_id` (FK→users, unique), `support_set_stats` (JSONB — total_samples, class_distribution), `mandatory_traits` (JSONB array), `preferred_traits` (JSONB array), `aversion_traits` (JSONB array), `adapted_weights_key` (string — Redis cache key for adapted generator weights), `created_at`, `updated_at`.

**visual_ratings** — `id` (UUID, PK), `user_id` (FK→users), `image_id` (string), `image_path` (string), `rating` (int 1-5), `created_at`. Index on (user_id, rating).

**hla_data** — `id` (UUID, PK), `user_id` (FK→users, unique), `encrypted_data` (binary — Fernet encrypted), `source` (string, e.g. "23andMe_v5"), `imputation_confidence` (float), `ancestry_model` (string), `snp_count` (int), `created_at`.

**matches** — `id` (UUID, PK), `user_a_id` (FK→users), `user_b_id` (FK→users), `s_vis_a_to_b` (float), `s_vis_b_to_a` (float), `s_psych` (float), `s_bio` (float, nullable), `wtm_score` (float), `reasoning_chain` (JSONB — Level 3 full calculation), `customer_summary` (JSONB — Level 1 sanitised), `created_at`. Unique constraint on (user_a_id, user_b_id) with ordering.

**swipes** — `id` (UUID, PK), `swiper_id` (FK→users), `target_id` (FK→users), `direction` (enum: left/right/superlike), `created_at`. Unique constraint on (swiper_id, target_id).

**parsing_evidence** — `id` (UUID, PK), `user_id` (FK→users), `question_number` (int 1-6), `sin` (string, one of the 7 sins), `score` (float -5 to +5), `confidence` (float 0-1), `evidence_snippet` (text — the exact quoted fragment from the user's response that triggered the sin recognition), `snippet_start_index` (int — character offset in the response text where the snippet begins), `snippet_end_index` (int — character offset where it ends), `interpretation` (text — a one-line explanation of what the snippet reveals about the trait, generated by Gemini during parsing, e.g. "Conflict avoidance — prefers social harmony over addressing unfairness"), `observer_persona` (string, nullable — "neutral"/"empathetic"/"skeptical" if multi-observer was used), `gemini_model_used` (string — which model in the fallback chain produced this result), `created_at`. Compound index on (user_id, question_number, sin). This table is the granular audit trail: every sin recognition can be traced back to the exact words the user wrote, which question they were answering, and which Gemini model detected it. The evidence snippets flow into Level 3 and Level 2 reports but are never exposed in Level 1 customer-facing output.

**calibration_examples** — `id` (UUID, PK), `question_number` (int 1-6), `response_text` (text — the full response generated by Claude Haiku), `sin` (string), `gemini_raw_score` (float — what Gemini originally assigned), `gemini_raw_confidence` (float), `gemini_raw_evidence` (text), `validated_score` (float, nullable — the admin-corrected score, or NULL if not yet reviewed), `validated_by` (string, nullable — admin username), `validated_at` (timestamp, nullable), `review_status` (enum: pending/approved/corrected/rejected), `review_notes` (text, nullable — admin can explain why they corrected a score), `source_profile_id` (UUID, FK→personality_profiles, nullable — links to the synthetic profile this came from), `created_at`. Compound index on (question_number, sin, review_status). This table is the Gemini calibration database: Haiku generates responses → Gemini parses → admins review and validate → approved pairs become few-shot examples injected into future Gemini prompts to anchor scoring and reduce hallucination. Only rows with `review_status = 'approved'` or `review_status = 'corrected'` are used as few-shot examples.

4. **Create the config.py** using Pydantic Settings, loading from environment variables:

```python
# Key config values from the V3 spec:
GEMINI_API_KEY: str
GEMINI_MODEL_PRIMARY: str = "gemini-3-pro-preview"
GEMINI_MODEL_FALLBACK: str = "gemini-3-flash-preview"
GEMINI_MODEL_STABLE: str = "gemini-2.5-flash"
ANTHROPIC_API_KEY: str = ""  # For cluster generation
DATABASE_URL: str             # Cloud SQL via Unix socket or private IP
REDIS_URL: str                # Memorystore for Redis
VISUAL_WEIGHT: float = 0.4      # S_vis weight in WtM
PERSONALITY_WEIGHT: float = 0.3  # S_psych weight in WtM
HLA_WEIGHT: float = 0.3          # S_bio weight in WtM
ADAPTATION_STRENGTH_LAMBDA: float = 0.01
INNER_LOOP_STEPS: int = 5
INNER_LR: float = 0.01
FEATURE_DIM: int = 512
METAFBP_EXTRACTOR_PATH: str = "models/universal_extractor.pth"
METAFBP_GENERATOR_PATH: str = "models/meta_generator.pth"
FERNET_KEY: str  # For HLA encryption
ENVIRONMENT: str = "development"
LOG_LEVEL: str = "INFO"

# Google Cloud Platform
GCP_PROJECT_ID: str = ""
GCP_REGION: str = "europe-west2"           # London — closest to your location
GCS_BUCKET_NAME: str = ""                  # For user photos, model weights, HLA uploads
CLOUD_SQL_INSTANCE_CONNECTION: str = ""    # project:region:instance format
CLOUD_SQL_USE_UNIX_SOCKET: bool = True     # True for Cloud Run, False for local dev
GCS_MODEL_WEIGHTS_PREFIX: str = "models/"  # GCS path prefix for .pth files
```

5. **Create the FastAPI main.py** with lifespan management, CORS middleware, timeout middleware (70s), structured logging, health check endpoints (`/health` and `/health/deep`), graceful shutdown with active request tracking, and the main API router mounted at `/api/v1`.

6. **Create the Google Cloud deployment files:**
   - **`Dockerfile`** — Multi-stage build: Stage 1 installs Python dependencies into a virtual env, Stage 2 copies the venv + app code into a slim runtime image (python:3.11-slim). Must install `libpq5` for asyncpg and `libgl1` for OpenCV/torch vision ops. Expose `$PORT` (Cloud Run injects this). Use `gunicorn` with `uvicorn.workers.UvicornWorker` as the entrypoint (Cloud Run expects gunicorn-style process management), with `--timeout 120` to survive 60s Gemini calls, `--graceful-timeout 120` for drain, and `--keep-alive 75`.
   - **`.dockerignore`** — Exclude `.git`, `__pycache__`, `.env`, `*.pth`, `alembic/versions/*.pyc`, `tests/`, `.venv`, `node_modules`.
   - **`.gcloudignore`** — Same as `.dockerignore` plus `infra/`, `README.md`, `docs/`.
   - **`infra/cloudbuild.yaml`** — Cloud Build pipeline: build Docker image → push to Artifact Registry → deploy to Cloud Run with `--max-instances=10`, `--min-instances=1`, `--cpu=2`, `--memory=2Gi`, `--timeout=120`, `--concurrency=20`, `--set-cloudsql-instances=$CLOUD_SQL_INSTANCE`, `--set-secrets` for all sensitive env vars from Secret Manager. Include a step to run Alembic migrations before deploy.
   - **`infra/cloud-run-service.yaml`** — Knative service spec with startup probe at `/health` (10s interval, 3 failures), liveness probe at `/health` (30s interval), Cloud SQL sidecar connection, VPC connector for Memorystore access, and the 120s request timeout annotation (`run.googleapis.com/request-timeout: "120"`).
   - **`infra/setup-gcp.sh`** — One-shot bootstrap script that enables required APIs (`run`, `sqladmin`, `redis`, `cloudbuild`, `secretmanager`, `storage`, `artifactregistry`), creates the Artifact Registry Docker repo, provisions Cloud SQL PostgreSQL 15 instance (db-f1-micro for dev, db-custom-2-4096 for prod), creates the database + user, provisions Memorystore Redis (basic tier, 1GB), creates the GCS bucket with lifecycle rules, creates Secret Manager entries for all sensitive values (`GEMINI_API_KEY`, `ANTHROPIC_API_KEY`, `FERNET_KEY`, `DATABASE_URL`), and grants the Cloud Run service account the necessary IAM roles (`cloudsql.client`, `secretmanager.secretAccessor`, `storage.objectAdmin`).
   - **`.env.example`** — Template of all environment variables with descriptions and safe defaults.

7. **Create requirements.txt** with pinned versions:
```
# Core framework
fastapi==0.115.0
uvicorn[standard]==0.30.0
gunicorn==22.0.0
sqlalchemy[asyncio]==2.0.36
asyncpg==0.30.0
alembic==1.14.0
pydantic==2.9.0
pydantic-settings==2.6.0

# Caching
redis[hiredis]==5.2.0

# AI/ML services
google-generativeai==0.8.0
anthropic==0.40.0
torch==2.5.0
torchvision==0.20.0
Pillow==11.0.0

# Google Cloud Platform
google-cloud-storage==2.18.0       # GCS for photos, model weights, HLA uploads
google-cloud-secret-manager==2.21.0 # Secret Manager for production env vars
cloud-sql-python-connector[asyncpg]==1.12.0  # Cloud SQL Auth Proxy connector

# Security & encryption
cryptography==43.0.0

# Utilities
python-dotenv==1.0.1
structlog==24.4.0
tenacity==9.0.0
jsonrepair==0.1.0
httpx==0.28.0
python-multipart==0.0.12
```

8. **Set up Alembic** for database migrations with the initial migration creating all tables.

9. **Create the seed script** (`scripts/seed_questions.py`) that inserts the 6 Felix questions into a `questions` reference table so they're consistent across the system.

10. **Build a comprehensive `README.md`** that serves as the single source of truth for anyone onboarding to this codebase — whether they're deploying the backend, contributing code, or building the frontend. The README must be detailed, well-organised, and self-contained. Structure it as follows:

#### Section 1: Project Overview
A clear explanation of what Harmonia V3 is: an AI-powered dating/matching engine with three matching phases (Visual Intelligence via MetaFBP, Psychometric Profiling via PIIP/Gemini, Biological Compatibility via HLA analysis). Explain that the system calculates a Willingness to Meet (WtM) score from these three phases and generates multi-level reports. Mention the tech stack (Python 3.11, FastAPI, PostgreSQL, Redis, PyTorch, Gemini API, Claude Haiku API) and the deployment target (Google Cloud Platform).

#### Section 2: Complete File & Directory Reference
A detailed table/listing of EVERY file and directory in the project with a clear explanation of what each one does and why it exists. Group by directory. For example:

```
app/main.py — FastAPI application entry point. Defines the lifespan handler (startup: 
  loads MetaFBP weights from GCS, initialises DB connection pool, connects to Redis; 
  shutdown: drains active requests, closes connections). Mounts all API routers under 
  /api/v1, configures CORS, timeout middleware (70s), and structured logging.

app/config.py — Pydantic Settings class that loads all configuration from environment 
  variables. Contains every tunable constant in the system: Gemini model chain, MetaFBP 
  hyperparameters (λ=0.01, k=5, α=0.01), sin weights (Wrath 1.5×, Sloth 1.3×, etc.), 
  WtM phase weights (0.4/0.3/0.3), GCP project settings, and all API keys.

app/services/gemini_service.py — The core AI parsing engine. Takes a user's free-text 
  response to one of the 6 Felix questions and runs 7 parallel Gemini calls (one per sin) 
  with trait-specific prompts and few-shot examples from the calibration database. Stores 
  every sin recognition with the exact evidence snippet and character offsets in the 
  parsing_evidence table.
  
app/services/calibration_service.py — Manages the Gemini calibration database. Handles 
  ingestion of parsed results from synthetic profiles, the admin review workflow 
  (approve/correct/reject), and retrieval of validated few-shot examples for injection 
  into Gemini prompts. This is the self-improving loop that reduces Gemini hallucination 
  over time.

scripts/cluster_generator.py — Standalone script that calls Claude Haiku 4.5 to generate 
  diverse synthetic personality profiles (responses to the 6 Felix questions). Supports 
  both real-time API calls and the Batch API for high-volume generation. Each generated 
  profile is submitted to the PIIP pipeline for Gemini parsing and automatically enters 
  the calibration review queue.

infra/setup-gcp.sh — One-shot GCP project bootstrap. Run this ONCE to provision all 
  Google Cloud resources: Cloud SQL PostgreSQL, Memorystore Redis, Cloud Storage bucket, 
  Secret Manager secrets, Artifact Registry repo, VPC connector, service account + IAM 
  roles. Outputs all connection strings and next steps.

infra/cloudbuild.yaml — CI/CD pipeline triggered by GitHub pushes to main. Builds Docker 
  image, pushes to Artifact Registry, runs Alembic migrations, deploys new revision to 
  Cloud Run with zero downtime.
```

Include EVERY file — don't skip utilities, schemas, models, infra, or config files. The goal is that someone reading only the README can understand the purpose of every file without opening it.

#### Section 3: MetaFBP Training Files — What You Need & Where to Find Them
This section is critical. Explain clearly:

**What MetaFBP is**: A meta-learning facial beauty prediction model (ACM MM 2023 paper, arXiv:2311.13929) trained using the MetaVisionLab/MetaFBP repository (https://github.com/MetaVisionLab/MetaFBP), which is built on dragen1860/MAML-Pytorch.

**The two-stage training pipeline**:
- Stage 1 (Universal Feature Extractor): ResNet-18 trained on generic beauty datasets using mode rating labels. Produces a ~44 MB `.pth` checkpoint containing the frozen 512-dimensional feature extractor weights. This was trained by running `train_fea.py` (or `train_fea.sh`) in the MetaFBP repository.
- Stage 2 (Personalized Meta-Learner): Meta-learning trains a Predictor (FC layer, 512→1) and a Parameter Generator MLP (FC→ReLU→FC, 512→512). Produces a ~1-5 MB `.pth` checkpoint. This was trained by running `train.py` (or `train.sh`). Two variants exist: MetaFBP-R (parameter rebirth) and MetaFBP-T (parameter tuning, λ=0.01).

**Files to transfer from your local training machine (USB / local computer where you ran MetaFBP training)**:

| File | What It Is | Where to Find It | Where It Goes |
|------|-----------|-------------------|---------------|
| Stage 1 checkpoint (`.pth`) | ResNet-18 feature extractor weights | Output directory of `train_fea.py` — look for the largest `.pth` file (~44 MB) in your training output folder | Upload to GCS: `gs://{bucket}/models/universal_extractor.pth` AND keep a local copy at `models/universal_extractor.pth` for local dev |
| Stage 2 checkpoint (`.pth`) | Predictor + Parameter Generator weights | Output directory of `train.py` — smaller `.pth` file (~1-5 MB) | Upload to GCS: `gs://{bucket}/models/meta_generator.pth` AND keep a local copy at `models/meta_generator.pth` |
| `model/meta.py` | Meta-learner class definition (`Meta` class) | `MetaFBP/model/meta.py` in the cloned repo | Copy to `app/ml/metafbp/meta.py` |
| `model/learner.py` | Learner architecture (network layer definitions, `nn.ParameterList`) | `MetaFBP/model/learner.py` | Copy to `app/ml/metafbp/learner.py` |
| `data/` directory | Data loading and image preprocessing utilities | `MetaFBP/data/` in the cloned repo | Reference only — the preprocessing logic is reimplemented in `app/ml/metafbp/preprocessing.py` using the same ImageNet normalisation (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], 224×224, center crop) |
| `test.py` | Inference entry point (shows how to load and run the model) | `MetaFBP/test.py` in the cloned repo | Reference only — the inference logic is reimplemented in `app/ml/metafbp/inference.py` |

**Files you do NOT need**: `train.py`, `train_fea.py`, `train_fea.sh`, `train.sh`, `test_fea.py`, raw dataset folders (SCUT-FBP5500, US10K, etc.), optimizer state dicts, TensorBoard logs.

**How to identify your checkpoint files**: Look in the output directory where you ran training. MetaFBP (built on MAML-Pytorch) saves checkpoints as `.pth` files. The Stage 1 file is the larger one (~44 MB, full ResNet-18). The Stage 2 file is smaller (~1-5 MB, just the predictor + generator). Checkpoint keys follow patterns like `vars.0`, `vars.1`, ..., `vars_bn.0`, `vars_bn.1` (stored in `nn.ParameterList` objects called `self.vars` and `self.vars_bn`).

**How to upload to Google Cloud Storage**:
```bash
gsutil cp models/universal_extractor.pth gs://{YOUR_BUCKET}/models/universal_extractor.pth
gsutil cp models/meta_generator.pth gs://{YOUR_BUCKET}/models/meta_generator.pth
```

**Critical: MetaFBP requires a user calibration step.** The model cannot score a single face in isolation. Each user must first rate a set of sample face images (the "support set"). The model performs inner-loop gradient adaptation on those ratings to personalise the predictor weights. Only then can it score new faces. This is handled by the `/api/v1/visual/calibrate` endpoint.

#### Section 4: GCP Requirements & Prerequisites
Everything someone needs before running `infra/setup-gcp.sh`:

**Required tools (install locally)**:
- `gcloud` CLI (Google Cloud SDK) — install from https://cloud.google.com/sdk/docs/install, then run `gcloud init` and `gcloud auth login`
- `gsutil` (included with `gcloud` CLI) — for uploading model weights to Cloud Storage
- Docker Desktop or Docker Engine — for local development and building container images
- Python 3.11+ — for local development and running scripts
- Git — for version control

**GCP account requirements**:
- A Google Cloud project with billing enabled (free trial: $300 credit for 90 days)
- Owner or Editor role on the project (needed for `setup-gcp.sh` to provision resources)
- The following APIs must be enabled (the setup script does this automatically, but listing them here for reference): Cloud Run, Cloud SQL Admin, Memorystore for Redis, Cloud Build, Secret Manager, Cloud Storage, Artifact Registry, VPC Access

**API keys you need to obtain before deployment**:
- `GEMINI_API_KEY` — From Google AI Studio (https://aistudio.google.com/apikey). Free tier: 15 RPM for Gemini Pro, 60 RPM for Flash. For production: switch to Vertex AI Gemini for higher rate limits.
- `ANTHROPIC_API_KEY` — From Anthropic Console (https://console.anthropic.com). Used only for the cluster generation agent (Claude Haiku 4.5). Pricing: $0.50/MTok input, $2.50/MTok output at Batch API rates.
- `FERNET_KEY` — Auto-generated by the setup script using `python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"`. Used for HLA genetic data encryption.

**Estimated GCP costs** (development with 1,000 users/day):
| Service | Spec | Monthly Cost |
|---------|------|--------------|
| Cloud Run | 2 vCPU, 2GB RAM, min 1 instance | ~$50-70 |
| Cloud SQL | db-f1-micro (dev) | ~$8 |
| Memorystore Redis | Basic 1GB | ~$35 |
| Cloud Storage | ~50GB | ~$1-2 |
| Gemini API | ~42K calls/day | ~$5-15 |
| Total (dev) | | ~$100-130 |

#### Section 5: Local Development Setup
Step-by-step instructions for running the system locally:
```bash
# 1. Clone the repo
git clone https://github.com/{your-org}/harmonia-v3.git
cd harmonia-v3

# 2. Create virtual environment
python3.11 -m venv .venv
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set up local PostgreSQL and Redis
# (Docker compose or local installs)
docker run -d --name harmonia-pg -e POSTGRES_DB=harmonia -e POSTGRES_PASSWORD=dev -p 5432:5432 postgres:15
docker run -d --name harmonia-redis -p 6379:6379 redis:7-alpine

# 5. Copy .env.example to .env and fill in your API keys
cp .env.example .env
# Edit .env: set GEMINI_API_KEY, ANTHROPIC_API_KEY, DATABASE_URL, REDIS_URL

# 6. Copy MetaFBP model weights to models/ directory
# (See Section 3 for where to find these files)

# 7. Run database migrations
alembic upgrade head

# 8. Seed the 6 Felix questions
python scripts/seed_questions.py

# 9. Start the development server
uvicorn app.main:app --reload --port 8000

# 10. Verify: visit http://localhost:8000/docs for Swagger UI
```

#### Section 6: GCP Deployment
Step-by-step instructions for production deployment:
```bash
# 1. Authenticate with GCP
gcloud auth login
gcloud config set project YOUR_PROJECT_ID

# 2. Run the bootstrap script (provisions all GCP resources)
chmod +x infra/setup-gcp.sh
./infra/setup-gcp.sh YOUR_PROJECT_ID europe-west2

# 3. Upload MetaFBP model weights to Cloud Storage
gsutil cp models/universal_extractor.pth gs://YOUR_BUCKET/models/universal_extractor.pth
gsutil cp models/meta_generator.pth gs://YOUR_BUCKET/models/meta_generator.pth

# 4. Store API keys in Secret Manager
echo -n "your-gemini-key" | gcloud secrets versions add GEMINI_API_KEY --data-file=-
echo -n "your-anthropic-key" | gcloud secrets versions add ANTHROPIC_API_KEY --data-file=-

# 5. Connect GitHub repo to Cloud Build
# (Follow interactive prompts in GCP Console → Cloud Build → Triggers)

# 6. Push to main to trigger first deployment
git push origin main

# 7. Verify deployment
gcloud run services describe harmonia-api --region=europe-west2
curl https://YOUR_CLOUD_RUN_URL/health
```

#### Section 7: Frontend Integration Guide
This section is for when you (or another developer) build the frontend application. The backend exposes a RESTful JSON API at `/api/v1/`. Here is everything the frontend will need:

**API documentation**: Visit `/docs` (Swagger UI) or `/redoc` (ReDoc) on the running backend for interactive, auto-generated API documentation with request/response schemas and a "Try it out" button for every endpoint.

**Authentication**: The current backend does NOT implement user authentication — this must be added before production launch. Recommended approach for a mobile dating app: Firebase Authentication (Google, Apple, email sign-in) with JWT tokens passed in the `Authorization: Bearer {token}` header. The backend will need a middleware to verify Firebase JWTs against the Firebase Admin SDK. This is the single most important pre-launch addition.

**CORS configuration**: The backend already has CORS middleware configured. You will need to update the allowed origins in `app/config.py` to include your frontend domain(s) (e.g., `https://app.harmonia.com`, `http://localhost:3000` for local dev).

**Recommended frontend tech stack** (not prescribed, but these pair well with the API):
- **React Native** (if building iOS + Android) or **Next.js** (if building web-first). React Native is recommended for a dating app because: swipe gestures, push notifications, camera access for profile photos, biometric auth, and offline support are all critical UX features that work better as native experiences.
- **React Query / TanStack Query** — for server state management, caching API responses, optimistic updates on swipes.
- **Zustand or Redux Toolkit** — for client state (current user profile, navigation state, cached match data).
- **Firebase** — for authentication (see above), push notifications (FCM), and optionally real-time chat (Firestore).

**Key API flows the frontend must implement**:

1. **Onboarding flow**: `POST /api/v1/users` → `POST /api/v1/visual/calibrate` (user rates K sample face images for MetaFBP personalisation) → 6× `POST /api/v1/questionnaire/submit` (one per Felix question, 25-150 words each) → optionally `POST /api/v1/hla/upload` (genetic data, if user opts in).

2. **Discovery/swipe flow**: Backend serves candidate profiles → frontend displays cards → user swipes → `POST /api/v1/swipe` records the action → if mutual right-swipe detected, `POST /api/v1/match/calculate/{a}/{b}` triggers the full matching engine.

3. **Match reveal flow**: `GET /api/v1/reports/{match_id}/summary` fetches the Level 1 customer summary → frontend renders compatibility score, shared traits, badges, conversation starters. Level 1 NEVER contains raw sin scores, evidence snippets, or sin labels — it's all friendly natural language.

4. **Photo handling**: User uploads photos → frontend sends to backend → backend stores in GCS → backend returns a signed URL (expires after 60 minutes) → frontend displays the signed URL. Frontend should pre-fetch new signed URLs before expiry.

**Frontend environment variables** the frontend will need:
- `API_BASE_URL` — the Cloud Run service URL (e.g., `https://harmonia-api-xxxxx-ew.a.run.app`)
- `FIREBASE_CONFIG` — if using Firebase Auth (API key, project ID, etc.)

**What the frontend does NOT need to know about**: Gemini API calls, sin scores, evidence snippets, CWMV aggregation, MetaFBP inner-loop adaptation, HLA allele counting, calibration database, or any admin-level report data. All of this complexity is encapsulated behind the API. The frontend only interacts with the user-facing endpoints.

#### Section 8: API Endpoint Quick Reference
A complete table of all endpoints with method, path, purpose, auth level (public/user/admin), and which session built it. (Use the Quick Reference table from the build plan.)

#### Section 9: Environment Variables Reference
A complete table of every environment variable with its description, required/optional status, example value, and which service uses it. (Use the `.env.example` as the source of truth.)

#### Section 10: Troubleshooting
Common issues and solutions: GCS model weight download fails (check service account IAM roles), Gemini API returns 429 (rate limited — check quota, consider Vertex AI), Cloud SQL connection refused (check VPC connector and `CLOUD_SQL_USE_UNIX_SOCKET` flag), Redis timeout (check Memorystore firewall rules and VPC access), MetaFBP inference fails (check checkpoint file integrity with SHA256 hash).

### Verification

Run `uvicorn app.main:app --reload`, confirm `/health` returns 200, confirm `/docs` shows the Swagger UI, and confirm the database migrations run cleanly.

---

## Session 1: PIIP Gemini Parsing Service (Phase 2 — Core)

**Goal:** Implement the complete GeminiService that takes a user's free-text response to one of the 6 Felix questions and returns 7 sin scores with confidence and evidence. This is the foundational AI parsing layer for the entire personality system.

**Context for Claude Code:**
Implement `app/services/gemini_service.py` following the PIIP specification exactly. The service uses trait-specific decomposition — 7 separate Gemini calls per question (one per sin), each with anchored scale definitions. The model fallback chain is gemini-3-pro-preview → gemini-3-flash-preview → gemini-2.5-flash. Use the google-generativeai Python SDK with structured JSON output enforcement.

### Tasks

1. **Implement the GeminiService class** with these exact methods:
   - `__init__(api_key, calibration_service)` — Configures the Gemini client, stores the model chain, sin definitions, trait anchors, safety settings (all BLOCK_NONE since we're parsing authentic emotional content), and generation config (`thinking_level: 'low'`, `max_output_tokens: 1024`, `response_mime_type: 'application/json'`). Accepts a reference to the CalibrationService for retrieving few-shot examples.
   - `parse_single_response(question, answer, question_number, user_id) → dict` — Parses one question-answer pair into 7 sin scores by running trait-specific parsing in parallel (7 concurrent calls using `asyncio.gather`). Also extracts LIWC signals and detects discrepancies. **After parsing, stores every (question, sin, score, confidence, evidence_snippet) tuple in the `parsing_evidence` table with character offsets** so that each sin recognition is traceable back to the exact words in the user's response.
   - `_parse_single_trait(question, answer, sin, question_number) → dict` — Builds the trait-specific prompt with scale anchors (-5 to +5), **injects 2-3 few-shot examples from the calibration database** (retrieved via `calibration_service.get_examples(question_number, sin)`), and calls Gemini with the fallback chain. Returns `{"score": float, "confidence": float, "evidence": str, "evidence_start": int, "evidence_end": int}` where the start/end indices are character offsets into the original response text.
   - `_build_trait_prompt(question, answer, sin, few_shot_examples) → str` — Constructs the focused prompt for a specific sin with anchored examples at -5, 0, and +5. **If few-shot examples are available from the calibration database, inserts them between the scale definition and the user's response** as "Reference Examples" — showing the model what validated (response → score) mappings look like for this specific question and sin combination. This is the key mechanism that reduces hallucination: Gemini isn't scoring in a vacuum, it's scoring relative to admin-validated anchor points.
   - `_extract_liwc_signals(text) → dict` — Lightweight linguistic marker extraction (first person singular/plural, negative/positive emotion, certainty, hedging, future orientation).
   - `_detect_discrepancies(text, sins) → list[str]` — Detects mismatches between claimed traits and linguistic behaviour (e.g., claims low wrath but uses anger language).
   - `_locate_evidence_in_response(response_text, evidence_quote) → tuple[int, int]` — Finds the character start and end offsets of the evidence snippet within the original response text using fuzzy matching (since Gemini may slightly alter the quote). Returns (-1, -1) if no match found.
   - `parse_all_responses(responses: list[dict]) → dict` — Parses all 6 question-response pairs and returns per-question results plus aggregated data. The full parsing_evidence records are bulk-inserted into the database after all 6 questions are parsed.

2. **Implement the trait anchors** exactly as specified:
   - Greed: "Extremely generous, prioritises others" ↔ "Highly materialistic, accumulates at others' expense"
   - Pride: "Deeply humble, deflects credit" ↔ "Ego-driven, seeks status and validation constantly"
   - Lust: "Very restrained, deliberate, avoids spontaneity" ↔ "Highly impulsive, novelty-seeking"
   - Wrath: "Extreme conflict avoidance, never expresses anger" ↔ "Quick to anger, confrontational"
   - Gluttony: "Highly moderate, strict self-control" ↔ "Strongly indulgent, struggles with restraint"
   - Envy: "Deeply content, never compares" ↔ "Constantly competitive, resentful of success"
   - Sloth: "Extremely proactive, takes initiative" ↔ "Avoidant, passive, procrastinates"

3. **Implement the multi-observer consensus framework** for high-bias-risk traits (Wrath, Envy, Pride). When a trait score has confidence < 0.70 or the trait is flagged as ambiguous, run 3 parallel evaluations with different observer personas: Neutral Evaluator, Empathetic Therapist, Skeptical Critic. Aggregate using mean with standard deviation thresholds: `std_dev < 0.3 → multiplier 1.0`, `0.3-0.5 → 0.85`, `> 0.5 → 0.6 + flag "observer_disagreement"`.

4. **Implement robust error handling** with the retry strategy from Section 8 of the PIIP spec: exponential backoff (1s initial, 2× multiplier, 60s max, 0-1s jitter, 5 max retries, 300s total deadline), model fallback chain on rate limit / timeout / 500 errors, JSON extraction pipeline (direct parse → markdown extraction → prefix removal → jsonrepair), and validation of score ranges (-5 to +5), confidence ranges (0 to 1), and evidence strings.

5. **Implement the social desirability bias detection**: prompt-level reverse coding for Wrath (check if response contains anger words but claims calm demeanour), discrepancy flagging for Pride (claims humility but self-promotes), and cross-response consistency checking (flag if max delta > 5 for any sin across questions).

6. **Write the API endpoint** `POST /api/v1/questionnaire/submit` that accepts `{"user_id": str, "question_number": int, "response_text": str}`, validates word count (25-150), stores the response in the database, triggers Gemini parsing, and returns the parsed sin scores with evidence.

7. **Write a batch endpoint** `POST /api/v1/questionnaire/submit-all` that accepts all 6 responses at once: `{"user_id": str, "responses": [{"question_number": int, "response_text": str}, ...]}`. This is used by both the frontend and the cluster generation agent.

### Verification

Test with the worked example from the spec — the "group dinner check" response: "I'd probably suggest we just split it evenly to keep things simple. I hate those awkward moments where everyone's trying to calculate their exact share. Life's too short for that kind of pettiness. If someone ordered way more, they can throw in extra if they want, but I'm not going to be the one calling them out." Expected: Greed -2 (conf 0.78), Wrath -3 (conf 0.88), Sloth +1 (conf 0.72).

---

## Session 2: Profile Aggregation Pipeline (Phase 2 — Aggregation)

**Goal:** Implement the ProfileService that takes the raw 42 data points (7 sins × 6 questions) from the GeminiService and aggregates them into a quality-controlled personality profile with confidence weighting, variance analysis, response style detection, and a composite quality score.

**Context for Claude Code:**
Implement `app/services/profile_service.py` following Section 5 of the PIIP spec. The aggregation pipeline has 5 stages: question-level validation → trait-level aggregation (CWMV or simple mean) → outlier detection → response style detection (ERS, MRS, patterns) → profile quality scoring.

### Tasks

1. **Implement the ProfileService class** with these constants:
   - `SIN_NAMES = ["greed", "pride", "lust", "wrath", "gluttony", "envy", "sloth"]`
   - `SIN_WEIGHTS = {"wrath": 1.5, "sloth": 1.3, "pride": 1.2, "lust": 1.0, "greed": 0.9, "gluttony": 0.8, "envy": 0.7}`
   - `MIN_QUESTIONS = 4`, `MIN_WORD_COUNT = 25`, `LOW_WORD_COUNT = 50`
   - `HIGH_VARIANCE_THRESHOLD = 3.0` (SD on the 11-point scale)
   - `ERS_THRESHOLD = 0.40`, `MRS_THRESHOLD = 0.50`

2. **Implement `build_profile(user_id, parsed_responses, response_metadata) → dict`** that orchestrates the full pipeline:
   - Step 1: Validate each Q&A pair (word count ≥ 25, not placeholder, has signal)
   - Step 2: Organise scores by trait (7 lists, each containing up to 6 score dicts)
   - Step 3: Aggregate each trait using CWMV if confidence SD ≥ 0.10, else simple mean
   - Step 4: Detect response styles (ERS, MRS, patterns)
   - Step 5: Calculate composite quality score (0-100)
   - Step 6: Compile final profile JSON with all metadata

3. **Implement Confidence-Weighted Mean Voting (CWMV):** `Score_trait = Σ(Item_score × Confidence) / Σ(Confidence)`. Include variance penalty: if score SD > 3.0, reduce confidence by `min(0.3, (SD - 3.0) × 0.1)`.

4. **Implement outlier detection** (Tier 1 — simple flags): zero variance (all 6 scores identical), extreme scores (±5 with confidence > 0.9), and score reversals (adjacent questions with delta ≥ 8).

5. **Implement response style detection**: ERS (>40% of all scores at ±4 or ±5), MRS (>50% near midpoint AND fast completion < 50% median time), and pattern detection (all 7 sins getting identical score for a question, alternating extreme patterns).

6. **Implement the quality score calculator** with 4 equal-weighted components: internal consistency (mean confidence normalised to 0-100), response variance (optimal range 1.0-6.0), response style (deduct 25 per flag), engagement (word count + response time). Tiers: ≥80 = high, 60-79 = moderate, <60 = low.

7. **Implement profile versioning**: when a user retakes the questionnaire, create a new profile version and archive the old one.

8. **Wire the ProfileService into the API**: after the GeminiService parses all 6 responses, automatically trigger profile aggregation and store the result in the `personality_profiles` table.

### Verification

Test with the full worked example from Section 7.2 of the master spec: verify that the Wrath aggregation across 6 questions produces -1.57 (rounded to -1.6) using CWMV, and that the profile quality score falls in the expected range.

---

## Session 3: Similarity Calculation & Match Explanation (Phase 2 — Matching)

**Goal:** Implement the SimilarityService that calculates perceived similarity between two personality profiles using positive overlap detection, and generates natural language match explanations.

**Context for Claude Code:**
Implement `app/services/similarity_service.py` following Section 6 of the PIIP spec. The core principle is perceived similarity — only counting traits where both users share the same direction (both virtuous OR both vice-leaning), never penalising differences. This creates the "astrology effect."

### Tasks

1. **Implement the SimilarityService class** with the positive overlap algorithm:
   - `NEUTRAL_THRESHOLD = 0.5` (scores between -0.5 and +0.5 treated as no signal)
   - `TOTAL_WEIGHT = 7.4` (sum of all sin weights)
   - `DEFAULT_STAGE2_THRESHOLD = 0.40`

2. **Implement `calculate_similarity(profile_a, profile_b) → dict`** that returns raw score, adjusted score, quality multiplier, breakdown (list of shared traits with contributions), overlap count, tier, and display mode.

3. **Implement the 5-step similarity calculation:**
   - Step 1: Determine shared direction (`both > +0.5` = shared vice, `both < -0.5` = shared virtue)
   - Step 2: Calculate trait similarity: `1 - (|score_a - score_b| / 10)`
   - Step 3: Apply confidence weighting: `trait_similarity × avg_confidence`
   - Step 4: Apply sin-specific weights (Wrath 1.5×, Sloth 1.3×, Pride 1.2×, Lust 1.0×, Greed 0.9×, Gluttony 0.8×, Envy 0.7×)
   - Step 5: Normalise: `Σ(contributions) / 7.4`

4. **Implement quality-adjusted similarity** using the quality multiplier table: High/High = 1.0, High/Moderate = 0.9, Moderate/Moderate = 0.8, High/Low = 0.7, Moderate/Low = 0.6, Low/Low = 0.5.

5. **Implement threshold evaluation** (soft gate): strong_fit (≥threshold+0.20, highlight mode, 4 traits), good_fit (≥threshold, standard mode, 3 traits), moderate_fit (≥threshold-0.15, minimal mode, 2 traits), low_fit (below, chemistry_focus mode, 1 trait).

6. **Implement the match explanation generator** with the trait-to-description mapping:
   - Greed virtue: "generous and easygoing about money" / vice: "practical and thoughtful about resources"
   - Pride virtue: "humble and down-to-earth" / vice: "confident and self-assured"
   - Lust virtue: "thoughtful and deliberate" / vice: "spontaneous and adventurous"
   - Wrath virtue: "easygoing and harmony-seeking" / vice: "direct and unafraid of confrontation"
   - Gluttony virtue: "balanced and moderate" / vice: "fun-loving and indulgent"
   - Envy virtue: "content and secure" / vice: "ambitious and driven"
   - Sloth virtue: "proactive and energetic" / vice: "relaxed and laid-back"

7. **Implement HLA display logic** (referenced by similarity service): scores ≥75 show 🔥 "Strong chemistry signal", 50-74 show ✨ "Good chemistry", 25-49 show 💫 "Some chemistry", <25 = HIDE (never show negative).

8. **Implement the match card assembly function** that combines personality explanation + HLA display into the customer-facing JSON.

### Verification

Test with the worked example from Section 8.4 of the master spec: User A [-1.8, +0.3, +2.5, -2.1, +1.2, -0.8, -1.5] vs User B [-2.2, -1.5, +3.1, -1.8, -0.3, -1.2, +1.8]. Expected: 4 of 7 traits shared, raw similarity = 3.156 / 7.4 = 0.426 (42.6%).

---

## Session 4: MetaFBP Visual Intelligence (Phase 1)

**Goal:** Integrate your locally-trained MetaFBP model into the production system. Build the calibration flow (users rate sample faces to build their support set), the per-user adaptation pipeline (inner loop gradient descent), weight caching (Redis), and the visual scoring endpoint.

**Context for Claude Code:**
Implement `app/services/visual_service.py` and `app/ml/metafbp/inference.py`. The MetaFBP model has two components: a frozen ResNet-18 feature extractor (E_θ_c) that produces 512-dim embeddings, and a Parameter Generator MLP (G_θ_g) that dynamically modifies predictor weights based on the user's support set. The inference pipeline is: (1) load user's support set, (2) run k-step inner loop adaptation to get personalised generator weights, (3) for each target face: extract features → generate dynamic weights → compute prediction. The user's checkpoint files should be placed in the `models/` directory.

### Tasks

1. **Create `app/ml/metafbp/preprocessing.py`** — Image preprocessing utility that loads an image, resizes to 224×224, applies center crop, and normalises with ImageNet stats (`mean=[0.485, 0.456, 0.406]`, `std=[0.229, 0.224, 0.225]`). Returns a PyTorch tensor.

2. **Create `app/ml/metafbp/inference.py`** — The production inference wrapper that:
   - Loads the Stage 1 checkpoint (frozen ResNet-18 feature extractor) at startup
   - Loads the Stage 2 checkpoint (meta-generator base weights) at startup
   - Provides `adapt_user_predictor(support_images, support_ratings) → adapted_generator_state_dict` — runs k=5 inner-loop SGD steps (lr=0.01) on the user's support set to produce personalised generator weights
   - Provides `score_target(adapted_generator, target_image_path) → float` — extracts features from the target image through the frozen extractor, generates dynamic weights via the adapted generator, applies adaptation strength (λ=0.01), and computes the raw prediction. Scales from the model's output range to 1-5, then normalises to 0-100.
   - Handle the Cyclically Re-sampling Strategy for imbalanced support sets (detailed in Section 3.3 of the master spec).

3. **Create `app/ml/trait_extraction/extractor.py`** — Visual trait extraction service that analyses detected facial traits in the support set images to build preference profiles. This uses a combination of the MetaFBP embeddings and Gemini vision calls (or a simpler heuristic approach) to detect traits like glasses, facial hair, hairstyle, complexion, etc. For each trait, calculate frequency in liked (4-5) vs disliked (1-2) faces, compute weight using `W_trait = F_trait × R_corr × sign(avg_rating - 3.0)`, and classify as MANDATORY (>80%), PREFERRED (60-80%), AVERSION (>80% in dislikes), or NEGATIVE (60-80% in dislikes).

4. **Implement `app/services/visual_service.py`** — The VisualService class that:
   - `calibrate_user(user_id, ratings: list[{image_id, rating}])` — Stores ratings in `visual_ratings` table, runs trait extraction to build the preference model, runs MetaFBP inner-loop adaptation, serialises adapted weights to Redis (`f"metafbp:adapted:{user_id}"`, TTL = session duration), and stores the trait preference model in `visual_preferences`.
   - `score_target(user_id, target_image_path) → dict` — Loads cached adapted weights from Redis, scores the target image, detects traits in the target, calculates T_match_positive and T_match_negative, and computes `S_vis = (MetaFBP_Component × 0.6) + (T_match_positive × 0.25) + (T_match_negative × 0.15)`.
   - `invalidate_cache(user_id)` — Called when user swipes (new data changes preference model). Triggers async background re-adaptation.

5. **Create the API endpoints:**
   - `POST /api/v1/visual/calibrate` — Accept `{"user_id": str, "ratings": [{image_id, rating}, ...]}`. Trigger calibration. Return the trait preference model summary.
   - `POST /api/v1/visual/score` — Accept `{"user_id": str, "target_user_id": str}`. Load target's photos, score using cached adapted model. Return S_vis score and trait match breakdown.
   - `POST /api/v1/swipe` — Accept `{"swiper_id": str, "target_id": str, "direction": "left"|"right"|"superlike"}`. Store swipe, update support set, invalidate MetaFBP cache for background re-adaptation.

6. **Implement session caching** for latency mitigation (Section 14.2 of master spec): run the inner loop once when user opens the app, serialise adapted generator weights to Redis, use forward-pass-only for scoring during the session, invalidate on new swipes and trigger async background re-adaptation.

### Verification

Create a test script that loads your checkpoint files, creates a mock support set of 10 images with ratings, runs adaptation, and scores a target image. Verify the output is in the 1-5 range and the 0-100 normalised score is sensible.

---

## Session 5: HLA Biological Compatibility (Phase 3)

**Goal:** Implement the HLA scoring system that calculates genetic compatibility based on MHC allele dissimilarity, predicts olfactory attraction, and estimates reproductive fitness.

**Context for Claude Code:**
Implement `app/services/hla_service.py` following Section 9 of the master spec. The biological scoring formula is `S_bio = (N_unique / N_total) × 100` where N_unique is the count of unique alleles in both users' combined allele pool and N_total is the total number of allele slots (typically 12 for 3 loci × 2 alleles × 2 users). The olfactory prediction model uses the Heterozygosity Index.

### Tasks

1. **Implement the HLA data upload endpoint** `POST /api/v1/hla/upload` that accepts genetic data (alleles for HLA-A, HLA-B, HLA-DRB1), encrypts it with Fernet, validates allele format (e.g., `A*01:01`), and stores it in the `hla_data` table.

2. **Implement `app/services/hla_service.py`** with:
   - `calculate_compatibility(user_a_id, user_b_id) → dict` — Decrypts both users' HLA data, counts unique alleles, calculates S_bio, computes Heterozygosity Index, generates olfactory prediction, and produces the display tier (≥75 show, 50-74 show, 25-49 show, <25 HIDE).
   - `_calculate_s_bio(alleles_a, alleles_b) → float` — `(N_unique / N_total) × 100`
   - `_calculate_heterozygosity(alleles_a, alleles_b) → float` — `N_unique / N_total`
   - `_predict_olfactory_attraction(heterozygosity_index) → dict` — Returns intensity score (0-100) and assessment text.
   - `_generate_peptide_analysis(alleles) → dict` — Maps known alleles to their binding characteristics and disease associations (from the reference data in Section 9.4 of the master spec).

3. **Implement Fernet encryption utilities** in `app/utils/encryption.py`: `encrypt_hla_data(data: dict) → bytes` and `decrypt_hla_data(encrypted: bytes) → dict`.

4. **Implement the display logic** — never show negative HLA results. If score <25, return `{"show": False}`. The API endpoint should respect this: `GET /api/v1/hla/compatibility/{user_a_id}/{user_b_id}`.

### Verification

Test with the worked example from Section 10.4: User A alleles [A*01:01, A*02:01, B*08:01, B*07:02, DRB1*15:01, DRB1*03:01] vs User B alleles [A*03:01, A*24:02, B*44:02, B*35:01, DRB1*04:01, DRB1*07:01]. Expected: 11 unique of 12 total → S_bio = 91.6, Heterozygosity = 0.916.

---

## Session 6: Three-Stage Matching Engine & WtM Calculation

**Goal:** Implement the MatchingService that orchestrates all three phases into the cascaded pipeline: Visual Gate → Personality Reveal → Genetics Info, and calculates the final Willingness to Meet (WtM) score.

**Context for Claude Code:**
Implement `app/services/matching_service.py` following Sections 10 and 7 of the master spec and PIIP document. The system uses a sequential cascade, not a weighted average formula. The 60/30/10 weights represent decision importance at each stage. Visual scores are asymmetric (A→B ≠ B→A), while personality and genetics are symmetric. The final reciprocal score uses geometric mean: `√(score_A→B × score_B→A)`.

### Tasks

1. **Implement the MatchingService class** with injected dependencies: `visual_service`, `similarity_service` (personality), `hla_service`.

2. **Implement Stage 1: Visual Gate** — Check if both users have swiped right on each other (query `swipes` table). If not mutual, pipeline terminates. If mutual, calculate `s_vis_a_to_b` and `s_vis_b_to_a` using the VisualService.

3. **Implement Stage 2: Personality Reveal** — Calculate perceived similarity using the SimilarityService. This is a soft gate — low similarity changes the display messaging but doesn't block the match.

4. **Implement Stage 3: Genetics Info** — Calculate HLA compatibility if both users have uploaded genetic data. This is informational only, never blocks.

5. **Implement the WtM calculation:**
   ```
   For each direction:
   combined_A_to_B = (0.4 × s_vis_a_to_b) + (0.3 × s_psych) + (0.3 × s_bio)
   combined_B_to_A = (0.4 × s_vis_b_to_a) + (0.3 × s_psych) + (0.3 × s_bio)
   
   reciprocal_wtm = √(combined_A_to_B × combined_B_to_A)
   ```
   Handle missing signals by redistributing weights proportionally (e.g., no genetics → 57%/43%/0%).

6. **Implement the friction flag system** from the psychometric evidence: detect when sin deltas exceed 0.3 between matched users, flag specific conflicts (sloth_delta, bluntness_delta), and calculate P_friction penalty (applied to S_psych).

7. **Implement match creation and storage**: when a mutual swipe is detected, run the full pipeline, store the match with reasoning_chain (Level 3), and return the customer_summary (Level 1).

8. **Create the match API endpoints:**
   - `POST /api/v1/match/calculate/{user_a_id}/{user_b_id}` — Trigger full match calculation
   - `GET /api/v1/match/{match_id}` — Retrieve match results
   - `GET /api/v1/matches/{user_id}` — List all matches for a user

### Verification

Test with the complete worked example from Section 10.4: S_vis = 79.7, S_psych = 52.0, S_bio = 91.6. Expected WtM = (0.4 × 79.7) + (0.3 × 52.0) + (0.3 × 91.6) = 74.96 ≈ 75.

---

## Session 7: Report Generation Architecture

**Goal:** Implement the multi-level reporting system with full evidence traceability: Level 3 (reasoning_chain.json — strict math, no AI, with complete evidence map), Level 2A (gemini_narrative.md — psych/visual analysis with evidence citations), Level 2B (hla_gemini_analysis.md — bio report), and Level 1 (customer_summary.json — sanitised UI with NO evidence exposure).

**Context for Claude Code:**
Implement `app/services/report_service.py` following Sections 11-13 of the master spec. The key addition is the **evidence map** — a structured trail that lets any admin trace every sin score back to the exact snippet from the user's response that triggered it. This evidence is compiled from the `parsing_evidence` table (populated during Session 1's Gemini parsing) and woven into Level 3 and Level 2 reports. Level 1 (customer-facing) explicitly does NOT include any evidence snippets, raw scores, or sin labels.

### Tasks

1. **Implement the EvidenceMapBuilder** utility that queries the `parsing_evidence` table for a given user and compiles a structured evidence map:
   ```
   {
     "user_id": "abc123",
     "evidence_map": {
       "q1_group_dinner": {
         "response_text": "I'd probably suggest we just split it evenly...",
         "sin_recognitions": [
           {
             "sin": "wrath",
             "score": -3,
             "confidence": 0.88,
             "evidence_snippet": "I hate those awkward moments where everyone's trying to calculate their exact share",
             "snippet_location": {"start": 67, "end": 142},
             "interpretation": "Conflict avoidance — prefers social harmony over addressing unfairness"
           },
           {
             "sin": "greed",
             "score": -2,
             "confidence": 0.78,
             "evidence_snippet": "Life's too short for that kind of pettiness",
             "snippet_location": {"start": 144, "end": 184},
             "interpretation": "Dismisses resource-tracking as beneath them — generous orientation"
           },
           {
             "sin": "sloth",
             "score": +1,
             "confidence": 0.72,
             "evidence_snippet": "just split it evenly to keep things simple",
             "snippet_location": {"start": 22, "end": 62},
             "interpretation": "Path of least resistance — easy solution preferred over precise accounting"
           }
         ]
       },
       "q2_unexpected_expense": { ... },
       // ... all 6 questions
     }
   }
   ```
   Each sin recognition entry shows: which sin was detected, the score assigned, the confidence, the exact quoted fragment from the user's response, where in the response it appears (character offsets), and a one-line interpretation of what the snippet reveals about the trait. The interpretation line is generated by Gemini at parse time (added to the `parsing_evidence` table as an additional column `interpretation`).

2. **Implement Level 3: reasoning_chain.json** — Pure mathematical output with no AI interpretation, but now including the **complete evidence map for both users** in the match. The Level 3 report structure is:
   - `phase_1_visual`: MetaFBP scores, trait preference models, target trait detections
   - `phase_2_psychometric`: **Full evidence map for User A** (every question → every sin recognition with snippet + location), **Full evidence map for User B**, aggregated sin vectors with CWMV calculations shown, similarity breakdown with per-sin contributions, friction flags with deltas
   - `phase_3_biological`: Allele data with per-allele confidence, S_bio calculation
   - `final_calculation`: WtM formula with all intermediate values
   
   The evidence map makes this the complete audit trail — an admin can read a sin score, see the exact words that produced it, verify the interpretation, and trace the full chain from raw text to final WtM number.

3. **Implement Level 2A: gemini_narrative.md** — Use Gemini with Protocol A system prompt ("You are the Harmonia Engine, a cynical evolutionary psychologist...") to generate a forensic audit report. **The evidence map is injected into the Gemini prompt** so the narrative can cite specific snippets. The report should:
   - Quote the relevant evidence snippets when discussing each sin score (e.g., *"User A's Wrath score of -3 derives from their statement: 'I hate those awkward moments where everyone's trying to calculate their exact share' — classic conflict avoidance dressed up as nonchalance."*)
   - Cover every sin where the delta between the two users exceeds 0.3, citing both users' evidence
   - Build the perceived similarity analysis using the overlapping trait evidence
   - Produce the friction analysis with probability estimates grounded in specific snippets
   - Deliver the verdict (Viable/Dead on Arrival) with evidence-backed reasoning

4. **Implement Level 2B: hla_gemini_analysis.md** — Use Gemini with Protocol B system prompt ("You are an expert Geneticist specializing in the Major Histocompatibility Complex...") to generate: data integrity audit, allelic dissimilarity assessment, peptide-binding groove analysis, olfactory/pheromonal prediction, reproductive fitness estimate, and summary verdict. (No personality evidence needed here — this is purely biological.)

5. **Implement Level 1: customer_summary.json** — Sanitised, friendly output with: display_score (0-100), badges (array of strings like "Instant Spark", "Visual Type Match"), synopsis (headline + body), compatibility_breakdown (physical/personality/chemistry with scores and labels), shared_traits (natural language list), and conversation_starters. **CRITICAL: Level 1 contains NO evidence snippets, NO raw sin scores, NO sin labels, and NO quoted fragments from the user's responses.** The customer never sees the scoring mechanics — only friendly natural language.

6. **Create report API endpoints:**
   - `GET /api/v1/reports/{match_id}/reasoning-chain` — Level 3 with full evidence maps (admin only)
   - `GET /api/v1/reports/{match_id}/narrative` — Level 2A with evidence citations (admin only)
   - `GET /api/v1/reports/{match_id}/hla-analysis` — Level 2B (admin only)
   - `GET /api/v1/reports/{match_id}/summary` — Level 1 (user-facing, NO evidence)
   - `GET /api/v1/reports/{match_id}/evidence-map/{user_id}` — Standalone evidence map for a specific user in a match (admin only, useful for debugging individual profiles)

### Verification

Generate a complete set of reports for the worked example match. Verify that Level 3 contains the full evidence map with snippet locations, Level 2A cites specific snippets in its narrative, and Level 1 contains zero evidence fragments or sin labels.

---

## Session 8: PIIP Cluster Generation Agent & Gemini Calibration Database

**Goal:** Build two interconnected systems: (1) the Claude Haiku agent that generates diverse personality clusters, and (2) the calibration database pipeline where those clusters are Gemini-parsed, admin-reviewed, and fed back as few-shot examples into future Gemini prompts. This is both database seeding and model calibration — each reviewed example makes Gemini more consistent and less likely to hallucinate.

**Context for Claude Code:**
This session implements three components: `scripts/cluster_generator.py` (the Haiku generation loop), `app/services/calibration_service.py` (the calibration database manager that handles storage, admin review workflow, and few-shot retrieval), and `app/api/admin/calibration.py` (the admin review endpoints). The full pipeline is:

```
Claude Haiku generates 6 responses
    ↓
Responses submitted to PIIP pipeline (Gemini parses into sin scores)
    ↓
Each (question, response, sin, score, confidence, evidence) tuple
is automatically inserted into calibration_examples table
with review_status = "pending"
    ↓
Admin reviews via dashboard/API:
  - APPROVE: Gemini got it right → becomes a few-shot example as-is
  - CORRECT: Gemini was close but off → admin provides corrected score + notes → corrected version becomes the few-shot example
  - REJECT: Response was garbage or Gemini failed completely → excluded from few-shot pool
    ↓
When GeminiService parses a NEW user's response,
CalibrationService retrieves 2-3 approved/corrected examples
for that specific (question_number, sin) pair and injects them
into the Gemini prompt as "Reference Examples"
    ↓
Gemini scores relative to validated anchors → reduced hallucination
```

The loop is self-improving: more reviewed examples → better few-shot selection → more consistent Gemini scoring → less admin correction needed over time.

### Tasks

#### Part A: Claude Haiku Generation Agent

1. **Implement the system prompt** for Claude Haiku that instructs it to silently generate a random personality archetype (age, temperament, values, communication style) before answering — do NOT include this in the response. Answer each question in 50-100 words as that person would, in first person, casual written tone. Show realistic, mixed motivations (someone can be generous but quietly resentful). Do NOT label traits, sins, or personality types in the response. Vary style each time — different vocabulary, emotional tone, sentence structure. Return ONLY a JSON object with keys q1 through q6.

2. **Implement the generation loop** (`scripts/cluster_generator.py`): call Claude Haiku API (`claude-haiku-4-5`) with the system prompt + all 6 questions, parse the JSON response, validate each response is 25-150 words (the PIIP design constraint), generate a synthetic user_id (with `source: "claude_agent"` flag), submit to `POST /api/v1/questionnaire/submit-all` (which triggers Gemini parsing), and log success/failure. Brief interval between iterations to respect rate limits.

3. **Implement the Batch API option** for high-volume generation: build batch requests (up to 100,000 per batch), submit via `POST /v1/messages/batches`, poll for completion, process results from `.jsonl` file, and use prompt caching (same system prompt across all requests) for up to 93% cost savings.

4. **Implement quality controls**: word count validation (reject and retry if outside 25-150), diversity checking (compare new responses against last N to detect repetition), JSON format validation, and error handling with retry logic.

5. **Mark all generated profiles as synthetic** in the database (`source: "claude_agent"`) so they can be distinguished from real user data.

#### Part B: Calibration Database Service

6. **Implement `app/services/calibration_service.py`** — the CalibrationService class that manages the full lifecycle of calibration examples:

   - `ingest_from_parsing(user_id, question_number, response_text, sin, gemini_score, gemini_confidence, gemini_evidence, source_profile_id) → calibration_example_id` — Called automatically by the GeminiService after parsing any synthetic profile (detected via `source: "claude_agent"` on the profile). Creates a new row in `calibration_examples` with `review_status = "pending"`. This is the bridge between the generation pipeline and the review queue.

   - `get_review_queue(filters: {status, question_number, sin, limit, offset}) → list[dict]` — Returns pending examples for admin review, sorted by creation date. Supports filtering by question number and sin so admins can review systematically (e.g., "show me all pending Wrath scores for Question 4").

   - `review_example(example_id, action: "approve"|"correct"|"reject", validated_score: float|None, validated_by: str, review_notes: str|None)` — Admin action on a calibration example. For "approve", the `validated_score` is set equal to `gemini_raw_score` (Gemini got it right). For "correct", the admin provides their own score and optionally notes explaining the correction (e.g., "Gemini scored Wrath at +3 but the response is clearly sarcastic, not genuinely angry — should be -1"). For "reject", the example is excluded from the few-shot pool entirely.

   - `get_examples(question_number: int, sin: str, n: int = 3) → list[dict]` — **This is the method called by GeminiService during prompt construction.** Retrieves the top N approved or corrected examples for a specific (question_number, sin) combination, prioritised by: (1) corrected examples first (these represent cases where Gemini was wrong and an admin provided the right answer — the most valuable training signal), (2) approved examples with highest confidence, (3) diversity of score values (try to include examples spanning the -5 to +5 range). Each returned example contains `{"response_text": str, "validated_score": float, "evidence_snippet": str, "review_notes": str|None}`.

   - `get_calibration_stats() → dict` — Returns summary statistics: total examples, pending count, approved count, corrected count, rejected count, breakdown by question and sin, average correction magnitude (how far off Gemini tends to be), and the "coverage map" showing which (question, sin) combinations have sufficient examples (target: at least 5 validated examples per cell = 6 questions × 7 sins = 42 cells).

7. **Implement the few-shot injection format** in GeminiService. When `calibration_service.get_examples()` returns results, they are formatted into the trait prompt as:

   ```
   Reference Examples (validated by human reviewers):
   
   Example 1:
   Response: "I'd probably just throw in my card and cover it. Money stuff 
   between friends gets weird and I'd rather just eat the cost than make 
   it a whole thing. Last time someone tried to itemize everything it 
   killed the vibe for like twenty minutes."
   Wrath Score: -2 (conflict avoidance — prefers social harmony)
   
   Example 2:  
   Response: "Honestly I'd speak up. If someone ordered three cocktails 
   and a steak and I had a salad, I'm not subsidizing that. I'd say it 
   nicely but I'd say it."
   Wrath Score: +2 (willing to confront, direct about fairness)
   
   Now analyze this NEW response for WRATH signals:
   ```

   This grounds Gemini's scoring in concrete, validated examples rather than relying solely on abstract scale anchors. The corrected examples are especially valuable because they show Gemini exactly where its default interpretation goes wrong.

#### Part C: Admin Review API

8. **Implement `app/api/admin/calibration.py`** with these endpoints:

   - `GET /api/v1/admin/calibration/queue` — Returns the pending review queue. Query params: `status` (pending/approved/corrected/rejected), `question_number` (1-6), `sin` (one of 7), `limit`, `offset`. Response includes the full response text, Gemini's raw score, confidence, and evidence for context.

   - `POST /api/v1/admin/calibration/{example_id}/review` — Submit a review action. Body: `{"action": "approve"|"correct"|"reject", "validated_score": float|null, "notes": str|null, "reviewer": str}`. Returns the updated example.

   - `GET /api/v1/admin/calibration/stats` — Returns the calibration coverage map and statistics. Shows which (question, sin) cells have enough validated examples and which need more generation/review.

   - `POST /api/v1/admin/calibration/bulk-review` — Accept an array of `[{example_id, action, validated_score, notes}]` for batch review. This speeds up the admin workflow significantly when reviewing dozens of similar examples.

   - `GET /api/v1/admin/calibration/effectiveness` — Returns metrics on how the calibration database is performing: average Gemini correction magnitude over time (should decrease as the few-shot pool grows), score distribution histograms per sin, and the "drift report" showing whether Gemini's scoring is converging toward admin expectations.

9. **Create a management script** (`scripts/calibration_manager.py`) that can: run N Haiku generation iterations, report on generation statistics (success rate, average word counts, diversity metrics), estimate cost per profile, show the calibration coverage map, and identify which (question, sin) combinations need more examples.

### The Self-Improving Loop

The system is designed to bootstrap itself. Initially, with zero calibration examples, Gemini scores using only the abstract scale anchors (the trait definitions from the PIIP spec). These early scores will have the highest variance and the most hallucination. As admins review the first batch of synthetic profiles and approve/correct scores, the few-shot pool begins to populate. Gemini's second batch of scoring will be anchored against these validated examples and should show less drift. Over successive batches, the correction rate should decrease and the scoring consistency should increase.

The target state is at least 5 validated examples per cell in the 6×7 matrix (42 cells total = 210 minimum validated examples). At the Batch API price of ~$28.50 per 10,000 profiles for Haiku generation, plus Gemini parsing costs, the calibration database can be fully populated for under $50 — but the admin review time is the real bottleneck. The bulk review endpoint and systematic queue filtering (review all Wrath examples for Q1, then all Wrath for Q2, etc.) are designed to make this as efficient as possible.

### Verification

Generate 50+ profiles and run 10 through full admin review (approve some, correct some, reject some). Then generate 10 more profiles and verify that the few-shot examples appear in the Gemini prompts. Compare the scoring consistency (standard deviation across similar response types) between the pre-calibration and post-calibration batches.

---

## Session 9: Testing, Integration & End-to-End Validation

**Goal:** Write comprehensive tests for all services, perform end-to-end integration testing of the full pipeline, and validate against the worked examples in the master spec.

### Tasks

1. **Unit tests** for each service: GeminiService (mock API calls, verify few-shot injection from calibration DB, verify evidence snippet extraction with character offsets), ProfileService (test CWMV, outlier detection, quality scoring), SimilarityService (test overlap calculation, quality adjustment, threshold evaluation), VisualService (test scoring formula), HLAService (test allele counting, encryption), MatchingService (test WtM calculation, weight redistribution), ReportService (test all 4 report levels — verify Level 3 contains evidence maps, Level 2A cites snippets, Level 1 contains NO evidence), CalibrationService (test ingestion from parsing, review workflow state transitions, few-shot retrieval ranking, coverage map calculation).

2. **Integration tests** for the full pipeline: create two test users → submit questionnaire responses → verify profile creation → verify `parsing_evidence` table has entries with valid character offsets for every sin recognition → trigger match calculation → verify Level 3 report contains both users' evidence maps → verify Level 2A narrative cites specific snippets → verify Level 1 contains zero evidence fragments → verify match card assembly.

3. **Calibration pipeline integration test**: run 5 Haiku generation cycles → verify `calibration_examples` table has pending entries → approve 3, correct 1, reject 1 via admin API → generate a new real-user-style profile → verify the GeminiService prompt now contains the approved/corrected examples as few-shot references → verify the corrected example appears with the admin's score (not Gemini's original).

4. **Validate against master spec worked examples:**
   - Phase 1: MetaFBP score 4.18 → component 79.5, T_match_positive 88.4, T_match_negative 66.0, S_vis = 79.7
   - Phase 2: Sin distance 0.35, P_friction 0.80, S_psych = 52.0
   - Phase 3: 11 unique / 12 total, S_bio = 91.6
   - WtM: (0.4 × 79.7) + (0.3 × 52.0) + (0.3 × 91.6) = 74.96 ≈ 75

4. **Load test** the cluster generator: run 100 profiles through the full pipeline and verify database integrity, score distributions, and system stability.

5. **Create a comprehensive Postman / HTTP test collection** for all API endpoints.

---

## Session 10: Google Cloud Deployment & Production Readiness

**Goal:** Deploy the complete Harmonia V3 system to Google Cloud Platform using Cloud Run (stateless API), Cloud SQL (PostgreSQL), Memorystore (Redis), Cloud Storage (photos, model weights, HLA uploads), Secret Manager (credentials), and Cloud Build (CI/CD). Configure monitoring, alerting, and production validation.

**Context for Claude Code:**
The deployment architecture uses Cloud Run as the compute layer because it supports request timeouts up to 3600s (critical for 60s Gemini calls), scales to zero when idle, and provides a managed TLS/HTTPS endpoint. Cloud SQL PostgreSQL provides the relational database with automatic backups and private networking. Memorystore Redis provides the in-memory cache for MetaFBP adapted weights with sub-millisecond latency over VPC. Cloud Storage handles all binary assets (user photos, MetaFBP checkpoint files, encrypted HLA uploads). Secret Manager holds all sensitive credentials and is accessed at runtime by the Cloud Run service account. Cloud Build provides the CI/CD pipeline triggered by GitHub pushes.

The GCP region should be `europe-west2` (London) for lowest latency to your user base.

### Tasks

#### Part A: GCP Project Bootstrap

1. **Implement `infra/setup-gcp.sh`** — A comprehensive one-shot bootstrap script that configures the entire GCP project from scratch. The script should:

   - Accept the GCP project ID and region as arguments, defaulting region to `europe-west2`
   - Enable all required APIs: `run.googleapis.com`, `sqladmin.googleapis.com`, `redis.googleapis.com`, `cloudbuild.googleapis.com`, `secretmanager.googleapis.com`, `storage.googleapis.com`, `artifactregistry.googleapis.com`, `vpcaccess.googleapis.com`
   - Create an Artifact Registry Docker repository (`harmonia-v3-repo`) for storing built container images
   - Provision a **Cloud SQL PostgreSQL 15** instance: `db-f1-micro` tier for development (can be upgraded to `db-custom-2-4096` for production), private IP networking enabled, automatic daily backups with 7-day retention, point-in-time recovery enabled, `europe-west2-a` zone, 10GB SSD storage with auto-resize. Create the `harmonia` database and `harmonia_user` with a generated password
   - Provision a **Memorystore Redis** instance: basic tier, 1GB memory, `europe-west2-a` zone, Redis version 7.x, connected to the default VPC
   - Create a **Serverless VPC Access connector** (`harmonia-vpc-connector`) so Cloud Run can reach Memorystore and Cloud SQL via private IP — this is required because both Memorystore and Cloud SQL private IP are VPC-internal resources
   - Create a **Cloud Storage bucket** (`{project_id}-harmonia-v3`) with: Standard storage class, `europe-west2` location, uniform bucket-level access, lifecycle rule to move objects to Nearline after 90 days, CORS configuration allowing your frontend domain
   - Create **Secret Manager secrets** for: `GEMINI_API_KEY`, `ANTHROPIC_API_KEY`, `FERNET_KEY`, `DATABASE_URL` (constructed from the Cloud SQL credentials), `REDIS_URL` (from Memorystore IP). Each secret should have an initial version set from either user input or auto-generation (Fernet key)
   - Create a **Cloud Run service account** (`harmonia-v3-runner@{project}.iam.gserviceaccount.com`) and grant it: `roles/cloudsql.client` (database access), `roles/secretmanager.secretAccessor` (credentials), `roles/storage.objectAdmin` (bucket read/write), `roles/logging.logWriter` (structured logging to Cloud Logging), `roles/monitoring.metricWriter` (custom metrics)
   - Output a summary of all provisioned resources with connection strings and next steps

2. **Implement the `Dockerfile`** for Cloud Run deployment. Multi-stage build:

   ```dockerfile
   # Stage 1: Build dependencies
   FROM python:3.11-slim AS builder
   WORKDIR /build
   COPY requirements.txt .
   RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

   # Stage 2: Runtime
   FROM python:3.11-slim
   # Install system deps: libpq5 for asyncpg, libgomp1 for PyTorch
   RUN apt-get update && apt-get install -y --no-install-recommends \
       libpq5 libgomp1 && rm -rf /var/lib/apt/lists/*
   WORKDIR /app
   COPY --from=builder /install /usr/local
   COPY . .
   # Cloud Run injects PORT env var (default 8080)
   CMD exec gunicorn app.main:app \
       --worker-class uvicorn.workers.UvicornWorker \
       --bind 0.0.0.0:$PORT \
       --workers 2 \
       --timeout 120 \
       --graceful-timeout 120 \
       --keep-alive 75 \
       --access-logfile - \
       --error-logfile -
   ```

   Note: MetaFBP `.pth` files are NOT baked into the image. They are downloaded from GCS at startup (implemented in the FastAPI lifespan handler). This keeps the image small (~1.2GB with PyTorch vs ~1.25GB with checkpoints) and means model updates don't require a full redeploy.

3. **Implement GCS model weight loading** in `app/main.py` lifespan handler. On startup, the service checks if the `.pth` files exist locally (for local dev) and if not, downloads them from the configured GCS bucket path to a local `/tmp/models/` directory. This runs once per cold start. Include a SHA256 integrity check: the expected hash is stored as a custom metadata field on the GCS object, and the service verifies the downloaded file matches before loading it into PyTorch.

4. **Implement GCS integration for user photos and HLA uploads** in a new utility `app/utils/storage.py`. Methods: `upload_file(bucket, path, file_bytes, content_type) → gcs_url`, `download_file(bucket, path) → bytes`, `generate_signed_url(bucket, path, expiry_minutes=60) → str` (for frontend to display photos without exposing the bucket directly), `delete_file(bucket, path)`. All HLA uploads must be server-side encrypted using Cloud Storage's default encryption plus the application-layer Fernet encryption already implemented.

5. **Implement Cloud SQL connection** in `app/database.py`. For Cloud Run, use the `cloud-sql-python-connector` library with IAM authentication (no password in connection string — the service account authenticates automatically). For local development, fall back to a standard `asyncpg` connection string from the `DATABASE_URL` env var. The connector creates a Unix socket connection through the Cloud SQL Auth Proxy sidecar that Cloud Run manages automatically.

   ```python
   from google.cloud.sql.connector import Connector
   import sqlalchemy
   
   async def get_cloud_sql_engine():
       connector = Connector()
       async def getconn():
           return await connector.connect_async(
               settings.CLOUD_SQL_INSTANCE_CONNECTION,
               "asyncpg",
               user=settings.DB_USER,
               password=settings.DB_PASSWORD,
               db=settings.DB_NAME,
           )
       engine = sqlalchemy.ext.asyncio.create_async_engine(
           "postgresql+asyncpg://",
           async_creator=getconn,
           pool_size=10,
           max_overflow=5,
           pool_timeout=30,
           pool_recycle=1800,  # Recycle connections every 30min
       )
       return engine
   ```

#### Part B: CI/CD Pipeline

6. **Implement `infra/cloudbuild.yaml`** — The Cloud Build pipeline triggered by GitHub pushes to `main`:

   ```yaml
   steps:
     # Step 1: Build the Docker image
     - name: 'gcr.io/cloud-builders/docker'
       args: ['build', '-t', '${_REGION}-docker.pkg.dev/${PROJECT_ID}/harmonia-v3-repo/harmonia-api:${SHORT_SHA}', '.']
     
     # Step 2: Push to Artifact Registry
     - name: 'gcr.io/cloud-builders/docker'
       args: ['push', '${_REGION}-docker.pkg.dev/${PROJECT_ID}/harmonia-v3-repo/harmonia-api:${SHORT_SHA}']
     
     # Step 3: Run Alembic migrations (using Cloud SQL Proxy sidecar)
     - name: 'gcr.io/cloud-builders/docker'
       args: ['run', '--network=cloudbuild',
              '${_REGION}-docker.pkg.dev/${PROJECT_ID}/harmonia-v3-repo/harmonia-api:${SHORT_SHA}',
              'alembic', 'upgrade', 'head']
       secretEnv: ['DATABASE_URL']
     
     # Step 4: Deploy to Cloud Run
     - name: 'gcr.io/google.com/cloudsdktool/cloud-sdk'
       args: ['gcloud', 'run', 'deploy', 'harmonia-api',
              '--image=${_REGION}-docker.pkg.dev/${PROJECT_ID}/harmonia-v3-repo/harmonia-api:${SHORT_SHA}',
              '--region=${_REGION}',
              '--platform=managed',
              '--cpu=2', '--memory=2Gi',
              '--min-instances=1', '--max-instances=10',
              '--timeout=120',
              '--concurrency=20',
              '--set-cloudsql-instances=${_CLOUD_SQL_INSTANCE}',
              '--vpc-connector=harmonia-vpc-connector',
              '--service-account=harmonia-v3-runner@${PROJECT_ID}.iam.gserviceaccount.com',
              '--set-secrets=GEMINI_API_KEY=GEMINI_API_KEY:latest,ANTHROPIC_API_KEY=ANTHROPIC_API_KEY:latest,FERNET_KEY=FERNET_KEY:latest,DATABASE_URL=DATABASE_URL:latest,REDIS_URL=REDIS_URL:latest',
              '--set-env-vars=GCP_PROJECT_ID=${PROJECT_ID},GCP_REGION=${_REGION},GCS_BUCKET_NAME=${PROJECT_ID}-harmonia-v3,ENVIRONMENT=production,LOG_LEVEL=INFO']

   substitutions:
     _REGION: europe-west2
     _CLOUD_SQL_INSTANCE: ''  # Set in Cloud Build trigger config

   availableSecrets:
     secretManager:
       - versionName: projects/${PROJECT_ID}/secrets/DATABASE_URL/versions/latest
         env: 'DATABASE_URL'
   ```

7. **Configure the Cloud Build GitHub trigger**: connect the GitHub repository to Cloud Build, set the trigger to fire on pushes to `main`, and configure the substitution variables (`_REGION`, `_CLOUD_SQL_INSTANCE`) in the trigger settings.

#### Part C: Monitoring & Alerting

8. **Configure Cloud Monitoring:**
   - Create a custom dashboard for Harmonia V3 with panels: Cloud Run request count, latency (P50/P95/P99), error rate, instance count, container CPU/memory utilisation, Cloud SQL connections/query latency/storage, Memorystore hit rate/memory usage/connections
   - Create alerting policies: Cloud Run error rate >2% for 5 minutes (notification: email + Slack), Cloud Run P95 latency >45s for 5 minutes, Cloud SQL CPU >80% for 10 minutes, Memorystore memory >80% for 10 minutes, Cloud Run instance count at max (scaling ceiling hit)
   - Configure structured logging: the `structlog` JSON logger output is automatically ingested by Cloud Logging. Create log-based metrics for: `@gemini_latency_ms > 30000` (slow Gemini calls), `@level = "error"` (error count), `@calibration_review` (calibration activity tracking). Create saved log queries in Cloud Logging for common debugging patterns

9. **Configure Cloud Run domain mapping**: map your custom domain to the Cloud Run service, provision the managed SSL certificate (automatic via Cloud Run), and configure DNS records (CNAME to `ghs.googlehosted.com`).

#### Part D: Production Validation

10. **Production validation checklist:**
    - Cloud Run service is `SERVING` with min 1 instance warm
    - `/health` returns 200 (liveness — confirms the app is running)
    - `/health/deep` returns 200 (confirms Gemini API connectivity, Cloud SQL reachable, Redis reachable, GCS bucket accessible)
    - Model weights loaded from GCS with SHA256 integrity check passing
    - Submit test questionnaire → verify profile creation → verify `parsing_evidence` table has entries with valid snippet offsets
    - Upload test HLA data → verify Fernet encryption + GCS storage + scoring
    - Trigger test match → verify Level 3 report contains evidence maps, Level 2A cites snippets, Level 1 has NO evidence
    - Run 10 cluster generation iterations → verify `calibration_examples` table populated with pending entries
    - Review 5 calibration examples via admin API → verify few-shot injection in subsequent Gemini prompts
    - Cloud Build pipeline: push a commit to `main` → verify image builds, migrations run, new revision deploys with zero downtime
    - Cloud Logging: verify structured JSON logs are queryable (`jsonPayload.gemini_latency_ms > 0`)
    - Cloud Monitoring: verify dashboard panels are populating and alerting policies are active
    - Verify signed URL generation for user photos (URL expires after 60 minutes)

#### GCP Cost Estimation (1,000 users/day)

| Service | Spec | Est. Monthly Cost |
|---------|------|-------------------|
| Cloud Run | 2 vCPU, 2GB RAM, min 1 instance, avg 2 instances | ~$50-70 |
| Cloud SQL | db-f1-micro (dev) / db-custom-2-4096 (prod) | ~$8 (dev) / ~$55 (prod) |
| Memorystore Redis | Basic 1GB | ~$35 |
| Cloud Storage | ~50GB photos + models, Standard class | ~$1-2 |
| Artifact Registry | Container images | ~$1 |
| Cloud Build | ~30 builds/month | ~$0 (free tier: 120 min/day) |
| Secret Manager | ~10 secrets, ~1000 accesses/day | ~$0 (free tier: 10K accesses) |
| Gemini API | ~1000 profiles/day × 42 calls each | ~$5-15 |
| **Total (dev)** | | **~$100-130/month** |
| **Total (prod)** | | **~$160-180/month** |

#### Part E: README Verification & Update

11. **Review and update the `README.md`** created in Session 0 (task 10). The Session 0 README was written as a comprehensive template with 10 sections covering project overview, complete file map, MetaFBP training file locations, GCP requirements, local development setup, GCP deployment instructions, frontend integration guide, API reference, environment variables, and troubleshooting. Now that the full system is built and deployed, review every section and:
    - Replace any placeholder values with actuals (GCS bucket name, Cloud Run URL, Cloud SQL instance connection string, custom domain)
    - Verify the complete file map is accurate — add any files created during Sessions 1-9 that weren't in the original Session 0 scaffolding
    - Update the API endpoint table to match the actual Swagger output at `/docs` (compare against the auto-generated OpenAPI spec)
    - Add an **Architecture Diagram** (ASCII or Mermaid) showing the full data flow: Client → Cloud Run (FastAPI) → Services → Cloud SQL / Memorystore / GCS / Gemini API / Anthropic API, with the match calculation flow highlighted
    - Flesh out the **Troubleshooting** section with real issues encountered during Sessions 0-10
    - Add the **Calibration Database Workflow** explanation for non-technical team members
    - Verify the README renders correctly on GitHub with no broken links or placeholder text

### Verification

Deploy the full system via `infra/setup-gcp.sh`, trigger a Cloud Build pipeline, and run through the entire production validation checklist. Verify that all GCP services are communicating correctly over private networking and that no credentials are exposed in logs or environment variable dumps. Verify the README renders correctly on GitHub and contains no broken links or placeholder text.

---

## Quick Reference: API Endpoint Summary

| Method | Endpoint | Purpose |
|--------|----------|---------|
| GET | `/health` | Liveness check |
| GET | `/health/deep` | Deep check (tests Gemini) |
| POST | `/api/v1/users` | Create user |
| POST | `/api/v1/questionnaire/submit` | Submit single question response |
| POST | `/api/v1/questionnaire/submit-all` | Submit all 6 responses at once |
| GET | `/api/v1/profile/{user_id}` | Get personality profile |
| POST | `/api/v1/visual/calibrate` | Submit face ratings for MetaFBP calibration |
| POST | `/api/v1/visual/score` | Score a target face for a user |
| POST | `/api/v1/swipe` | Record a swipe action |
| POST | `/api/v1/hla/upload` | Upload HLA genetic data |
| GET | `/api/v1/hla/compatibility/{a}/{b}` | Get HLA compatibility score |
| POST | `/api/v1/match/calculate/{a}/{b}` | Calculate full match |
| GET | `/api/v1/match/{match_id}` | Get match details |
| GET | `/api/v1/matches/{user_id}` | List user's matches |
| GET | `/api/v1/reports/{match_id}/summary` | Customer summary (Level 1, NO evidence) |
| GET | `/api/v1/reports/{match_id}/reasoning-chain` | Admin reasoning chain + evidence maps (Level 3) |
| GET | `/api/v1/reports/{match_id}/narrative` | Psych/visual narrative with evidence citations (Level 2A) |
| GET | `/api/v1/reports/{match_id}/hla-analysis` | HLA analysis (Level 2B) |
| GET | `/api/v1/reports/{match_id}/evidence-map/{user_id}` | Standalone evidence map for one user (admin) |
| GET | `/api/v1/admin/calibration/queue` | Pending calibration examples for review |
| POST | `/api/v1/admin/calibration/{id}/review` | Approve/correct/reject a calibration example |
| POST | `/api/v1/admin/calibration/bulk-review` | Batch review multiple examples at once |
| GET | `/api/v1/admin/calibration/stats` | Calibration coverage map and statistics |
| GET | `/api/v1/admin/calibration/effectiveness` | Drift report and convergence metrics |

---

## Frontend Readiness: Everything the Frontend Developer Needs

This section is a complete specification for building the Harmonia V3 frontend application. It documents every API contract, user flow, data shape, and real-time requirement so that a frontend developer can work independently from the backend team. The backend API is designed to be frontend-agnostic — it works equally well with React Native (mobile), Next.js (web), Flutter, or any framework that can make HTTP requests and handle JSON responses.

### Recommended Frontend Stack

The backend is framework-agnostic, but given the nature of the application (dating app with image-heavy UI, swipe interactions, real-time match notifications), the following stack is recommended:

**For mobile-first (recommended for a dating app):** React Native with Expo, or Flutter. The swipe UX, camera integration for photos, and push notifications are all critical and work better as native experiences.

**For web:** Next.js 14+ with App Router, Tailwind CSS, and Framer Motion for animations. The API is REST-based so no GraphQL layer is needed. Use `next/image` for optimised photo loading from GCS signed URLs.

**For both:** You will need an authentication layer — the backend currently does not implement user auth (this is deliberately scoped out of the backend build plan). The frontend should implement auth (Firebase Auth, Clerk, Auth0, or Supabase Auth are all compatible) and send the authenticated user's ID in API requests. The `user_id` in all endpoints corresponds to whatever ID your auth provider assigns.

### GCP Requirements for Frontend

If deploying the frontend on GCP alongside the backend, you will need:

| Service | Purpose | Notes |
|---------|---------|-------|
| Firebase Hosting or Cloud Run | Host the web frontend | Firebase for static/SSG, Cloud Run for SSR (Next.js) |
| Firebase Auth (optional) | User authentication | Free tier covers 50K MAU. Integrates with GCP IAM |
| Cloud CDN | Photo delivery | Sits in front of GCS signed URLs for faster photo loading |
| Cloud DNS | Domain management | Only if not using an external DNS provider |

The backend's CORS middleware is configured to allow your frontend domain — you'll set this in the `ALLOWED_ORIGINS` environment variable on the Cloud Run backend service.

### User Flows & Corresponding API Calls

**Flow 1: Onboarding (New User Registration)**

The onboarding flow has 4 sequential phases. Each phase must complete before the next becomes available. The frontend should show a progress indicator (e.g., "Step 2 of 4").

Step 1 — Account Creation:
```
POST /api/v1/users
Body: { "email": "...", "display_name": "...", "age": 28, "gender": "female", "location": "London" }
Response: { "id": "uuid", "email": "...", ... }
```
Store the returned `id` — this is the `user_id` for all subsequent calls.

Step 2 — Photo Upload:
Upload 1-6 photos. The backend stores them in GCS and returns signed URLs for display.
```
POST /api/v1/users/{user_id}/photos
Body: multipart/form-data with image files
Response: { "photos": ["https://storage.googleapis.com/...", ...] }
```
The signed URLs expire after 60 minutes. The frontend should re-fetch them via `GET /api/v1/users/{user_id}` when needed rather than caching expired URLs.

Step 3 — Visual Calibration (MetaFBP):
The user must rate a set of face images to calibrate their personal beauty prediction model. The backend serves a stream of calibration images and the user rates each 1-5 (or swipes left/right with an intensity slider). A minimum of 10 ratings is required, 20+ recommended for better personalisation.
```
GET /api/v1/visual/calibration-set
Response: { "images": [{"id": "img_001", "url": "https://..."}, ...] }

POST /api/v1/visual/calibrate
Body: { "user_id": "...", "ratings": [{"image_id": "img_001", "rating": 4}, {"image_id": "img_002", "rating": 2}, ...] }
Response: { "status": "calibrated", "support_set_size": 15, "adapted_weights_cached": true }
```
This triggers the MetaFBP inner-loop adaptation (k=10 gradient steps) on the backend. The adapted weights are cached in Redis. This step takes 2-5 seconds — show a loading animation like "Learning your preferences..."

Step 4 — Personality Questionnaire (PIIP):
Present the 6 Felix questions one at a time. Each question is a scenario with a free-text response field. Enforce the 25-150 word constraint in the frontend with a live word counter. Show a soft warning at 25 words ("Almost there — a bit more detail helps us understand you better") and a hard stop at 150 words ("That's enough detail — keep it concise").

You can submit one at a time (useful for save-as-you-go) or all at once:
```
POST /api/v1/questionnaire/submit
Body: { "user_id": "...", "question_number": 1, "response_text": "I'd probably suggest..." }
Response: { "status": "parsed", "question_number": 1, "word_count": 47 }
```
Or batch:
```
POST /api/v1/questionnaire/submit-all
Body: { "user_id": "...", "responses": [{"question_number": 1, "response_text": "..."}, ...] }
Response: { "status": "profile_created", "profile_id": "...", "quality_tier": "high", "quality_score": 84.2 }
```
The batch endpoint triggers full Gemini parsing (42 API calls — 7 sins × 6 questions) and profile aggregation. This takes 15-30 seconds. Show a multi-step progress animation: "Analysing your responses..." → "Building your personality profile..." → "Profile complete!"

The 6 questions (in order, not randomised):
1. "You're at a group dinner with friends. The bill arrives and it's not split evenly..."
2. "You receive an unexpected expense notification — your car needs repairs / your laptop dies..."
3. "You get a surprise day off this weekend with no obligations..."
4. "You and a friend worked equally on a project, but they received more credit..."
5. "A close friend calls you at midnight in a crisis. You have an important meeting at 8am..."
6. "Your manager gives you mixed feedback — praise for one thing, criticism for another..."
(The full question texts are stored in the `questions` table and can be fetched via `GET /api/v1/questions`.)

**Flow 2: Discovery & Swiping**

After onboarding, the user enters the discovery feed. The backend serves candidate profiles ranked by predicted visual attractiveness (MetaFBP scores the user's photos through each candidate's personalised model).

```
GET /api/v1/discover/{user_id}?limit=20
Response: {
  "candidates": [
    {
      "user_id": "...",
      "display_name": "Sarah",
      "age": 27,
      "photos": ["https://signed-url-1...", "https://signed-url-2..."],
      "visual_score": 82.3,   // How attractive THIS user finds the candidate (not shown to user)
      "distance_km": 4.2
    }, ...
  ]
}
```
The `visual_score` is used for ranking but should NOT be displayed to the user. The frontend presents the candidates as swipeable cards (Tinder-style).

When the user swipes:
```
POST /api/v1/swipe
Body: { "swiper_id": "...", "target_id": "...", "direction": "right" }
Response: { "status": "recorded", "is_mutual_match": true|false }
```
If `is_mutual_match: true`, the backend has already detected that both users swiped right on each other. This triggers the full matching pipeline (personality similarity + HLA compatibility + report generation). The frontend should show a "It's a Match!" screen.

**Flow 3: Match Results & Compatibility Report**

When a mutual match occurs, the frontend fetches the customer-facing summary (Level 1 only — never Level 2 or 3):

```
GET /api/v1/reports/{match_id}/summary
Response: {
  "display_score": 78,
  "badges": ["Strong Chemistry", "Personality Match"],
  "synopsis": {
    "headline": "You two have real potential",
    "body": "You share a similar approach to conflict and both value spontaneity..."
  },
  "compatibility_breakdown": {
    "physical": { "score": 82, "label": "Strong attraction" },
    "personality": { "score": 71, "label": "Good alignment" },
    "chemistry": { "score": 92, "label": "Strong chemistry signal" }
  },
  "shared_traits": [
    "You're both generous and easygoing about money",
    "You're both spontaneous and up for adventure",
    "You handle conflict in similar ways"
  ],
  "conversation_starters": [
    "Ask about their ideal spontaneous weekend",
    "Share your thoughts on splitting bills with friends"
  ]
}
```

**CRITICAL:** The Level 1 summary contains NO sin labels, NO raw scores, NO evidence snippets, and NO quoted fragments from the user's responses. It uses only friendly natural language. The frontend should never attempt to fetch Level 2 or Level 3 reports — those are admin-only and the API will return 403.

The personality traits are revealed progressively based on the similarity tier:
- `strong_fit` (≥0.60): Show 4 shared traits with highlight animation
- `good_fit` (≥0.40): Show 3 shared traits
- `moderate_fit` (≥0.25): Show 2 shared traits
- `low_fit` (<0.25): Show 1 trait + emphasise physical chemistry instead

The HLA/chemistry badge display follows the same principle — only show positive signals:
- ≥75: 🔥 "Strong chemistry signal"
- 50-74: ✨ "Good chemistry"
- 25-49: 💫 "Some chemistry"
- <25: Don't show anything (never display negative chemistry)

**Flow 4: HLA Upload (Optional)**

HLA genetic data upload is optional and can happen at any point after onboarding. The frontend should present this as a "Boost your compatibility insights" feature, not a requirement.

```
POST /api/v1/hla/upload
Body: multipart/form-data with genetic data file (23andMe, AncestryDNA, etc.)
Response: { "status": "processed", "snp_count": 248, "imputation_confidence": 0.87 }
```

If the user hasn't uploaded HLA data, the matching engine automatically redistributes the 10% biological weight proportionally to visual (67%) and personality (33%). The Level 1 summary will simply not include the chemistry badge.

**Flow 5: Admin Dashboard (Calibration Review)**

The admin dashboard is a separate frontend (can be a simple internal web app) for reviewing calibration examples. The flows are:

```
GET /api/v1/admin/calibration/queue?status=pending&question_number=1&sin=wrath&limit=20
Response: {
  "examples": [
    {
      "id": "...",
      "question_number": 1,
      "response_text": "I'd probably suggest we just split it evenly...",
      "sin": "wrath",
      "gemini_raw_score": -3.0,
      "gemini_raw_confidence": 0.88,
      "gemini_raw_evidence": "I hate those awkward moments...",
      "review_status": "pending"
    }, ...
  ],
  "total_pending": 142
}
```

The admin reads the response text, sees Gemini's score and evidence, and decides: approve (Gemini was right), correct (provide the right score + notes), or reject (garbage example).

```
POST /api/v1/admin/calibration/{example_id}/review
Body: { "action": "correct", "validated_score": -1.0, "notes": "Sarcasm, not genuine anger", "reviewer": "avery" }
Response: { "id": "...", "review_status": "corrected", "validated_score": -1.0 }
```

### Response Shapes & Error Handling

All API responses follow a consistent shape. Success responses return the data directly (not wrapped in a `data` envelope). Error responses return:

```json
{
  "detail": "Human-readable error message",
  "error_code": "MACHINE_READABLE_CODE",
  "field": "optional_field_name"
}
```

Common HTTP status codes: 200 (success), 201 (created), 400 (validation error — show the `detail` to the user), 403 (forbidden — user doesn't have access), 404 (not found), 409 (conflict — e.g., duplicate swipe), 422 (unprocessable entity — Pydantic validation failure, `detail` contains field-level errors), 429 (rate limited — retry after the `Retry-After` header), 500 (server error — show a generic "Something went wrong" message), 504 (gateway timeout — Gemini took too long, suggest retrying).

For the questionnaire submission specifically, the 422 error will include word count validation details:
```json
{ "detail": "Response too short: 18 words (minimum 25)", "error_code": "WORD_COUNT_BELOW_MINIMUM", "field": "response_text" }
```

### Real-Time Considerations

The current backend is REST-only (no WebSocket support). For real-time match notifications ("It's a Match!" appearing instantly when the other person swipes right), you have two options:

**Option A — Polling (simpler, recommended for MVP):** The frontend polls `GET /api/v1/matches/{user_id}?since={last_check_timestamp}` every 10-15 seconds while the user is in the discovery feed. New matches appear in the response.

**Option B — WebSocket (better UX, add later):** Add a WebSocket endpoint (`ws /api/v1/ws/{user_id}`) to the backend that pushes match notifications in real time. This requires adding `websockets` to the backend dependencies and a Redis pub/sub channel for cross-instance communication (since Cloud Run may have multiple instances). This should be implemented as a follow-up to the core backend build.

### Photo Requirements for the Frontend

User photos must be validated client-side before upload: JPEG or PNG format, minimum 400×400 pixels, maximum 10MB per image, 1-6 photos per user. The first photo is the primary display photo. Photos are stored in GCS and served via signed URLs that expire after 60 minutes — the frontend should re-fetch user data rather than caching photo URLs long-term.

For MetaFBP calibration images, the backend serves pre-processed images at 224×224. The frontend just displays them and collects ratings — no image processing is needed client-side.

### Authentication Architecture (Not Implemented in Backend — Frontend Responsibility)

The backend currently uses `user_id` as a path/body parameter with no authentication middleware. Before production launch, you need to:

1. Implement an auth provider (Firebase Auth recommended for GCP alignment) in the frontend.
2. Add auth middleware to the backend that validates the JWT token from the auth provider and extracts the `user_id`.
3. Add the JWT token as a `Bearer` token in the `Authorization` header of every API request.
4. The admin endpoints (`/api/v1/admin/*`) need a separate role-based check — only users with the `admin` role should be able to access calibration review, evidence maps, and Level 2/3 reports.

This is deliberately left as a frontend-driven integration because the auth choice (Firebase, Clerk, Auth0, etc.) depends on your frontend stack and business requirements.

### CORS Configuration

The backend's CORS middleware allows all origins in development (`ALLOWED_ORIGINS=*`). For production, set `ALLOWED_ORIGINS` to your frontend domain(s) in the Cloud Run environment variables:
```
ALLOWED_ORIGINS=https://app.harmonia.com,https://admin.harmonia.com
```

---

## Key Formulas Reference

**Visual Score:** `S_vis = (MetaFBP_Component × 0.6) + (T_match_positive × 0.25) + (T_match_negative × 0.15)`

**MetaFBP Component:** `(raw_score - 1) × 25` mapping [1,5] → [0,100]

**MetaFBP Adaptation:** `θ_f_dynamic = θ'_f + λ × G_θ_g(x)` where λ = 0.01

**Trait Weight:** `W_trait = F_trait × R_corr × sign(avg_rating - 3.0)`

**CWMV Aggregation:** `Score_trait = Σ(Item_score × Confidence) / Σ(Confidence)`

**Similarity:** `Σ(trait_similarity × avg_confidence × sin_weight) / 7.4`

**Trait Similarity:** `1 - (|score_a - score_b| / 10)`

**Biological Score:** `S_bio = (N_unique / N_total) × 100`

**WtM:** `(0.4 × S_vis) + (0.3 × S_psych) + (0.3 × S_bio)`

**Reciprocal Score:** `√(score_A→B × score_B→A)`

---

*Document generated from: Harmonia V3 Master Technical Specification v8, PIIP Specification (ilovepdf_merged.pdf), and PIIP Cluster Generation Task v3. All section references, formulas, schemas, and worked examples are traced directly to these source documents. Version 1.1 added: (1) Evidence Traceability — every sin recognition stored with exact response snippet and character offsets, flowing into Level 3 and Level 2 reports but never Level 1; (2) Gemini Calibration Database — a self-improving loop where Haiku-generated responses are Gemini-parsed, admin-reviewed, and fed back as few-shot examples to anchor future Gemini scoring. Version 1.2 added: (3) Comprehensive README specification — a complete file-by-file map, GCP requirements table, step-by-step local and cloud deployment instructions, and troubleshooting guide, all generated as part of Session 10; (4) Frontend Integration Guide — complete API contract documentation, user flow specifications with request/response shapes, progressive trait reveal logic, photo handling, real-time considerations, authentication architecture, and admin dashboard flow, designed so a frontend developer can build the client application independently from the backend team.*
