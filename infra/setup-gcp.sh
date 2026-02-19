#!/usr/bin/env bash
# =============================================================================
# Harmonia V3 — GCP Infrastructure Bootstrap Script
# =============================================================================
# Usage:
#   ./setup-gcp.sh <PROJECT_ID> [REGION]
#
# Example:
#   ./setup-gcp.sh my-harmonia-project europe-west2
# =============================================================================

set -euo pipefail

# ---------------------------------------------------------------------------
# Arguments
# ---------------------------------------------------------------------------
PROJECT_ID="${1:?Usage: $0 <PROJECT_ID> [REGION]}"
REGION="${2:-europe-west2}"

# ---------------------------------------------------------------------------
# Derived constants
# ---------------------------------------------------------------------------
SERVICE_NAME="harmonia-api"
SA_NAME="harmonia-v3-runner"
SA_EMAIL="${SA_NAME}@${PROJECT_ID}.iam.gserviceaccount.com"
ARTIFACT_REPO="harmonia-v3-repo"
SQL_INSTANCE="harmonia-v3-pg"
SQL_DB="harmonia"
SQL_USER="harmonia_user"
SQL_PASSWORD="$(openssl rand -base64 24 | tr -dc 'a-zA-Z0-9' | head -c 32)"
REDIS_INSTANCE="harmonia-v3-redis"
VPC_CONNECTOR="harmonia-vpc-connector"
GCS_BUCKET="${PROJECT_ID}-harmonia-v3"
NETWORK="default"

echo "============================================================"
echo "  Harmonia V3 — GCP Bootstrap"
echo "============================================================"
echo "  Project : ${PROJECT_ID}"
echo "  Region  : ${REGION}"
echo "============================================================"
echo ""

# ---------------------------------------------------------------------------
# 0. Set active project
# ---------------------------------------------------------------------------
echo "[0/10] Setting active project to ${PROJECT_ID} ..."
gcloud config set project "${PROJECT_ID}"
echo ""

# ---------------------------------------------------------------------------
# 1. Enable required APIs
# ---------------------------------------------------------------------------
echo "[1/10] Enabling required GCP APIs ..."
gcloud services enable \
  cloudbuild.googleapis.com \
  run.googleapis.com \
  sqladmin.googleapis.com \
  redis.googleapis.com \
  secretmanager.googleapis.com \
  artifactregistry.googleapis.com \
  vpcaccess.googleapis.com \
  compute.googleapis.com \
  iam.googleapis.com \
  cloudresourcemanager.googleapis.com \
  storage.googleapis.com
echo "  -> APIs enabled."
echo ""

# ---------------------------------------------------------------------------
# 2. Create Artifact Registry repository
# ---------------------------------------------------------------------------
echo "[2/10] Creating Artifact Registry repository '${ARTIFACT_REPO}' ..."
if gcloud artifacts repositories describe "${ARTIFACT_REPO}" \
     --location="${REGION}" --format="value(name)" 2>/dev/null; then
  echo "  -> Repository already exists, skipping."
else
  gcloud artifacts repositories create "${ARTIFACT_REPO}" \
    --repository-format=docker \
    --location="${REGION}" \
    --description="Harmonia V3 Docker images"
  echo "  -> Repository created."
fi
echo ""

# ---------------------------------------------------------------------------
# 3. Provision Cloud SQL (PostgreSQL 15)
# ---------------------------------------------------------------------------
echo "[3/10] Provisioning Cloud SQL instance '${SQL_INSTANCE}' ..."
if gcloud sql instances describe "${SQL_INSTANCE}" --format="value(name)" 2>/dev/null; then
  echo "  -> Instance already exists, skipping creation."
else
  gcloud sql instances create "${SQL_INSTANCE}" \
    --database-version=POSTGRES_15 \
    --tier=db-f1-micro \
    --region="${REGION}" \
    --storage-type=SSD \
    --storage-size=10GB \
    --storage-auto-increase \
    --backup-start-time=03:00 \
    --enable-point-in-time-recovery \
    --availability-type=zonal \
    --network="${NETWORK}" \
    --no-assign-ip
  echo "  -> Cloud SQL instance created."
fi

# Get the connection name
SQL_CONNECTION_NAME="$(gcloud sql instances describe "${SQL_INSTANCE}" \
  --format='value(connectionName)')"
echo "  -> Connection name: ${SQL_CONNECTION_NAME}"
echo ""

# ---------------------------------------------------------------------------
# 4. Create database and user
# ---------------------------------------------------------------------------
echo "[4/10] Creating database '${SQL_DB}' and user '${SQL_USER}' ..."

# Create database (ignore error if exists)
gcloud sql databases create "${SQL_DB}" \
  --instance="${SQL_INSTANCE}" 2>/dev/null \
  && echo "  -> Database '${SQL_DB}' created." \
  || echo "  -> Database '${SQL_DB}' already exists."

# Create user (ignore error if exists)
gcloud sql users create "${SQL_USER}" \
  --instance="${SQL_INSTANCE}" \
  --password="${SQL_PASSWORD}" 2>/dev/null \
  && echo "  -> User '${SQL_USER}' created." \
  || echo "  -> User '${SQL_USER}' already exists (password NOT changed)."

DATABASE_URL="postgresql+asyncpg://${SQL_USER}:${SQL_PASSWORD}@/${SQL_DB}?host=/cloudsql/${SQL_CONNECTION_NAME}"
echo ""

# ---------------------------------------------------------------------------
# 5. Provision Memorystore Redis (Basic, 1 GB)
# ---------------------------------------------------------------------------
echo "[5/10] Provisioning Memorystore Redis '${REDIS_INSTANCE}' ..."
if gcloud redis instances describe "${REDIS_INSTANCE}" \
     --region="${REGION}" --format="value(name)" 2>/dev/null; then
  echo "  -> Redis instance already exists, skipping."
else
  gcloud redis instances create "${REDIS_INSTANCE}" \
    --size=1 \
    --region="${REGION}" \
    --tier=basic \
    --redis-version=redis_7_0 \
    --network="${NETWORK}"
  echo "  -> Redis instance created."
fi

REDIS_HOST="$(gcloud redis instances describe "${REDIS_INSTANCE}" \
  --region="${REGION}" --format='value(host)')"
REDIS_PORT="$(gcloud redis instances describe "${REDIS_INSTANCE}" \
  --region="${REGION}" --format='value(port)')"
REDIS_URL="redis://${REDIS_HOST}:${REDIS_PORT}/0"
echo "  -> Redis URL: ${REDIS_URL}"
echo ""

# ---------------------------------------------------------------------------
# 6. Create Serverless VPC Access connector
# ---------------------------------------------------------------------------
echo "[6/10] Creating VPC connector '${VPC_CONNECTOR}' ..."
if gcloud compute networks vpc-access connectors describe "${VPC_CONNECTOR}" \
     --region="${REGION}" --format="value(name)" 2>/dev/null; then
  echo "  -> VPC connector already exists, skipping."
else
  gcloud compute networks vpc-access connectors create "${VPC_CONNECTOR}" \
    --region="${REGION}" \
    --network="${NETWORK}" \
    --range="10.8.0.0/28" \
    --min-instances=2 \
    --max-instances=3 \
    --machine-type=e2-micro
  echo "  -> VPC connector created."
fi
echo ""

# ---------------------------------------------------------------------------
# 7. Create GCS bucket with lifecycle rules and CORS
# ---------------------------------------------------------------------------
echo "[7/10] Creating GCS bucket 'gs://${GCS_BUCKET}' ..."
if gsutil ls -b "gs://${GCS_BUCKET}" 2>/dev/null; then
  echo "  -> Bucket already exists, skipping creation."
else
  gsutil mb -l "${REGION}" -c STANDARD "gs://${GCS_BUCKET}"
  echo "  -> Bucket created."
fi

# Lifecycle: delete temp objects after 30 days
cat > /tmp/harmonia-lifecycle.json <<'LIFECYCLE_EOF'
{
  "lifecycle": {
    "rule": [
      {
        "action": {"type": "Delete"},
        "condition": {
          "age": 30,
          "matchesPrefix": ["tmp/"]
        }
      },
      {
        "action": {"type": "Delete"},
        "condition": {
          "age": 90,
          "matchesPrefix": ["logs/"]
        }
      }
    ]
  }
}
LIFECYCLE_EOF
gsutil lifecycle set /tmp/harmonia-lifecycle.json "gs://${GCS_BUCKET}"
echo "  -> Lifecycle rules applied."

# CORS configuration
cat > /tmp/harmonia-cors.json <<'CORS_EOF'
[
  {
    "origin": ["*"],
    "method": ["GET", "PUT", "POST"],
    "responseHeader": ["Content-Type"],
    "maxAgeSeconds": 3600
  }
]
CORS_EOF
gsutil cors set /tmp/harmonia-cors.json "gs://${GCS_BUCKET}"
echo "  -> CORS policy applied."

# Create expected prefixes
gsutil cp /dev/null "gs://${GCS_BUCKET}/models/.keep"
gsutil cp /dev/null "gs://${GCS_BUCKET}/uploads/.keep"
echo "  -> Folder stubs created (models/, uploads/)."
echo ""

# ---------------------------------------------------------------------------
# 8. Create Secret Manager secrets
# ---------------------------------------------------------------------------
echo "[8/10] Creating Secret Manager secrets ..."

FERNET_KEY="$(python3 -c 'from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())' 2>/dev/null \
  || openssl rand -base64 32)"

declare -A SECRETS=(
  ["GEMINI_API_KEY"]="PLACEHOLDER-change-me"
  ["ANTHROPIC_API_KEY"]="PLACEHOLDER-change-me"
  ["FERNET_KEY"]="${FERNET_KEY}"
  ["DATABASE_URL"]="${DATABASE_URL}"
  ["REDIS_URL"]="${REDIS_URL}"
)

for secret_name in "${!SECRETS[@]}"; do
  if gcloud secrets describe "${secret_name}" --format="value(name)" 2>/dev/null; then
    echo "  -> Secret '${secret_name}' already exists, adding new version ..."
    printf '%s' "${SECRETS[$secret_name]}" | \
      gcloud secrets versions add "${secret_name}" --data-file=-
  else
    printf '%s' "${SECRETS[$secret_name]}" | \
      gcloud secrets create "${secret_name}" --data-file=- --replication-policy=automatic
    echo "  -> Secret '${secret_name}' created."
  fi
done
echo ""

# ---------------------------------------------------------------------------
# 9. Create service account and bind IAM roles
# ---------------------------------------------------------------------------
echo "[9/10] Creating service account '${SA_NAME}' and binding IAM roles ..."
if gcloud iam service-accounts describe "${SA_EMAIL}" --format="value(email)" 2>/dev/null; then
  echo "  -> Service account already exists."
else
  gcloud iam service-accounts create "${SA_NAME}" \
    --display-name="Harmonia V3 Cloud Run Runner"
  echo "  -> Service account created."
fi

ROLES=(
  "roles/cloudsql.client"
  "roles/secretmanager.secretAccessor"
  "roles/storage.objectAdmin"
  "roles/redis.editor"
  "roles/logging.logWriter"
  "roles/monitoring.metricWriter"
  "roles/run.invoker"
)

for role in "${ROLES[@]}"; do
  gcloud projects add-iam-policy-binding "${PROJECT_ID}" \
    --member="serviceAccount:${SA_EMAIL}" \
    --role="${role}" \
    --condition=None \
    --quiet
  echo "  -> Bound ${role}"
done

# Grant Cloud Build SA permission to deploy to Cloud Run
CB_SA="$(gcloud projects describe "${PROJECT_ID}" --format='value(projectNumber)')@cloudbuild.gserviceaccount.com"
gcloud projects add-iam-policy-binding "${PROJECT_ID}" \
  --member="serviceAccount:${CB_SA}" \
  --role="roles/run.admin" \
  --condition=None \
  --quiet
gcloud iam service-accounts add-iam-policy-binding "${SA_EMAIL}" \
  --member="serviceAccount:${CB_SA}" \
  --role="roles/iam.serviceAccountUser" \
  --quiet
echo "  -> Cloud Build SA permissions configured."
echo ""

# ---------------------------------------------------------------------------
# 10. Summary
# ---------------------------------------------------------------------------
echo "============================================================"
echo "  Harmonia V3 — GCP Bootstrap COMPLETE"
echo "============================================================"
echo ""
echo "  Project ID          : ${PROJECT_ID}"
echo "  Region              : ${REGION}"
echo ""
echo "  Artifact Registry   : ${REGION}-docker.pkg.dev/${PROJECT_ID}/${ARTIFACT_REPO}"
echo "  Cloud SQL Instance  : ${SQL_CONNECTION_NAME}"
echo "  Cloud SQL Database  : ${SQL_DB}"
echo "  Cloud SQL User      : ${SQL_USER}"
echo "  Cloud SQL Password  : ${SQL_PASSWORD}"
echo "  Redis Host          : ${REDIS_HOST}:${REDIS_PORT}"
echo "  VPC Connector       : ${VPC_CONNECTOR}"
echo "  GCS Bucket          : gs://${GCS_BUCKET}"
echo "  Service Account     : ${SA_EMAIL}"
echo ""
echo "  Secrets stored in Secret Manager:"
echo "    - GEMINI_API_KEY      (update with real key)"
echo "    - ANTHROPIC_API_KEY   (update with real key)"
echo "    - FERNET_KEY"
echo "    - DATABASE_URL"
echo "    - REDIS_URL"
echo ""
echo "  Next steps:"
echo "    1. Update GEMINI_API_KEY and ANTHROPIC_API_KEY secrets:"
echo "       echo -n 'your-key' | gcloud secrets versions add GEMINI_API_KEY --data-file=-"
echo "    2. Upload model weights to GCS:"
echo "       gsutil cp models/*.pth gs://${GCS_BUCKET}/models/"
echo "    3. Connect Cloud Build to your repo and trigger a build."
echo "    4. Update infra/cloudbuild.yaml _CLOUD_SQL_INSTANCE:"
echo "       _CLOUD_SQL_INSTANCE: '${SQL_CONNECTION_NAME}'"
echo ""
echo "============================================================"
