# =============================================================================
# Harmonia V3 — Terraform Main Configuration
# =============================================================================
# Provisions: Cloud SQL, Memorystore Redis, Cloud Run, GCS, Secret Manager,
#             VPC Connector, Artifact Registry, IAM service account.
# =============================================================================

terraform {
  required_version = ">= 1.5.0"

  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 5.0"
    }
    random = {
      source  = "hashicorp/random"
      version = "~> 3.5"
    }
  }

  # Uncomment and configure for remote state:
  # backend "gcs" {
  #   bucket = "your-tf-state-bucket"
  #   prefix = "harmonia-v3"
  # }
}

provider "google" {
  project = var.project_id
  region  = var.region
}

# ---------------------------------------------------------------------------
# Locals
# ---------------------------------------------------------------------------
locals {
  gcs_bucket_name = var.gcs_bucket_name != "" ? var.gcs_bucket_name : "${var.project_id}-harmonia-v3"
  sa_email        = google_service_account.harmonia_runner.email
}

# ---------------------------------------------------------------------------
# Enable APIs
# ---------------------------------------------------------------------------
resource "google_project_service" "apis" {
  for_each = toset([
    "cloudbuild.googleapis.com",
    "run.googleapis.com",
    "sqladmin.googleapis.com",
    "redis.googleapis.com",
    "secretmanager.googleapis.com",
    "artifactregistry.googleapis.com",
    "vpcaccess.googleapis.com",
    "compute.googleapis.com",
    "iam.googleapis.com",
    "storage.googleapis.com",
  ])

  project = var.project_id
  service = each.value

  disable_dependent_services = false
  disable_on_destroy         = false
}

# ---------------------------------------------------------------------------
# Artifact Registry
# ---------------------------------------------------------------------------
resource "google_artifact_registry_repository" "harmonia_repo" {
  location      = var.region
  repository_id = "harmonia-v3-repo"
  format        = "DOCKER"
  description   = "Harmonia V3 Docker images"

  depends_on = [google_project_service.apis]
}

# ---------------------------------------------------------------------------
# Cloud SQL — PostgreSQL 15
# ---------------------------------------------------------------------------
resource "random_password" "sql_password" {
  length  = 32
  special = false
}

resource "google_sql_database_instance" "harmonia_pg" {
  name             = var.sql_instance_name
  database_version = "POSTGRES_15"
  region           = var.region

  settings {
    tier              = var.sql_tier
    availability_type = "ZONAL"
    disk_type         = "PD_SSD"
    disk_size         = 10
    disk_autoresize   = true

    backup_configuration {
      enabled                        = true
      start_time                     = "03:00"
      point_in_time_recovery_enabled = true
      transaction_log_retention_days = 7
    }

    ip_configuration {
      ipv4_enabled    = false
      private_network = "projects/${var.project_id}/global/networks/${var.network}"
    }

    database_flags {
      name  = "max_connections"
      value = "100"
    }
  }

  deletion_protection = true

  depends_on = [google_project_service.apis]
}

resource "google_sql_database" "harmonia_db" {
  name     = var.sql_database_name
  instance = google_sql_database_instance.harmonia_pg.name
}

resource "google_sql_user" "harmonia_user" {
  name     = var.sql_user
  instance = google_sql_database_instance.harmonia_pg.name
  password = random_password.sql_password.result
}

# ---------------------------------------------------------------------------
# Memorystore Redis
# ---------------------------------------------------------------------------
resource "google_redis_instance" "harmonia_redis" {
  name           = var.redis_instance_name
  tier           = var.redis_tier
  memory_size_gb = var.redis_memory_size_gb
  region         = var.region
  redis_version  = "REDIS_7_0"

  authorized_network = "projects/${var.project_id}/global/networks/${var.network}"

  depends_on = [google_project_service.apis]
}

# ---------------------------------------------------------------------------
# Serverless VPC Access Connector
# ---------------------------------------------------------------------------
resource "google_vpc_access_connector" "harmonia_connector" {
  name          = var.vpc_connector_name
  region        = var.region
  network       = var.network
  ip_cidr_range = var.vpc_connector_cidr
  min_instances = 2
  max_instances = 3
  machine_type  = "e2-micro"

  depends_on = [google_project_service.apis]
}

# ---------------------------------------------------------------------------
# GCS Bucket
# ---------------------------------------------------------------------------
resource "google_storage_bucket" "harmonia_bucket" {
  name          = local.gcs_bucket_name
  location      = var.region
  storage_class = "STANDARD"

  uniform_bucket_level_access = true

  lifecycle_rule {
    condition {
      age                = 30
      matches_prefix     = ["tmp/"]
    }
    action {
      type = "Delete"
    }
  }

  lifecycle_rule {
    condition {
      age                = 90
      matches_prefix     = ["logs/"]
    }
    action {
      type = "Delete"
    }
  }

  cors {
    origin          = ["*"]
    method          = ["GET", "PUT", "POST"]
    response_header = ["Content-Type"]
    max_age_seconds = 3600
  }

  depends_on = [google_project_service.apis]
}

# ---------------------------------------------------------------------------
# Secret Manager Secrets
# ---------------------------------------------------------------------------
resource "google_secret_manager_secret" "gemini_api_key" {
  secret_id = "GEMINI_API_KEY"

  replication {
    auto {}
  }

  depends_on = [google_project_service.apis]
}

resource "google_secret_manager_secret" "anthropic_api_key" {
  secret_id = "ANTHROPIC_API_KEY"

  replication {
    auto {}
  }

  depends_on = [google_project_service.apis]
}

resource "google_secret_manager_secret" "fernet_key" {
  secret_id = "FERNET_KEY"

  replication {
    auto {}
  }

  depends_on = [google_project_service.apis]
}

resource "google_secret_manager_secret" "database_url" {
  secret_id = "DATABASE_URL"

  replication {
    auto {}
  }

  depends_on = [google_project_service.apis]
}

resource "google_secret_manager_secret_version" "database_url_value" {
  secret      = google_secret_manager_secret.database_url.id
  secret_data = "postgresql+asyncpg://${var.sql_user}:${random_password.sql_password.result}@/${var.sql_database_name}?host=/cloudsql/${google_sql_database_instance.harmonia_pg.connection_name}"
}

resource "google_secret_manager_secret" "redis_url" {
  secret_id = "REDIS_URL"

  replication {
    auto {}
  }

  depends_on = [google_project_service.apis]
}

resource "google_secret_manager_secret_version" "redis_url_value" {
  secret      = google_secret_manager_secret.redis_url.id
  secret_data = "redis://${google_redis_instance.harmonia_redis.host}:${google_redis_instance.harmonia_redis.port}/0"
}

# ---------------------------------------------------------------------------
# Service Account + IAM
# ---------------------------------------------------------------------------
resource "google_service_account" "harmonia_runner" {
  account_id   = "harmonia-v3-runner"
  display_name = "Harmonia V3 Cloud Run Runner"

  depends_on = [google_project_service.apis]
}

locals {
  sa_roles = [
    "roles/cloudsql.client",
    "roles/secretmanager.secretAccessor",
    "roles/storage.objectAdmin",
    "roles/redis.editor",
    "roles/logging.logWriter",
    "roles/monitoring.metricWriter",
    "roles/run.invoker",
  ]
}

resource "google_project_iam_member" "harmonia_runner_roles" {
  for_each = toset(local.sa_roles)

  project = var.project_id
  role    = each.value
  member  = "serviceAccount:${local.sa_email}"
}

# ---------------------------------------------------------------------------
# Cloud Run Service
# ---------------------------------------------------------------------------
resource "google_cloud_run_v2_service" "harmonia_api" {
  name     = var.cloud_run_service_name
  location = var.region
  ingress  = "INGRESS_TRAFFIC_ALL"

  template {
    scaling {
      min_instance_count = var.cloud_run_min_instances
      max_instance_count = var.cloud_run_max_instances
    }

    vpc_access {
      connector = google_vpc_access_connector.harmonia_connector.id
      egress    = "PRIVATE_RANGES_ONLY"
    }

    service_account = local.sa_email
    timeout         = "120s"

    max_instance_request_concurrency = var.cloud_run_concurrency

    volumes {
      name = "cloudsql"
      cloud_sql_instance {
        instances = [google_sql_database_instance.harmonia_pg.connection_name]
      }
    }

    containers {
      image = var.container_image != "" ? var.container_image : "${var.region}-docker.pkg.dev/${var.project_id}/harmonia-v3-repo/harmonia-api:latest"

      ports {
        container_port = 8080
      }

      resources {
        limits = {
          cpu    = var.cloud_run_cpu
          memory = var.cloud_run_memory
        }
        cpu_idle = false
      }

      env {
        name  = "ENVIRONMENT"
        value = var.environment
      }
      env {
        name  = "LOG_LEVEL"
        value = "INFO"
      }
      env {
        name  = "GCP_PROJECT_ID"
        value = var.project_id
      }
      env {
        name  = "GCP_REGION"
        value = var.region
      }
      env {
        name  = "GCS_BUCKET_NAME"
        value = local.gcs_bucket_name
      }

      # Secrets injected from Secret Manager
      env {
        name = "GEMINI_API_KEY"
        value_source {
          secret_key_ref {
            secret  = google_secret_manager_secret.gemini_api_key.secret_id
            version = "latest"
          }
        }
      }
      env {
        name = "ANTHROPIC_API_KEY"
        value_source {
          secret_key_ref {
            secret  = google_secret_manager_secret.anthropic_api_key.secret_id
            version = "latest"
          }
        }
      }
      env {
        name = "FERNET_KEY"
        value_source {
          secret_key_ref {
            secret  = google_secret_manager_secret.fernet_key.secret_id
            version = "latest"
          }
        }
      }
      env {
        name = "DATABASE_URL"
        value_source {
          secret_key_ref {
            secret  = google_secret_manager_secret.database_url.secret_id
            version = "latest"
          }
        }
      }
      env {
        name = "REDIS_URL"
        value_source {
          secret_key_ref {
            secret  = google_secret_manager_secret.redis_url.secret_id
            version = "latest"
          }
        }
      }

      volume_mounts {
        name       = "cloudsql"
        mount_path = "/cloudsql"
      }

      startup_probe {
        http_get {
          path = "/health"
          port = 8080
        }
        initial_delay_seconds = 5
        period_seconds        = 10
        failure_threshold     = 3
      }

      liveness_probe {
        http_get {
          path = "/health"
          port = 8080
        }
        period_seconds = 30
      }
    }
  }

  depends_on = [
    google_project_service.apis,
    google_project_iam_member.harmonia_runner_roles,
  ]
}

# Allow unauthenticated access (public API)
resource "google_cloud_run_v2_service_iam_member" "public_access" {
  name     = google_cloud_run_v2_service.harmonia_api.name
  location = var.region
  role     = "roles/run.invoker"
  member   = "allUsers"
}
