# =============================================================================
# Harmonia V3 — Terraform Variables
# =============================================================================

variable "project_id" {
  description = "GCP project ID"
  type        = string
}

variable "region" {
  description = "GCP region for all resources"
  type        = string
  default     = "europe-west2"
}

variable "environment" {
  description = "Deployment environment (development, staging, production)"
  type        = string
  default     = "production"
}

# ---------------------------------------------------------------------------
# Cloud SQL
# ---------------------------------------------------------------------------
variable "sql_instance_name" {
  description = "Cloud SQL instance name"
  type        = string
  default     = "harmonia-v3-pg"
}

variable "sql_tier" {
  description = "Cloud SQL machine tier"
  type        = string
  default     = "db-f1-micro"
}

variable "sql_database_name" {
  description = "PostgreSQL database name"
  type        = string
  default     = "harmonia"
}

variable "sql_user" {
  description = "PostgreSQL user name"
  type        = string
  default     = "harmonia_user"
}

# ---------------------------------------------------------------------------
# Redis
# ---------------------------------------------------------------------------
variable "redis_instance_name" {
  description = "Memorystore Redis instance name"
  type        = string
  default     = "harmonia-v3-redis"
}

variable "redis_memory_size_gb" {
  description = "Redis memory size in GB"
  type        = number
  default     = 1
}

variable "redis_tier" {
  description = "Redis tier (BASIC or STANDARD_HA)"
  type        = string
  default     = "BASIC"
}

# ---------------------------------------------------------------------------
# Cloud Run
# ---------------------------------------------------------------------------
variable "cloud_run_service_name" {
  description = "Cloud Run service name"
  type        = string
  default     = "harmonia-api"
}

variable "cloud_run_cpu" {
  description = "CPU limit for Cloud Run containers"
  type        = string
  default     = "2"
}

variable "cloud_run_memory" {
  description = "Memory limit for Cloud Run containers"
  type        = string
  default     = "2Gi"
}

variable "cloud_run_min_instances" {
  description = "Minimum number of Cloud Run instances"
  type        = number
  default     = 1
}

variable "cloud_run_max_instances" {
  description = "Maximum number of Cloud Run instances"
  type        = number
  default     = 10
}

variable "cloud_run_concurrency" {
  description = "Maximum concurrent requests per container"
  type        = number
  default     = 20
}

variable "container_image" {
  description = "Full container image URL (e.g. europe-west2-docker.pkg.dev/PROJECT/harmonia-v3-repo/harmonia-api:latest)"
  type        = string
  default     = ""
}

# ---------------------------------------------------------------------------
# GCS
# ---------------------------------------------------------------------------
variable "gcs_bucket_name" {
  description = "GCS bucket name (defaults to PROJECT_ID-harmonia-v3)"
  type        = string
  default     = ""
}

# ---------------------------------------------------------------------------
# Networking
# ---------------------------------------------------------------------------
variable "vpc_connector_name" {
  description = "Serverless VPC Access connector name"
  type        = string
  default     = "harmonia-vpc-connector"
}

variable "vpc_connector_cidr" {
  description = "CIDR range for VPC connector"
  type        = string
  default     = "10.8.0.0/28"
}

variable "network" {
  description = "VPC network name"
  type        = string
  default     = "default"
}
