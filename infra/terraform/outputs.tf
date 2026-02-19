# =============================================================================
# Harmonia V3 — Terraform Outputs
# =============================================================================

output "cloud_run_url" {
  description = "URL of the deployed Cloud Run service"
  value       = google_cloud_run_v2_service.harmonia_api.uri
}

output "cloud_run_service_name" {
  description = "Cloud Run service name"
  value       = google_cloud_run_v2_service.harmonia_api.name
}

output "cloud_sql_connection_name" {
  description = "Cloud SQL instance connection name (project:region:instance)"
  value       = google_sql_database_instance.harmonia_pg.connection_name
}

output "cloud_sql_instance_ip" {
  description = "Cloud SQL private IP address"
  value       = google_sql_database_instance.harmonia_pg.private_ip_address
}

output "redis_host" {
  description = "Memorystore Redis host IP"
  value       = google_redis_instance.harmonia_redis.host
}

output "redis_port" {
  description = "Memorystore Redis port"
  value       = google_redis_instance.harmonia_redis.port
}

output "gcs_bucket_name" {
  description = "GCS bucket name"
  value       = google_storage_bucket.harmonia_bucket.name
}

output "gcs_bucket_url" {
  description = "GCS bucket URL"
  value       = google_storage_bucket.harmonia_bucket.url
}

output "artifact_registry_url" {
  description = "Artifact Registry repository URL"
  value       = "${var.region}-docker.pkg.dev/${var.project_id}/${google_artifact_registry_repository.harmonia_repo.repository_id}"
}

output "service_account_email" {
  description = "Cloud Run service account email"
  value       = google_service_account.harmonia_runner.email
}

output "vpc_connector_name" {
  description = "VPC Access connector name"
  value       = google_vpc_access_connector.harmonia_connector.name
}

output "database_url_secret_id" {
  description = "Secret Manager secret ID for DATABASE_URL"
  value       = google_secret_manager_secret.database_url.secret_id
}

output "redis_url_secret_id" {
  description = "Secret Manager secret ID for REDIS_URL"
  value       = google_secret_manager_secret.redis_url.secret_id
}
