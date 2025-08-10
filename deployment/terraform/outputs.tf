output "ceo_service_url" {
  description = "URL of the CEO orchestrator service"
  value       = google_cloud_run_service.ceo_orchestrator.status[0].url
}

output "trend_discovery_service_url" {
  description = "URL of the trend discovery service"
  value       = google_cloud_run_service.trend_discovery.status[0].url
}

output "content_generation_service_url" {
  description = "URL of the content generation service"
  value       = google_cloud_run_service.content_generation.status[0].url
}

output "publication_handler_service_url" {
  description = "URL of the publication handler service"
  value       = google_cloud_run_service.publication_handler.status[0].url
}

output "firestore_database_name" {
  description = "Name of the Firestore database"
  value       = google_firestore_database.helios_data.name
}

output "cloud_storage_bucket" {
  description = "Name of the Cloud Storage bucket for product assets"
  value       = google_storage_bucket.helios_product_assets.name
}

output "pubsub_topics" {
  description = "Names of the Pub/Sub topics"
  value = {
    trend_discovered     = google_pubsub_topic.trend_discovered.name
    analysis_complete    = google_pubsub_topic.analysis_complete.name
    content_ready        = google_pubsub_topic.content_ready.name
    publication_success  = google_pubsub_topic.publication_success.name
  }
}

output "project_id" {
  description = "The ID of the project"
  value       = var.project_id
}

output "region" {
  description = "The region where resources are deployed"
  value       = var.region
}
