terraform {
  required_version = ">= 1.0"
  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 5.0"
    }
  }
  
  backend "gcs" {
    bucket = "helios-terraform-state"
    prefix = "terraform/state"
  }
}

provider "google" {
  project = var.project_id
  region  = var.region
  zone    = var.zone
}

# Enable required APIs
resource "google_project_service" "required_apis" {
  for_each = toset([
    "aiplatform.googleapis.com",
    "cloudfunctions.googleapis.com",
    "cloudrun.googleapis.com",
    "firestore.googleapis.com",
    "cloudstorage.googleapis.com",
    "sheets.googleapis.com",
    "drive.googleapis.com",
    "secretmanager.googleapis.com",
    "cloudscheduler.googleapis.com",
    "pubsub.googleapis.com",
    "cloudbuild.googleapis.com",
    "iam.googleapis.com"
  ])
  
  service = each.value
  disable_dependent_services = false
  disable_on_destroy = false
}

# Create service account for automation
resource "google_service_account" "helios_automation" {
  account_id   = "helios-automation-sa"
  display_name = "Helios Autonomous Store Service Account"
  description  = "Service account for Helios automation services"
  
  depends_on = [google_project_service.required_apis]
}

# Grant necessary roles to service account
resource "google_project_iam_member" "helios_roles" {
  for_each = toset([
    "roles/aiplatform.user",
    "roles/storage.admin",
    "roles/sheets.editor",
    "roles/drive.file",
    "roles/secretmanager.secretAccessor",
    "roles/pubsub.publisher",
    "roles/pubsub.subscriber",
    "roles/cloudrun.developer",
    "roles/cloudbuild.builds.builder"
  ])
  
  project = var.project_id
  role    = each.value
  member  = "serviceAccount:${google_service_account.helios_automation.email}"
  
  depends_on = [google_service_account.helios_automation]
}

# Create Firestore database
resource "google_firestore_database" "helios_data" {
  name        = "helios-data"
  location_id = var.region
  type        = "FIRESTORE_NATIVE"
  
  depends_on = [google_project_service.required_apis]
}

# Create Cloud Storage bucket for product assets
resource "google_storage_bucket" "helios_product_assets" {
  name          = "helios-product-assets-${var.project_id}"
  location      = var.region
  force_destroy = false
  
  uniform_bucket_level_access = true
  
  versioning {
    enabled = true
  }
  
  lifecycle_rule {
    condition {
      age = 365
    }
    action {
      type = "Delete"
    }
  }
  
  depends_on = [google_project_service.required_apis]
}

# Create Pub/Sub topics
resource "google_pubsub_topic" "trend_discovered" {
  name = "trend-discovered"
  
  depends_on = [google_project_service.required_apis]
}

resource "google_pubsub_topic" "analysis_complete" {
  name = "analysis-complete"
  
  depends_on = [google_project_service.required_apis]
}

resource "google_pubsub_topic" "content_ready" {
  name = "content-ready"
  
  depends_on = [google_project_service.required_apis]
}

resource "google_pubsub_topic" "publication_success" {
  name = "publication-success"
  
  depends_on = [google_project_service.required_apis]
}

# Create Pub/Sub subscriptions
resource "google_pubsub_subscription" "trend_discovered_sub" {
  name  = "trend-discovered-sub"
  topic = google_pubsub_topic.trend_discovered.name
  
  ack_deadline_seconds = 20
  
  depends_on = [google_pubsub_topic.trend_discovered]
}

resource "google_pubsub_subscription" "analysis_complete_sub" {
  name  = "analysis-complete-sub"
  topic = google_pubsub_topic.analysis_complete.name
  
  ack_deadline_seconds = 20
  
  depends_on = [google_pubsub_topic.analysis_complete]
}

resource "google_pubsub_subscription" "content_ready_sub" {
  name  = "content-ready-sub"
  topic = google_pubsub_topic.content_ready.name
  
  ack_deadline_seconds = 20
  
  depends_on = [google_pubsub_topic.content_ready]
}

resource "google_pubsub_subscription" "publication_success_sub" {
  name  = "publication-success-sub"
  topic = google_pubsub_topic.publication_success.name
  
  ack_deadline_seconds = 20
  
  depends_on = [google_pubsub_topic.publication_success]
}

# Create Cloud Scheduler job for trend discovery
resource "google_cloud_scheduler_job" "trend_discovery_job" {
  name        = "trend-discovery-job"
  description = "Scheduled job for trend discovery"
  schedule    = "0 */6 * * *"  # Every 6 hours
  
  pubsub_target {
    topic_name = google_pubsub_topic.trend_discovered.id
    data       = base64encode(jsonencode({
      "trigger": "scheduled",
      "timestamp": "{{.Timestamp}}"
    }))
  }
  
  depends_on = [google_project_service.required_apis, google_pubsub_topic.trend_discovered]
}

# Create Secret Manager secrets
resource "google_secret_manager_secret" "printify_api_token" {
  secret_id = "printify-api-token"
  
  replication {
    auto {}
  }
  
  depends_on = [google_project_service.required_apis]
}

resource "google_secret_manager_secret" "etsy_api_credentials" {
  secret_id = "etsy-api-credentials"
  
  replication {
    auto {}
  }
  
  depends_on = [google_project_service.required_apis]
}

# Create Cloud Run services (basic configuration - detailed config in YAML)
resource "google_cloud_run_service" "ceo_orchestrator" {
  name     = "ceo-orchestrator"
  location = var.region
  
  template {
    spec {
      containers {
        image = "gcr.io/${var.project_id}/ceo-orchestrator:latest"
        
        resources {
          limits = {
            cpu    = "2000m"
            memory = "4Gi"
          }
        }
        
        env {
          name  = "GOOGLE_CLOUD_PROJECT"
          value = var.project_id
        }
        
        env {
          name  = "GOOGLE_CLOUD_REGION"
          value = var.region
        }
      }
      
      service_account_name = google_service_account.helios_automation.email
    }
  }
  
  depends_on = [google_project_service.required_apis, google_service_account.helios_automation]
}

# Outputs
output "project_id" {
  value = var.project_id
}

output "region" {
  value = var.region
}

output "service_account_email" {
  value = google_service_account.helios_automation.email
}

output "firestore_database" {
  value = google_firestore_database.helios_data.name
}

output "storage_bucket" {
  value = google_storage_bucket.helios_product_assets.name
}

output "pubsub_topics" {
  value = {
    trend_discovered    = google_pubsub_topic.trend_discovered.name
    analysis_complete   = google_pubsub_topic.analysis_complete.name
    content_ready       = google_pubsub_topic.content_ready.name
    publication_success = google_pubsub_topic.publication_success.name
  }
}
