variable "project_id" {
  description = "The Google Cloud Project ID"
  type        = string
  default     = "helios-autonomous-store"
}

variable "region" {
  description = "The Google Cloud region for resources"
  type        = string
  default     = "us-central1"
}

variable "zone" {
  description = "The Google Cloud zone for zonal resources"
  type        = string
  default     = "us-central1-a"
}

variable "environment" {
  description = "Environment (dev, staging, prod)"
  type        = string
  default     = "dev"
  
  validation {
    condition     = contains(["dev", "staging", "prod"], var.environment)
    error_message = "Environment must be one of: dev, staging, prod."
  }
}

variable "enable_monitoring" {
  description = "Enable Cloud Monitoring and Logging"
  type        = bool
  default     = true
}

variable "enable_tracing" {
  description = "Enable Cloud Trace"
  type        = bool
  default     = true
}

variable "ceo_cpu" {
  description = "CPU allocation for CEO orchestrator service"
  type        = string
  default     = "2"
}

variable "ceo_memory" {
  description = "Memory allocation for CEO orchestrator service"
  type        = string
  default     = "4Gi"
}

variable "trend_discovery_cpu" {
  description = "CPU allocation for trend discovery service"
  type        = string
  default     = "4"
}

variable "trend_discovery_memory" {
  description = "Memory allocation for trend discovery service"
  type        = string
  default     = "8Gi"
}

variable "content_generation_cpu" {
  description = "CPU allocation for content generation service"
  type        = string
  default     = "4"
}

variable "content_generation_memory" {
  description = "Memory allocation for content generation service"
  type        = string
  default     = "8Gi"
}

variable "publication_handler_cpu" {
  description = "CPU allocation for publication handler service"
  type        = string
  default     = "2"
}

variable "publication_handler_memory" {
  description = "Memory allocation for publication handler service"
  type        = string
  default     = "4Gi"
}

variable "min_instances" {
  description = "Minimum number of instances for Cloud Run services"
  type        = map(number)
  default = {
    ceo_orchestrator    = 1
    trend_discovery      = 0
    content_generation   = 0
    publication_handler  = 0
  }
}

variable "max_instances" {
  description = "Maximum number of instances for Cloud Run services"
  type        = map(number)
  default = {
    ceo_orchestrator    = 10
    trend_discovery      = 5
    content_generation   = 8
    publication_handler  = 3
  }
}

variable "timeout_seconds" {
  description = "Timeout for Cloud Run services in seconds"
  type        = map(number)
  default = {
    ceo_orchestrator    = 900
    trend_discovery      = 300
    content_generation   = 600
    publication_handler  = 300
  }
}

variable "container_concurrency" {
  description = "Container concurrency for Cloud Run services"
  type        = map(number)
  default = {
    ceo_orchestrator    = 80
    trend_discovery      = 40
    content_generation   = 60
    publication_handler  = 30
  }
}
