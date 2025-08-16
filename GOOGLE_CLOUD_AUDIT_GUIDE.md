# 🚀 **COMPREHENSIVE GOOGLE CLOUD PROJECT AUDIT GUIDE**

## 📋 **Complete Review Scope & Methodology**

This guide provides a systematic approach to auditing your Google Cloud Project for cost optimization, security, and performance. Use this as your checklist for thorough project reviews.

---

## 🎯 **1. ACTIVE SERVICES & APIs AUDIT**

### **1.1 API Discovery & Analysis**
```bash
# List ALL enabled APIs
gcloud services list --enabled --format="table(name,config.name,usage)"

# Find APIs with no usage (potential candidates for disable)
gcloud services list --enabled --format="table(name,config.name,usage)" | grep -E "(0|null|N/A)"

# Check API quotas and limits
gcloud compute regions describe us-central1 --format="table(quotas)"

# List APIs by category
gcloud services list --enabled --filter="config.name:compute*"
gcloud services list --enabled --filter="config.name:storage*"
gcloud services list --enabled --filter="config.name:run*"
```

### **1.2 API Cleanup Commands**
```bash
# Disable unused APIs (use --force for dependent services)
gcloud services disable SERVICE_NAME --force

# Check API dependencies before disabling
gcloud services list --enabled --filter="config.name:SERVICE_NAME"

# Monitor API usage over time
gcloud services list --enabled --format="table(name,config.name,usage)" > api_usage_$(date +%Y%m%d).txt
```

### **1.3 Critical APIs to Keep**
- **Compute Engine API** (`compute.googleapis.com`)
- **Cloud Run API** (`run.googleapis.com`)
- **Cloud Storage API** (`storage.googleapis.com`)
- **Cloud Firestore API** (`firestore.googleapis.com`)
- **Cloud Build API** (`cloudbuild.googleapis.com`)
- **Secret Manager API** (`secretmanager.googleapis.com`)
- **Cloud Logging API** (`logging.googleapis.com`)
- **Cloud Monitoring API** (`monitoring.googleapis.com`)

---

## 🗄️ **2. STORAGE & DATA AUDIT**

### **2.1 Cloud Storage Analysis**
```bash
# List ALL storage buckets
gsutil ls -b

# Check bucket sizes and costs
gsutil du -sh gs://*

# Find buckets with no lifecycle policies
gsutil lifecycle get gs://BUCKET_NAME

# Check bucket access logs
gsutil logging get gs://BUCKET_NAME

# Find public buckets (security risk)
gsutil iam get gs://BUCKET_NAME
```

### **2.2 Storage Optimization Commands**
```bash
# Set lifecycle policy for automatic cleanup
gsutil lifecycle set lifecycle.json gs://BUCKET_NAME

# Move objects to cheaper storage class
gsutil rewrite -s NEARLINE gs://BUCKET_NAME/OBJECT_PATH

# Enable object versioning (if needed)
gsutil versioning set on gs://BUCKET_NAME

# Check for duplicate objects
gsutil ls -l gs://BUCKET_NAME | sort -k3 | uniq -d -f2
```

### **2.3 Database & Disk Analysis**
```bash
# List ALL Compute Engine disks
gcloud compute disks list --format="table(name,sizeGb,users,lastAttachTimestamp)"

# Find orphaned disks (no attached instances)
gcloud compute disks list --filter="users=null"

# Check Cloud SQL instances
gcloud sql instances list --format="table(name,state,settings.tier,settings.dataDiskSizeGb)"

# List Firestore databases
gcloud firestore databases list --format="table(name,locationId,type,state)"

# Check BigQuery datasets
bq ls --project_id=PROJECT_ID
```

---

## 💻 **3. COMPUTE RESOURCES AUDIT**

### **3.1 Cloud Run Services**
```bash
# List ALL Cloud Run services
gcloud run services list --format="table(metadata.name,status.conditions[0].status,spec.template.spec.containers[0].image)"

# Check service revisions and history
gcloud run revisions list --service=SERVICE_NAME --format="table(metadata.name,status.conditions[0].status,spec.template.spec.containers[0].image)"

# Monitor service metrics
gcloud run services describe SERVICE_NAME --region=REGION --format="value(status.conditions[0].status)"

# Check service scaling
gcloud run services describe SERVICE_NAME --region=REGION --format="value(spec.template.metadata.annotations.autoscaling.knative.dev/minScale,spec.template.metadata.annotations.autoscaling.knative.dev/maxScale)"
```

### **3.2 Compute Engine Instances**
```bash
# List ALL instances (running and stopped)
gcloud compute instances list --format="table(name,status,zone,machineType,lastStartTimestamp)"

# Find stopped instances (costing money for disks)
gcloud compute instances list --filter="status!=RUNNING"

# Check instance metadata and labels
gcloud compute instances describe INSTANCE_NAME --zone=ZONE --format="value(labels,metadata.items[].key,metadata.items[].value)"

# Monitor instance usage
gcloud compute instances get-serial-port-output INSTANCE_NAME --zone=ZONE
```

### **3.3 Cloud Functions & App Engine**
```bash
# List ALL Cloud Functions
gcloud functions list --format="table(name,status,entryPoint,runtime,updateTime)"

# Check function versions
gcloud functions versions list --function=FUNCTION_NAME

# List App Engine applications
gcloud app services list --format="table(id,version,traffic_split"

# Check App Engine versions
gcloud app versions list --service=SERVICE_NAME --format="table(id,traffic_split,createdBy,createTime"
```

---

## 🐳 **4. CONTAINER REGISTRY AUDIT**

### **4.1 GCR & Artifact Registry Analysis**
```bash
# List ALL container repositories
gcloud container images list --repository=gcr.io/PROJECT_ID

# Check image sizes and tags
gcloud container images list-tags gcr.io/PROJECT_ID/IMAGE_NAME --format="table(tags,timestamp.datetime,digest,imageSizeBytes)"

# Find untagged images (orphaned)
gcloud container images list-tags gcr.io/PROJECT_ID/IMAGE_NAME --format="table(tags,timestamp.datetime,digest)" | grep "^[[:space:]]*[0-9]"

# Find duplicate images by digest
gcloud container images list-tags gcr.io/PROJECT_ID/IMAGE_NAME --format="table(tags,timestamp.datetime,digest)" | sort -k3 | uniq -d -f2

# Check for large images
gcloud container images list --repository=gcr.io/PROJECT_ID --format="table(name,diskSizeGb,imageSizeBytes)"
```

### **4.2 Container Cleanup Commands**
```bash
# Remove specific image tags
gcloud container images delete gcr.io/PROJECT_ID/IMAGE_NAME:TAG --quiet

# Remove images by digest
gcloud container images delete gcr.io/PROJECT_ID/IMAGE_NAME@sha256:DIGEST --quiet

# Remove entire repository (if unused)
gcloud container images delete gcr.io/PROJECT_ID/IMAGE_NAME --recursive --quiet

# Clean up orphaned images
gcloud container images list-tags gcr.io/PROJECT_ID/IMAGE_NAME --format="table(tags,timestamp.datetime,digest)" | awk '$1=="" {print $3}' | xargs -I {} gcloud container images delete gcr.io/PROJECT_ID/IMAGE_NAME@sha256:{} --quiet
```

---

## 🌐 **5. NETWORKING & SECURITY AUDIT**

### **5.1 VPC & Network Analysis**
```bash
# List ALL VPC networks
gcloud compute networks list --format="table(name,x_gcloud_subnet_mode,autoCreateSubnetworks"

# Check subnet usage
gcloud compute networks subnets list --format="table(name,network,region,ipCidrRange,stackType"

# List firewall rules
gcloud compute firewall-rules list --format="table(name,network,sourceRanges.list(),allowed[].ports[],direction"

# Check load balancers
gcloud compute forwarding-rules list --format="table(name,region,IPAddress,IPProtocol,portRange"

# List Cloud NAT gateways
gcloud compute routers nats list --router=ROUTER_NAME --region=REGION
```

### **5.2 Security & IAM Analysis**
```bash
# List service accounts
gcloud iam service-accounts list --format="table(email,displayName,disabled"

# Check service account permissions
gcloud projects get-iam-policy PROJECT_ID --flatten="bindings[].members" --format="table(bindings.role,bindings.members)"

# List secrets in Secret Manager
gcloud secrets list --format="table(name,createTime,labels)"

# Check Cloud KMS keys
gcloud kms keys list --keyring=KEYRING_NAME --location=LOCATION --format="table(name,primary.name,primary.state,primary.createTime)"
```

---

## 📊 **6. MONITORING & LOGGING AUDIT**

### **6.1 Log Analysis**
```bash
# Check log buckets
gcloud logging buckets list --location=LOCATION --format="table(name,location,retentionDays,locked)"

# Check log retention policies
gcloud logging sinks list --format="table(name,destination,filter,writerIdentity)"

# Monitor log volume
gcloud logging read "resource.type=cloud_run_revision" --limit=100 --format="table(timestamp,severity,textPayload)"

# Check for expensive log queries
gcloud logging read "resource.type=cloud_run_revision AND severity>=ERROR" --limit=1000 --format="table(timestamp,severity,textPayload)"
```

### **6.2 Monitoring & Metrics**
```bash
# List monitoring workspaces
gcloud monitoring workspaces list --format="table(name,displayName,project)"

# Check custom metrics
gcloud monitoring metrics list --filter="metric.type:custom.googleapis.com" --format="table(metric.type,valueType,metricKind)"

# List alerting policies
gcloud alpha monitoring policies list --format="table(displayName,conditions[0].displayName,enabled)"
```

---

## 🔧 **7. DEVELOPMENT TOOLS AUDIT**

### **7.1 Cloud Build Analysis**
```bash
# List build triggers
gcloud builds triggers list --format="table(name,filename,github.name,github.owner,github.push.branch)"

# Check build history
gcloud builds list --limit=100 --format="table(id,status,createTime,source.repoSource.branchName,images)"

# Find build artifacts
gcloud builds list --limit=100 --format="table(id,status,artifacts.objects.name,artifacts.objects.timing)"

# Check build logs
gcloud builds log BUILD_ID
```

### **7.2 Source Repository & Cloud Shell**
```bash
# List source repositories
gcloud source repos list --format="table(name,url,size)"

# Check Cloud Shell instances
gcloud cloud-shell ssh --command="df -h"

# List development environments
gcloud compute instances list --filter="labels.environment=development" --format="table(name,status,zone,machineType"
```

---

## 💰 **8. COST ANALYSIS & OPTIMIZATION**

### **8.1 Billing & Usage Analysis**
```bash
# Check current billing
gcloud billing accounts list --format="table(name,displayName,open,accountId)"

# Get project billing info
gcloud billing projects describe PROJECT_ID --format="table(billingAccountName,billingEnabled)"

# Check resource quotas
gcloud compute regions describe us-central1 --format="table(quotas[].metric,quotas[].limit,quotas[].usage)"

# Monitor cost trends
gcloud billing budgets list --billing-account=BILLING_ACCOUNT_ID
```

### **8.2 Cost Optimization Commands**
```bash
# Enable billing export to BigQuery
gcloud billing export-logs create-billing-export BUCKET_NAME --dataset-name=BILLING_DATASET

# Set up budget alerts
gcloud billing budgets create --billing-account=BILLING_ACCOUNT_ID --display-name="Monthly Budget" --budget-amount=1000USD

# Check for unused resources
gcloud compute instances list --filter="status!=RUNNING" --format="table(name,status,zone,lastStartTimestamp)"

# Find over-provisioned resources
gcloud compute instances list --format="table(name,machineType,status,zone)"
```

---

## 🚨 **9. COMMON HIDDEN COST DRIVERS**

### **9.1 Storage-Related Costs**
- **Log Storage**: Cloud Logging can accumulate massive amounts of data
- **Backup Storage**: Automated backups may have long retention periods
- **Snapshot Storage**: Old snapshots may not be automatically cleaned up
- **Cross-Region Replication**: Automatic replication can double storage costs
- **Object Versioning**: Multiple versions of the same object
- **Lifecycle Policies**: Missing or overly generous retention policies

### **9.2 Compute-Related Costs**
- **Stopped Instances**: Still incur disk storage costs
- **Preemptible Instances**: May restart frequently, increasing costs
- **Over-provisioned Resources**: Instances with more resources than needed
- **Development Environments**: Test environments left running
- **Orphaned Resources**: Resources created during testing/debugging

### **9.3 Network-Related Costs**
- **Egress Costs**: Cross-region data transfer
- **Load Balancer**: Using higher tier than needed
- **VPN Connections**: Always-on VPN gateways
- **CDN Usage**: Origin requests when caching could be used

---

## 📋 **10. COMPREHENSIVE AUDIT CHECKLIST**

### **10.1 Pre-Audit Preparation**
- [ ] **Gather Project Information**
  - [ ] Project ID and name
  - [ ] Billing account details
  - [ ] Team access and permissions
  - [ ] Current cost baseline

- [ ] **Set Up Monitoring Tools**
  - [ ] Enable billing export
  - [ ] Set up cost alerts
  - [ ] Configure resource monitoring
  - [ ] Enable audit logging

### **10.2 Resource Discovery**
- [ ] **Services & APIs**
  - [ ] List all enabled APIs
  - [ ] Identify unused APIs
  - [ ] Check API quotas
  - [ ] Review API usage patterns

- [ ] **Storage Resources**
  - [ ] Audit all storage buckets
  - [ ] Check database instances
  - [ ] Review disk usage
  - [ ] Analyze backup policies

- [ ] **Compute Resources**
  - [ ] List all instances
  - [ ] Check Cloud Run services
  - [ ] Review Cloud Functions
  - [ ] Audit Kubernetes clusters

### **10.3 Cost Analysis**
- [ ] **Current Costs**
  - [ ] Monthly spending breakdown
  - [ ] Cost trends over time
  - [ ] Resource cost allocation
  - [ ] Unusual cost spikes

- [ ] **Optimization Opportunities**
  - [ ] Identify over-provisioned resources
  - [ ] Find unused resources
  - [ ] Check for reserved instance opportunities
  - [ ] Review storage class usage

### **10.4 Security & Compliance**
- [ ] **Access Control**
  - [ ] Review IAM roles
  - [ ] Check service account permissions
  - [ ] Audit API access
  - [ ] Review network security

- [ ] **Data Protection**
  - [ ] Check encryption settings
  - [ ] Review backup policies
  - [ ] Audit data retention
  - [ ] Verify compliance requirements

---

## 🛠️ **11. AUTOMATED CLEANUP SCRIPTS**

### **11.1 Storage Cleanup Script**
```bash
#!/bin/bash
# Automated storage cleanup script

echo "🧹 Starting Google Cloud Storage cleanup..."

# Find and remove orphaned disks
echo "📀 Cleaning up orphaned disks..."
gcloud compute disks list --filter="users=null" --format="value(name,zone)" | while read -r disk zone; do
    if [ ! -z "$disk" ]; then
        echo "Removing orphaned disk: $disk in zone: $zone"
        gcloud compute disks delete "$disk" --zone="$zone" --quiet
    fi
done

# Clean up old snapshots
echo "📸 Cleaning up old snapshots..."
gcloud compute snapshots list --filter="creationTimestamp<$(date -d '30 days ago' -I)" --format="value(name)" | while read -r snapshot; do
    if [ ! -z "$snapshot" ]; then
        echo "Removing old snapshot: $snapshot"
        gcloud compute snapshots delete "$snapshot" --quiet
    fi
done

# Clean up untagged container images
echo "🐳 Cleaning up untagged container images..."
gcloud container images list --repository=gcr.io/PROJECT_ID --format="table(name)" | tail -n +2 | while read -r repo; do
    if [ ! -z "$repo" ]; then
        echo "Cleaning untagged images in: $repo"
        gcloud container images list-tags "$repo" --format="table(tags,timestamp.datetime,digest)" | grep "^[[:space:]]*[0-9]" | awk '{print $3}' | head -10 | xargs -I {} gcloud container images delete "$repo@sha256:{}" --quiet
    fi
done

echo "✅ Storage cleanup completed!"
```

### **11.2 API Cleanup Script**
```bash
#!/bin/bash
# Automated API cleanup script

echo "🔌 Starting Google Cloud API cleanup..."

# List of APIs to keep (essential for Helios system)
ESSENTIAL_APIS=(
    "compute.googleapis.com"
    "run.googleapis.com"
    "storage.googleapis.com"
    "firestore.googleapis.com"
    "cloudbuild.googleapis.com"
    "secretmanager.googleapis.com"
    "logging.googleapis.com"
    "monitoring.googleapis.com"
    "cloudscheduler.googleapis.com"
    "pubsub.googleapis.com"
    "iam.googleapis.com"
    "cloudkms.googleapis.com"
)

# Get all enabled APIs
echo "📋 Checking enabled APIs..."
gcloud services list --enabled --format="value(config.name)" > enabled_apis.txt

# Check for unused APIs
echo "🔍 Identifying potentially unused APIs..."
for api in $(cat enabled_apis.txt); do
    if [[ ! " ${ESSENTIAL_APIS[@]} " =~ " ${api} " ]]; then
        echo "Potential candidate for disable: $api"
        # Check usage before disabling
        usage=$(gcloud services list --enabled --filter="config.name=$api" --format="value(usage)")
        if [[ "$usage" == "0" || "$usage" == "null" ]]; then
            echo "Disabling unused API: $api"
            gcloud services disable "$api" --quiet
        fi
    fi
done

echo "✅ API cleanup completed!"
```

---

## 📊 **12. MONITORING & ALERTS SETUP**

### **12.1 Cost Monitoring Dashboard**
```bash
# Create cost monitoring dashboard
gcloud monitoring dashboards create --config-from-file=cost_dashboard.json

# Set up budget alerts
gcloud billing budgets create \
    --billing-account=BILLING_ACCOUNT_ID \
    --display-name="Monthly Budget Alert" \
    --budget-amount=1000USD \
    --threshold-rules=threshold=0.5,basis=BASIS_CURRENT_SPEND \
    --threshold-rules=threshold=0.9,basis=BASIS_CURRENT_SPEND \
    --threshold-rules=threshold=1.0,basis=BASIS_CURRENT_SPEND
```

### **12.2 Resource Monitoring**
```bash
# Create resource usage alerts
gcloud alpha monitoring policies create \
    --policy-from-file=resource_monitoring_policy.json

# Set up log-based metrics
gcloud logging metrics create "high_cost_operations" \
    --description="Operations that may incur high costs" \
    --log-filter="resource.type=cloud_run_revision AND severity>=WARNING"
```

---

## 🎯 **13. PREVENTION STRATEGIES**

### **13.1 Resource Tagging Strategy**
```bash
# Implement consistent resource tagging
gcloud compute instances add-labels INSTANCE_NAME \
    --labels=environment=production,team=helios,project=autonomous-store

# Create tagging policy
cat > tagging_policy.md << EOF
# Resource Tagging Policy

## Required Tags
- environment: production|staging|development
- team: helios|backend|frontend
- project: autonomous-store|ai-agent|trend-discovery
- cost-center: compute|storage|network

## Tag Enforcement
- All resources must have required tags
- Tags are enforced via IAM policies
- Regular audits ensure compliance
EOF
```

### **13.2 Automated Cleanup Policies**
```bash
# Set up automated cleanup for Cloud Storage
gsutil lifecycle set lifecycle_policy.json gs://BUCKET_NAME

# Configure automatic disk cleanup
gcloud compute disks list --filter="users=null AND creationTimestamp<$(date -d '7 days ago' -I)" --format="value(name,zone)" | while read -r disk zone; do
    gcloud compute disks delete "$disk" --zone="$zone" --quiet
done
```

---

## 📚 **14. REFERENCES & RESOURCES**

### **14.1 Official Google Cloud Documentation**
- [Cloud Run Best Practices](https://cloud.google.com/run/docs/best-practices)
- [Cost Optimization Guide](https://cloud.google.com/architecture/cost-optimization-on-gcp)
- [Resource Management](https://cloud.google.com/resource-manager/docs)
- [Billing & Cost Management](https://cloud.google.com/billing/docs)

### **14.2 Cost Optimization Resources**
- [Google Cloud Cost Optimization](https://www.cloudwards.net/google-cloud-cost-optimization/)
- [GCP Cost Optimization Best Practices](https://medium.com/@bossiwriter/gcp-cost-optimization-best-practices-126bda50da0b)
- [Cloud Cost Management](https://edgedelta.com/company/blog/gcp-cost-optimization)

### **14.3 Security & Compliance**
- [Google Cloud Security](https://cloud.google.com/security)
- [IAM Best Practices](https://cloud.google.com/iam/docs/best-practices)
- [Data Protection](https://cloud.google.com/security/data-protection)

---

## 🎉 **CONCLUSION**

This comprehensive audit guide provides a systematic approach to reviewing your Google Cloud Project. Regular audits using this guide will help you:

- **Optimize Costs**: Identify and eliminate unnecessary expenses
- **Improve Security**: Ensure proper access controls and data protection
- **Enhance Performance**: Optimize resource allocation and usage
- **Maintain Compliance**: Meet regulatory and organizational requirements

**Recommended Audit Frequency:**
- **Daily**: Cost monitoring and alerts
- **Weekly**: Resource usage review
- **Monthly**: Comprehensive security audit
- **Quarterly**: Full project review using this guide

Remember: **Prevention is better than cure**. Implement the prevention strategies and automated cleanup policies to maintain an optimized Google Cloud environment continuously.

---

**Last Updated**: August 16, 2025  
**Version**: 1.0  
**Status**: ✅ Production Ready - Comprehensive Audit Guide
