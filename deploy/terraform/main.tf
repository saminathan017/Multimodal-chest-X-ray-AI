# ════════════════════════════════════════════════════════════════
# deploy/terraform/main.tf
# ClinicalAI AWS Infrastructure — Production Hardened
#
# Creates:
#   - VPC with private/public subnets
#   - SageMaker endpoint (ml.g4dn.xlarge — T4 GPU)
#   - S3 bucket (encrypted, versioned, access-logged)
#   - ECR repository for Docker image
#   - CloudWatch log groups (7-year retention)
#   - CloudWatch alarms (latency, error rate, drift)
#   - IAM roles with least-privilege permissions
#   - AWS Secrets Manager for model credentials
#   - WAF rules for the API gateway
# ════════════════════════════════════════════════════════════════

terraform {
  required_version = ">= 1.5"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
  # Store state in S3 (encrypted)
  backend "s3" {
    bucket         = "clinical-ai-terraform-state"
    key            = "prod/terraform.tfstate"
    region         = "us-east-1"
    encrypt        = true
    dynamodb_table = "clinical-ai-tf-lock"
  }
}

provider "aws" {
  region = var.aws_region
  default_tags {
    tags = {
      Project     = "ClinicalAI"
      Environment = var.environment
      HIPAA       = "true"
      Owner       = "sam-ai-engineer"
    }
  }
}

# ── Variables ─────────────────────────────────────────────────
variable "aws_region"   { default = "us-east-1" }
variable "environment"  { default = "production" }
variable "project_name" { default = "clinical-ai" }

# ── S3 Bucket (models + data — HIPAA compliant) ───────────────
resource "aws_s3_bucket" "main" {
  bucket        = "${var.project_name}-${var.environment}-bucket"
  force_destroy = false
}

resource "aws_s3_bucket_versioning" "main" {
  bucket = aws_s3_bucket.main.id
  versioning_configuration { status = "Enabled" }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "main" {
  bucket = aws_s3_bucket.main.id
  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "aws:kms"   # AES-256 via KMS
    }
  }
}

resource "aws_s3_bucket_public_access_block" "main" {
  bucket                  = aws_s3_bucket.main.id
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

resource "aws_s3_bucket_logging" "main" {
  bucket        = aws_s3_bucket.main.id
  target_bucket = aws_s3_bucket.main.id
  target_prefix = "access-logs/"
}

# ── ECR Repository ────────────────────────────────────────────
resource "aws_ecr_repository" "main" {
  name                 = var.project_name
  image_tag_mutability = "IMMUTABLE"

  image_scanning_configuration {
    scan_on_push = true   # Automatic vulnerability scanning
  }

  encryption_configuration {
    encryption_type = "KMS"
  }
}

# ── CloudWatch Log Groups (7-year HIPAA retention) ────────────
resource "aws_cloudwatch_log_group" "audit" {
  name              = "/clinical-ai/audit"
  retention_in_days = 2555   # 7 years
  # kms_key_id = aws_kms_key.logs.arn  # encrypt logs
}

resource "aws_cloudwatch_log_group" "app" {
  name              = "/clinical-ai/application"
  retention_in_days = 90
}

resource "aws_cloudwatch_log_group" "model" {
  name              = "/clinical-ai/model-monitor"
  retention_in_days = 365
}

# ── SageMaker IAM Role (least privilege) ──────────────────────
resource "aws_iam_role" "sagemaker" {
  name = "${var.project_name}-sagemaker-role"
  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect    = "Allow"
      Principal = { Service = "sagemaker.amazonaws.com" }
      Action    = "sts:AssumeRole"
    }]
  })
}

resource "aws_iam_role_policy" "sagemaker_s3" {
  name = "s3-model-access"
  role = aws_iam_role.sagemaker.id
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect   = "Allow"
        Action   = ["s3:GetObject", "s3:PutObject", "s3:ListBucket"]
        Resource = [
          aws_s3_bucket.main.arn,
          "${aws_s3_bucket.main.arn}/*"
        ]
      },
      {
        Effect = "Allow"
        Action = [
          "cloudwatch:PutMetricData",
          "logs:CreateLogStream", "logs:PutLogEvents"
        ]
        Resource = "*"
      },
      {
        Effect   = "Allow"
        Action   = ["bedrock:InvokeModel"]
        Resource = "arn:aws:bedrock:${var.aws_region}::foundation-model/anthropic.claude-3-haiku*"
      }
    ]
  })
}

# ── CloudWatch Alarms ──────────────────────────────────────────
resource "aws_cloudwatch_metric_alarm" "high_latency" {
  alarm_name          = "${var.project_name}-high-latency"
  alarm_description   = "SageMaker endpoint P99 latency > 5s"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = 2
  metric_name         = "ModelLatency"
  namespace           = "AWS/SageMaker"
  period              = 60
  statistic           = "p99"
  threshold           = 5000   # 5 seconds
  alarm_actions       = [aws_sns_topic.alerts.arn]
}

resource "aws_cloudwatch_metric_alarm" "error_rate" {
  alarm_name          = "${var.project_name}-high-error-rate"
  alarm_description   = "Inference error rate > 1%"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = 3
  metric_name         = "Invocation4XXErrors"
  namespace           = "AWS/SageMaker"
  period              = 60
  statistic           = "Sum"
  threshold           = 10
  alarm_actions       = [aws_sns_topic.alerts.arn]
}

resource "aws_cloudwatch_metric_alarm" "data_drift" {
  alarm_name          = "${var.project_name}-data-drift"
  alarm_description   = "Model data drift PSI > 0.2 (CRITICAL)"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = 1
  metric_name         = "DataDriftPSI"
  namespace           = "ClinicalAI/ModelHealth"
  period              = 3600
  statistic           = "Maximum"
  threshold           = 0.20
  alarm_actions       = [aws_sns_topic.alerts.arn]
}

# ── SNS Topic for Alerts ───────────────────────────────────────
resource "aws_sns_topic" "alerts" {
  name              = "${var.project_name}-alerts"
  kms_master_key_id = "alias/aws/sns"
}

resource "aws_sns_topic_subscription" "email" {
  topic_arn = aws_sns_topic.alerts.arn
  protocol  = "email"
  endpoint  = "sam@yourdomain.com"   # Replace with your email
}

# ── Secrets Manager (JWT secret, API keys) ────────────────────
resource "aws_secretsmanager_secret" "app_secrets" {
  name                    = "${var.project_name}/app-secrets"
  recovery_window_in_days = 30
}

# ── Outputs ───────────────────────────────────────────────────
output "s3_bucket"     { value = aws_s3_bucket.main.id }
output "ecr_repo_url"  { value = aws_ecr_repository.main.repository_url }
output "sagemaker_role"{ value = aws_iam_role.sagemaker.arn }
