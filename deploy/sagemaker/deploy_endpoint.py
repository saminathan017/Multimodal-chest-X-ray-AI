"""
deploy/sagemaker/deploy_endpoint.py
─────────────────────────────────────────────────────────────────────
Deploy the ClinicalAI inference pipeline as an AWS SageMaker
real-time endpoint.

Usage:
    python deploy/sagemaker/deploy_endpoint.py \
        --action deploy \
        --role arn:aws:iam::ACCOUNT_ID:role/SageMakerExecutionRole \
        --region us-east-1

    python deploy/sagemaker/deploy_endpoint.py --action delete
─────────────────────────────────────────────────────────────────────
"""

import argparse
import boto3
import sagemaker
from sagemaker.pytorch import PyTorchModel
from sagemaker.serializers import JSONSerializer
from sagemaker.deserializers import JSONDeserializer
from loguru import logger


ENDPOINT_NAME    = "clinical-cds-endpoint"
MODEL_NAME       = "clinical-ai-multimodal"
S3_MODEL_URI     = "s3://clinical-ai-demo-bucket/models/model.tar.gz"
INFERENCE_IMAGE  = "763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-inference:2.2.0-gpu-py311"


def deploy_endpoint(role: str, region: str, instance_type: str = "ml.g4dn.xlarge"):
    """
    Deploy the model to a SageMaker real-time endpoint.
    g4dn.xlarge = NVIDIA T4 GPU, ~$0.74/hr — cheapest GPU option.
    """
    session = sagemaker.Session(boto_session=boto3.Session(region_name=region))

    model = PyTorchModel(
        model_data=S3_MODEL_URI,
        role=role,
        framework_version="2.2",
        py_version="py311",
        entry_point="inference_handler.py",
        source_dir="deploy/sagemaker/",
        name=MODEL_NAME,
        sagemaker_session=session,
    )

    logger.info(f"Deploying to {instance_type}...")
    predictor = model.deploy(
        initial_instance_count=1,
        instance_type=instance_type,
        endpoint_name=ENDPOINT_NAME,
        serializer=JSONSerializer(),
        deserializer=JSONDeserializer(),
    )

    logger.info(f"✅ Endpoint deployed: {ENDPOINT_NAME}")
    logger.info("Test with: python deploy/sagemaker/test_endpoint.py")
    return predictor


def delete_endpoint(region: str):
    client = boto3.client("sagemaker", region_name=region)
    try:
        client.delete_endpoint(EndpointName=ENDPOINT_NAME)
        logger.info(f"Endpoint {ENDPOINT_NAME} deleted")
    except client.exceptions.ClientError as e:
        logger.warning(f"Could not delete endpoint: {e}")


def package_model_artifacts(output_path: str = "model.tar.gz"):
    """
    Package model weights + inference code into a .tar.gz for S3.
    Run this before deploying to upload fresh weights.
    """
    import tarfile, os
    with tarfile.open(output_path, "w:gz") as tar:
        for path in ["models/", "src/", "configs/"]:
            if os.path.exists(path):
                tar.add(path)
    logger.info(f"Model packaged: {output_path}")

    # Upload to S3
    s3 = boto3.client("s3")
    bucket = "clinical-ai-demo-bucket"
    key    = "models/model.tar.gz"
    s3.upload_file(output_path, bucket, key)
    logger.info(f"Uploaded to s3://{bucket}/{key}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--action", choices=["deploy", "delete", "package"], default="deploy")
    parser.add_argument("--role", default="arn:aws:iam::ACCOUNT_ID:role/SageMakerExecutionRole")
    parser.add_argument("--region", default="us-east-1")
    parser.add_argument("--instance", default="ml.g4dn.xlarge")
    args = parser.parse_args()

    if args.action == "deploy":
        deploy_endpoint(args.role, args.region, args.instance)
    elif args.action == "delete":
        delete_endpoint(args.region)
    elif args.action == "package":
        package_model_artifacts()
