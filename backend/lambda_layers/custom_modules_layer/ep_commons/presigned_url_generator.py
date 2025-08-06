import boto3
import logging
from botocore.exceptions import BotoCoreError, ClientError

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.ERROR)

class PresignedUrlGenerationError(Exception):
    """Custom exception for errors during presigned URL generation."""
    pass


class PresignedUrlGenerator:
    def __init__(self, bucket_name: str, region: str = "eu-central-1"):
        """
        Initializes the PresignedUrlGenerator.

        :param bucket_name: Name of the S3 bucket.
        :param region: AWS region where the bucket is located (default: "eu-central-1").
        """
        self.bucket_name = bucket_name
        self.s3_client = boto3.client("s3", region_name=region)

    def generate_presigned_url(self, object_key: str, expiration: int = 3600) -> str:
        """
        Generates a pre-signed URL to grant temporary access to an S3 object.

        :param object_key: Key of the S3 object (file path).
        :param expiration: Time in seconds until the URL expires (default: 3600s = 1 hour).
        :return: Pre-signed URL as a string.
        :raises PresignedUrlGenerationError: If URL generation fails.
        """
        try:
            self.s3_client.head_object(Bucket=self.bucket_name, Key=object_key)
        except ClientError as e:
            logger.error(f"AWS S3 ClientError: {e.response['Error']['Message']}")
            raise PresignedUrlGenerationError(f"Object with key {object_key} not found.") from e
        except Exception as e:
            logger.error(f"Unexpected error: {str(e)}")
            raise PresignedUrlGenerationError("An unexpected error occurred while generating the presigned URL.") from e

        try:
            url = self.s3_client.generate_presigned_url(
                "get_object",
                Params={"Bucket": self.bucket_name, "Key": object_key},
                ExpiresIn=expiration
            )
            return url
        except ClientError as e:
            logger.error(f"AWS S3 ClientError: {e.response['Error']['Message']}")
            raise PresignedUrlGenerationError("Failed to generate presigned URL due to a client error.") from e
        except BotoCoreError as e:
            logger.error(f"AWS BotoCoreError: {str(e)}")
            raise PresignedUrlGenerationError("Failed to generate presigned URL due to a Boto3 error.") from e
        except Exception as e:
            logger.error(f"Unexpected error: {str(e)}")
            raise PresignedUrlGenerationError("An unexpected error occurred while generating the presigned URL.") from e
