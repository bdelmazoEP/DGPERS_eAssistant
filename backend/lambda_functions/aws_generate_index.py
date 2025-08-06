import json
import requests
from requests_aws4auth import AWS4Auth
import boto3
import time  
import os

# Configuration for OpenSearch and AWS
index_name = os.environ.get('INDEX_NAME')
region = "eu-central-1"
session = boto3.Session()
credentials = session.get_credentials()
auth = AWS4Auth(
    credentials.access_key,
    credentials.secret_key,
    region,
    "aoss",
    session_token=credentials.token,
)

# Custom CloudFormation response function
def send_cfn_response(event, context, response_status, response_data, physical_resource_id=None):
    response_url = event.get('ResponseURL')
    
    if response_url:
        response_body = {
            'Status': response_status,
            'Reason': f'See the details in CloudWatch Log Stream: {context.log_stream_name}',
            'PhysicalResourceId': physical_resource_id or context.log_stream_name,
            'StackId': event['StackId'],
            'RequestId': event['RequestId'],
            'LogicalResourceId': event['LogicalResourceId'],
            'Data': response_data
        }
        
        json_response_body = json.dumps(response_body)
        
        headers = {
            'content-type': '',
            'content-length': str(len(json_response_body))
        }
        
        try:
            response = requests.put(response_url, data=json_response_body, headers=headers)
            response.raise_for_status()
        except Exception as e:
            print(f"Failed to send response to CloudFormation: {e}")
    else:
        print("No ResponseURL in event; skipping CloudFormation response.")

# Function to get the collection endpoint from OpenSearch Serverless
def get_collection_endpoint():
    collection_name = os.environ.get('COLLECTION_NAME')
    client = boto3.client('opensearchserverless')
    try:
        response = client.list_collections()
        if 'collectionSummaries' in response and response['collectionSummaries']:
            for collection in response['collectionSummaries']:
                if collection['name'] == collection_name:
                    return collection.get('id')
        else:
            print("'collectionSummaries' was not found in the response or is empty.")
    except Exception as e:
        print(f"Error retrieving the collection endpoint: {e}")
    return None

# Function to repeatedly check the index status until it is created or a maximum number of retries is reached
def get_index_status(url):
    max_retries = 10  # Maximum number of retries
    retry_delay = 0   # Delay in seconds between retries (currently 0 seconds)
    retries = 0

    while retries < max_retries:
        try:
            response = requests.get(url, auth=auth)
            response.raise_for_status()

            response_data = response.json()
            print(f"Full API response: {response_data}")

            # Process the JSON response to get index data
            index_data = response_data.get(index_name, {})
            
            # Check if the index creation fields exist (creation_date and uuid)
            if index_data.get("settings", {}).get("index", {}).get("creation_date") and \
               index_data.get("settings", {}).get("index", {}).get("uuid"):
                print("Index created successfully.")
                return "CREATED"
            
            print("Index not yet created, retrying...")
            retries += 1
            time.sleep(retry_delay)  # Wait before retrying
        except requests.exceptions.RequestException as e:
            print(f"Error querying the index: {e}, retrying...")
            retries += 1
            time.sleep(retry_delay)  # Wait before retrying

    print("Reached maximum number of retries.")
    return None

# Main lambda handler function
def lambda_handler(event, context):
    try:
        # Get the OpenSearch collection endpoint
        OPENSEARCH_ENDPOINT = get_collection_endpoint()
        if OPENSEARCH_ENDPOINT:
            # Build the URL for creating the index
            url = f"https://{OPENSEARCH_ENDPOINT}.eu-central-1.aoss.amazonaws.com/{index_name}"
            headers = {"Content-Type": "application/json"}
            data = {
                "settings": {
                    "index": {
                        "knn": "true",
                        "knn.algo_param.ef_search": 512
                    }
                },
                "mappings": {
                    "properties": {
                        "bedrock-knowledge-base-default-vector": {
                            "type": "knn_vector",
                            "dimension": 1024,
                            "method": {
                                "name": "hnsw",
                                "engine": "faiss",
                                "parameters": {"ef_construction": 256, "m": 48},
                                "space_type": "innerproduct"
                            }
                        },
                        "AMAZON_BEDROCK_TEXT_CHUNK": {
                            "type": "text",
                            "index": "true"
                        },
                        "AMAZON_BEDROCK_METADATA": {
                            "type": "text",
                            "index": "false"
                        }
                    }
                }
            }
            print("Start-delay")
            time.sleep(60)  # Wait for 60 seconds before proceeding
            print("End-delay")
            
            # Send a PUT request to create the index
            response = requests.put(url, auth=auth, json=data, headers=headers)
            response.raise_for_status()

            # Check the index status
            get_index_status(url)

            # Send SUCCESS response to CloudFormation
            send_cfn_response(event, context, "SUCCESS", {"Message": "Index created successfully."})

            return {
                "statusCode": 200,
                "body": json.dumps("Index created successfully.")
            }

        else:
            # If collection endpoint not found, send FAILED response
            send_cfn_response(event, context, "FAILED", {"Message": "Collection endpoint not found."})
            return {
                "statusCode": 404,
                "body": json.dumps("Collection endpoint not found.")
            }

    except Exception as e:
        print(f"Error creating index: {e}")
        send_cfn_response(event, context, "FAILED", {"Error": str(e)})
        return {
            "statusCode": 500,
            "body": json.dumps(f"Error creating index: {e}")
        }
