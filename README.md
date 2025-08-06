# DG-PERS Archibot Chatbot

An AWS-powered, serverless chatbot application for document-driven question answering. This project supports multi-turn (_not yet_) conversational flow using Claude Sonnet 3.5, backed by vector-based semantic retrieval.

## Project Overview

This system is composed of the following major components:

- **Frontend**  
  Prebuilt React application hosted via S3 and CloudFront.

- **Backend Functions (Lambda)**  
  - `ask-the-docs-answer-question`: Answers user queries via RAG + Claude  

- **Infrastructure**  
  CloudFormation templates to provision all resources including:
  - S3 buckets  
  - OpenSearch Serverless vector store  
  - Bedrock KnowledgeBase and DataSource  
  - Lambda functions and IAM roles  
  - API Gateway and CloudFront CDN

- **Embedding & Indexing**  
  A deployment-time Lambda function (`createindexlambda`) processes document embeddings into OpenSearch.

## Repo Structure

```plaintext
chatbot-project/
├── frontend/              # Prebuilt React assets for hosting
├── lambda_functions/      # Lambda handlers for chatbot and context
├── layers/                # Lambda layers (requirements.txt only)
├── config/                # Prompt configs and environment settings
├── infrastructure/        # CloudFormation templates
├── scripts/               # CLI helpers (e.g. build_layer.sh, deploy.sh)
├── tests/                 # (Optional) unit/integration tests
├── README.md
└── .gitignore
```

## Development Environment

This project is built and deployed entirely on AWS. No local logic is executed during runtime.  
Current tooling:

- **Code Editor**: Visual Studio Code  
- **Repository Management**: GitLab  
- **Layer packaging**: Dependencies installed locally and zipped manually  
- **Environment**: Lambda functions target Python 3.11

## TODOs / Next Steps

- [ ] Add shell script to automate dependency layer packaging  
- [ ] Split `requirements.txt` by layer to reduce cold-start  
- [ ] Set up basic test coverage for Lambda logic  
- [ ] Begin logging and metrics integration via CloudWatch  
- [ ] Version and tag infrastructure templates

