"""Choose the judge model that scores the DeepEval metrics.

Set the JUDGE environment variable:
    offline  (default) deterministic rules, no API key, used in CI   -> see offline_judge.py
    openai   OpenAI;       needs OPENAI_API_KEY, optional JUDGE_MODEL
    azure    Azure OpenAI; needs AZURE_OPENAI_API_KEY, AZURE_OPENAI_ENDPOINT,
             AZURE_DEPLOYMENT_NAME, AZURE_MODEL_NAME and OPENAI_API_VERSION
    bedrock  AWS Bedrock;  needs AWS credentials, AWS_BEDROCK_MODEL_NAME and AWS_BEDROCK_REGION
             (and `pip install aiobotocore`)

The real judges are DeepEval's own model classes, which read the settings above themselves.
"""

import os

from evals.offline_judge import OfflineJudge


def get_judge_name():
    return os.environ.get("JUDGE", "offline").lower()


def get_judge():
    name = get_judge_name()
    if name == "offline":
        return OfflineJudge()
    if name == "openai":
        from deepeval.models import OpenAIModel

        return OpenAIModel(model=os.environ.get("JUDGE_MODEL") or None)
    if name == "azure":
        from deepeval.models import AzureOpenAIModel

        return AzureOpenAIModel()
    if name == "bedrock":
        from deepeval.models import AmazonBedrockModel

        return AmazonBedrockModel()
    raise ValueError(f"Unknown JUDGE {name!r}. Use offline, openai, azure or bedrock.")
