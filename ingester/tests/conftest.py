import os
from typing import Generator, TypeVar

import pytest
from botocore.exceptions import ClientError
from fastapi.testclient import TestClient
from elasticsearch import Elasticsearch
from sentence_transformers import SentenceTransformer

from ingester.src.worker import env, app
from redbox.models import File

T = TypeVar("T")

YieldFixture = Generator[T, None, None]


@pytest.fixture
def s3_client():
    _client = env.s3_client()
    try:
        _client.create_bucket(
            Bucket=env.bucket_name,
            CreateBucketConfiguration={"LocationConstraint": env.aws_region},
        )
    except ClientError as e:
        if e.response["Error"]["Code"] != "BucketAlreadyOwnedByYou":
            raise e

    yield _client


@pytest.fixture
def es_client() -> YieldFixture[Elasticsearch]:
    yield env.elasticsearch_client()


@pytest.fixture
def embedding_model() -> YieldFixture[SentenceTransformer]:
    yield SentenceTransformer(env.embedding_model)


@pytest.fixture
def file_pdf_path() -> YieldFixture[str]:
    path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..",
        "..",
        "tests",
        "data",
        "pdf",
        "Cabinet Office - Wikipedia.pdf",
    )
    yield path


@pytest.fixture
def file(s3_client, file_pdf_path):
    file_name = os.path.basename(file_pdf_path)
    file_type = f'.{file_name.split(".")[-1]}'

    with open(file_pdf_path, "rb") as f:
        s3_client.put_object(
            Bucket=env.bucket_name,
            Body=f.read(),
            Key=file_name,
            Tagging=f"file_type={file_type}",
        )

    file_record = File(
        key=file_name,
        bucket=env.bucket_name,
    )

    yield file_record


@pytest.fixture
def app_client():
    yield TestClient(app)
