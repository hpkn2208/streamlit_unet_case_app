"""Cloudflare R2 (S3-compatible) object storage for case images.

Credentials come from st.secrets["r2"]: account_id, access_key_id,
secret_access_key, bucket. Objects are stored under PREFIX so this app can
share a bucket with streamlit_app without key collisions.
"""

import io
from functools import lru_cache

import numpy as np
import streamlit as st
from PIL import Image

PREFIX = "case-app"


@lru_cache(maxsize=1)
def _client():
    r2 = st.secrets["r2"]
    import boto3

    return boto3.client(
        "s3",
        endpoint_url=f"https://{r2['account_id']}.r2.cloudflarestorage.com",
        aws_access_key_id=r2["access_key_id"],
        aws_secret_access_key=r2["secret_access_key"],
        region_name="auto",
    )


def _bucket() -> str:
    return st.secrets["r2"]["bucket"]


def upload_image(image_array: np.ndarray, key: str) -> str:
    """Uploads an RGB array as PNG. Returns the full object key to store in the DB."""
    full_key = f"{PREFIX}/{key}"
    buf = io.BytesIO()
    Image.fromarray(image_array).save(buf, format="PNG")
    _client().put_object(Bucket=_bucket(), Key=full_key, Body=buf.getvalue(), ContentType="image/png")
    return full_key


def get_image_url(key: str, expires_in: int = 3600) -> str:
    """Pre-signed URL so st.image() can load the (private) object directly."""
    return _client().generate_presigned_url(
        "get_object", Params={"Bucket": _bucket(), "Key": key}, ExpiresIn=expires_in,
    )


def delete_image(key: str) -> None:
    _client().delete_object(Bucket=_bucket(), Key=key)
