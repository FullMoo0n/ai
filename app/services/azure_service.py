import os
import logging
from typing import Union
import io
from urllib.parse import urlparse
from datetime import datetime, timedelta

from azure.storage.blob import BlobServiceClient, generate_blob_sas, BlobSasPermissions
from azure.core.exceptions import ResourceNotFoundError

logger = logging.getLogger(__name__)


class AzureServiceError(Exception):
    """Azure 처리 관련 에러 기본 클래스"""

    pass


class AzureURLError(AzureServiceError):
    """Azure Blob URL 관련 에러"""

    pass


class AzureAccessError(AzureServiceError):
    """Azure 접근 권한 에러"""

    pass


def get_blob_service_client() -> BlobServiceClient:
    """Azure Blob Storage 클라이언트 생성 및 반환"""
    connection_string = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
    if not connection_string:
        # Retry with quotes removed if present
        connection_string = os.getenv("AZURE_STORAGE_CONNECTION_STRING", "").strip(
            "'\""
        )

    if not connection_string:
        raise AzureAccessError(
            "AZURE_STORAGE_CONNECTION_STRING 환경변수가 설정되지 않았습니다."
        )

    try:
        return BlobServiceClient.from_connection_string(connection_string)
    except Exception as e:
        logger.error(f"Azure Blob 클라이언트 생성 실패: {str(e)}")
        raise AzureAccessError(f"Azure 연결 오류: {str(e)}")


def parse_azure_url(blob_url: str) -> tuple[str, str]:
    """Azure Blob URL을 파싱하여 컨테이너 이름과 블롭 이름 추출

    Args:
        blob_url: Azure Blob URL (예: https://stdevbinary.blob.core.windows.net/blob-binary/image.jpg)

    Returns:
        tuple[str, str]: (container_name, blob_name)
    """
    if not blob_url or not isinstance(blob_url, str):
        raise AzureURLError("유효하지 않은 Azure Blob URL입니다.")

    try:
        parsed = urlparse(blob_url)
        if not parsed.netloc.endswith(".blob.core.windows.net"):
            raise AzureURLError(f"지원되지 않는 URL 스키마/호스트입니다: {blob_url}")

        path_parts = parsed.path.strip("/").split("/", 1)
        if len(path_parts) < 2:
            raise AzureURLError("컨테이너 이름 또는 블롭 이름이 없습니다.")

        container_name, blob_name = path_parts
        return container_name, blob_name

    except Exception as e:
        if isinstance(e, AzureURLError):
            raise
        raise AzureURLError(f"URL 파싱 중 오류 발생: {str(e)}")


def check_azure_blob_exists(blob_url: str) -> bool:
    """Azure Blob이 실제로 존재하는지 확인"""
    try:
        container_name, blob_name = parse_azure_url(blob_url)
        blob_service_client = get_blob_service_client()
        blob_client = blob_service_client.get_blob_client(
            container=container_name, blob=blob_name
        )

        return blob_client.exists()

    except ResourceNotFoundError:
        return False
    except AzureURLError:
        raise
    except Exception as e:
        logger.error(f"Azure Blob 확인 중 오류 발생: {str(e)}")
        return False


def get_azure_public_url(blob_url: str, expires_in_hours: int = 1) -> str:
    """Azure Blob의 공개 접근 가능한 URL (SAS 토큰 포함) 생성
    주의: Blob 컨테이너가 public 읽기를 허용하는 경우 SAS 토큰 없이도 접근 가능합니다.
    이 함수는 컨테이너 설정에 관계없이 접근 가능한 SAS URL을 생성합니다.
    """
    try:
        container_name, blob_name = parse_azure_url(blob_url)
        blob_service_client = get_blob_service_client()
        blob_client = blob_service_client.get_blob_client(
            container=container_name, blob=blob_name
        )

        if not blob_client.exists():
            raise AzureURLError(f"Azure Blob을 찾을 수 없습니다: {blob_url}")

        account_name = blob_service_client.account_name
        account_key = blob_service_client.credential.account_key

        sas_token = generate_blob_sas(
            account_name=account_name,
            container_name=container_name,
            blob_name=blob_name,
            account_key=account_key,
            permission=BlobSasPermissions(read=True),
            expiry=datetime.utcnow() + timedelta(hours=expires_in_hours),
        )

        sas_url = f"{blob_client.url}?{sas_token}"
        return sas_url

    except (AzureURLError, AzureAccessError):
        raise
    except Exception as e:
        logger.error(f"Azure SAS URL 생성 실패: {str(e)}")
        raise AzureAccessError(f"SAS URL 생성 중 오류 발생: {str(e)}")


async def upload_stream_to_azure(
    stream: Union[io.BytesIO, bytes], blob_name: str, content_type: str = "video/mp4"
) -> str:
    """바이트 스트림을 Azure Blob Storage에 업로드합니다.

    Args:
        stream: 업로드할 파일 데이터
        blob_name: 저장할 Blob 이름 (경로 포함)
        content_type: MIME 타입

    Returns:
        str: 업로드된 파일의 원본(public 가정 시) Blob URL
    """
    try:
        container_name = os.getenv("AZURE_CONTAINER_NAME", "blob-binary")
        blob_service_client = get_blob_service_client()
        blob_client = blob_service_client.get_blob_client(
            container=container_name, blob=blob_name
        )

        if isinstance(stream, bytes):
            data = stream
        else:
            data = stream.getvalue()

        # Upload data
        blob_client.upload_blob(
            data,
            blob_type="BlockBlob",
            overwrite=True,
            content_settings={"content_type": content_type},
        )

        # public 접근이 가능한 컨테이너를 가정하고 기본 URL 반환
        logger.info(f"✅ Azure 업로드 완료: {blob_client.url}")
        return blob_client.url

    except Exception as e:
        logger.error(f"❌ Azure 업로드 실패: {str(e)}")
        raise AzureServiceError(f"Azure 업로드 중 오류: {str(e)}")
