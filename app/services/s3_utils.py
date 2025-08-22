import re
import os
import logging
from typing import Tuple, Optional
from urllib.parse import urlparse
import boto3
from botocore.exceptions import ClientError, NoCredentialsError

logger = logging.getLogger(__name__)


class S3URLError(Exception):
    """S3 URL 관련 에러 기본 클래스"""
    pass


class S3InvalidURLError(S3URLError):
    """잘못된 S3 URL 형식 에러"""
    pass


class S3AccessError(S3URLError):
    """S3 접근 권한 에러"""
    pass


def create_s3_client():
    """S3 클라이언트 생성 및 반환
    
    Returns:
        boto3.client: S3 클라이언트 객체
        
    Raises:
        NoCredentialsError: AWS 자격 증명이 없을 경우
        Exception: 기타 S3 클라이언트 생성 오류
    """
    try:
        return boto3.client(
            's3',
            aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
            aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
            region_name=os.getenv('AWS_REGION', 'ap-northeast-2')
        )
    except NoCredentialsError:
        logger.error("AWS 자격 증명이 설정되지 않았습니다.")
        raise
    except Exception as e:
        logger.error(f"S3 클라이언트 생성 실패: {str(e)}")
        raise


def parse_s3_url(s3_url: str) -> Tuple[str, str]:
    """S3 URL을 파싱하여 버킷 이름과 객체 키 추출
    
    Args:
        s3_url: S3 URL (다양한 형식 지원)
            - s3://bucket-name/path/to/object.jpg
            - https://bucket-name.s3.amazonaws.com/path/to/object.jpg
            - https://bucket-name.s3.ap-northeast-2.amazonaws.com/path/to/object.jpg
            - https://s3.amazonaws.com/bucket-name/path/to/object.jpg
            
    Returns:
        Tuple[str, str]: (bucket_name, object_key)
        
    Raises:
        S3InvalidURLError: URL 형식이 올바르지 않을 경우
    """
    if not s3_url or not isinstance(s3_url, str):
        raise S3InvalidURLError("유효하지 않은 S3 URL입니다.")
    
    s3_url = s3_url.strip()
    
    # s3:// 형식 URL 파싱
    s3_pattern = r'^s3://([^/]+)/(.+)$'
    match = re.match(s3_pattern, s3_url)
    if match:
        bucket_name, object_key = match.groups()
        return bucket_name, object_key
    
    # HTTPS 형식 URL 파싱
    try:
        parsed = urlparse(s3_url)
        if not parsed.netloc or not parsed.scheme == 'https':
            raise S3InvalidURLError(f"지원되지 않는 URL 스키마입니다: {s3_url}")
        
        # bucket-name.s3.amazonaws.com 형식
        bucket_s3_pattern = r'^([^.]+)\.s3\.([^.]+\.)?amazonaws\.com$'
        match = re.match(bucket_s3_pattern, parsed.netloc)
        if match:
            bucket_name = match.group(1)
            object_key = parsed.path.lstrip('/')
            if not object_key:
                raise S3InvalidURLError("객체 키가 없습니다.")
            return bucket_name, object_key
        
        # s3.amazonaws.com/bucket-name 형식
        if parsed.netloc in ['s3.amazonaws.com', 's3-ap-northeast-2.amazonaws.com']:
            path_parts = parsed.path.strip('/').split('/', 1)
            if len(path_parts) < 2:
                raise S3InvalidURLError("버킷 이름 또는 객체 키가 없습니다.")
            bucket_name, object_key = path_parts
            return bucket_name, object_key
            
    except Exception as e:
        if isinstance(e, S3InvalidURLError):
            raise
        raise S3InvalidURLError(f"URL 파싱 중 오류 발생: {str(e)}")
    
    raise S3InvalidURLError(f"지원되지 않는 S3 URL 형식입니다: {s3_url}")


def validate_s3_url_format(s3_url: str) -> bool:
    """S3 URL 형식이 유효한지 검증 (실제 접근 테스트 없음)
    
    Args:
        s3_url: 검증할 S3 URL
        
    Returns:
        bool: 형식이 유효하면 True, 아니면 False
    """
    try:
        bucket_name, object_key = parse_s3_url(s3_url)
        
        # 버킷 이름 규칙 검증
        if not bucket_name or len(bucket_name) < 3 or len(bucket_name) > 63:
            return False
            
        # 버킷 이름은 소문자, 숫자, 하이픈만 허용
        if not re.match(r'^[a-z0-9][a-z0-9\-]*[a-z0-9]$', bucket_name):
            return False
            
        # 객체 키가 비어있지 않은지 확인
        if not object_key:
            return False
            
        return True
        
    except S3InvalidURLError:
        return False


def check_s3_object_exists(s3_url: str) -> bool:
    """S3 객체가 실제로 존재하고 접근 가능한지 확인
    
    Args:
        s3_url: 확인할 S3 URL
        
    Returns:
        bool: 객체가 존재하고 접근 가능하면 True, 아니면 False
        
    Raises:
        S3InvalidURLError: URL 형식이 잘못된 경우
        S3AccessError: AWS 자격 증명 또는 권한 문제
    """
    try:
        bucket_name, object_key = parse_s3_url(s3_url)
        s3_client = create_s3_client()
        
        # head_object를 사용하여 객체 존재 여부 확인 (다운로드 없음)
        s3_client.head_object(Bucket=bucket_name, Key=object_key)
        logger.info(f"S3 객체 확인 성공: {s3_url}")
        return True
        
    except ClientError as e:
        error_code = e.response.get('Error', {}).get('Code', '')
        
        if error_code == 'NoSuchKey':
            logger.warning(f"S3 객체를 찾을 수 없습니다: {s3_url}")
            return False
        elif error_code == 'NoSuchBucket':
            logger.warning(f"S3 버킷을 찾을 수 없습니다: {bucket_name}")
            return False
        elif error_code in ['AccessDenied', 'Forbidden']:
            raise S3AccessError(f"S3 객체에 접근 권한이 없습니다: {s3_url}")
        else:
            logger.error(f"S3 객체 확인 중 오류: {error_code} - {str(e)}")
            return False
            
    except NoCredentialsError:
        raise S3AccessError("AWS 자격 증명이 설정되지 않았습니다.")
    
    except S3InvalidURLError:
        raise
        
    except Exception as e:
        logger.error(f"S3 객체 확인 중 예상치 못한 오류: {str(e)}")
        return False


def get_s3_public_url(s3_url: str, expires_in: int = 3600) -> str:
    """S3 객체의 공개 URL 또는 presigned URL 생성
    
    Args:
        s3_url: S3 URL
        expires_in: presigned URL 만료 시간(초), 기본값 1시간
        
    Returns:
        str: 공개 접근 가능한 URL
        
    Raises:
        S3InvalidURLError: URL 형식이 잘못된 경우
        S3AccessError: AWS 자격 증명 문제
    """
    try:
        bucket_name, object_key = parse_s3_url(s3_url)
        s3_client = create_s3_client()
        
        # 먼저 객체가 존재하는지 확인
        if not check_s3_object_exists(s3_url):
            raise S3InvalidURLError(f"S3 객체를 찾을 수 없습니다: {s3_url}")
        
        # presigned URL 생성
        presigned_url = s3_client.generate_presigned_url(
            'get_object',
            Params={'Bucket': bucket_name, 'Key': object_key},
            ExpiresIn=expires_in
        )
        
        logger.info(f"S3 presigned URL 생성 성공: {s3_url}")
        return presigned_url
        
    except (S3InvalidURLError, S3AccessError):
        raise
        
    except Exception as e:
        logger.error(f"S3 presigned URL 생성 실패: {str(e)}")
        raise S3AccessError(f"S3 URL 생성 중 오류 발생: {str(e)}") 