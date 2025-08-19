import os
import httpx
from typing import Optional
import xml.etree.ElementTree as ET


async def get_sign_description(keyword: str) -> Optional[str]:
    """
    한국문화정보원 API를 호출하여 수어 설명을 가져옵니다.
    
    Args:
        keyword: 검색할 키워드
        
    Returns:
        첫 번째 검색 결과의 signDescription 또는 None
    """
    culture_api_key = os.getenv("CULTURE_API_KEY")
    if not culture_api_key:
        raise ValueError("CULTURE_API_KEY 환경변수가 설정되지 않았습니다.")
    
    url = "https://api.kcisa.kr/openapi/service/rest/meta13/getCTE01701"
    params = {
        "serviceKey": culture_api_key,
        "keyword": keyword
    }
    
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(url, params=params)
            response.raise_for_status()
            
            # XML 응답 파싱
            xml_text = response.text
            root = ET.fromstring(xml_text)
            
            # XML 구조: <response><body><items><item>...</item></items></body></response>
            items = root.find('.//items')
            if items is not None:
                item = items.find('item')
                if item is not None:
                    sign_description_elem = item.find('signDescription')
                    if sign_description_elem is not None and sign_description_elem.text:
                        return sign_description_elem.text.strip()
            
            return None
            
    except httpx.HTTPError as e:
        raise Exception(f"API 호출 실패: {str(e)}")
    except ET.ParseError as e:
        raise Exception(f"XML 파싱 실패: {str(e)}")
    except Exception as e:
        raise Exception(f"데이터 처리 실패: {str(e)}") 