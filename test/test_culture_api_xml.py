import asyncio
import sys
import os

# 프로젝트 루트 경로 추가
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from dotenv import load_dotenv
from app.services.culture_api import get_sign_description

# .env 파일 로드
load_dotenv()

async def test_xml_parsing():
    """XML 파싱이 제대로 작동하는지 테스트"""
    
    test_keywords = ["공주", "안녕", "사랑", "물", "음식", "학교"]
    
    print("🧪 XML 파싱 테스트 시작\n")
    
    for keyword in test_keywords:
        print(f"🔍 테스트 키워드: '{keyword}'")
        
        try:
            result = await get_sign_description(keyword)
            
            if result:
                print(f"✅ 성공: {result}")
            else:
                print("❌ 결과 없음")
                
        except Exception as e:
            print(f"❌ 오류: {e}")
        
        print("-" * 50)

if __name__ == "__main__":
    asyncio.run(test_xml_parsing()) 