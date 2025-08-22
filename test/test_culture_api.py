import asyncio
import httpx
import os
from dotenv import load_dotenv

# .env 파일 로드
load_dotenv()

async def test_culture_api_direct():
    """문화 API를 직접 호출해서 응답 형태를 확인"""
    
    culture_api_key = os.getenv("CULTURE_API_KEY")
    if not culture_api_key:
        print("❌ CULTURE_API_KEY 환경변수가 설정되지 않았습니다.")
        return
    
    print(f"✅ API Key: {culture_api_key[:10]}..." if len(culture_api_key) > 10 else culture_api_key)
    
    url = "https://api.kcisa.kr/openapi/service/rest/meta13/getCTE01701"
    params = {
        "serviceKey": culture_api_key,
        "keyword": "공주"
    }
    
    print(f"🔍 요청 URL: {url}")
    print(f"🔍 요청 파라미터: {params}")
    
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            print("📡 API 호출 중...")
            response = await client.get(url, params=params)
            
            print(f"📊 응답 상태 코드: {response.status_code}")
            print(f"📊 응답 헤더: {dict(response.headers)}")
            
            # 원시 응답 텍스트 확인
            raw_text = response.text
            print(f"📄 응답 길이: {len(raw_text)} 문자")
            print(f"📄 응답 시작 100자: {raw_text[:100]}")
            
            if raw_text.strip().startswith('<'):
                print("⚠️  응답이 XML 형태인 것 같습니다:")
                print(raw_text[:500])
                return
            
            # JSON 파싱 시도
            try:
                data = response.json()
                print("✅ JSON 파싱 성공!")
                print(f"📊 응답 구조: {list(data.keys()) if isinstance(data, dict) else type(data)}")
                
                # 응답 구조 자세히 확인
                if isinstance(data, dict) and "response" in data:
                    resp = data["response"]
                    print(f"📊 response 키: {list(resp.keys()) if isinstance(resp, dict) else type(resp)}")
                    
                    if isinstance(resp, dict) and "header" in resp:
                        header = resp["header"]
                        print(f"📊 header: {header}")
                        
                    if isinstance(resp, dict) and "body" in resp:
                        body = resp["body"]
                        print(f"📊 body 키: {list(body.keys()) if isinstance(body, dict) else type(body)}")
                        
                        if isinstance(body, dict) and "items" in body:
                            items = body["items"]
                            print(f"📊 items 키: {list(items.keys()) if isinstance(items, dict) else type(items)}")
                            
                            if isinstance(items, dict) and "item" in items:
                                item_list = items["item"]
                                print(f"📊 item 개수: {len(item_list) if isinstance(item_list, list) else 'Not a list'}")
                                
                                if isinstance(item_list, list) and len(item_list) > 0:
                                    first_item = item_list[0]
                                    print(f"📊 첫 번째 아이템 키: {list(first_item.keys()) if isinstance(first_item, dict) else type(first_item)}")
                                    
                                    if isinstance(first_item, dict) and "signDescription" in first_item:
                                        sign_desc = first_item["signDescription"]
                                        print(f"✅ signDescription: {sign_desc}")
                                    else:
                                        print("❌ signDescription 키가 없습니다.")
                                        print(f"📄 첫 번째 아이템 전체: {first_item}")
                else:
                    print("❌ 예상되는 응답 구조가 아닙니다.")
                    print(f"📄 전체 응답: {data}")
                    
            except Exception as json_error:
                print(f"❌ JSON 파싱 실패: {json_error}")
                print("📄 원시 응답 텍스트:")
                print(raw_text)
                
    except httpx.HTTPError as e:
        print(f"❌ HTTP 오류: {e}")
    except Exception as e:
        print(f"❌ 예상치 못한 오류: {e}")


async def test_different_keywords():
    """다양한 키워드로 테스트"""
    keywords = ["공주", "안녕", "사랑", "물", "음식"]
    
    culture_api_key = os.getenv("CULTURE_API_KEY")
    if not culture_api_key:
        print("❌ CULTURE_API_KEY 환경변수가 설정되지 않았습니다.")
        return
    
    url = "https://api.kcisa.kr/openapi/service/rest/meta13/getCTE01701"
    
    for keyword in keywords:
        print(f"\n🔍 테스트 키워드: {keyword}")
        params = {
            "serviceKey": culture_api_key,
            "keyword": keyword
        }
        
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(url, params=params)
                print(f"  상태 코드: {response.status_code}")
                
                if response.status_code == 200:
                    try:
                        data = response.json()
                        # 결과 개수 확인
                        if (data.get("response") and 
                            data["response"].get("body") and 
                            data["response"]["body"].get("items") and 
                            data["response"]["body"]["items"].get("item")):
                            
                            items = data["response"]["body"]["items"]["item"]
                            print(f"  결과 개수: {len(items) if isinstance(items, list) else 1}")
                            
                            if isinstance(items, list) and len(items) > 0:
                                first_item = items[0]
                                sign_desc = first_item.get("signDescription")
                                if sign_desc:
                                    print(f"  ✅ signDescription: {sign_desc[:50]}...")
                                else:
                                    print("  ❌ signDescription 없음")
                            else:
                                print("  ❌ 결과 없음")
                        else:
                            print("  ❌ 예상 구조 없음")
                    except Exception as e:
                        print(f"  ❌ JSON 파싱 실패: {e}")
                        print(f"  📄 응답 시작: {response.text[:100]}")
                else:
                    print(f"  ❌ HTTP 오류: {response.status_code}")
                    print(f"  📄 응답: {response.text[:200]}")
                    
        except Exception as e:
            print(f"  ❌ 오류: {e}")


if __name__ == "__main__":
    print("🧪 문화 API 테스트 시작\n")
    
    # 기본 API 호출 테스트
    print("=" * 50)
    print("1. 기본 API 호출 테스트")
    print("=" * 50)
    asyncio.run(test_culture_api_direct())
    
    # 다양한 키워드 테스트
    print("\n" + "=" * 50)
    print("2. 다양한 키워드 테스트")
    print("=" * 50)
    asyncio.run(test_different_keywords()) 