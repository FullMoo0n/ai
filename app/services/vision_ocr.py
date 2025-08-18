import os
import base64
import math
import re
from typing import Any, Dict, List, Tuple, Optional

import httpx


# =========================
# 공통 유틸
# =========================
def _get_api_key() -> str:
    key = os.getenv("VISION_API_KEY")
    if not key:
        raise RuntimeError("환경변수 VISION_API_KEY가 필요합니다.")
    return key

def _get_api_url() -> str:
    return f"https://vision.googleapis.com/v1/images:annotate?key={_get_api_key()}"

def encode_bytes_to_b64(content: bytes) -> str:
    return base64.b64encode(content).decode("utf-8")

def build_payload(
    image_b64: str,
    feature: str = "TEXT_DETECTION",  # 동화책 기본: 단어 박스 위주
    language_hints: Optional[List[str]] = None,
) -> Dict[str, Any]:
    req: Dict[str, Any] = {"image": {"content": image_b64}, "features": [{"type": feature}]}
    if language_hints:
        req["imageContext"] = {"languageHints": language_hints}
    return {"requests": [req]}

async def call_vision_api(payload: Dict[str, Any]) -> Dict[str, Any]:
    url = _get_api_url()
    async with httpx.AsyncClient(timeout=30.0) as client:
        res = await client.post(url, json=payload)
        res.raise_for_status()
        return res.json()

def extract_full_text(resp_json: Dict[str, Any]) -> str:
    resp = resp_json.get("responses", [{}])[0]
    text = resp.get("fullTextAnnotation", {}).get("text")
    if text:
        return text.strip()
    ann = resp.get("textAnnotations", [])
    if ann:
        return ann[0].get("description", "").strip()
    return ""

def _poly_to_box(vertices: List[Dict[str, int]]) -> Tuple[int, int, int, int]:
    xs = [v.get("x", 0) for v in vertices]
    ys = [v.get("y", 0) for v in vertices]
    return (min(xs or [0]), min(ys or [0]), max(xs or [0]), max(ys or [0]))  # x1,y1,x2,y2


# =========================
# 단어 박스 & 노이즈 제거
# =========================
def extract_word_boxes(resp_json: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    textAnnotations[1:]을 단어(토큰)로 취급
    [{'text': '...', 'box': [x1,y1,x2,y2]}]
    """
    out: List[Dict[str, Any]] = []
    resp = resp_json.get("responses", [{}])[0]
    for i, item in enumerate(resp.get("textAnnotations", [])):
        if i == 0:
            continue  # [0]은 전체 텍스트
        txt = item.get("description", "")
        if not txt:
            continue
        x1, y1, x2, y2 = _poly_to_box(item.get("boundingPoly", {}).get("vertices", []))
        out.append({"text": txt, "box": [x1, y1, x2, y2]})
    return out

_HANGUL = range(0xAC00, 0xD7A4)

def _is_hangul(s: str) -> bool:
    return any(ord(ch) in _HANGUL for ch in s)

def _box_area(b):  # [x1,y1,x2,y2]
    return max(0, b[2]-b[0]) * max(0, b[3]-b[1])

def _center(b):
    return ((b[0]+b[2]) / 2.0, (b[1]+b[3]) / 2.0)

def filter_noise_words(words: List[Dict[str, Any]], img_w: int, img_h: int) -> List[Dict[str, Any]]:
    """
    - 극소 박스 제거(이미지/페이지 규모 대비)
    - 한 글자 대문자/기호(B, ., , … 등) 제거(한국어 문맥에서 노이즈 확률 높음)
    """
    if not words:
        return []

    areas = sorted([_box_area(w["box"]) for w in words if _box_area(w["box"]) > 0])
    med_area = areas[len(areas)//2] if areas else 0
    min_area = max(12*12, int(0.02 * med_area))  # 필요시 조정

    out = []
    for w in words:
        t = w["text"].strip()
        b = w["box"]

        if _box_area(b) < min_area:
            continue

        if len(t) == 1 and not _is_hangul(t):
            # 알파벳 한 글자/점 등은 기본적으로 제거
            if t.isalpha() or t in {".", ",", "•", "·", "…"}:
                continue

        if len(t) == 1 and not t.isdigit() and not _is_hangul(t):
            continue

        out.append(w)
    return out


# =========================
# 세로/가로 방향 인식
# =========================
def _infer_orientation(words: List[Dict[str, any]]) -> str:
    """
    'vertical' / 'horizontal' 판정
    - 세로쓰기: h/w 비율↑, x 분산 < y 분산
    """
    if not words:
        return "horizontal"

    ratios = []
    xs, ys = [], []
    for w in words:
        x1,y1,x2,y2 = w["box"]
        w_ = max(1, x2-x1)
        h_ = max(1, y2-y1)
        ratios.append(h_/w_)
        cx, cy = _center(w["box"])
        xs.append(cx); ys.append(cy)

    med_ratio = sorted(ratios)[len(ratios)//2]
    var_x = sum((x - sum(xs)/len(xs))**2 for x in xs) / max(1, len(xs)-1)
    var_y = sum((y - sum(ys)/len(ys))**2 for y in ys) / max(1, len(ys)-1)

    if med_ratio >= 1.3 and (var_y > var_x*1.6):
        return "vertical"
    return "horizontal"


# =========================
# 줄/문단 병합 (가로쓰기)
# =========================
def _group_words_into_lines(words: List[Dict[str, Any]], line_tol: int = 12) -> List[List[Dict[str, Any]]]:
    """
    y1 근접 단어를 같은 줄로 묶는다.
    """
    if not words:
        return []
    words = sorted(words, key=lambda w: (w["box"][1], w["box"][0]))  # y,x 정렬
    lines: List[List[Dict[str, Any]]] = []
    cur: List[Dict[str, Any]] = [words[0]]
    for w in words[1:]:
        prev_y = cur[-1]["box"][1]
        if abs(w["box"][1] - prev_y) <= line_tol:
            cur.append(w)
        else:
            lines.append(sorted(cur, key=lambda ww: ww["box"][0]))
            cur = [w]
    lines.append(sorted(cur, key=lambda ww: ww["box"][0]))
    return lines

def _merge_lines_into_paragraphs(
    lines: List[List[Dict[str, Any]]],
    para_gap: int = 28,
    horiz_overlap_min: float = 0.1,
    para_x_join_tol: int = 30,
) -> List[List[Dict[str, Any]]]:
    """
    세로 간격 + 수평 겹침/근접을 이용해 줄들을 문단으로 병합
    """
    if not lines:
        return []

    def line_bbox(line):
        xs1 = [w["box"][0] for w in line]; ys1 = [w["box"][1] for w in line]
        xs2 = [w["box"][2] for w in line]; ys2 = [w["box"][3] for w in line]
        return min(xs1), min(ys1), max(xs2), max(ys2)

    paragraphs: List[List[Dict[str, Any]]] = []
    curp = lines[0]
    c_x1, c_y1, c_x2, c_y2 = line_bbox(curp)

    for line in lines[1:]:
        l_x1, l_y1, l_x2, l_y2 = line_bbox(line)
        vert_gap = l_y1 - c_y2

        inter_left = max(c_x1, l_x1); inter_right = min(c_x2, l_x2)
        inter = max(0, inter_right - inter_left)
        union = max(c_x2, l_x2) - min(c_x1, l_x1)
        overlap_ratio = (inter / union) if union > 0 else 0.0
        x_adjacent = (abs(l_x1 - c_x1) <= para_x_join_tol) or (abs(l_x2 - c_x2) <= para_x_join_tol)

        same_para = (vert_gap <= para_gap) and (overlap_ratio >= horiz_overlap_min or x_adjacent)

        if same_para:
            curp += line
            c_x1, c_y1, c_x2, c_y2 = min(c_x1, l_x1), min(c_y1, l_y1), max(c_x2, l_x2), max(c_y2, l_y2)
        else:
            paragraphs.append(curp)
            curp = line
            c_x1, c_y1, c_x2, c_y2 = l_x1, l_y1, l_x2, l_y2
    paragraphs.append(curp)
    return paragraphs


# =========================
# 세로쓰기: 열(column) 묶기
# =========================
def _group_words_into_columns_vertical(words: List[Dict[str, any]], col_tol: int = 18) -> List[List[Dict[str, any]]]:
    """
    세로쓰기: x 중심이 비슷한 단어를 동일 '열'로 묶음 (오른쪽 열부터 읽기)
    """
    if not words:
        return []
    words = sorted(words, key=lambda w: _center(w["box"])[0], reverse=True)  # x 내림차순(오른쪽→왼쪽)

    columns: List[List[Dict[str, any]]] = []
    cur = [words[0]]
    cur_x = _center(words[0]["box"])[0]

    for w in words[1:]:
        cx = _center(w["box"])[0]
        if abs(cx - cur_x) <= col_tol:
            cur.append(w)
        else:
            columns.append(sorted(cur, key=lambda ww: _center(ww["box"])[1]))  # 위→아래
            cur = [w]
            cur_x = cx
    columns.append(sorted(cur, key=lambda ww: _center(ww["box"])[1]))
    return columns

def _columns_to_paragraphs(columns: List[List[Dict[str, any]]]) -> List[List[Dict[str, any]]]:
    """
    세로쓰기에선 보통 열 하나가 한 문단인 경우가 많으므로 열 단위로 문단화
    """
    paragraphs: List[List[Dict[str, any]]] = []
    for col in columns:
        if col:
            paragraphs.append(col)
    return paragraphs


# =========================
# 가벼운 문장 정리
# =========================
def _tidy_korean(s: str) -> str:
    s = re.sub(r'\s+', ' ', s)              # 중복 공백
    s = re.sub(r'\s+([,\.!?…])', r'\1', s)  # 구두점 앞 공백 제거
    return s.strip()


# =========================
# 간단 클러스터링 (덩어리 나누기)
# =========================
def _cluster_by_distance(words: List[Dict[str, Any]], eps_px: int) -> List[List[Dict[str, Any]]]:
    """
    단어 중심 거리 eps_px 이내면 같은 클러스터로 묶는 연결 기반 클러스터링
    """
    if not words:
        return []
    centers = [_center(w["box"]) for w in words]
    n = len(words)
    parent = list(range(n))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for i in range(n):
        xi, yi = centers[i]
        for j in range(i+1, n):
            xj, yj = centers[j]
            if (xi - xj)**2 + (yi - yj)**2 <= eps_px**2:
                union(i, j)

    groups = {}
    for i in range(n):
        r = find(i)
        groups.setdefault(r, []).append(words[i])

    return list(groups.values())


# =========================
# 최종: 고급 문단 추출(세로/가로 자동)
# =========================
def extract_paragraphs_spatial_proximity_advanced(
    resp_json: Dict[str, Any],
    img_w: int,
    img_h: int,
    line_tol: int = 12,
    para_gap: int = 28,
    horiz_overlap_min: float = 0.1,
    para_x_join_tol: int = 30,
    cluster_scale: float = 0.06,
) -> List[Dict[str, Any]]:
    """
    1) 단어 박스 추출 → 2) 노이즈 제거 → 3) 거리 기반 클러스터링
    4) 클러스터별 세로/가로 판정 → 5) 해당 방식으로 줄/열 정렬 → 6) 문단 생성
    """
    words = extract_word_boxes(resp_json)
    words = filter_noise_words(words, img_w, img_h)
    if not words:
        return []

    diag = math.hypot(img_w, img_h)
    eps_px = max(24, int(diag * cluster_scale))
    clusters = _cluster_by_distance(words, eps_px=eps_px)

    paragraphs: List[Dict[str, Any]] = []
    for cluster in clusters:
        orient = _infer_orientation(cluster)

        if orient == "vertical":
            columns = _group_words_into_columns_vertical(cluster, col_tol=max(14, int(img_w * 0.008)))
            para_lines = _columns_to_paragraphs(columns)
            for line_group in para_lines:
                xs1 = [w["box"][0] for w in line_group]; ys1 = [w["box"][1] for w in line_group]
                xs2 = [w["box"][2] for w in line_group]; ys2 = [w["box"][3] for w in line_group]
                box = [min(xs1), min(ys1), max(xs2), max(ys2)]
                text = " ".join(w["text"] for w in line_group)
                paragraphs.append({"text": _tidy_korean(text), "box": box})

        else:
            lines = _group_words_into_lines(cluster, line_tol=line_tol)
            para_lines = _merge_lines_into_paragraphs(
                lines,
                para_gap=para_gap,
                horiz_overlap_min=horiz_overlap_min,
                para_x_join_tol=para_x_join_tol,
            )
            for line_group in para_lines:
                xs1 = [w["box"][0] for w in line_group]; ys1 = [w["box"][1] for w in line_group]
                xs2 = [w["box"][2] for w in line_group]; ys2 = [w["box"][3] for w in line_group]
                box = [min(xs1), min(ys1), max(xs2), max(ys2)]
                text = " ".join(w["text"] for w in line_group)
                paragraphs.append({"text": _tidy_korean(text), "box": box})

    paragraphs.sort(key=lambda p: (p["box"][1], p["box"][0]))  # 위→아래, 왼→오
    return paragraphs
