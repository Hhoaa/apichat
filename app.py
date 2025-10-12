# pip install flask flask-cors google-generativeai supabase python-dotenv werkzeug

import os
import sys
import json
import tempfile
import mimetypes
from typing import List, Dict, Any, Tuple, Optional
import re

from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename

import google.generativeai as genai
from supabase import create_client, Client

# Ensure UTF-8 output
sys.stdout.reconfigure(encoding='utf-8')

# -----------------------------
# Config via environment vars
# -----------------------------
GEMINI_API_KEY = "AIzaSyC_mgnRedmnOFnhgL6vLZmKNTAHUudK0pc"
SUPABASE_URL ="https://acddbjalchiruigappqg.supabase.co"
SUPABASE_ANON_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImFjZGRiamFsY2hpcnVpZ2FwcHFnIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NTkwMzAzMTQsImV4cCI6MjA3NDYwNjMxNH0.Psefs-9-zIwe8OjhjQOpA19MddU3T9YMcfFtMcYQQS4"

if not GEMINI_API_KEY or not SUPABASE_URL or not SUPABASE_ANON_KEY:
    print("Please set GEMINI_API_KEY, SUPABASE_URL, SUPABASE_ANON_KEY in environment.")
    sys.exit(1)

genai.configure(api_key=GEMINI_API_KEY)
supabase: Client = create_client(SUPABASE_URL, SUPABASE_ANON_KEY)

model = genai.GenerativeModel("gemini-2.0-flash")

app = Flask(__name__)
CORS(app)

MAX_FILE_SIZE = 20 * 1024 * 1024  # 20MB
ALLOWED_IMAGE_EXT = {"jpg", "jpeg", "png", "webp"}


def is_allowed_image(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_IMAGE_EXT


def get_mime_type(filename: str) -> str:
    mtype, _ = mimetypes.guess_type(filename)
    return mtype or "application/octet-stream"


# ===================================
# IMPROVED EXTRACTION PROMPT - WITH SMART PRICE ANALYSIS
# ===================================
EXTRACTION_PROMPT = """Bạn là trợ lý thông minh phân tích yêu cầu mua sắm thời trang.

NHIỆM VỤ: Phân tích tin nhắn để xác định ý định của người dùng VÀ PHÂN TÍCH GIÁ CHÍNH XÁC.

PHÂN LOẠI Ý ĐỊNH:
1. "greeting" - Chào hỏi: "xin chào", "hi", "hello", "chào shop"
2. "general_question" - Hỏi chung: "shop ở đâu", "giao hàng thế nào", "có uy tín không"
3. "style_advice" - Xin tư vấn: "mặc gì đẹp", "phối đồ thế nào", "hợp với tôi không"
4. "product_search" - Tìm sản phẩm: "tìm váy", "có áo sơ mi không", "xem quần jean"
5. "product_question" - Hỏi về sản phẩm cụ thể: "size nào", "còn màu đen không", "giá bao nhiêu"

⚠️ PHÂN TÍCH GIÁ THÔNG MINH (QUAN TRỌNG):
Khi người dùng nói về giá, hãy PHÂN TÍCH CHÍNH XÁC:

1. **GIÁ TỐI ĐA (max only):**
   - "300k trở xuống" → min: null, max: 300000
   - "dưới 500k" → min: null, max: 500000
   - "không quá 400k" → min: null, max: 400000
   - "tối đa 300k" → min: null, max: 300000

2. **GIÁ TỐI THIỂU (min only):**
   - "500k trở lên" → min: 500000, max: null
   - "trên 300k" → min: 300000, max: null
   - "ít nhất 400k" → min: 400000, max: null

3. **KHOẢNG GIÁ (range):**
   - "300k-500k" → min: 300000, max: 500000
   - "từ 200k đến 400k" → min: 200000, max: 400000

4. **GIÁ KHOẢNG (around ±20%):**
   - "tầm 300k" → min: 240000, max: 360000
   - "khoảng 500k" → min: 400000, max: 600000

5. **ĐƠN VỊ:**
   - "300" hoặc "300k" → 300,000 VNĐ
   - "1tr" hoặc "1 triệu" → 1,000,000 VNĐ

CHỈ TRÍCH XUẤT thông tin sản phẩm KHI user_intent là "product_search" hoặc "product_question".

TRẢ VỀ JSON:
{
  "user_intent": "greeting | general_question | style_advice | product_search | product_question",
  "confidence": 0.0-1.0,
  "should_search_products": true/false,
  "type": "váy | áo | quần | chân váy | áo sơ mi | áo thun | quần jean | đầm | ...",
  "colors": ["đen", "trắng", "xanh navy", ...],
  "material": "cotton | jeans | lụa | len | da | ...",
  "pattern": "trơn | kẻ sọc | caro | hoa | chấm bi | ...",
  "style": ["công sở", "dạo phố", "dự tiệc", "thể thao", ...],
  "length": "ngắn | midi | dài | qua gối | ...",
  "sleeve": "sát nách | ngắn tay | dài tay | ...",
  "fit": "ôm | suông | rộng | ...",
  "price_range": {"min": 200000, "max": 500000},
  "price_analysis": "Giải thích cách bạn phân tích giá",
  "keywords": ["từ khóa tìm kiếm"],
  "conversation_context": "phân tích ngắn về ngữ cảnh"
}

QUY TẮC QUAN TRỌNG:
- should_search_products = true CHỈ KHI có ý định tìm/mua sản phẩm rõ ràng
- keywords PHẢI rỗng [] nếu không có ý định tìm sản phẩm
- keywords KHÔNG BAO GIỜ chứa "không rõ" hoặc giá trị mơ hồ
- price_range PHẢI chính xác dựa trên ý định người dùng
- Nếu chỉ chào hỏi: user_intent="greeting", should_search_products=false, keywords=[]
- Nếu hỏi tư vấn chung: user_intent="style_advice", should_search_products=false

VÍ DỤ:
Input: "xin chào"
Output: {"user_intent": "greeting", "should_search_products": false, "keywords": []}

Input: "tìm váy đen giá 300k trở xuống"
Output: {"user_intent": "product_search", "should_search_products": true, "type": "váy", "colors": ["đen"], "price_range": {"min": null, "max": 300000}, "price_analysis": "300k trở xuống = tối đa 300k", "keywords": ["váy đen"]}

Input: "váy tầm 300k"
Output: {"user_intent": "product_search", "should_search_products": true, "type": "váy", "price_range": {"min": 240000, "max": 360000}, "price_analysis": "tầm 300k = khoảng ±20%", "keywords": ["váy"]}

Input: "mặc gì đẹp?"
Output: {"user_intent": "style_advice", "should_search_products": false, "keywords": []}
"""


# ===================================
# IMPROVED CHAT PROMPT
# ===================================
CHAT_PROMPT = """Bạn là Mina - trợ lý thời trang thân thiện, nhiệt tình của Zamy Shop. 

🎯 TÍNH CÁCH CỦA BẠN:
- Thân thiện, gần gũi như người bạn (không khách sáo)
- Nhiệt tình nhưng không áp đặt
- Tinh tế, hiểu tâm lý phụ nữ
- Sử dụng emoji tự nhiên (1-2/câu): 😊 💕 ✨ 👗 
- Đặt câu hỏi mở để hiểu rõ khách hàng
- Trả lời ngắn gọn, súc tích (2-4 câu)

👤 THÔNG TIN KHÁCH HÀNG:
- Tên: {customer_name}
- Chiều cao: {height} | Cân nặng: {weight}  
- Màu yêu thích: {favorite_colors}

📝 LỊCH SỬ TRÒ CHUYỆN:
{chat_history}

🎨 Ý ĐỊNH HIỆN TẠI: {current_intent}
📦 SẢN PHẨM TÌM ĐƯỢC: {products_found}

---

HƯỚNG DẪN TRẢ LỜI THEO TỪNG TÌNH HUỐNG:

1️⃣ CHÀO HỎI LẦN ĐẦU (user_intent = "greeting"):
✅ Làm:
- Chào thân mật, ấm áp
- Giới thiệu ngắn gọn về mình
- Hỏi tên khách (nếu chưa biết)
- Hỏi mở: "Hôm nay bạn cần tìm gì?" hoặc "Mình có thể giúp gì cho bạn?"

❌ Không làm:
- Không liệt kê sản phẩm ngay
- Không hỏi quá nhiều câu cùng lúc
- Không dùng ngôn ngữ marketing

Ví dụ tốt:
"Chào bạn! Mình là Mina, trợ lý thời trang của Zamy Shop đây 😊 Rất vui được hỗ trợ bạn hôm nay! Bạn tên gì nhỉ?"

2️⃣ TƯ VẤN PHONG CÁCH (user_intent = "style_advice"):
✅ Làm:
- Đặt câu hỏi để hiểu rõ: dịp gì? phong cách nào?
- Phân tích dựa trên chiều cao/cân nặng
- Gợi ý 2-3 style cụ thể với LÝ DO
- Kết thúc bằng câu hỏi tiếp theo

Ví dụ tốt:
"Với chiều cao {height} và vóc dáng cân đối của bạn, mình nghĩ bạn sẽ rất hợp với:
- Váy chữ A midi: tôn dáng và thanh lịch 
- Quần ống suông + áo croptop: trẻ trung, năng động

Bạn định mặc đi đâu nhỉ? Công sở hay đi chơi?"

3️⃣ TÌM SẢN PHẨM - CÓ KẾT QUẢ (products_found > 0):
✅ Làm:
- Xác nhận đã hiểu nhu cầu
- Thông báo tìm được sản phẩm một cách tự nhiên
- Nhấn mạnh ưu điểm nổi bật (1-2 điểm)
- Hỏi xem có cần filter thêm không

❌ Không làm:
- Không nói "Tuyệt vời! Tôi đã tìm thấy..."
- Không dùng câu template cứng nhắc

Ví dụ tốt:
"Mình tìm được mấy em váy đẹp trong tầm giá bạn cần luôn! 😍 Có cả màu đen và trắng, vừa túi tiền mà chất lượng tốt nè.

Bạn thích dáng nào hơn: ôm hay suông?"

4️⃣ TÌM SẢN PHẨM - KHÔNG CÓ KẾT QUẢ (products_found = 0):
❌ TUYỆT ĐỐI KHÔNG ĐƯỢC:
- Hỏi thêm thông tin hoặc yêu cầu làm rõ
- Nói "hiện tại không có" rồi dừng lại
- Chỉ xin lỗi mà không đưa ra giải pháp

✅ BẮT BUỘC PHẢI LÀM:
- Giải thích ngắn gọn tại sao không tìm thấy (hết hàng, giá không phù hợp...)
- NGAY LẬP TỨC đề xuất 2-3 sản phẩm thay thế cụ thể
- Nhấn mạnh ưu điểm của sản phẩm thay thế
- Hỏi xem khách có muốn xem không (câu đóng, dễ trả lời)

Ví dụ tốt:
"Váy đen ôm body giá dưới 300k hiện tại đang hết hàng rồi bạn ơi 😢 

Nhưng mình có mấy lựa chọn tương tự cũng đẹp lắm nè:
- Váy xanh navy ôm dáng: thanh lịch, giá 280k
- Váy đen suông: dễ mặc hơn, 250k

Bạn muốn xem không? 😊"

5️⃣ TRẢ LỜI CÂU HỎI CHUNG (user_intent = "general_question"):
✅ Làm:
- Trả lời trực tiếp, ngắn gọn
- Thêm thông tin hữu ích liên quan
- Hỏi liệu còn thắc mắc gì không

Ví dụ tốt:
"Zamy Shop giao hàng toàn quốc trong 2-3 ngày bạn nhé! Freeship cho đơn từ 300k 🚚

Bạn ở tỉnh nào? Mình check giúp thời gian giao cụ thể nha!"

---

⚠️ LƯU Ý QUAN TRỌNG:
- LUÔN gọi khách hàng là "bạn" (không dùng "chị", "cô", "anh")
- SỬ DỤNG ngôi "mình" thay vì "tôi" hoặc "em"
- TRÁNH các cụm từ AI: "Tôi có thể giúp", "Tôi là trợ lý AI"
- MỖI câu trả lời NÊN có 1 câu hỏi mở ở cuối
- ĐỌC kỹ lịch sử chat, KHÔNG lặp lại câu hỏi đã hỏi
- NẾU khách đã cung cấp thông tin, HÃY sử dụng ngay (tên, sở thích...)
- KHÔNG đề cập "sản phẩm trong database" - nói tự nhiên
- CHỈ gợi ý sản phẩm CÓ THẬT, không bịa ra
- 🚨 TUYỆT ĐỐI KHÔNG BỊA GIÁ - Chỉ dùng giá từ danh sách sản phẩm được cung cấp
- 🚨 KHI nói về giá, PHẢI dùng CHÍNH XÁC số tiền từ thông tin sản phẩm, KHÔNG làm tròn

---

HÃY TRẢ LỜI TIN NHẮN SAU ĐÂY:
User: {user_message}
"""


def extract_keywords_with_gemini(user_message: str, file_path: str | None, mime_type: str | None) -> Dict[str, Any]:
    """Extract keywords and intent from user message with SMART PRICE DETECTION"""
    try:
        if file_path:
            uploaded_file = genai.upload_file(file_path, mime_type=mime_type)
            parts = [uploaded_file, user_message or "Hãy phân tích ảnh này."]
        else:
            parts = [user_message or ""]

        response = model.generate_content([EXTRACTION_PROMPT, *parts])
        text = (response.text or "").strip()
        
        # Remove markdown code blocks if present
        if text.startswith("```"):
            lines = text.split("\n")
            text = "\n".join(lines[1:-1]) if len(lines) > 2 else text
        text = text.strip()

        def try_parse_json(s: str):
            try:
                return json.loads(s)
            except Exception:
                start = s.find("{")
                end = s.rfind("}")
                if start != -1 and end != -1 and end > start:
                    return json.loads(s[start:end + 1])
                raise

        data = try_parse_json(text)
        
        # Ensure keywords is a list and doesn't contain "không rõ"
        keywords = data.get("keywords", [])
        if not isinstance(keywords, list):
            keywords = []
        keywords = [k for k in keywords if k and isinstance(k, str) and k.strip() and "không rõ" not in k.lower()]
        
        # Build keywords from structured data if empty
        if not keywords and data.get("should_search_products", False):
            type_ = data.get("type") or ""
            colors = data.get("colors") or []
            material = data.get("material") or ""
            pattern = data.get("pattern") or ""
            style = data.get("style") or []
            
            candidates = []
            base = type_.strip()
            if base:
                candidates.append(base)
                if colors:
                    for c in colors[:2]:
                        candidates.append(f"{base} {c}")
                if pattern and pattern != "không rõ":
                    candidates.append(f"{base} {pattern}")
                if material and material != "không rõ":
                    candidates.append(f"{base} {material}")
            
            keywords = candidates[:6]
        
        # Remove duplicates while preserving order
        keywords = list(dict.fromkeys([k.strip() for k in keywords if k.strip()]))[:8]
        
        # Extract price range from AI analysis
        price_range = data.get("price_range", {})
        min_price = price_range.get("min") if isinstance(price_range, dict) else None
        max_price = price_range.get("max") if isinstance(price_range, dict) else None
        price_analysis = data.get("price_analysis", "")
        
        out = {
            "keywords": keywords,
            "user_intent": data.get("user_intent", "general_question"),
            "should_search_products": data.get("should_search_products", False),
            "confidence": data.get("confidence", 0.5),
            "conversation_context": data.get("conversation_context", ""),
            "price_min": min_price,
            "price_max": max_price,
            "price_analysis": price_analysis,
        }
        
        # Copy other fields
        for k in ["type","colors","material","pattern","style","length","sleeve","fit"]:
            if k in data:
                out[k] = data[k]
        
        print(f"🔍 [EXTRACTION] Intent: {out['user_intent']}, Should search: {out['should_search_products']}, Keywords: {keywords}")
        print(f"💰 [PRICE AI] Min: {min_price}, Max: {max_price} - {price_analysis}")
        
        return out
        
    except Exception as e:
        print(f"[extract_keywords_with_gemini] error: {e}")
        # Safe fallback
        return {
            "keywords": [],
            "user_intent": "general_question",
            "should_search_products": False,
            "confidence": 0.0,
            "conversation_context": "Lỗi phân tích",
            "price_min": None,
            "price_max": None,
            "price_analysis": ""
        }


def build_or_clause_for_keywords(columns: List[str], keywords: List[str]) -> str:
    parts = []
    for kw in keywords:
        pattern = f"%{kw}%"
        for col in columns:
            parts.append(f"{col}.ilike.{pattern}")
    return ",".join(parts)


def score_product(product: Dict[str, Any], keywords: List[str]) -> int:
    text = f"{product.get('ten_san_pham','')} {product.get('mo_ta_san_pham','')}".lower()
    cnt = 0
    for kw in keywords:
        if kw.lower() in text:
            cnt += 1
    return cnt


def map_product_row(row: Dict[str, Any]) -> Dict[str, Any]:
    images = []
    if isinstance(row.get("product_images"), list):
        for img in row["product_images"]:
            url = img.get("duong_dan_anh")
            if isinstance(url, str):
                images.append(url)

    return {
        "id": int(row.get("ma_san_pham")) if row.get("ma_san_pham") else None,
        "name": row.get("ten_san_pham"),
        "description": row.get("mo_ta_san_pham"),
        "price": float(row.get("gia_ban") or 0.0),
        "original_price": float(row.get("muc_gia_goc") or 0.0),
        "images": images,
    }


@app.route("/api/search_products", methods=["POST"])
def search_products():
    try:
        user_message = ""
        file_path = None
        mime_type = None

        if request.content_type and "multipart/form-data" in request.content_type:
            user_message = request.form.get("message", "") or ""
            file = request.files.get("file")
            if file and file.filename:
                if not is_allowed_image(file.filename):
                    return jsonify({"error": "Chỉ chấp nhận ảnh .jpg, .jpeg, .png, .webp"}), 400
                file.seek(0, os.SEEK_END)
                size = file.tell()
                file.seek(0)
                if size > MAX_FILE_SIZE:
                    return jsonify({"error": "File quá lớn (tối đa 20MB)"}), 400

                filename = secure_filename(file.filename)
                mime_type = get_mime_type(filename)
                suffix = "." + filename.rsplit(".", 1)[1].lower()
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                    file.save(tmp.name)
                    file_path = tmp.name
        else:
            data = request.get_json(silent=True) or {}
            user_message = data.get("message", "") or ""

        extracted = extract_keywords_with_gemini(user_message, file_path, mime_type)
        keywords: List[str] = extracted.get("keywords", [])[:6]
        
        # Clean up temp file
        if file_path and os.path.exists(file_path):
            try:
                os.unlink(file_path)
            except Exception:
                pass

        # If no product search intent, return empty
        if not extracted.get("should_search_products", False) or not keywords:
            return jsonify({
                "keywords": keywords,
                "products": [],
                "user_intent": extracted.get("user_intent"),
                "notes": extracted.get("conversation_context", "")
            })

        # Get price from AI analysis
        min_price = extracted.get("price_min")
        max_price = extracted.get("price_max")
        det_type = (extracted.get("type") or "").strip().lower() or None
        det_colors = [c.strip().lower() for c in (extracted.get("colors") or []) if isinstance(c, str) and c.strip()]

        # Build search query
        columns = ["ten_san_pham", "mo_ta_san_pham"]
        or_clause = build_or_clause_for_keywords(columns, keywords)
        
        q = supabase.table("products").select(
            "ma_san_pham,ten_san_pham,mo_ta_san_pham,gia_ban,muc_gia_goc,product_images(duong_dan_anh)"
        )
        
        # Apply filters
        if min_price is not None:
            q = q.gte("gia_ban", min_price)
            print(f"💰 [FILTER] Min price: {min_price}")
        if max_price is not None:
            q = q.lte("gia_ban", max_price)
            print(f"💰 [FILTER] Max price: {max_price}")
        if det_type:
            type_clause = build_or_clause_for_keywords(["ten_san_pham", "mo_ta_san_pham"], [det_type])
            if type_clause:
                q = q.or_(type_clause)
        if det_colors:
            color_clause = build_or_clause_for_keywords(["ten_san_pham", "mo_ta_san_pham"], det_colors[:2])
            if color_clause:
                q = q.or_(color_clause)
        if or_clause:
            q = q.or_(or_clause)
        
        resp = q.limit(20).execute()
        rows = resp.data or []

        # Fallback search if no results
        if not rows and keywords:
            single_tokens = [t for t in keywords if len(t.split()) == 1]
            if single_tokens:
                or_clause_2 = build_or_clause_for_keywords(["ten_san_pham", "mo_ta_san_pham"], single_tokens)
                q2 = supabase.table("products").select(
                    "ma_san_pham,ten_san_pham,mo_ta_san_pham,gia_ban,muc_gia_goc,product_images(duong_dan_anh)"
                )
                if min_price is not None:
                    q2 = q2.gte("gia_ban", min_price)
                if max_price is not None:
                    q2 = q2.lte("gia_ban", max_price)
                if or_clause_2:
                    q2 = q2.or_(or_clause_2)
                resp2 = q2.limit(20).execute()
                rows = resp2.data or []
        
        # Sort by relevance
        def rank_row(r: Dict[str, Any]) -> tuple:
            name = (r.get("ten_san_pham") or "").lower()
            desc = (r.get("mo_ta_san_pham") or "").lower()
            txt = name + " " + desc
            score = score_product(r, keywords)
            type_bonus = 3 if det_type and det_type in txt else 0
            color_bonus = 0
            if det_colors:
                color_bonus = sum(1 for c in det_colors if c in txt)
            price_penalty = 0
            price = float(r.get("gia_ban") or 0)
            if min_price is not None or max_price is not None:
                center = ((min_price or price) + (max_price or price)) / 2.0
                price_penalty = abs(price - center) / max(center, 1.0)
            return (score + type_bonus + color_bonus, -price_penalty)

        rows_sorted = sorted(rows, key=lambda r: rank_row(r), reverse=True)
        products = [map_product_row(r) for r in rows_sorted]

        return jsonify({
            "keywords": keywords,
            "user_intent": extracted.get("user_intent"),
            "notes": extracted.get("conversation_context", ""),
            "price_analysis": extracted.get("price_analysis", ""),
            "count": len(products),
            "products": products
        })

    except Exception as e:
        print(f"❌ Error /api/search_products: {e}")
        return jsonify({"error": f"Lỗi máy chủ: {str(e)}"}), 500


def generate_ai_response(
    user_message: str,
    chat_history: List[Dict],
    user_profile: Dict,
    file_path: str | None = None,
    mime_type: str | None = None,
) -> Dict[str, Any]:
    """Generate AI response for chat with improved natural conversation and ALWAYS show products"""
    try:
        # Extract user info with safe defaults
        customer_name = user_profile.get('name', 'bạn') if user_profile else 'bạn'
        height = user_profile.get('height', 'chưa rõ') if user_profile else 'chưa rõ'
        weight = user_profile.get('weight', 'chưa rõ') if user_profile else 'chưa rõ'
        favorite_colors = user_profile.get('favorite_colors', []) if user_profile else []
        
        if isinstance(favorite_colors, list):
            favorite_colors_str = ', '.join(favorite_colors) if favorite_colors else 'chưa rõ'
        else:
            favorite_colors_str = str(favorite_colors) if favorite_colors else 'chưa rõ'
        
        # Build chat history context (last 5 messages)
        chat_context = []
        if chat_history and isinstance(chat_history, list):
            for msg in chat_history[-5:]:
                try:
                    if not isinstance(msg, dict):
                        continue
                    
                    msg_type = msg.get('type', '') or msg.get('role', '') or msg.get('sender', '')
                    message = msg.get('message', '') or msg.get('content', '') or msg.get('text', '')
                    
                    if msg_type.lower() in ('user', 'human', 'customer'):
                        role = "Khách hàng"
                    elif msg_type.lower() in ('ai', 'assistant', 'bot', 'mina'):
                        role = "Mina"
                    else:
                        role = "Khách hàng"
                    
                    if message and isinstance(message, str) and message.strip():
                        chat_context.append(f"{role}: {message.strip()}")
                except Exception as msg_error:
                    print(f"⚠️ [Chat History] Error processing message: {msg_error}")
                    continue
        
        chat_history_str = '\n'.join(chat_context) if chat_context else "Chưa có lịch sử"
        
        # Extract keywords and intent first
        extracted = extract_keywords_with_gemini(user_message, file_path, mime_type)
        keywords = extracted.get("keywords", [])[:6]
        user_intent = extracted.get("user_intent", "general_question")
        should_search = extracted.get("should_search_products", False)
        
        print(f"🤖 [AI] Intent: {user_intent}, Should search: {should_search}")
        
        # Search for products if needed
        suggested_products = []
        search_fallback_level = 0
        
        if should_search and keywords:
            print(f"🔍 [SEARCH] Keywords: {keywords}")
            
            # Get price from AI analysis
            min_price = extracted.get("price_min")
            max_price = extracted.get("price_max")
            det_type = (extracted.get("type") or "").strip().lower()
            det_colors = [c.strip().lower() for c in (extracted.get("colors") or []) if isinstance(c, str) and c.strip()]
            
            # Build query
            columns = ["ten_san_pham", "mo_ta_san_pham"]
            or_clause = build_or_clause_for_keywords(columns, keywords)
            
            q = supabase.table("products").select(
                "ma_san_pham,ten_san_pham,mo_ta_san_pham,gia_ban,muc_gia_goc,product_images(duong_dan_anh)"
            )
            
            if min_price is not None:
                q = q.gte("gia_ban", min_price)
                print(f"💰 [FILTER] Min price: {min_price}")
            if max_price is not None:
                q = q.lte("gia_ban", max_price)
                print(f"💰 [FILTER] Max price: {max_price}")
            if det_type:
                type_clause = build_or_clause_for_keywords(["ten_san_pham", "mo_ta_san_pham"], [det_type])
                if type_clause:
                    q = q.or_(type_clause)
            if det_colors:
                color_clause = build_or_clause_for_keywords(["ten_san_pham", "mo_ta_san_pham"], det_colors[:2])
                if color_clause:
                    q = q.or_(color_clause)
            if or_clause:
                q = q.or_(or_clause)
            
            resp = q.limit(8).execute()
            rows = resp.data or []
            
            # FALLBACK LEVEL 1: Single tokens
            if not rows and keywords:
                search_fallback_level = 1
                print(f"🔄 [FALLBACK 1] Trying with single tokens")
                single_tokens = [t for t in keywords if len(t.split()) == 1 and len(t) > 2]
                if single_tokens:
                    or_clause_2 = build_or_clause_for_keywords(["ten_san_pham", "mo_ta_san_pham"], single_tokens)
                    q2 = supabase.table("products").select(
                        "ma_san_pham,ten_san_pham,mo_ta_san_pham,gia_ban,muc_gia_goc,product_images(duong_dan_anh)"
                    )
                    if min_price is not None:
                        q2 = q2.gte("gia_ban", min_price)
                    if max_price is not None:
                        q2 = q2.lte("gia_ban", max_price)
                    if or_clause_2:
                        q2 = q2.or_(or_clause_2)
                    resp2 = q2.limit(8).execute()
                    rows = resp2.data or []
            
            # FALLBACK LEVEL 2: Remove price constraints
            if not rows and (min_price is not None or max_price is not None):
                search_fallback_level = 2
                print(f"🔄 [FALLBACK 2] Removing price constraints")
                q3 = supabase.table("products").select(
                    "ma_san_pham,ten_san_pham,mo_ta_san_pham,gia_ban,muc_gia_goc,product_images(duong_dan_anh)"
                )
                if or_clause:
                    q3 = q3.or_(or_clause)
                if det_type:
                    type_clause = build_or_clause_for_keywords(["ten_san_pham", "mo_ta_san_pham"], [det_type])
                    if type_clause:
                        q3 = q3.or_(type_clause)
                resp3 = q3.limit(8).execute()
                rows = resp3.data or []
            
            # FALLBACK LEVEL 3: Products by type only
            if not rows and det_type:
                search_fallback_level = 3
                print(f"🔄 [FALLBACK 3] Getting products by type: {det_type}")
                type_clause = build_or_clause_for_keywords(["ten_san_pham", "mo_ta_san_pham"], [det_type])
                if type_clause:
                    q4 = supabase.table("products").select(
                        "ma_san_pham,ten_san_pham,mo_ta_san_pham,gia_ban,muc_gia_goc,product_images(duong_dan_anh)"
                    ).or_(type_clause).limit(8)
                    resp4 = q4.execute()
                    rows = resp4.data or []
            
            # FALLBACK LEVEL 4: ANY products (last resort)
            if not rows:
                search_fallback_level = 4
                print(f"🔄 [FALLBACK 4] Getting random popular products")
                q5 = supabase.table("products").select(
                    "ma_san_pham,ten_san_pham,mo_ta_san_pham,gia_ban,muc_gia_goc,product_images(duong_dan_anh)"
                ).limit(8)
                resp5 = q5.execute()
                rows = resp5.data or []
            
            # Sort by relevance
            def rank_row(r: Dict[str, Any]) -> tuple:
                name = (r.get("ten_san_pham") or "").lower()
                desc = (r.get("mo_ta_san_pham") or "").lower()
                txt = name + " " + desc
                score = score_product(r, keywords)
                type_bonus = 3 if det_type and det_type in txt else 0
                color_bonus = sum(1 for c in det_colors if c in txt) if det_colors else 0
                price_penalty = 0
                price = float(r.get("gia_ban") or 0)
                if min_price is not None or max_price is not None:
                    center = ((min_price or price) + (max_price or price)) / 2.0
                    price_penalty = abs(price - center) / max(center, 1.0)
                return (score + type_bonus + color_bonus, -price_penalty)

            rows_sorted = sorted(rows, key=lambda r: rank_row(r), reverse=True)
            suggested_products = [map_product_row(r) for r in rows_sorted[:6]]
            
            print(f"🔍 [SEARCH] Found {len(suggested_products)} products (fallback level: {search_fallback_level})")
        
        # Format the chat prompt with context
        formatted_prompt = CHAT_PROMPT.format(
            customer_name=customer_name,
            height=height,
            weight=weight,
            favorite_colors=favorite_colors_str,
            chat_history=chat_history_str,
            current_intent=user_intent,
            products_found=len(suggested_products),
            user_message=user_message
        )
        
        # ADD PRODUCT DETAILS TO PROMPT (so AI knows exact prices)
        if suggested_products:
            formatted_prompt += "\n\n📦 **SẢN PHẨM TÌM ĐƯỢC** (PHẢI dùng thông tin này, KHÔNG được bịa):\n"
            for i, prod in enumerate(suggested_products[:6], 1):
                price = prod.get('price', 0)
                name = prod.get('name', 'Không rõ tên')
                formatted_prompt += f"{i}. {name} - Giá: {int(price):,}đ\n"
            formatted_prompt += "\n⚠️ QUAN TRỌNG: Khi đề cập giá, PHẢI dùng CHÍNH XÁC giá trên, KHÔNG được làm tròn hoặc bịa số khác!\n"
        
        # ADD FALLBACK INFO TO PROMPT
        if search_fallback_level > 0:
            fallback_notes = {
                1: "Sản phẩm tìm được bằng cách mở rộng từ khóa",
                2: "Sản phẩm tìm được sau khi bỏ giới hạn giá",
                3: "Sản phẩm tìm được theo loại tương tự",
                4: "Sản phẩm gợi ý phổ biến cho bạn"
            }
            formatted_prompt += f"\n\n⚠️ LƯU Ý: {fallback_notes[search_fallback_level]}. Hãy GIẢI THÍCH rõ ràng cho khách hàng tại sao không tìm thấy sản phẩm chính xác, và ĐỀ XUẤT các sản phẩm thay thế một cách TỰ NHIÊN, TÍCH CỰC."
        
        # Prepare content for Gemini
        if file_path:
            uploaded_file = genai.upload_file(file_path, mime_type=mime_type)
            parts = [uploaded_file, formatted_prompt]
        else:
            parts = [formatted_prompt]
        
        # Generate AI response
        response = model.generate_content(parts)
        ai_message = (response.text or "").strip()
        
        # Ensure natural response
        if not ai_message:
            if user_intent == "greeting":
                ai_message = f"Chào {customer_name}! Mình là Mina đây 😊 Hôm nay mình có thể giúp gì cho bạn?"
            elif user_intent == "product_search" and suggested_products:
                ai_message = f"Mình tìm được {len(suggested_products)} sản phẩm phù hợp cho bạn! Xem thử nhé 😍"
            elif user_intent == "product_search" and not suggested_products:
                ai_message = "Úi, mình chưa tìm thấy sản phẩm phù hợp. Bạn có thể mô tả rõ hơn được không?"
            else:
                ai_message = "Mình đang sẵn sàng tư vấn cho bạn! Bạn muốn hỏi gì nào? 😊"
        
        return {
            "ai_message": ai_message,
            "suggested_products": suggested_products,
            "keywords": keywords,
            "user_intent": user_intent,
            "notes": extracted.get("conversation_context", "")
        }
        
    except Exception as e:
        print(f"[generate_ai_response] error: {e}")
        import traceback
        traceback.print_exc()
        return {
            "ai_message": "Ôi, mình gặp chút vấn đề kỹ thuật. Bạn thử lại sau nhé! 😅",
            "suggested_products": [],
            "keywords": [],
            "user_intent": "error",
            "notes": "Lỗi hệ thống"
        }


@app.route("/api/chat", methods=["POST"])
def chat():
    try:
        data = request.get_json(silent=True) or {}
        user_message = data.get("message", "").strip()
        chat_history = data.get("chat_history", [])
        user_profile = data.get("user_profile", {})
        
        if not user_message:
            return jsonify({"error": "Tin nhắn không được để trống"}), 400
        
        print(f"💬 [Chat] User: {user_message}")
        print(f"💬 [Chat] History: {len(chat_history)} messages")
        
        # Generate AI response
        result = generate_ai_response(
            user_message, chat_history, user_profile, None, None
        )
        
        return jsonify(result)
        
    except Exception as e:
        print(f"❌ Error /api/chat: {e}")
        return jsonify({"error": f"Lỗi máy chủ: {str(e)}"}), 500


@app.route("/api/chat_with_image", methods=["POST"])
def chat_with_image():
    try:
        user_message = request.form.get("message", "").strip()
        chat_history_str = request.form.get("chat_history", "[]")
        user_profile_str = request.form.get("user_profile", "{}")
        file_path = None
        mime_type = None
        
        # Parse JSON strings
        try:
            chat_history = json.loads(chat_history_str) if chat_history_str else []
            user_profile = json.loads(user_profile_str) if user_profile_str else {}
        except json.JSONDecodeError:
            chat_history = []
            user_profile = {}
        
        if not user_message:
            return jsonify({"error": "Tin nhắn không được để trống"}), 400
        
        # Handle image upload
        file = request.files.get("image")
        if file and file.filename:
            if not is_allowed_image(file.filename):
                return jsonify({"error": "Chỉ chấp nhận ảnh .jpg, .jpeg, .png, .webp"}), 400
            
            file.seek(0, os.SEEK_END)
            size = file.tell()
            file.seek(0)
            if size > MAX_FILE_SIZE:
                return jsonify({"error": "File quá lớn (tối đa 20MB)"}), 400
            
            filename = secure_filename(file.filename)
            mime_type = get_mime_type(filename)
            suffix = "." + filename.rsplit(".", 1)[1].lower()
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                file.save(tmp.name)
                file_path = tmp.name
        
        print(f"📸 [Chat Image] User: {user_message}, Has image: {file_path is not None}")
        
        # Generate AI response with image
        result = generate_ai_response(user_message, chat_history, user_profile, file_path, mime_type)
        
        # Clean up temp file
        if file_path and os.path.exists(file_path):
            try:
                os.unlink(file_path)
            except Exception:
                pass
        
        return jsonify(result)
        
    except Exception as e:
        print(f"❌ Error /api/chat_with_image: {e}")
        return jsonify({"error": f"Lỗi máy chủ: {str(e)}"}), 500


# -----------------------------
# Size recommendation helpers
# -----------------------------

def _parse_number(s: Any) -> float | None:
    try:
        if s is None:
            return None
        if isinstance(s, (int, float)):
            return float(s)
        s = str(s).strip().lower()
        if not s:
            return None
        s = s.replace('cm', '').replace('kg', '').replace('m', ' ').replace(',', '.')
        s = ''.join(ch for ch in s if ch.isdigit() or ch == '.' or ch == ' ')
        s = s.strip()
        if not s:
            return None
        return float(s)
    except Exception:
        return None


def parse_height_cm(height: Any) -> float | None:
    h = _parse_number(height)
    if h is None:
        return None
    if 1.3 <= h <= 2.3:
        return h * 100.0
    if 130 <= h <= 230:
        return float(h)
    return None


def parse_weight_kg(weight: Any) -> float | None:
    w = _parse_number(weight)
    if w is None:
        return None
    if 30 <= w <= 200:
        return float(w)
    return None


def recommend_size(
    *,
    height_cm: float | None,
    weight_kg: float | None,
    bust_cm: float | None = None,
    waist_cm: float | None = None,
    hip_cm: float | None = None,
    category: str | None = None,
    gender: str | None = None,
) -> dict:
    """Heuristic size recommendation"""
    size = 'M'
    reasons: list[str] = []

    top_chart = [
        {'size': 'S', 'bust_max': 84, 'waist_max': 66},
        {'size': 'M', 'bust_max': 88, 'waist_max': 70},
        {'size': 'L', 'bust_max': 92, 'waist_max': 74},
        {'size': 'XL', 'bust_max': 96, 'waist_max': 78},
    ]
    bottom_chart = [
        {'size': 'S', 'waist_max': 66, 'hip_max': 90},
        {'size': 'M', 'waist_max': 70, 'hip_max': 94},
        {'size': 'L', 'waist_max': 74, 'hip_max': 98},
        {'size': 'XL', 'waist_max': 78, 'hip_max': 102},
    ]

    if height_cm and weight_kg:
        h_m = height_cm / 100.0
        bmi = weight_kg / (h_m * h_m)
        reasons.append(f"BMI≈{bmi:.1f}")
        if bmi < 18.5:
            size = 'S'
        elif bmi < 23:
            size = 'M'
        elif bmi < 27.5:
            size = 'L'
        else:
            size = 'XL'

    cat = (category or '').lower()
    if cat in ('top', 'dress') and (bust_cm or waist_cm):
        for row in top_chart:
            ok_bust = (bust_cm is None) or (bust_cm <= row['bust_max'])
            ok_waist = (waist_cm is None) or (waist_cm <= row['waist_max'])
            if ok_bust and ok_waist:
                size = row['size']
                reasons.append(f"ngực≤{row['bust_max']}cm, eo≤{row['waist_max']}cm")
                break
    if cat in ('bottom',) and (waist_cm or hip_cm):
        for row in bottom_chart:
            ok_waist = (waist_cm is None) or (waist_cm <= row['waist_max'])
            ok_hip = (hip_cm is None) or (hip_cm <= row['hip_max'])
            if ok_waist and ok_hip:
                size = row['size']
                reasons.append(f"eo≤{row['waist_max']}cm, mông≤{row['hip_max']}cm")
                break

    if height_cm:
        if height_cm < 155 and size in ('M', 'L', 'XL'):
            reasons.append('thấp, giảm 1 size')
            size = 'S' if size == 'M' else ('M' if size == 'L' else 'L')
        if height_cm > 170 and size in ('S', 'M'):
            reasons.append('cao, tăng 1 size')
            size = 'M' if size == 'S' else 'L'

    return {
        'size': size,
        'notes': ', '.join(reasons) if reasons else 'Dựa trên số đo cung cấp',
        'inputs': {
            'height_cm': height_cm,
            'weight_kg': weight_kg,
            'bust_cm': bust_cm,
            'waist_cm': waist_cm,
            'hip_cm': hip_cm,
            'category': category,
            'gender': gender,
        }
    }


@app.route("/api/recommend_size", methods=["POST"])
def recommend_size_api():
    try:
        data = request.get_json(silent=True) or {}
        height_raw = data.get('height')
        weight_raw = data.get('weight')
        bust = _parse_number(data.get('bust'))
        waist = _parse_number(data.get('waist'))
        hip = _parse_number(data.get('hip'))
        category = data.get('category')
        gender = data.get('gender')
        use_gemini = bool(data.get('use_gemini'))

        height = parse_height_cm(height_raw)
        weight = parse_weight_kg(weight_raw)

        if use_gemini:
            try:
                prompt = (
                    "Bạn là stylist. Hãy gợi ý size cho phụ nữ (S/M/L/XL) và lý do dựa trên số đo sau\n"
                    f"Chiều cao: {height_raw}, Cân nặng: {weight_raw}, Ngực: {bust}cm, Eo: {waist}cm, Mông: {hip}cm\n"
                    f"Danh mục: {category or 'không rõ'}, Giới tính: {gender or 'không rõ'}\n"
                    "Trả về JSON duy nhất: {\"size\":\"S|M|L|XL\", \"notes\":\"lý do ngắn\"}"
                )
                resp = model.generate_content([prompt])
                text = (resp.text or '').strip()
                try:
                    rec = json.loads(text)
                    if isinstance(rec, dict) and rec.get('size'):
                        size = str(rec.get('size')).upper()
                        if size not in ('S','M','L','XL'):
                            size = 'M'
                        return jsonify({
                            'size': size,
                            'notes': rec.get('notes') or 'Theo Gemini',
                            'source': 'gemini'
                        })
                except Exception:
                    pass
            except Exception as e:
                print(f"[recommend_size_api] gemini error: {e}")

        result = recommend_size(
            height_cm=height,
            weight_kg=weight,
            bust_cm=bust,
            waist_cm=waist,
            hip_cm=hip,
            category=category,
            gender=gender,
        )
        result['source'] = 'heuristic'
        return jsonify(result)
    except Exception as e:
        print(f"❌ Error /api/recommend_size: {e}")
        return jsonify({'error': f'Lỗi máy chủ: {str(e)}'}), 500


@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({"ok": True})

if __name__ == "__main__":
    # set GEMINI_API_KEY=... && set SUPABASE_URL=... && set SUPABASE_ANON_KEY=... && python app_gemini_product_search.py
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port, debug=False)







