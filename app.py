# pip install flask flask-cors google-generativeai supabase python-dotenv werkzeug

import os
import sys
import json
import tempfile
import mimetypes
from typing import List, Dict, Any

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


EXTRACTION_PROMPT = """Bạn là trợ lý tìm kiếm sản phẩm thời trang. Hãy phân tích ảnh (nếu có) và/hoặc mô tả văn bản để TRẢ VỀ JSON CHUẨN, KHÔNG kèm văn bản khác, theo thứ tự ưu tiên: loại -> màu sắc -> đặc tính.

YÊU CẦU JSON (tiếng Việt, không viết hoa toàn bộ):
{
  "type": "váy | áo sơ mi | quần jean | chân váy | áo thun | ...",
  "colors": ["đen", "trắng", "xanh navy", ...],
  "material": "cotton | jeans | lụa | len | da | ...",
  "pattern": "trơn | kẻ sọc | caro | hoa | chấm bi | ...",
  "style": ["công sở", "dạo phố", "dự tiệc", "thể thao", ...],
  "length": "ngắn | midi | dài | qua gối | ...",
  "sleeve": "sát nách | ngắn tay | dài tay | ...",
  "fit": "ôm | suông | rộng | ...",
  "gender": "nữ | nam | unisex | không rõ",
  "additional_keywords": ["cổ tròn", "cổ bẻ", "cổ V", "xếp ly", ...],
  "keywords": ["chuỗi tìm kiếm tổng hợp tối đa 8 mục"],
  "notes": "mô tả ngắn gọn (tùy chọn)"
}

QUY TẮC:
- Ưu tiên nhận diện loại (type) trước, sau đó là màu sắc (colors), rồi đến material/pattern/style/length/sleeve/fit.
- Không ghi thương hiệu trừ khi quá rõ.
- keywords gồm các cụm ngắn gọn ghép từ các trường trên (ví dụ: "váy đen chữ A", "áo sơ mi trắng dài tay"), tối đa 8.
- Nếu thông tin mơ hồ, điền "không rõ" hoặc mảng rỗng.

Hãy chỉ trả về JSON hợp lệ duy nhất.
"""

CHAT_PROMPT = """Bạn là AI tư vấn thời trang chuyên nghiệp của Zamy Shop. Nhiệm vụ của bạn là:

1. TƯ VẤN THỜI TRANG: Giúp khách hàng chọn trang phục phù hợp dựa trên:
   - Chiều cao, cân nặng, vóc dáng
   - Màu sắc yêu thích
   - Dịp sử dụng (công sở, dạo phố, dự tiệc, thể thao...)
   - Phong cách cá nhân
   - Ngân sách

2. PHÂN TÍCH HÌNH ẢNH: Nếu có hình ảnh, phân tích để:
   - Nhận diện loại trang phục
   - Xác định màu sắc, chất liệu, phong cách
   - Gợi ý sản phẩm tương tự

3. TRẢ LỜI THÂN THIỆN: Sử dụng giọng điệu:
   - Nhiệt tình, thân thiện
   - Chuyên nghiệp nhưng gần gũi
   - Sử dụng emoji phù hợp
   - Đưa ra lời khuyên cụ thể, thực tế

4. GỢI Ý SẢN PHẨM: 
   - CHỈ gợi ý sản phẩm có thật trong cửa hàng
   - KHÔNG được "bịa" hoặc tạo ra sản phẩm không tồn tại
   - Nếu không tìm thấy sản phẩm phù hợp, hãy thừa nhận và gợi ý sản phẩm tương tự có sẵn
   - Luôn kết thúc bằng việc gợi ý sản phẩm cụ thể từ shop

THÔNG TIN KHÁCH HÀNG:
- Chiều cao: {height}
- Cân nặng: {weight}  
- Màu sắc yêu thích: {favorite_colors}
- Lịch sử chat: {chat_history}

QUAN TRỌNG: Bạn CHỈ được gợi ý sản phẩm có thật trong database của shop. Nếu không tìm thấy sản phẩm chính xác, hãy gợi ý sản phẩm tương tự gần nhất có sẵn.

Hãy trả lời ngắn gọn, hữu ích và gợi ý sản phẩm phù hợp."""


def extract_keywords_with_gemini(user_message: str, file_path: str | None, mime_type: str | None) -> Dict[str, Any]:
    try:
        if file_path:
            uploaded_file = genai.upload_file(file_path, mime_type=mime_type)
            parts = [uploaded_file, user_message or "Hãy trích xuất từ khóa từ ảnh này."]
        else:
            parts = [user_message or "Hãy trích xuất từ khóa từ mô tả này."]

        response = model.generate_content([EXTRACTION_PROMPT, *parts])
        text = (response.text or "").strip()

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
        keywords = data.get("keywords", [])
        if not isinstance(keywords, list):
            keywords = []
        
        if not keywords:
            type_ = data.get("type") or ""
            colors = data.get("colors") or []
            material = data.get("material") or ""
            pattern = data.get("pattern") or ""
            style = data.get("style") or []
            length = data.get("length") or ""
            sleeve = data.get("sleeve") or ""
            add = data.get("additional_keywords") or []

            candidates = []
            base = type_.strip()
            if base:
                if colors:
                    for c in colors[:2]:
                        candidates.append(f"{base} {c}")
                if sleeve:
                    candidates.append(f"{base} {sleeve}")
                if length:
                    candidates.append(f"{base} {length}")
                if pattern:
                    candidates.append(f"{base} {pattern}")
                if material:
                    candidates.append(f"{base} {material}")
                for s in (style if isinstance(style, list) else [style]):
                    if s:
                        candidates.append(f"{base} {s}")
            for a in add[:3]:
                if base:
                    candidates.append(f"{base} {a}")
                else:
                    candidates.append(a)

            keywords = candidates

        keywords = list(dict.fromkeys([k.strip() for k in keywords if isinstance(k, str) and k.strip()]))
        keywords = keywords[:8]
        
        out = {
            "keywords": keywords,
            "notes": data.get("notes", ""),
        }
        for k in ["type","colors","material","pattern","style","length","sleeve","fit","gender","additional_keywords"]:
            if k in data:
                out[k] = data[k]
        return out
    except Exception as e:
        print(f"[extract_keywords_with_gemini] error: {e}")
        fallback = [t.strip() for t in (user_message or "").split() if len(t.strip()) > 1][:6]
        return {"keywords": fallback, "notes": "fallback"}


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


def calculate_similarity(text1: str, text2: str) -> float:
    if not text1 or not text2:
        return 0.0
    
    text1 = text1.lower().strip()
    text2 = text2.lower().strip()
    
    if text1 == text2:
        return 1.0
    
    common_chars = 0
    max_len = max(len(text1), len(text2))
    
    for char in text1:
        if char in text2:
            common_chars += 1
    
    return common_chars / max_len if max_len > 0 else 0.0


def find_similar_products(products: List[Dict[str, Any]], keywords: List[str]) -> List[Dict[str, Any]]:
    if not products or not keywords:
        return []
    
    similar_products = []
    min_similarity = 0.3
    
    for product in products:
        product_name = product.get("ten_san_pham", "")
        if not product_name:
            continue
            
        max_similarity = 0.0
        best_match_keyword = ""
        
        for keyword in keywords:
            similarity = calculate_similarity(product_name, keyword)
            if similarity > max_similarity:
                max_similarity = similarity
                best_match_keyword = keyword
        
        if max_similarity >= min_similarity:
            product_with_score = product.copy()
            product_with_score["_similarity_score"] = max_similarity
            product_with_score["_matched_keyword"] = best_match_keyword
            similar_products.append(product_with_score)
            print(f"🔍 [SIMILARITY] '{product_name}' matches '{best_match_keyword}' with {max_similarity:.2f} similarity")
    
    similar_products.sort(key=lambda x: x.get("_similarity_score", 0), reverse=True)
    
    for product in similar_products:
        product.pop("_similarity_score", None)
        product.pop("_matched_keyword", None)
    
    return similar_products


def find_partial_matches(products: List[Dict[str, Any]], keywords: List[str]) -> List[Dict[str, Any]]:
    if not products or not keywords:
        return []
    
    partial_matches = []
    
    for product in products:
        product_name = product.get("ten_san_pham", "").lower()
        product_desc = product.get("mo_ta_san_pham", "").lower()
        
        match_score = 0
        matched_keywords = []
        
        for keyword in keywords:
            keyword_lower = keyword.lower()
            
            if keyword_lower in product_name:
                match_score += 3
                matched_keywords.append(keyword)
            elif any(word in product_name for word in keyword_lower.split()):
                match_score += 2
                matched_keywords.append(keyword)
            elif keyword_lower in product_desc:
                match_score += 1
                matched_keywords.append(keyword)
        
        if match_score > 0:
            product_with_score = product.copy()
            product_with_score["_match_score"] = match_score
            product_with_score["_matched_keywords"] = matched_keywords
            partial_matches.append(product_with_score)
            print(f"🔍 [PARTIAL] '{product.get('ten_san_pham', '')}' matches keywords: {matched_keywords} (score: {match_score})")
    
    partial_matches.sort(key=lambda x: x.get("_match_score", 0), reverse=True)
    
    for product in partial_matches:
        product.pop("_match_score", None)
        product.pop("_matched_keywords", None)
    
    return partial_matches


# SỬA: Hàm map_product_row - ma_san_pham giờ là INTEGER
def map_product_row(row: Dict[str, Any]) -> Dict[str, Any]:
    images = []
    if isinstance(row.get("product_images"), list):
        for img in row["product_images"]:
            url = img.get("duong_dan_anh")
            if isinstance(url, str):
                images.append(url)

    return {
        "id": int(row.get("ma_san_pham")) if row.get("ma_san_pham") else None,  # SỬA: Đảm bảo là INTEGER
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

        token_set = []
        for kw in keywords:
            token_set.append(kw)
            for t in kw.replace(",", " ").split():
                t = t.strip()
                if len(t) > 1 and t.lower() not in [x.lower() for x in token_set]:
                    token_set.append(t)
        keywords = token_set[:10]

        if file_path and os.path.exists(file_path):
            try:
                os.unlink(file_path)
            except Exception:
                pass

        if not keywords:
            return jsonify({"keywords": [], "products": [], "notes": extracted.get("notes", "")})

        columns = ["ten_san_pham", "mo_ta_san_pham"]
        or_clause = build_or_clause_for_keywords(columns, keywords)
        q = supabase.table("products").select(
            "ma_san_pham,ten_san_pham,mo_ta_san_pham,gia_ban,muc_gia_goc,product_images(duong_dan_anh)"
        )
        if or_clause:
            q = q.or_(or_clause)
        resp = q.limit(20).execute()
        rows = resp.data or []

        if not rows and keywords:
            single_tokens = [t for t in keywords if len(t.split()) == 1]
            if single_tokens:
                or_clause_2 = build_or_clause_for_keywords(["ten_san_pham", "mo_ta_san_pham"], single_tokens)
                q2 = supabase.table("products").select(
                    "ma_san_pham,ten_san_pham,mo_ta_san_pham,gia_ban,muc_gia_goc,product_images(duong_dan_anh)"
                )
                if or_clause_2:
                    q2 = q2.or_(or_clause_2)
                resp2 = q2.limit(20).execute()
                rows = resp2.data or []
        
        rows_sorted = sorted(rows, key=lambda r: score_product(r, keywords), reverse=True)
        products = [map_product_row(r) for r in rows_sorted]

        return jsonify({
            "keywords": keywords,
            "notes": extracted.get("notes", ""),
            "count": len(products),
            "products": products
        })

    except Exception as e:
        print(f"❌ Error /api/search_products: {e}")
        return jsonify({"error": f"Lỗi máy chủ: {str(e)}"}), 500


def generate_ai_response(user_message: str, chat_history: List[Dict], user_profile: Dict, file_path: str | None = None, mime_type: str | None = None) -> Dict[str, Any]:
    """Generate AI response for chat"""
    try:
        # Build context from user profile and chat history
        height = user_profile.get('height', 'không rõ')
        weight = user_profile.get('weight', 'không rõ')
        favorite_colors = user_profile.get('favorite_colors', [])
        if isinstance(favorite_colors, list):
            favorite_colors_str = ', '.join(favorite_colors)
        else:
            favorite_colors_str = str(favorite_colors)
        
        # Build chat history context
        chat_context = []
        for msg in chat_history[-5:]:  # Last 5 messages for context
            if msg.get('type') == 'user':
                chat_context.append(f"Khách hàng: {msg.get('message', '')}")
        
        chat_history_str = '\n'.join(chat_context) if chat_context else "Chưa có lịch sử chat"
        
        # Format the chat prompt
        formatted_prompt = CHAT_PROMPT.format(
            height=height,
            weight=weight,
            favorite_colors=favorite_colors_str,
            chat_history=chat_history_str
        )
        
        # Prepare content for Gemini
        if file_path:
            uploaded_file = genai.upload_file(file_path, mime_type=mime_type)
            parts = [uploaded_file, f"{formatted_prompt}\n\nTin nhắn khách hàng: {user_message}"]
        else:
            parts = [f"{formatted_prompt}\n\nTin nhắn khách hàng: {user_message}"]
        
        # Extract keywords for product search first
        extracted = extract_keywords_with_gemini(user_message, file_path, mime_type)
        keywords = extracted.get("keywords", [])[:6]
        
        # Generate AI response after we know the search results
        response = model.generate_content(parts)
        ai_message = (response.text or "").strip()
        
        # Search for products based on keywords
        suggested_products = []
        if keywords:
            print(f"🔍 [DEBUG] Searching with keywords: {keywords}")
            
            # First, try exact keyword search
            columns = ["ten_san_pham", "mo_ta_san_pham"]
            or_clause = build_or_clause_for_keywords(columns, keywords)
            print(f"🔍 [DEBUG] OR clause: {or_clause}")
            
            q = supabase.table("products").select(
                "ma_san_pham,ten_san_pham,mo_ta_san_pham,gia_ban,muc_gia_goc,product_images(duong_dan_anh)"
            )
            if or_clause:
                q = q.or_(or_clause)
            resp = q.limit(6).execute()
            rows = resp.data or []
            print(f"🔍 [DEBUG] Found {len(rows)} products with multi-keyword search")
            
            # If no results, try with main keywords immediately
            if not rows:
                print("🔍 [DEBUG] Trying main keywords search immediately...")
                main_keywords = []
                for keyword in keywords:
                    words = keyword.split()
                    for word in words:
                        if len(word) > 2:
                            main_keywords.append(word)
                
                if main_keywords:
                    print(f"🔍 [DEBUG] Main keywords: {main_keywords}")
                    main_clause = build_or_clause_for_keywords(["ten_san_pham", "mo_ta_san_pham"], main_keywords)
                    if main_clause:
                        q_main = supabase.table("products").select(
                            "ma_san_pham,ten_san_pham,mo_ta_san_pham,gia_ban,muc_gia_goc,product_images(duong_dan_anh)"
                        ).or_(main_clause).limit(6).execute()
                        rows = q_main.data or []
                        print(f"🔍 [DEBUG] Found {len(rows)} products with main keywords")
                        
                        # If found products with main keywords, update AI message
                        if rows:
                            ai_message += f"\n\nTôi tìm thấy 1 số sản phẩm '{', '.join(keywords)}' nhưng đã tìm thấy một số sản phẩm liên quan dựa trên từ khóa chính: {', '.join(main_keywords)} 😊"
            
            # Fallback search with single tokens
            if not rows:
                single_tokens = [t for t in keywords if len(t.split()) == 1]
                print(f"🔍 [DEBUG] Trying single tokens: {single_tokens}")
                if single_tokens:
                    or_clause_2 = build_or_clause_for_keywords(["ten_san_pham", "mo_ta_san_pham"], single_tokens)
                    q2 = supabase.table("products").select(
                        "ma_san_pham,ten_san_pham,mo_ta_san_pham,gia_ban,muc_gia_goc,product_images(duong_dan_anh)"
                    )
                    if or_clause_2:
                        q2 = q2.or_(or_clause_2)
                    resp2 = q2.limit(6).execute()
                    rows = resp2.data or []
                    print(f"🔍 [DEBUG] Found {len(rows)} products with single-token search")
            
            # If still no results, try broader search
            if not rows:
                print("🔍 [DEBUG] Trying broader search...")
                # Search for any product containing any keyword
                broader_keywords = [k for k in keywords if len(k) > 2]  # Only longer keywords
                if broader_keywords:
                    broader_clause = build_or_clause_for_keywords(["ten_san_pham", "mo_ta_san_pham"], broader_keywords)
                    q3 = supabase.table("products").select(
                        "ma_san_pham,ten_san_pham,mo_ta_san_pham,gia_ban,muc_gia_goc,product_images(duong_dan_anh)"
                    )
                    if broader_clause:
                        q3 = q3.or_(broader_clause)
                    resp3 = q3.limit(6).execute()
                    rows = resp3.data or []
                    print(f"🔍 [DEBUG] Found {len(rows)} products with broader search")
            
            # If still no results, try fuzzy/similarity search
            if not rows:
                print("🔍 [DEBUG] Trying fuzzy/similarity search...")
                # Get all products and find similar ones
                all_products_resp = supabase.table("products").select(
                    "ma_san_pham,ten_san_pham,mo_ta_san_pham,gia_ban,muc_gia_goc,product_images(duong_dan_anh)"
                ).limit(100).execute()  # Get more products for better matching
                all_products = all_products_resp.data or []
                
                if all_products:
                    # Find products with similar names
                    similar_products = find_similar_products(all_products, keywords)
                    rows = similar_products[:6]  # Limit to 6 results
                    print(f"🔍 [DEBUG] Found {len(rows)} products with similarity search")
                    
                    # If still no results, try partial keyword matching
                    if not rows:
                        print("🔍 [DEBUG] Trying partial keyword matching...")
                        partial_products = find_partial_matches(all_products, keywords)
                        rows = partial_products[:6]
                        print(f"🔍 [DEBUG] Found {len(rows)} products with partial matching")
            
            # Sort by relevance
            rows_sorted = sorted(rows, key=lambda r: score_product(r, keywords), reverse=True)
            suggested_products = [map_product_row(r) for r in rows_sorted]
            print(f"🔍 [DEBUG] Final suggested products: {len(suggested_products)}")
            
            # Log product names for debugging
            for i, product in enumerate(suggested_products):
                print(f"🔍 [DEBUG] Product {i+1}: {product.get('name', 'Unknown')}")
        
        # Update AI message based on search results
        if suggested_products:
            # If products found, update AI message to be more positive
            product_names = [p.get('name', 'Sản phẩm') for p in suggested_products[:3]]
            
            # Simple positive response when products are found
            ai_message = f"Tuyệt vời! Tôi đã tìm thấy một số sản phẩm phù hợp với yêu cầu của bạn. Dưới đây là các sản phẩm gợi ý cho bạn! 😊"
        elif keywords:
            # If no products found, suggest alternatives
            ai_message = f"Hiện tại shop chưa có sản phẩm phù hợp với yêu cầu của bạn. Bạn có thể thử tìm kiếm với từ khóa khác hoặc xem các sản phẩm có sẵn nhé! 😊"
        
        return {
            "ai_message": ai_message,
            "suggested_products": suggested_products,
            "keywords": keywords,
            "notes": extracted.get("notes", "")
        }
        
    except Exception as e:
        print(f"[generate_ai_response] error: {e}")
        return {
            "ai_message": "Xin lỗi, tôi gặp sự cố kỹ thuật. Vui lòng thử lại sau.",
            "suggested_products": [],
            "keywords": [],
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
        
        print(f"🤖 [Chat] User message: {user_message}")
        print(f"🤖 [Chat] Chat history length: {len(chat_history)}")
        print(f"🤖 [Chat] User profile: {user_profile}")
        
        # Generate AI response
        result = generate_ai_response(user_message, chat_history, user_profile)
        
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
        
        print(f"🤖 [Chat Image] User message: {user_message}")
        print(f"🤖 [Chat Image] Has image: {file_path is not None}")
        
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
        # Remove units and non-numeric except dot and comma
        s = s.replace('cm', '').replace('kg', '').replace('m', ' ').replace(',', '.')
        s = ''.join(ch for ch in s if ch.isdigit() or ch == '.' or ch == ' ')
        s = s.strip()
        if not s:
            return None
        # If looks like meters (e.g., 1.65), convert to cm later by caller
        return float(s)
    except Exception:
        return None


def parse_height_cm(height: Any) -> float | None:
    h = _parse_number(height)
    if h is None:
        return None
    # If clearly meters (e.g., 1.5 to 2.5), convert to cm
    if 1.3 <= h <= 2.3:
        return h * 100.0
    # If looks like centimeters
    if 130 <= h <= 230:
        return float(h)
    return None


def parse_weight_kg(weight: Any) -> float | None:
    w = _parse_number(weight)
    if w is None:
        return None
    # reasonable human weights
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
    category: str | None = None,  # 'top' | 'bottom' | 'dress' | None
    gender: str | None = None,
) -> dict:
    """Heuristic size recommendation. Returns dict with size, range, reasoning."""
    size = 'M'
    reasons: list[str] = []

    # Fallback charts (VN female general). Adjust as needed.
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

    # BMI baseline
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

    # Measurement overrides by category
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

    # Height-based nudge
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
        category = data.get('category')  # 'top' | 'bottom' | 'dress'
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
                        # Normalize size
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
                # fall through to heuristics

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





