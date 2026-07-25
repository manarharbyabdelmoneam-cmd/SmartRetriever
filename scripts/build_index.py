"""
🔨 بناء فهرس Chroma من knowledge_base
يشتغل تلقائياً عند بدء التطبيق لو الفهرس مش موجود
"""

import sys
import asyncio
from pathlib import Path
import numpy as np

sys.path.append(str(Path(__file__).parent.parent))

from core.config import settings
from database.docx_loader import DocxLoader
from database.embeddings import Embeddings
from database.chroma_loader import ChromaLoader
from utils.logger import logger


# ============================================================
# ✅ دالة تنظيف البيانات الوصفية لتكون متوافقة مع Chroma
# ============================================================

def clean_metadata(metadata: dict) -> dict:
    """
    تنظيف البيانات الوصفية لتكون متوافقة مع Chroma
    
    Chroma يقبل فقط:
    - str, int, float, bool
    - لا يقبل: None, list, dict, set, tuple
    
    Args:
        metadata: البيانات الوصفية غير النظيفة
        
    Returns:
        البيانات الوصفية النظيفة
    """
    cleaned = {}
    for key, value in metadata.items():
        # تخطي المفاتيح الفارغة
        if not key:
            continue
            
        # ============================================================
        # 1. معالجة القيم الفارغة (None)
        # ============================================================
        if value is None:
            cleaned[key] = "unknown"
            continue
        
        # ============================================================
        # 2. معالجة الأنواع الأساسية المدعومة
        # ============================================================
        if isinstance(value, (str, int, float, bool)):
            cleaned[key] = value
            continue
        
        # ============================================================
        # 3. معالجة القوائم والمجموعات
        # ============================================================
        if isinstance(value, (list, tuple, set)):
            # تحويل إلى نص مفصول بفواصل
            try:
                # استخراج القيم النصية فقط
                str_values = [str(v) for v in value if v is not None]
                if str_values:
                    cleaned[key] = ", ".join(str_values)
                else:
                    cleaned[key] = "empty_list"
            except:
                cleaned[key] = str(value)
            continue
        
        # ============================================================
        # 4. معالجة القواميس (dict)
        # ============================================================
        if isinstance(value, dict):
            # تحويل القاموس إلى نص key=value, key2=value2
            try:
                items = []
                for k, v in value.items():
                    if v is not None:
                        items.append(f"{k}={v}")
                if items:
                    cleaned[key] = "; ".join(items)
                else:
                    cleaned[key] = "empty_dict"
            except:
                cleaned[key] = str(value)
            continue
        
        # ============================================================
        # 5. أي نوع آخر - تحويل إلى نص
        # ============================================================
        try:
            cleaned[key] = str(value)
        except:
            cleaned[key] = "unknown_type"
    
    return cleaned


# ============================================================
# 🔨 بناء الفهرس
# ============================================================

async def build_index():
    logger.info("🔨 Starting index build with Chroma...")

    kb_path = settings.KNOWLEDGE_BASE_PATH
    if not kb_path.exists():
        logger.error(f"❌ knowledge_base not found: {kb_path}")
        return False

    # تحميل المستندات
    loader = DocxLoader()
    all_chunks = []

    for category_dir in kb_path.iterdir():
        if not category_dir.is_dir():
            continue
        for file_path in category_dir.glob("*.docx"):
            try:
                # استخدام load_file بدلاً من load
                result = loader.load_file(str(file_path))
                if result:
                    # load_file بترجع قاموس واحد
                    chunk = {
                        "text": result.get("text", ""),
                        "metadata": result.get("metadata", {})
                    }
                    chunk["metadata"]["category"] = category_dir.name
                    chunk["metadata"]["filename"] = file_path.name
                    
                    # ✅ تنظيف البيانات الوصفية قبل الإضافة
                    chunk["metadata"] = clean_metadata(chunk["metadata"])
                    
                    all_chunks.append(chunk)
                    logger.info(f"✅ Loaded: {file_path.name}")
                else:
                    logger.warning(f"⚠️ Failed to load: {file_path.name}")
            except Exception as e:
                logger.error(f"❌ Error loading {file_path.name}: {e}")

    if not all_chunks:
        logger.error("❌ No chunks loaded!")
        return False

    logger.info(f"📄 Total chunks: {len(all_chunks)}")

    # توليد المتجهات
    embeddings_model = Embeddings(
        model_name=settings.EMBEDDING_MODEL,
        device=settings.EMBEDDING_DEVICE
    )

    texts = [c.get("text", "") for c in all_chunks]
    vectors = await embeddings_model.encode(texts)

    logger.info(f"🧬 Generated {len(vectors)} embeddings")

    # ✅ بناء فهرس Chroma
    chroma_loader = ChromaLoader()
    chroma_loader.clear()  # مسح المجموعة القديمة

    # تجهيز المستندات للإضافة
    documents_to_add = []
    for i, (chunk, vector) in enumerate(zip(all_chunks, vectors)):
        doc_id = f"doc_{i}_{chunk['metadata'].get('filename', 'unknown')}"
        
        # ✅ تحويل المتجه إلى قائمة (إذا كان numpy array)
        if hasattr(vector, 'tolist'):
            vector = vector.tolist()
        elif hasattr(vector, 'numpy'):
            vector = vector.numpy().tolist()
        elif isinstance(vector, np.ndarray):
            vector = vector.tolist()
        elif not isinstance(vector, list):
            vector = list(vector)
        
        # ✅ التأكد من أن البيانات الوصفية نظيفة مرة أخرى
        clean_meta = clean_metadata(chunk.get("metadata", {}))
        
        documents_to_add.append({
            "id": doc_id,
            "text": chunk.get("text", ""),
            "embedding": vector,
            "metadata": clean_meta
        })

    # ✅ إضافة جميع المستندات دفعة واحدة
    if documents_to_add:
        added_count = chroma_loader.add_documents(documents_to_add)
        
        if added_count > 0:
            logger.info(f"✅ Index built and saved! ({added_count} documents in Chroma)")
            return True
        else:
            logger.error("❌ Failed to save index!")
            return False
    else:
        logger.error("❌ No documents to add!")
        return False


if __name__ == "__main__":
    asyncio.run(build_index())
