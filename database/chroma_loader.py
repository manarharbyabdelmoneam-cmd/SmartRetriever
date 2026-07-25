"""
🗄️ تحميل وإدارة قاعدة بيانات Chroma

يقوم بإدارة الاتصال بقاعدة بيانات Chroma والبحث فيها
"""

import chromadb
from chromadb.utils import embedding_functions
from typing import List, Dict, Any, Optional, Union
from pathlib import Path
import uuid
import numpy as np

from core.config import settings
from utils.logger import logger


# ============================================================
# ✅ دالة تنظيف البيانات الوصفية
# ============================================================

def clean_metadata_for_chroma(metadata: Dict[str, Any]) -> Dict[str, Any]:
    """
    تنظيف البيانات الوصفية لتكون متوافقة مع Chroma
    
    Chroma يقبل فقط:
    - str, int, float, bool
    - لا يقبل: None, list, dict, set, tuple
    
    Args:
        metadata: البيانات الوصفية
        
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
        # 5. معالجة numpy arrays
        # ============================================================
        if isinstance(value, np.ndarray):
            try:
                # تحويل إلى قائمة ثم إلى نص
                arr_list = value.tolist()
                if arr_list:
                    str_values = [str(v) for v in arr_list if v is not None]
                    cleaned[key] = ", ".join(str_values) if str_values else "empty_array"
                else:
                    cleaned[key] = "empty_array"
            except:
                cleaned[key] = str(value)
            continue
        
        # ============================================================
        # 6. أي نوع آخر - تحويل إلى نص
        # ============================================================
        try:
            cleaned[key] = str(value)
        except:
            cleaned[key] = "unknown_type"
    
    return cleaned


class ChromaLoader:
    """
    إدارة قاعدة بيانات Chroma
    
    يدعم:
    - تخزين دائم (Persistent) على القرص
    - البحث الدلالي باستخدام المتجهات
    - تصفية البيانات الوصفية
    - إضافة وحذف المستندات
    - استرجاع جميع المستندات
    """

    def __init__(
        self,
        collection_name: str = "documents",
        embedding_model: Optional[str] = None
    ):
        """
        تهيئة محمل Chroma
        
        Args:
            collection_name: اسم المجموعة
            embedding_model: نموذج التضمين (افتراضي من الإعدادات)
        """
        self.collection_name = collection_name
        self.embedding_model = embedding_model or settings.EMBEDDING_MODEL
        
        # مسار التخزين الدائم
        self.persist_path = settings.CHROMA_PATH
        
        # إنشاء المجلد إذا لم يكن موجوداً
        self.persist_path.mkdir(parents=True, exist_ok=True)
        
        # تهيئة العميل مع تخزين دائم
        self.client = chromadb.PersistentClient(
            path=str(self.persist_path)
        )
        
        # تهيئة دالة التضمين
        self.embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=self.embedding_model
        )
        
        # الحصول على المجموعة (أو إنشاؤها)
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            embedding_function=self.embedding_fn,
            metadata={"hnsw:space": "cosine"}
        )
        
        self.is_loaded = True
        self.stats = {
            "total_documents": self.collection.count(),
            "total_searches": 0,
            "avg_search_time": 0,
            "last_search_time": 0
        }
        
        logger.info(f"🗄️ ChromaLoader initialized with collection: {collection_name}")
        logger.info(f"📊 Documents in collection: {self.collection.count()}")

    # ============================================================
    # طرق البحث
    # ============================================================
    
    async def search(
        self,
        query_vector: List[float],
        top_k: int = 10,
        include_metadata: bool = True,
        filter_category: Optional[str] = None,
        filter_supplier: Optional[str] = None,
        min_score: float = 0.0
    ) -> List[Dict[str, Any]]:
        """
        البحث في قاعدة البيانات باستخدام المتجهات
        
        Args:
            query_vector: متجه السؤال
            top_k: عدد النتائج المطلوبة
            include_metadata: تضمين البيانات الوصفية
            filter_category: تصفية حسب التصنيف
            filter_supplier: تصفية حسب المورد
            min_score: الحد الأدنى لدرجة المطابقة
            
        Returns:
            قائمة النتائج
        """
        import time
        start_time = time.time()
        
        # ✅ التأكد من صيغة المتجه
        if isinstance(query_vector, np.ndarray):
            query_vector = query_vector.tolist()
        elif not isinstance(query_vector, list):
            query_vector = list(query_vector)
        
        # التأكد من أن المتجه هو قائمة مسطحة
        if query_vector and isinstance(query_vector[0], (list, np.ndarray)):
            query_vector = query_vector[0]
        
        # بناء شرط التصفية
        where_filter = {}
        if filter_category:
            where_filter["category"] = filter_category
        if filter_supplier:
            where_filter["supplier"] = filter_supplier
        
        try:
            # ✅ التحقق من وجود مستندات
            if self.collection.count() == 0:
                logger.warning("⚠️ Collection is empty")
                return []
            
            # البحث في Chroma
            results = self.collection.query(
                query_embeddings=[query_vector],
                n_results=min(top_k, self.collection.count()),
                where=where_filter if where_filter else None,
                include=["documents", "metadatas", "distances"]
            )
            
            # تنسيق النتائج
            formatted_results = []
            
            if results and results.get("ids") and len(results["ids"]) > 0:
                ids = results["ids"][0]
                documents = results.get("documents", [[]])[0]
                metadatas = results.get("metadatas", [[]])[0]
                distances = results.get("distances", [[]])[0]
                
                for i, doc_id in enumerate(ids):
                    # حساب درجة التشابه من المسافة
                    distance = distances[i] if i < len(distances) else 1.0
                    similarity = 1 / (1 + distance)
                    
                    # تصفية حسب الحد الأدنى للدرجة
                    if similarity < min_score:
                        continue
                    
                    result = {
                        "id": doc_id,
                        "text": documents[i] if i < len(documents) else "",
                        "metadata": metadatas[i] if i < len(metadatas) else {},
                        "relevance_score": similarity,
                        "distance": distance
                    }
                    formatted_results.append(result)
            
            # تحديث الإحصائيات
            elapsed = time.time() - start_time
            self._update_stats(len(formatted_results), elapsed)
            
            logger.info(f"🔍 Found {len(formatted_results)} results in {elapsed:.3f}s")
            
            return formatted_results
            
        except Exception as e:
            logger.error(f"❌ Error searching Chroma: {str(e)}")
            return []

    # ============================================================
    # طرق إدارة المستندات
    # ============================================================
    
    def add_document(
        self,
        doc_id: str,
        text: str,
        embedding: List[float],
        metadata: Dict[str, Any]
    ) -> bool:
        """
        إضافة مستند إلى قاعدة البيانات
        
        Args:
            doc_id: معرف المستند
            text: نص المستند
            embedding: متجه المستند
            metadata: البيانات الوصفية
            
        Returns:
            نجاح العملية
        """
        try:
            # ✅ تنظيف البيانات الوصفية
            clean_meta = clean_metadata_for_chroma(metadata)
            
            # ✅ التأكد من صيغة المتجه
            if isinstance(embedding, np.ndarray):
                embedding = embedding.tolist()
            elif not isinstance(embedding, list):
                embedding = list(embedding)
            
            self.collection.add(
                ids=[doc_id],
                embeddings=[embedding],
                documents=[text],
                metadatas=[clean_meta]
            )
            
            self.stats["total_documents"] = self.collection.count()
            logger.info(f"✅ Document added: {doc_id}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error adding document: {str(e)}")
            return False

    def add_documents(
        self,
        documents: List[Dict[str, Any]]
    ) -> int:
        """
        إضافة مستندات متعددة إلى قاعدة البيانات
        
        Args:
            documents: قائمة المستندات (كل مستند يحتوي على id, text, embedding, metadata)
            
        Returns:
            عدد المستندات المضافة
        """
        if not documents:
            return 0
        
        try:
            ids = []
            texts = []
            embeddings = []
            metadatas = []
            
            for doc in documents:
                # ✅ تنظيف البيانات الوصفية
                clean_meta = clean_metadata_for_chroma(doc.get("metadata", {}))
                
                # ✅ التأكد من صيغة المتجه
                embedding = doc.get("embedding", [])
                if isinstance(embedding, np.ndarray):
                    embedding = embedding.tolist()
                elif not isinstance(embedding, list):
                    embedding = list(embedding)
                
                ids.append(doc.get("id", str(uuid.uuid4())))
                texts.append(doc.get("text", ""))
                embeddings.append(embedding)
                metadatas.append(clean_meta)
            
            # ✅ التأكد من أن جميع القوائم بنفس الطول
            if not (len(ids) == len(texts) == len(embeddings) == len(metadatas)):
                logger.error("❌ Mismatched lengths in documents")
                return 0
            
            self.collection.add(
                ids=ids,
                documents=texts,
                embeddings=embeddings,
                metadatas=metadatas
            )
            
            self.stats["total_documents"] = self.collection.count()
            logger.info(f"✅ {len(documents)} documents added")
            return len(documents)
            
        except Exception as e:
            logger.error(f"❌ Error adding documents: {str(e)}")
            return 0

    def delete_document(self, doc_id: str) -> bool:
        """
        حذف مستند من قاعدة البيانات
        
        Args:
            doc_id: معرف المستند
            
        Returns:
            نجاح العملية
        """
        try:
            self.collection.delete(ids=[doc_id])
            self.stats["total_documents"] = self.collection.count()
            logger.info(f"🗑️ Document deleted: {doc_id}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error deleting document: {str(e)}")
            return False

    def update_document(
        self,
        doc_id: str,
        text: str,
        embedding: List[float],
        metadata: Dict[str, Any]
    ) -> bool:
        """
        تحديث مستند في قاعدة البيانات
        
        Args:
            doc_id: معرف المستند
            text: نص المستند الجديد
            embedding: متجه المستند الجديد
            metadata: البيانات الوصفية الجديدة
            
        Returns:
            نجاح العملية
        """
        try:
            # حذف المستند القديم
            self.collection.delete(ids=[doc_id])
            # إضافة المستند الجديد
            return self.add_document(doc_id, text, embedding, metadata)
            
        except Exception as e:
            logger.error(f"❌ Error updating document: {str(e)}")
            return False

    def clear(self) -> bool:
        """
        حذف جميع المستندات من المجموعة
        
        Returns:
            نجاح العملية
        """
        try:
            # حذف المجموعة وإعادة إنشاؤها
            self.client.delete_collection(self.collection_name)
            self.collection = self.client.create_collection(
                name=self.collection_name,
                embedding_function=self.embedding_fn,
                metadata={"hnsw:space": "cosine"}
            )
            self.stats["total_documents"] = 0
            logger.info("🗑️ Collection cleared")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error clearing collection: {str(e)}")
            return False

    # ============================================================
    # طرق الاستعلام
    # ============================================================
    
    def get_all_documents(self) -> List[Dict[str, Any]]:
        """
        الحصول على جميع المستندات في المجموعة
        
        Returns:
            قائمة المستندات
        """
        try:
            results = self.collection.get(
                include=["documents", "metadatas"]
            )
            
            documents = []
            if results and results.get("ids"):
                for i, doc_id in enumerate(results["ids"]):
                    documents.append({
                        "id": doc_id,
                        "text": results["documents"][i] if i < len(results["documents"]) else "",
                        "metadata": results["metadatas"][i] if i < len(results["metadatas"]) else {}
                    })
            
            return documents
            
        except Exception as e:
            logger.error(f"❌ Error getting all documents: {str(e)}")
            return []

    def get_document(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """
        الحصول على مستند محدد
        
        Args:
            doc_id: معرف المستند
            
        Returns:
            المستند أو None
        """
        try:
            results = self.collection.get(
                ids=[doc_id],
                include=["documents", "metadatas"]
            )
            
            if results and results.get("ids") and len(results["ids"]) > 0:
                return {
                    "id": results["ids"][0],
                    "text": results["documents"][0] if results.get("documents") else "",
                    "metadata": results["metadatas"][0] if results.get("metadatas") else {}
                }
            return None
            
        except Exception as e:
            logger.error(f"❌ Error getting document: {str(e)}")
            return None

    def get_index_size(self) -> int:
        """
        الحصول على عدد المستندات في المجموعة
        
        Returns:
            عدد المستندات
        """
        try:
            return self.collection.count()
        except Exception as e:
            logger.error(f"❌ Error getting index size: {str(e)}")
            return 0

    def get_index_info(self) -> Dict[str, Any]:
        """
        الحصول على معلومات المجموعة
        
        Returns:
            معلومات المجموعة
        """
        return {
            "status": "loaded" if self.is_loaded else "not_loaded",
            "collection_name": self.collection_name,
            "total_vectors": self.collection.count(),
            "persist_path": str(self.persist_path)
        }

    # ============================================================
    # طرق إضافية
    # ============================================================
    
    def save(self) -> bool:
        """
        حفظ البيانات (Chroma يحفظ تلقائياً، لكن هذه الدالة للتأكد)
        
        Returns:
            نجاح العملية
        """
        try:
            logger.info("✅ Chroma data saved (automatic)")
            return True
        except Exception as e:
            logger.error(f"❌ Error saving Chroma: {str(e)}")
            return False

    def _update_stats(self, count: int, elapsed: float) -> None:
        """
        تحديث الإحصائيات
        """
        self.stats["total_searches"] += 1
        self.stats["last_search_time"] = elapsed
        
        total = self.stats["total_searches"]
        if total > 0:
            self.stats["avg_search_time"] = (
                (self.stats["avg_search_time"] * (total - 1) + elapsed) / total
            )

    def get_stats(self) -> Dict[str, Any]:
        """
        الحصول على إحصائيات المحمل
        
        Returns:
            إحصائيات المحمل
        """
        return {
            **self.stats,
            "collection_name": self.collection_name,
            "persist_path": str(self.persist_path),
            "is_loaded": self.is_loaded
        }

    def reset_stats(self) -> None:
        """
        إعادة تعيين الإحصائيات
        """
        self.stats = {
            "total_documents": self.collection.count(),
            "total_searches": 0,
            "avg_search_time": 0,
            "last_search_time": 0
        }
        logger.info("🔄 ChromaLoader stats reset")
