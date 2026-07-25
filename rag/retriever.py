"""
🔍 استرجاع المستندات (Retriever)

يقوم بالبحث في قاعدة بيانات Chroma واسترجاع المستندات الأكثر تشابهاً مع السؤال
"""

from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import pickle
import os
import json
from pathlib import Path
import hashlib
from datetime import datetime

from core.config import settings
from database.chroma_loader import ChromaLoader  # ✅ استبدال FAISSLoader بـ ChromaLoader
from database.embeddings import Embeddings
from utils.logger import logger


class Retriever:
    """
    استرجاع المستندات من قاعدة بيانات Chroma
    
    يدعم:
    - البحث الدلالي باستخدام المتجهات
    - البحث بالبيانات الوصفية
    - دمج النتائج من طرق بحث متعددة
    - تصفية النتائج حسب التصنيف أو المورد
    """
    
    def __init__(
        self,
        chroma_loader: Optional[ChromaLoader] = None,  # ✅ تغيير الاسم
        embeddings: Optional[Embeddings] = None,
        top_k: int = 10
    ):
        """
        تهيئة أداة الاسترجاع
        
        Args:
            chroma_loader: محمل قاعدة بيانات Chroma
            embeddings: أداة توليد المتجهات
            top_k: عدد النتائج الافتراضي
        """
        self.chroma_loader = chroma_loader or ChromaLoader()  # ✅ استخدام ChromaLoader
        self.embeddings = embeddings or Embeddings()
        self.top_k = top_k
        
        # إحصائيات
        self.stats = {
            "total_queries": 0,
            "avg_retrieval_time": 0,
            "total_documents_retrieved": 0,
            "last_query_time": 0
        }
        
        logger.info("🔍 Retriever initialized with Chroma")
    
    # ============================================================
    # الطريقة الرئيسية
    # ============================================================
    
    async def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        filter_category: Optional[str] = None,
        filter_supplier: Optional[str] = None,
        min_score: float = 0.0,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        استرجاع المستندات الأكثر تشابهاً مع السؤال
        
        Args:
            query: سؤال المستخدم
            top_k: عدد النتائج المطلوبة
            filter_category: تصفية حسب التصنيف
            filter_supplier: تصفية حسب المورد
            min_score: الحد الأدنى لدرجة المطابقة
            
        Returns:
            قائمة المستندات المسترجعة
        """
        import time
        start_time = time.time()
        
        # 1. توليد متجه للسؤال
        query_vector = await self.embeddings.encode(query)
        
        # 2. البحث في Chroma (مع دعم التصفية المدمج)
        results = await self.chroma_loader.search(  # ✅ استخدام Chroma
            query_vector=query_vector.tolist() if hasattr(query_vector, 'tolist') else query_vector,
            top_k=top_k or self.top_k,
            filter_category=filter_category,  # ✅ تصفية مدمجة
            filter_supplier=filter_supplier,  # ✅ تصفية مدمجة
            min_score=min_score
        )
        
        # 3. تنسيق النتائج
        formatted_results = self._format_results(results)
        
        # 4. تحديث الإحصائيات
        elapsed = time.time() - start_time
        self._update_stats(len(formatted_results), elapsed)
        
        logger.info(f"✅ Retrieved {len(formatted_results)} documents in {elapsed:.3f}s")
        
        return formatted_results
    
    # ============================================================
    # طرق البحث المتقدمة
    # ============================================================
    
    async def retrieve_hybrid(
        self,
        query: str,
        top_k: Optional[int] = None,
        filter_category: Optional[str] = None,
        filter_supplier: Optional[str] = None,
        semantic_weight: float = 0.7,
        keyword_weight: float = 0.3
    ) -> List[Dict[str, Any]]:
        """
        بحث هجين (دلالي + كلمات مفتاحية)
        
        Args:
            query: سؤال المستخدم
            top_k: عدد النتائج المطلوبة
            filter_category: تصفية حسب التصنيف
            filter_supplier: تصفية حسب المورد
            semantic_weight: وزن البحث الدلالي
            keyword_weight: وزن البحث بالكلمات المفتاحية
            
        Returns:
            قائمة المستندات المسترجعة
        """
        import time
        start_time = time.time()
        
        # 1. البحث الدلالي
        semantic_results = await self.retrieve(
            query=query,
            top_k=top_k or self.top_k * 2,
            filter_category=filter_category,
            filter_supplier=filter_supplier
        )
        
        # 2. البحث بالكلمات المفتاحية
        keywords = self._extract_keywords(query)
        keyword_results = await self.retrieve_by_keywords(
            keywords=keywords,
            top_k=top_k or self.top_k * 2,
            filter_category=filter_category,
            filter_supplier=filter_supplier
        )
        
        # 3. دمج النتائج
        merged_results = self._merge_results(
            semantic_results,
            keyword_results,
            semantic_weight=semantic_weight,
            keyword_weight=keyword_weight
        )
        
        # 4. ترتيب النتائج المدمجة
        merged_results.sort(key=lambda x: x.get("combined_score", 0), reverse=True)
        
        # 5. تحديد عدد النتائج
        if top_k:
            merged_results = merged_results[:top_k]
        
        elapsed = time.time() - start_time
        logger.info(f"✅ Hybrid search returned {len(merged_results)} docs in {elapsed:.3f}s")
        
        return merged_results
    
    async def retrieve_by_keywords(
        self,
        keywords: List[str],
        top_k: Optional[int] = None,
        filter_category: Optional[str] = None,
        filter_supplier: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        البحث بالكلمات المفتاحية في البيانات الوصفية
        
        Args:
            keywords: قائمة الكلمات المفتاحية
            top_k: عدد النتائج المطلوبة
            filter_category: تصفية حسب التصنيف
            filter_supplier: تصفية حسب المورد
            
        Returns:
            قائمة المستندات المسترجعة
        """
        # جلب جميع المستندات مع البيانات الوصفية
        all_docs = self.chroma_loader.get_all_documents()  # ✅ استخدام Chroma
        
        # تصفية حسب التصنيف والمورد
        if filter_category:
            all_docs = [
                doc for doc in all_docs
                if doc.get("metadata", {}).get("category") == filter_category
            ]
        
        if filter_supplier:
            all_docs = [
                doc for doc in all_docs
                if doc.get("metadata", {}).get("supplier") == filter_supplier
            ]
        
        # حساب درجة المطابقة لكل مستند
        scored_docs = []
        for doc in all_docs:
            score = self._calculate_keyword_score(doc, keywords)
            if score > 0:
                scored_docs.append({
                    **doc,
                    "relevance_score": score,
                    "search_type": "keyword"
                })
        
        # ترتيب حسب الدرجة
        scored_docs.sort(key=lambda x: x["relevance_score"], reverse=True)
        
        # تحديد عدد النتائج
        if top_k:
            scored_docs = scored_docs[:top_k]
        
        return scored_docs
    
    # ============================================================
    # طرق مساعدة
    # ============================================================
    
    def _filter_results(
        self,
        results: List[Dict[str, Any]],
        filter_category: Optional[str] = None,
        filter_supplier: Optional[str] = None,
        min_score: float = 0.0
    ) -> List[Dict[str, Any]]:
        """
        تصفية النتائج (تستخدم كمكمل لتصفية Chroma)
        
        Args:
            results: قائمة النتائج
            filter_category: تصفية حسب التصنيف
            filter_supplier: تصفية حسب المورد
            min_score: الحد الأدنى للدرجة
            
        Returns:
            قائمة النتائج المصفاة
        """
        filtered = []
        
        for result in results:
            # تصفية حسب الدرجة
            score = result.get("relevance_score", 0)
            if score < min_score:
                continue
            
            metadata = result.get("metadata", {})
            
            # تصفية حسب التصنيف (تصفية إضافية)
            if filter_category:
                if metadata.get("category") != filter_category:
                    continue
            
            # تصفية حسب المورد (تصفية إضافية)
            if filter_supplier:
                if metadata.get("supplier") != filter_supplier:
                    continue
            
            filtered.append(result)
        
        return filtered
    
    def _format_results(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        تنسيق النتائج للعرض
        
        Args:
            results: قائمة النتائج
            
        Returns:
            قائمة النتائج المنسقة
        """
        formatted = []
        
        for i, result in enumerate(results, 1):
            formatted.append({
                "id": result.get("id", f"doc_{i}"),
                "text": result.get("text", ""),
                "metadata": result.get("metadata", {}),
                "relevance_score": result.get("relevance_score", 0),
                "rank": i,
                "search_type": result.get("search_type", "semantic")
            })
        
        return formatted
    
    def _extract_keywords(self, text: str) -> List[str]:
        """
        استخراج الكلمات المفتاحية من النص
        
        Args:
            text: النص
            
        Returns:
            قائمة الكلمات المفتاحية
        """
        # كلمات شائعة في المشتريات
        procurement_keywords = {
            "عقد", "مورد", "توريد", "شراء", "طلب", "فاتورة",
            "جودة", "تقييم", "سعر", "تكلفة", "ميزانية",
            "شروط", "بنود", "التزام", "ضمان", "تسليم",
            "منتج", "خدمة", "مواد", "معدات", "تخزين",
            "Alpha", "Beta", "Gamma", "Delta", "Epsilon",
            "Inc", "Co", "Supplies", "Logistics", "Group"
        }
        
        words = text.split()
        keywords = []
        
        for word in words:
            # إزالة علامات الترقيم
            clean_word = word.strip("،؛؟!.،")
            if clean_word in procurement_keywords:
                keywords.append(clean_word)
        
        return list(set(keywords))  # إزالة التكرار
    
    def _calculate_keyword_score(self, doc: Dict[str, Any], keywords: List[str]) -> float:
        """
        حساب درجة المطابقة للكلمات المفتاحية
        
        Args:
            doc: المستند
            keywords: قائمة الكلمات المفتاحية
            
        Returns:
            درجة المطابقة
        """
        if not keywords:
            return 0
        
        # جمع النصوص من المستند
        text = doc.get("text", "")
        metadata_text = " ".join([
            str(value) for value in doc.get("metadata", {}).values()
            if isinstance(value, (str, int, float))
        ])
        full_text = f"{text} {metadata_text}".lower()
        
        # حساب عدد المطابقات
        matches = 0
        for keyword in keywords:
            if keyword.lower() in full_text:
                matches += 1
        
        # حساب الدرجة
        score = matches / len(keywords)
        
        return min(score, 1.0)
    
    def _merge_results(
        self,
        semantic_results: List[Dict[str, Any]],
        keyword_results: List[Dict[str, Any]],
        semantic_weight: float = 0.7,
        keyword_weight: float = 0.3
    ) -> List[Dict[str, Any]]:
        """
        دمج نتائج البحث الدلالي والكلمات المفتاحية
        
        Args:
            semantic_results: نتائج البحث الدلالي
            keyword_results: نتائج البحث بالكلمات المفتاحية
            semantic_weight: وزن البحث الدلالي
            keyword_weight: وزن البحث بالكلمات المفتاحية
            
        Returns:
            قائمة النتائج المدمجة
        """
        # إنشاء قاموس لتجميع النتائج حسب المعرف
        merged = {}
        
        # إضافة النتائج الدلالية
        for doc in semantic_results:
            doc_id = doc.get("id", "")
            merged[doc_id] = {
                **doc,
                "semantic_score": doc.get("relevance_score", 0),
                "keyword_score": 0,
                "combined_score": doc.get("relevance_score", 0) * semantic_weight
            }
        
        # إضافة نتائج الكلمات المفتاحية
        for doc in keyword_results:
            doc_id = doc.get("id", "")
            keyword_score = doc.get("relevance_score", 0)
            
            if doc_id in merged:
                # تحديث المستند الموجود
                merged[doc_id]["keyword_score"] = keyword_score
                merged[doc_id]["combined_score"] = (
                    merged[doc_id]["semantic_score"] * semantic_weight +
                    keyword_score * keyword_weight
                )
                # دمج البيانات الوصفية
                merged[doc_id]["metadata"] = {
                    **merged[doc_id].get("metadata", {}),
                    **doc.get("metadata", {})
                }
            else:
                # إضافة مستند جديد
                merged[doc_id] = {
                    **doc,
                    "semantic_score": 0,
                    "keyword_score": keyword_score,
                    "combined_score": keyword_score * keyword_weight
                }
        
        # تحويل إلى قائمة
        results = list(merged.values())
        
        # ترتيب حسب الدرجة المدمجة
        results.sort(key=lambda x: x.get("combined_score", 0), reverse=True)
        
        return results
    
    def _update_stats(self, count: int, elapsed: float) -> None:
        """
        تحديث الإحصائيات
        
        Args:
            count: عدد المستندات المسترجعة
            elapsed: زمن المعالجة
        """
        self.stats["total_queries"] += 1
        self.stats["total_documents_retrieved"] += count
        self.stats["last_query_time"] = elapsed
        
        # تحديث المتوسط
        total = self.stats["total_queries"]
        if total > 0:
            self.stats["avg_retrieval_time"] = (
                (self.stats["avg_retrieval_time"] * (total - 1) + elapsed) / total
            )
    
    def get_stats(self) -> Dict[str, Any]:
        """
        الحصول على إحصائيات الاسترجاع
        
        Returns:
            إحصائيات الاسترجاع
        """
        return {
            **self.stats,
            "top_k": self.top_k,
            "index_size": self.chroma_loader.get_index_size()  # ✅ استخدام Chroma
        }
    
    def reset(self) -> None:
        """
        إعادة تعيين الإحصائيات
        """
        self.stats = {
            "total_queries": 0,
            "avg_retrieval_time": 0,
            "total_documents_retrieved": 0,
            "last_query_time": 0
        }
        logger.info("🔄 Retriever stats reset")
