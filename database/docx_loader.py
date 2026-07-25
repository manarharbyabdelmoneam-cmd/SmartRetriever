"""
📝 تحميل النصوص من ملفات DOCX

يقرأ محتوى ملفات Word (.docx) ويستخرج النص منها
"""

import os
import re
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import zipfile
import xml.etree.ElementTree as ET
from datetime import datetime

from utils.logger import logger


# ============================================================
# ✅ دالة تنظيف البيانات الوصفية
# ============================================================

def clean_metadata_for_chroma(metadata: Dict[str, Any]) -> Dict[str, Any]:
    """
    تنظيف البيانات الوصفية لتكون متوافقة مع Chroma
    
    Chroma يقبل فقط:
    - str, int, float, bool, None
    
    Args:
        metadata: البيانات الوصفية
        
    Returns:
        البيانات الوصفية النظيفة
    """
    cleaned = {}
    
    for key, value in metadata.items():
        # تخطي القيم الفارغة
        if value is None:
            cleaned[key] = None
        elif isinstance(value, (str, int, float, bool)):
            cleaned[key] = value
        elif isinstance(value, (list, tuple, set)):
            # تحويل المجموعات إلى سلسلة
            try:
                cleaned[key] = str(list(value))
            except:
                cleaned[key] = str(value)
        elif isinstance(value, dict):
            # تحويل القواميس إلى سلسلة
            try:
                cleaned[key] = str(value)
            except:
                cleaned[key] = str(value)
        elif isinstance(value, datetime):
            # تحويل التواريخ إلى سلسلة ISO
            try:
                cleaned[key] = value.isoformat()
            except:
                cleaned[key] = str(value)
        else:
            # أي نوع آخر يتم تحويله إلى سلسلة
            try:
                cleaned[key] = str(value)
            except:
                cleaned[key] = None
    
    return cleaned


class DocxLoader:
    """
    تحميل النصوص من ملفات DOCX

    يدعم:
    - استخراج النص من ملفات .docx
    - استخراج البيانات الوصفية
    - تنظيف النص المستخرج
    - قراءة الملفات في مجلدات
    """

    NAMESPACES = {
        'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main',
        'r': 'http://schemas.openxmlformats.org/officeDocument/2006/relationships',
        'dc': 'http://purl.org/dc/elements/1.1/',
        'cp': 'http://schemas.openxmlformats.org/package/2006/metadata/core-properties',
        'dcterms': 'http://purl.org/dc/terms/'
    }

    def __init__(self, clean_text: bool = True, remove_extra_spaces: bool = True):
        self.clean_text = clean_text
        self.remove_extra_spaces = remove_extra_spaces

        self.stats = {
            "total_loaded": 0,
            "total_failed": 0,
            "total_tokens": 0
        }

        logger.info("📝 DocxLoader initialized")

    # ============================================================
    # الطرق الرئيسية
    # ============================================================

    def load_file(self, file_path: str) -> Optional[Dict[str, Any]]:
        try:
            path = Path(file_path)

            if not path.exists():
                logger.error(f"❌ File not found: {file_path}")
                return None

            if not path.suffix.lower() == '.docx':
                logger.warning(f"⚠️ Not a DOCX file: {file_path}")
                return None

            text = self.extract_text(path)

            if not text:
                logger.warning(f"⚠️ No text extracted from: {file_path}")
                return None

            metadata = self.extract_metadata(path)
            
            # ✅ تنظيف البيانات الوصفية قبل الإرجاع
            metadata = clean_metadata_for_chroma(metadata)

            if self.clean_text:
                text = self._clean_extracted_text(text)

            self.stats["total_loaded"] += 1
            self.stats["total_tokens"] += len(text.split())

            return {
                "text": text,
                "metadata": metadata,
                "file_path": str(path),
                "filename": path.name,
                "file_size": path.stat().st_size,
                "loaded_at": datetime.now().isoformat()
            }

        except Exception as e:
            logger.error(f"❌ Error loading DOCX {file_path}: {str(e)}")
            self.stats["total_failed"] += 1
            return None

    def load_directory(
        self,
        directory_path: str,
        recursive: bool = True,
        max_files: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        documents = []
        path = Path(directory_path)

        if not path.exists():
            logger.error(f"❌ Directory not found: {directory_path}")
            return []

        pattern = "**/*.docx" if recursive else "*.docx"
        files = list(path.glob(pattern))

        logger.info(f"📁 Found {len(files)} DOCX files in {directory_path}")

        if max_files and len(files) > max_files:
            files = files[:max_files]
            logger.info(f"📁 Limiting to {max_files} files")

        for file_path in files:
            doc = self.load_file(str(file_path))
            if doc:
                documents.append(doc)

        logger.info(f"✅ Loaded {len(documents)} DOCX documents")

        return documents

    # ============================================================
    # طرق استخراج النص
    # ============================================================

    def extract_text(self, file_path: Path) -> str:
        try:
            with zipfile.ZipFile(file_path, 'r') as zip_file:
                if 'word/document.xml' not in zip_file.namelist():
                    logger.warning(f"⚠️ No document.xml found in {file_path}")
                    return ""

                with zip_file.open('word/document.xml') as xml_file:
                    content = xml_file.read().decode('utf-8')

                    text = self._extract_text_from_xml(content)
                    text += self._extract_headers_and_footers(zip_file)
                    text += self._extract_tables(zip_file)

                    return text

        except Exception as e:
            logger.error(f"❌ Error extracting text from {file_path}: {str(e)}")
            return ""

    def _extract_text_from_xml(self, xml_content: str) -> str:
        try:
            root = ET.fromstring(xml_content)

            texts = []
            for elem in root.iter():
                if elem.tag.endswith('t'):
                    if elem.text:
                        texts.append(elem.text)
                elif elem.tag.endswith('p'):
                    if texts and not texts[-1].endswith('\n'):
                        texts.append('\n')

            return ''.join(texts)

        except Exception as e:
            logger.error(f"❌ Error parsing XML: {str(e)}")
            return ""

    def _extract_headers_and_footers(self, zip_file: zipfile.ZipFile) -> str:
        text = ""

        paths = [
            'word/header1.xml',
            'word/header2.xml',
            'word/header3.xml',
            'word/footer1.xml',
            'word/footer2.xml',
            'word/footer3.xml'
        ]

        for path in paths:
            if path in zip_file.namelist():
                try:
                    with zip_file.open(path) as f:
                        content = f.read().decode('utf-8')
                        extracted = self._extract_text_from_xml(content)
                        if extracted:
                            text += f"\n[{path}]\n{extracted}\n"
                except Exception:
                    continue

        return text

    def _extract_tables(self, zip_file: zipfile.ZipFile) -> str:
        text = ""

        if 'word/document.xml' in zip_file.namelist():
            try:
                with zip_file.open('word/document.xml') as f:
                    content = f.read().decode('utf-8')
                    root = ET.fromstring(content)

                    for table in root.iter():
                        if table.tag.endswith('tbl'):
                            text += "\n[TABLE]\n"
                            for row in table.iter():
                                if row.tag.endswith('tr'):
                                    row_text = []
                                    for cell in row.iter():
                                        if cell.tag.endswith('tc'):
                                            cell_text = self._extract_text_from_xml(
                                                ET.tostring(cell, encoding='unicode')
                                            )
                                            if cell_text.strip():
                                                row_text.append(cell_text.strip())
                                    if row_text:
                                        text += " | ".join(row_text) + "\n"
                            text += "[/TABLE]\n"
            except Exception:
                pass

        return text

    # ============================================================
    # طرق استخراج البيانات الوصفية
    # ============================================================

    def extract_metadata(self, file_path: Path) -> Dict[str, Any]:
        metadata = {
            "filename": file_path.name,
            "file_path": str(file_path),
            "file_size": file_path.stat().st_size,
            "extension": file_path.suffix,
            "category": self._guess_category(str(file_path)),
            "created_at": None,
            "modified_at": None,
            "author": None,
            "title": None,
            "subject": None,
            "keywords": None,
            "comments": None,
            "page_count": None,
            "word_count": None,
            "paragraph_count": None
        }

        try:
            with zipfile.ZipFile(file_path, 'r') as zip_file:
                if 'docProps/core.xml' in zip_file.namelist():
                    with zip_file.open('docProps/core.xml') as f:
                        content = f.read().decode('utf-8')
                        metadata.update(self._extract_core_properties(content))

                if 'docProps/app.xml' in zip_file.namelist():
                    with zip_file.open('docProps/app.xml') as f:
                        content = f.read().decode('utf-8')
                        metadata.update(self._extract_app_properties(content))

        except Exception as e:
            logger.debug(f"⚠️ Could not extract metadata: {str(e)}")

        metadata["loaded_at"] = datetime.now().isoformat()
        
        # ✅ تنظيف البيانات الوصفية قبل الإرجاع
        return clean_metadata_for_chroma(metadata)

    def _extract_core_properties(self, content: str) -> Dict[str, Any]:
        properties = {}

        try:
            root = ET.fromstring(content)

            for child in root:
                tag = child.tag.split('}')[-1] if '}' in child.tag else child.tag

                if tag == 'title':
                    properties['title'] = child.text
                elif tag == 'subject':
                    properties['subject'] = child.text
                elif tag == 'creator':
                    properties['author'] = child.text
                elif tag == 'description':
                    properties['comments'] = child.text
                elif tag == 'keywords':
                    properties['keywords'] = child.text
                elif tag == 'created':
                    properties['created_at'] = child.text
                elif tag == 'modified':
                    properties['modified_at'] = child.text

        except Exception:
            pass

        return properties

    def _extract_app_properties(self, content: str) -> Dict[str, Any]:
        properties = {}

        try:
            root = ET.fromstring(content)

            for child in root:
                tag = child.tag.split('}')[-1] if '}' in child.tag else child.tag

                if tag == 'Pages':
                    properties['page_count'] = int(child.text) if child.text else None
                elif tag == 'Words':
                    properties['word_count'] = int(child.text) if child.text else None
                elif tag == 'Paragraphs':
                    properties['paragraph_count'] = int(child.text) if child.text else None

        except Exception:
            pass

        return properties

    def _guess_category(self, file_path: str) -> str:
        filename = Path(file_path).name.lower()

        if 'contract' in filename or 'عقد' in filename:
            return 'contracts'
        elif 'policy' in filename or 'سياسة' in filename:
            return 'policies'
        elif 'quotation' in filename or 'عرض' in filename or 'سعر' in filename:
            return 'quotations'
        elif 'quality' in filename or 'جودة' in filename or 'تقرير' in filename:
            return 'quality_reports'
        elif 'report' in filename:
            return 'reports'
        elif 'manual' in filename or 'دليل' in filename:
            return 'manuals'
        elif 'invoice' in filename or 'فاتورة' in filename:
            return 'invoices'
        else:
            return 'other'

    # ============================================================
    # طرق تنظيف النص
    # ============================================================

    def _clean_extracted_text(self, text: str) -> str:
        if not text:
            return ""

        text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text)
        text = re.sub(r'<[^>]+>', '', text)

        if self.remove_extra_spaces:
            text = re.sub(r'\s+', ' ', text)
            text = re.sub(r'\n\s*\n', '\n\n', text)

        text = re.sub(r'\n{3,}', '\n\n', text)
        text = text.strip()

        return text

    # ============================================================
    # طرق إضافية
    # ============================================================

    def get_stats(self) -> Dict[str, Any]:
        return {
            **self.stats,
            "total_loaded": self.stats["total_loaded"],
            "total_failed": self.stats["total_failed"],
            "total_tokens": self.stats["total_tokens"]
        }

    def reset_stats(self) -> None:
        self.stats = {
            "total_loaded": 0,
            "total_failed": 0,
            "total_tokens": 0
        }
        logger.info("🔄 DocxLoader stats reset")

    def is_supported(self, file_path: str) -> bool:
        return Path(file_path).suffix.lower() == '.docx'

    def get_file_info(self, file_path: str) -> Dict[str, Any]:
        path = Path(file_path)

        info = {
            "filename": path.name,
            "file_path": str(path),
            "file_size": path.stat().st_size,
            "extension": path.suffix,
            "is_supported": self.is_supported(file_path),
            "category": self._guess_category(file_path)
        }

        if info["is_supported"]:
            try:
                with zipfile.ZipFile(file_path, 'r') as zip_file:
                    if 'docProps/core.xml' in zip_file.namelist():
                        with zip_file.open('docProps/core.xml') as f:
                            content = f.read().decode('utf-8')
                            core = self._extract_core_properties(content)
                            info.update(core)
            except Exception:
                pass

        return clean_metadata_for_chroma(info)
