"""
Multimodal document processing: PDFs and images.
- PDF: Extract text and images via PyMuPDF
- Images: OCR via pytesseract (optional), description via Claude vision
"""

import base64
import io
import logging
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class ExtractedContent:
    text: str
    images: list
    metadata: dict
    source_path: str


def extract_pdf(filepath):
    """Extract text and images from a PDF file."""
    try:
        import fitz  # PyMuPDF
    except ImportError:
        logger.error("PyMuPDF not installed. Run: pip install PyMuPDF")
        return None

    try:
        doc = fitz.open(filepath)
        all_text = []
        images = []

        for page_num in range(len(doc)):
            page = doc[page_num]
            text = page.get_text("text")
            if text.strip():
                all_text.append("--- Page {} ---\n{}".format(page_num + 1, text))

            image_list = page.get_images(full=True)
            for img_idx, img_info in enumerate(image_list[:3]):
                try:
                    xref = img_info[0]
                    base_image = doc.extract_image(xref)
                    if base_image and base_image["image"]:
                        img_bytes = base_image["image"]
                        img_b64 = base64.b64encode(img_bytes).decode("utf-8")
                        images.append({
                            "data": img_b64,
                            "page": page_num + 1,
                            "mime_type": "image/{}".format(base_image.get("ext", "png")),
                            "description": "Image from page {}".format(page_num + 1),
                        })
                except Exception as e:
                    logger.debug("Could not extract image {} from page {}: {}".format(img_idx, page_num + 1, e))

        doc.close()

        return ExtractedContent(
            text="\n\n".join(all_text),
            images=images[:10],
            metadata={
                "type": "pdf",
                "pages": len(doc) if not doc.is_closed else 0,
                "has_images": len(images) > 0,
            },
            source_path=filepath,
        )

    except Exception as e:
        logger.error("PDF extraction failed for {}: {}".format(filepath, e))
        return None


def extract_image_text(filepath):
    """Extract text from an image via OCR (optional)."""
    try:
        from PIL import Image
    except ImportError:
        logger.error("Pillow not installed. Run: pip install Pillow")
        return None

    try:
        img = Image.open(filepath)

        ocr_text = ""
        try:
            import pytesseract
            ocr_text = pytesseract.image_to_string(img)
        except ImportError:
            ocr_text = "[Image file - install Tesseract OCR for text extraction]"
        except Exception as e:
            logger.warning("OCR failed: {}".format(e))
            ocr_text = "[Image - OCR unavailable]"

        buf = io.BytesIO()
        img_format = img.format or "PNG"
        img.save(buf, format=img_format)
        img_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

        mime_map = {"PNG": "image/png", "JPEG": "image/jpeg", "GIF": "image/gif", "WEBP": "image/webp"}
        mime_type = mime_map.get(img_format.upper(), "image/png")

        return ExtractedContent(
            text=ocr_text,
            images=[{"data": img_b64, "page": 1, "mime_type": mime_type, "description": "Uploaded image"}],
            metadata={
                "type": "image",
                "format": img_format,
                "size": "{}x{}".format(img.width, img.height),
                "has_ocr": bool(ocr_text.strip()),
            },
            source_path=filepath,
        )

    except Exception as e:
        logger.error("Image extraction failed for {}: {}".format(filepath, e))
        return None


MULTIMODAL_EXTENSIONS = {".pdf", ".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp", ".tiff"}


def is_multimodal_file(filepath):
    return Path(filepath).suffix.lower() in MULTIMODAL_EXTENSIONS


def extract_multimodal(filepath):
    ext = Path(filepath).suffix.lower()
    if ext == ".pdf":
        return extract_pdf(filepath)
    elif ext in {".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp", ".tiff"}:
        return extract_image_text(filepath)
    return None
