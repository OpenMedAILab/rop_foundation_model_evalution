"""Replace one embedded Word figure while preserving the rest of the DOCX package.

The target picture is resolved from its DrawingML ``pic:cNvPr`` name.  The script
updates both drawing extents to retain the new image's aspect ratio, then performs
basic package-level verification before atomically replacing the DOCX.
"""
from __future__ import annotations

import argparse
import html
import io
import re
import shutil
import tempfile
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET

from PIL import Image


REL_NS = {"pr": "http://schemas.openxmlformats.org/package/2006/relationships"}


def _replace_extent(inline: bytes, width_px: int, height_px: int) -> tuple[bytes, int]:
    extent = re.search(rb'<wp:extent cx="(\d+)" cy="(\d+)"\s*/>', inline)
    if extent is None:
        raise RuntimeError("Could not locate wp:extent in the target inline drawing")
    cx = int(extent.group(1))
    cy = round(cx * height_px / width_px)

    updated, n_wp = re.subn(
        rb'<wp:extent cx="\d+" cy="\d+"\s*/>',
        f'<wp:extent cx="{cx}" cy="{cy}"/>'.encode(),
        inline,
        count=1,
    )
    updated, n_a = re.subn(
        rb'<a:ext cx="\d+" cy="\d+"\s*/>',
        f'<a:ext cx="{cx}" cy="{cy}"/>'.encode(),
        updated,
        count=1,
    )
    if n_wp != 1 or n_a != 1:
        raise RuntimeError(f"Expected one wp:extent and one a:ext; found {n_wp} and {n_a}")
    return updated, cy


def replace_figure(docx: Path, image: Path, figure_name: str, backup: Path,
                   caption_text: str | None = None) -> None:
    image_bytes = image.read_bytes()
    with Image.open(io.BytesIO(image_bytes)) as im:
        width_px, height_px = im.size
        if im.format != "PNG":
            raise ValueError(f"Replacement image must be PNG, not {im.format}")

    with zipfile.ZipFile(docx, "r") as zin:
        document_xml = zin.read("word/document.xml")
        rels_xml = zin.read("word/_rels/document.xml.rels")

        escaped = re.escape(figure_name.encode())
        picture_match = re.search(
            rb'<pic:cNvPr\b[^>]*\bname="' + escaped + rb'"[^>]*/>.*?'
            rb'<a:blip\b[^>]*\br:embed="([^"]+)"[^>]*/>',
            document_xml,
            flags=re.DOTALL,
        )
        if picture_match is None:
            raise RuntimeError(f"Figure named {figure_name!r} was not found in word/document.xml")
        rel_id = picture_match.group(1).decode()

        rel_root = ET.fromstring(rels_xml)
        target = None
        for rel in rel_root.findall("pr:Relationship", REL_NS):
            if rel.attrib.get("Id") == rel_id:
                target = rel.attrib.get("Target")
                break
        if not target or not target.startswith("media/"):
            raise RuntimeError(f"Could not resolve image relationship {rel_id}")
        media_member = f"word/{target}"

        blip_pos = picture_match.end()
        inline_start = document_xml.rfind(b"<wp:inline", 0, blip_pos)
        inline_end = document_xml.find(b"</wp:inline>", blip_pos)
        if inline_start < 0 or inline_end < 0:
            raise RuntimeError("Could not isolate the target wp:inline block")
        inline_end += len(b"</wp:inline>")
        inline, cy = _replace_extent(document_xml[inline_start:inline_end], width_px, height_px)
        updated_document = document_xml[:inline_start] + inline + document_xml[inline_end:]

        if caption_text is not None:
            caption_pattern = re.compile(
                rb'(<w:t[^>]*>Figure 1\. ?</w:t>.*?<w:t[^>]*>)(.*?)(</w:t>)',
                flags=re.DOTALL,
            )
            escaped_caption = html.escape(caption_text, quote=False).encode("utf-8")
            updated_document, caption_count = caption_pattern.subn(
                lambda m: m.group(1) + escaped_caption + m.group(3),
                updated_document,
                count=1,
            )
            if caption_count != 1:
                raise RuntimeError(f"Expected one Figure 1 caption; found {caption_count}")

        if not backup.exists():
            shutil.copy2(docx, backup)

        with tempfile.NamedTemporaryFile(
            prefix=f".{docx.stem}.", suffix=".docx", dir=docx.parent, delete=False
        ) as tmp_handle:
            tmp_path = Path(tmp_handle.name)

        try:
            with zipfile.ZipFile(tmp_path, "w") as zout:
                for info in zin.infolist():
                    data = zin.read(info.filename)
                    if info.filename == "word/document.xml":
                        data = updated_document
                    elif info.filename == media_member:
                        data = image_bytes
                    zout.writestr(info, data)

            with zipfile.ZipFile(tmp_path, "r") as check:
                bad = check.testzip()
                if bad is not None:
                    raise RuntimeError(f"Corrupt ZIP member after replacement: {bad}")
                if check.read(media_member) != image_bytes:
                    raise RuntimeError("Replacement image verification failed")
                updated = check.read("word/document.xml")
                if f'cy="{cy}"'.encode() not in updated:
                    raise RuntimeError("Updated drawing height was not persisted")
                if caption_text is not None and html.escape(
                    caption_text, quote=False
                ).encode("utf-8") not in updated:
                    raise RuntimeError("Updated Figure 1 caption was not persisted")
            tmp_path.replace(docx)
        except Exception:
            tmp_path.unlink(missing_ok=True)
            raise

    print(f"Replaced {figure_name} ({width_px}×{height_px}px) in {docx}")
    print(f"Backup: {backup}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("docx", type=Path)
    parser.add_argument("image", type=Path)
    parser.add_argument("--figure-name", default="Fig1_protocol.png")
    parser.add_argument("--backup", type=Path)
    parser.add_argument("--caption-file", type=Path,
                        help="UTF-8 text file containing the replacement Figure 1 caption body")
    args = parser.parse_args()

    backup = args.backup or args.docx.with_name(
        f"{args.docx.stem}_before_Fig1_replacement{args.docx.suffix}"
    )
    caption_text = args.caption_file.read_text(encoding="utf-8").strip() if args.caption_file else None
    replace_figure(args.docx.resolve(), args.image.resolve(), args.figure_name,
                   backup.resolve(), caption_text=caption_text)


if __name__ == "__main__":
    main()
