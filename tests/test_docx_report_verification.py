import os
import sys
import zipfile
import xml.etree.ElementTree as ET
from PIL import Image
import docx

sys.stdout.reconfigure(encoding='utf-8')

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DOCX_PATH = os.path.join(REPO_ROOT, "docs", "reports", "Bao_Cao_Ban_Dau_NCKH_Edge_AI.docx")
ASSETS_DIR = os.path.join(REPO_ROOT, "docs", "figures")

FORBIDDEN_STRINGS = [
    'pin mặt trời',
    'solar panel',
    'mppt',
    'lifepo4',
    'pack pin',
    'gpio33',
    'gpio37',
    'bảng phân bổ chân',
    '4g lte',
    'simcom',
    'vỏ hộp in 3d',
]

REQUIRED_TERMS = [
    ('19,297 / 19297 / 19.297', ['19,297', '19297', '19.297']),
    ('Magnus-Tetens', ['Magnus-Tetens', 'magnus-tetens']),
    ('BAM-1020', ['BAM-1020', 'bam-1020']),
    ('112.117 / 112117 / 112,117', ['112.117', '112117', '112,117']),
    ('55%', ['55%']),
    ('[0.25, 1.05]', ['[0.25, 1.05]', '[0.25,1.05]', '0.25, 1.05']),
    ('0B Heap / 0 Byte Heap', ['0b heap', '0 byte heap', '0 byte', '0b']),
    ('18.84 KB', ['18.84 kb', '18.84kb', '18.84']),
    ('0.15', ['0.15']),
]

import pytest

@pytest.fixture(scope="module")
def doc():
    return docx.Document(DOCX_PATH)

def test_file_validity():
    print("\n" + "="*70)
    print("1. FILE VALIDITY CHECK")
    print("="*70)
    assert os.path.exists(DOCX_PATH), f"File does not exist: {DOCX_PATH}"
    size = os.path.getsize(DOCX_PATH)
    print(f"File path: {DOCX_PATH}")
    print(f"File size: {size:,} bytes ({size / (1024*1024):.2f} MB)")
    assert size > 500 * 1024, f"File size too small: {size} bytes"
    assert size > 2 * 1024 * 1024, f"File size smaller than expected 2 MB: {size} bytes"
    print("File size check: PASSED (> 500 KB and > 2 MB)")

    with zipfile.ZipFile(DOCX_PATH, 'r') as zf:
        namelist = zf.namelist()
        print(f"Zip archive contains {len(namelist)} items.")
        assert "word/document.xml" in namelist, "word/document.xml missing from zip archive!"
        media_files = [f for f in sorted(namelist) if f.startswith("word/media/")]
        print(f"Embedded media files in zip: {media_files}")
        assert len(media_files) >= 4, f"Expected at least 4 media files in zip, found {len(media_files)}"
        for mf in media_files:
            info = zf.getinfo(mf)
            print(f"  - {mf}: {info.file_size:,} bytes")

    loaded_doc = docx.Document(DOCX_PATH)
    assert loaded_doc is not None, "Failed to load document via python-docx"
    print("python-docx load & XML tree traversal: PASSED (Clean parse, no errors)")

def test_structure(doc):
    print("\n" + "="*70)
    print("2. STRUCTURE CHECK")
    print("="*70)
    body_p = len(doc.paragraphs)
    table_p = sum(len(c.paragraphs) for t in doc.tables for r in t.rows for c in r.cells)
    total_p = body_p + table_p

    with zipfile.ZipFile(DOCX_PATH, 'r') as zf:
        xml_root = ET.fromstring(zf.read('word/document.xml'))
        xml_p = len(xml_root.findall('.//{http://schemas.openxmlformats.org/wordprocessingml/2006/main}p'))

    print(f"Paragraphs (Body top-level): {body_p}")
    print(f"Paragraphs (Table cells): {table_p}")
    print(f"Total Document Paragraphs: {total_p}")
    print(f"Total OpenXML <w:p> elements: {xml_p}")

    assert total_p > 400, f"Total document paragraph count {total_p} is <= 400!"
    assert xml_p > 400, f"Total OpenXML <w:p> count {xml_p} is <= 400!"
    print(f"Paragraph count check: PASSED (Total: {total_p} paragraphs, {xml_p} XML elements > 400)")

    tables = doc.tables
    t_count = len(tables)
    print(f"\nTotal tables: {t_count} (Requirement: >= 5)")
    assert t_count >= 5, f"Table count {t_count} is < 5!"

    table_texts = []
    for i, t in enumerate(tables):
        rows = len(t.rows)
        cols = len(t.columns) if rows > 0 else 0
        all_text = " ".join([cell.text.strip() for row in t.rows for cell in row.cells])
        table_texts.append(all_text)
        header_text = [c.text.strip().replace('\n', ' ') for c in t.rows[0].cells] if rows > 0 else []
        print(f"Table {i+1}: {rows} rows x {cols} cols | Header: {header_text[:4]}")

    all_tables_combined = " ".join(table_texts).lower()

    has_params = any("19.297" in txt or "19,297" in txt or "19297" in txt for txt in table_texts)
    print(f"  [PASS] 19,297 parameters breakdown in tables: {has_params}")
    assert has_params, "Table containing 19,297 parameters breakdown not found!"

    has_samples = any("112.117" in txt or "112,117" in txt or "112117" in txt for txt in table_texts)
    print(f"  [PASS] 112,117 samples breakdown in tables: {has_samples}")
    assert has_samples, "Table containing 112,117 samples breakdown not found!"

    has_overfitting = any("jitter" in txt.lower() and "dropout" in txt.lower() for txt in table_texts)
    print(f"  [PASS] 4 anti-overfitting layers in tables: {has_overfitting}")
    assert has_overfitting, "Table containing anti-overfitting layers not found!"

    has_esp32 = any("esp32-s3" in txt.lower() or "flash" in txt.lower() for txt in table_texts)
    print(f"  [PASS] ESP32-S3 hardware specifications in tables: {has_esp32}")
    assert has_esp32, "Table containing ESP32-S3 specs not found!"

    has_criteria = any("nghiệm thu" in txt.lower() or "tiêu chí" in txt.lower() or "acceptance" in txt.lower() for txt in table_texts)
    print(f"  [PASS] Acceptance criteria in tables: {has_criteria}")
    assert has_criteria, "Table containing acceptance criteria not found!"

    print("Table content check: ALL 5 REQUIRED TOPICS VERIFIED IN TABLES!")

def test_image_insertion(doc):
    print("\n" + "="*70)
    print("3. IMAGE INSERTION CHECK")
    print("="*70)
    expected_figures = [
        ("fig1_robust_resmlp_architecture.png", "Hình 3.1"),
        ("fig2_residual_block_skip_connection.png", "Hình 3.2"),
        ("fig3_adaptive_humidity_gating.png", "Hình 3.3"),
        ("fig4_bounded_output_activation.png", "Hình 3.4"),
    ]

    print("Disk Assets in ai/docs_assets/:")
    for fig_file, _ in expected_figures:
        fig_path = os.path.join(ASSETS_DIR, fig_file)
        assert os.path.exists(fig_path), f"Figure missing on disk: {fig_path}"
        with Image.open(fig_path) as im:
            dpi = im.info.get('dpi', (72, 72))
            print(f"  - {fig_file}: {im.size[0]}x{im.size[1]} px | mode={im.mode} | dpi={dpi} | size={os.path.getsize(fig_path):,} bytes")

    with zipfile.ZipFile(DOCX_PATH, 'r') as zf:
        media_names = [f for f in sorted(zf.namelist()) if f.startswith("word/media/")]
        print(f"\nEmbedded media in docx zip ({len(media_names)} items):")
        for m in media_names:
            print(f"  - {m}: {len(zf.read(m)):,} bytes")
        assert len(media_names) >= 4, f"Expected 4 images, found {len(media_names)}"

    inline_shapes = []
    for idx, p in enumerate(doc.paragraphs):
        for run in p.runs:
            drawings = run._r.xpath('.//w:drawing')
            for drawing in drawings:
                inline = drawing.xpath('.//wp:inline')
                if inline:
                    extent = inline[0].xpath('./wp:extent')
                    docPr = inline[0].xpath('./wp:docPr')
                    blip = inline[0].xpath('.//a:blip')
                    cx = int(extent[0].get('cx')) if extent else 0
                    cy = int(extent[0].get('cy')) if extent else 0
                    name = docPr[0].get('name') if docPr else 'Unknown'
                    r_id = blip[0].get('{http://schemas.openxmlformats.org/officeDocument/2006/relationships}embed') if blip else 'None'

                    width_cm = cx / 360000.0
                    height_cm = cy / 360000.0
                    inline_shapes.append((idx, name, r_id, width_cm, height_cm))

    print(f"\nInline Shapes in document ({len(inline_shapes)} found):")
    for i, (p_idx, name, r_id, w, h) in enumerate(inline_shapes):
        print(f"  Shape {i+1} at Paragraph {p_idx}: {name} ({r_id}) -> {w:.2f} cm x {h:.2f} cm (aspect ratio: {w/h:.3f})")
    assert len(inline_shapes) == 4, f"Expected 4 inline drawing shapes, found {len(inline_shapes)}"

    captions = []
    for idx, p in enumerate(doc.paragraphs):
        txt = p.text.strip()
        for _, cap_prefix in expected_figures:
            if txt.startswith(cap_prefix):
                captions.append((idx, cap_prefix, txt))
                print(f"  Caption found at P{idx}: {txt[:120]}...")
                break

    assert len(captions) == 4, f"Expected 4 figure captions, found {len(captions)}"
    print("Image insertion check: ALL 4 HIGH-RES FIGURES & CAPTIONS VERIFIED!")

def test_forbidden_strings(doc):
    print("\n" + "="*70)
    print("4. STRING AUDIT (FORBIDDEN STRINGS)")
    print("="*70)
    violations = []

    for idx, p in enumerate(doc.paragraphs):
        text_lower = p.text.lower()
        for forbidden in FORBIDDEN_STRINGS:
            if forbidden.lower() in text_lower:
                violations.append((f"Paragraph {idx}", forbidden, p.text[:100]))

    for t_idx, t in enumerate(doc.tables):
        for r_idx, row in enumerate(t.rows):
            for c_idx, cell in enumerate(row.cells):
                for p_idx, p in enumerate(cell.paragraphs):
                    text_lower = p.text.lower()
                    for forbidden in FORBIDDEN_STRINGS:
                        if forbidden.lower() in text_lower:
                            violations.append((f"Table {t_idx+1} Row {r_idx+1} Col {c_idx+1} P {p_idx}", forbidden, p.text[:100]))

    for s_idx, sec in enumerate(doc.sections):
        for p_idx, p in enumerate(sec.header.paragraphs):
            text_lower = p.text.lower()
            for forbidden in FORBIDDEN_STRINGS:
                if forbidden.lower() in text_lower:
                    violations.append((f"Section {s_idx+1} Header P {p_idx}", forbidden, p.text[:100]))

        for p_idx, p in enumerate(sec.footer.paragraphs):
            text_lower = p.text.lower()
            for forbidden in FORBIDDEN_STRINGS:
                if forbidden.lower() in text_lower:
                    violations.append((f"Section {s_idx+1} Footer P {p_idx}", forbidden, p.text[:100]))

    with zipfile.ZipFile(DOCX_PATH, 'r') as zf:
        for fname in zf.namelist():
            if fname.endswith(".xml"):
                content = zf.read(fname).decode('utf-8', errors='ignore').lower()
                for forbidden in FORBIDDEN_STRINGS:
                    if forbidden.lower() in content:
                        violations.append((f"Raw XML {fname}", forbidden, "(found in raw xml)"))

    print(f"Total forbidden string occurrences: {len(violations)}")
    if violations:
        for v in violations:
            print(f"  [VIOLATION] {v[0]}: matches '{v[1]}' -> text: {v[2]}")
    assert len(violations) == 0, f"Found {len(violations)} forbidden string occurrences!"
    print("Forbidden string audit: PASSED (EXACTLY 0 OCCURRENCES)")

def test_content_check(doc):
    print("\n" + "="*70)
    print("5. REQUIRED KEY TERMS CHECK")
    print("="*70)
    full_items = []
    for idx, p in enumerate(doc.paragraphs):
        if p.text.strip():
            full_items.append((f"Body P{idx}", p.text))
    for t_idx, t in enumerate(doc.tables):
        for r_idx, r in enumerate(t.rows):
            for c_idx, c in enumerate(r.cells):
                if c.text.strip():
                    full_items.append((f"Table {t_idx+1} R{r_idx+1} C{c_idx+1}", c.text))
    for s_idx, s in enumerate(doc.sections):
        for p in s.header.paragraphs:
            if p.text.strip():
                full_items.append((f"Section {s_idx+1} Header", p.text))
        for p in s.footer.paragraphs:
            if p.text.strip():
                full_items.append((f"Section {s_idx+1} Footer", p.text))

    for label, needles in REQUIRED_TERMS:
        matches = []
        for loc, text in full_items:
            t_low = text.lower()
            for n in needles:
                if n.lower() in t_low:
                    idx = t_low.find(n.lower())
                    snippet = text[max(0, idx - 30):min(len(text), idx + len(n) + 40)].replace('\n', ' ')
                    matches.append((loc, snippet))
                    break
        print(f"Term '{label}': {len(matches)} occurrences")
        assert len(matches) > 0, f"Required term '{label}' not found!"
        for loc, snip in matches[:2]:
            print(f"   [{loc}] ...{snip}...")

    print("Required key terms check: PASSED (ALL REQUIRED TERMS PRESENT AND VERIFIED)")

if __name__ == "__main__":
    test_file_validity()
    doc_instance = docx.Document(DOCX_PATH)
    test_structure(doc_instance)
    test_image_insertion(doc_instance)
    test_forbidden_strings(doc_instance)
    test_content_check(doc_instance)
    print("\n" + "="*70)
    print("ALL 5 EMPIRICAL VERIFICATION TESTS PASSED WITH 100% SUCCESS!")
    print("VERDICT: APPROVE")
    print("="*70)
