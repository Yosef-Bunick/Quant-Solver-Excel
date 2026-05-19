"""
inject_ribbon.py
-----------------
Injects a custom ribbon (tab + button + LOGO) into an Excel .xlam add-in
for "Bunick Quant Solver".

A .xlam is just a ZIP. This script:
  1. Reads the existing .xlam
  2. Adds  customUI/customUI14.xml             (ribbon definition)
  3. Adds  customUI/images/bunick.png          (the logo, embedded)
  4. Adds  customUI/_rels/customUI14.xml.rels  (links button -> image)
  5. Patches  _rels/.rels                      (so Excel loads the ribbon)
  6. Writes a NEW .xlam (original untouched)

Requires the logo file 'bunick_logo.png' next to this script.
Excel must be CLOSED when you run this.

Usage:
    python inject_ribbon.py FastSolver.xlam
        -> creates FastSolver_ribbon.xlam
"""

import sys
import os
import zipfile
import xml.etree.ElementTree as ET

LOGO_FILE = "bunick_logo.png"

RIBBON_XML = '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<customUI xmlns="http://schemas.microsoft.com/office/2009/07/customui"
          loadImage="GetBunickLogo">
  <ribbon>
    <tabs>
      <tab idMso="TabData">
        <group id="bqsGroup" label=" "
               insertBeforeMso="GroupGetExternalData">
          <button id="bqsRun"
                  label="YosefBunick's Quant Solver"
                  size="large"
                  onAction="RunBunickSolver"
                  image="bunickLogo"
                  screentip="Yosef Bunick Quant Solver"
                  supertip="Run the Bunick Quant solver on this workbook."/>
        </group>
      </tab>
    </tabs>
  </ribbon>
</customUI>'''

CUSTOMUI_PART = "customUI/customUI14.xml"
IMAGE_PART = "customUI/images/bunick.png"
CUI_RELS_PART = "customUI/_rels/customUI14.xml.rels"
RELS_PART = "_rels/.rels"
CT_PART = "[Content_Types].xml"
CT_NS = "http://schemas.openxmlformats.org/package/2006/content-types"

REL_TYPE = "http://schemas.microsoft.com/office/2007/relationships/ui/extensibility"
IMG_REL_TYPE = "http://schemas.openxmlformats.org/officeDocument/2006/relationships/image"
REL_NS = "http://schemas.openxmlformats.org/package/2006/relationships"

IMG_REL_ID = "bunickLogo"

CUI_RELS_XML = f'''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="{REL_NS}">
  <Relationship Id="{IMG_REL_ID}" Type="{IMG_REL_TYPE}" Target="images/bunick.png"/>
</Relationships>'''


def _serialize(root) -> bytes:
    body = ET.tostring(root, encoding="unicode")
    decl = '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\r\n'
    return (decl + body).encode("utf-8")


def patch_content_types(ct_bytes: bytes) -> bytes:
    """Ensure [Content_Types].xml declares png + the customUI xml part.
    Without this Office reports the file as corrupt."""
    ET.register_namespace("", CT_NS)
    root = ET.fromstring(ct_bytes)

    have_png = False
    have_cui = False
    for el in root:
        tag = el.tag.split("}")[-1]
        if tag == "Default" and (el.get("Extension") or "").lower() == "png":
            have_png = True
        if tag == "Override" and el.get("PartName") == "/" + CUSTOMUI_PART:
            have_cui = True

    if not have_png:
        d = ET.SubElement(root, f"{{{CT_NS}}}Default")
        d.set("Extension", "png")
        d.set("ContentType", "image/png")
        print("  - registered png content type")

    if not have_cui:
        o = ET.SubElement(root, f"{{{CT_NS}}}Override")
        o.set("PartName", "/" + CUSTOMUI_PART)
        o.set("ContentType",
              "application/xml")
        print("  - registered customUI part content type")

    return _serialize(root)


def patch_rels(rels_bytes: bytes) -> bytes:
    ET.register_namespace("", REL_NS)
    root = ET.fromstring(rels_bytes)

    for rel in root:
        if rel.get("Type") == REL_TYPE:
            print("  - ribbon relationship already present, leaving as is")
            return _serialize(root)

    existing_ids = {rel.get("Id") for rel in root}
    new_id, i = "rIdCustomUI", 1
    while new_id in existing_ids:
        new_id = f"rIdCustomUI{i}"
        i += 1

    rel = ET.SubElement(root, f"{{{REL_NS}}}Relationship")
    rel.set("Id", new_id)
    rel.set("Type", REL_TYPE)
    rel.set("Target", CUSTOMUI_PART)
    print(f"  - added relationship {new_id} -> {CUSTOMUI_PART}")
    return _serialize(root)


def inject(src_path: str) -> str:
    if not os.path.isfile(src_path):
        raise FileNotFoundError(f"File not found: {src_path}")

    here = os.path.dirname(os.path.abspath(__file__))
    logo_path = os.path.join(here, LOGO_FILE)
    if not os.path.isfile(logo_path):
        logo_path = LOGO_FILE
    if not os.path.isfile(logo_path):
        raise FileNotFoundError(
            f"Logo '{LOGO_FILE}' not found. Keep it in the same folder as this script."
        )

    if not src_path.lower().endswith(".xlam"):
        print("WARNING: file does not end in .xlam - continuing anyway")

    base, ext = os.path.splitext(src_path)
    out_path = f"{base}_ribbon{ext}"

    with open(logo_path, "rb") as f:
        logo_bytes = f.read()

    print(f"Reading : {src_path}")
    print(f"Logo    : {logo_path} ({len(logo_bytes)} bytes)")

    skip = {RELS_PART, CUSTOMUI_PART, IMAGE_PART, CUI_RELS_PART, CT_PART}

    with zipfile.ZipFile(src_path, "r") as zin:
        names = zin.namelist()
        if RELS_PART not in names:
            raise RuntimeError(f"'{RELS_PART}' not found - not a valid Office package")
        if CT_PART not in names:
            raise RuntimeError(f"'{CT_PART}' not found - not a valid Office package")
        new_rels = patch_rels(zin.read(RELS_PART))
        new_ct = patch_content_types(zin.read(CT_PART))

        with zipfile.ZipFile(out_path, "w", zipfile.ZIP_DEFLATED) as zout:
            for item in zin.infolist():
                if item.filename in skip:
                    continue
                zout.writestr(item, zin.read(item.filename))

            zout.writestr(CT_PART, new_ct)
            zout.writestr(RELS_PART, new_rels)
            zout.writestr(CUSTOMUI_PART, RIBBON_XML)
            zout.writestr(CUI_RELS_PART, CUI_RELS_XML)
            zout.writestr(IMAGE_PART, logo_bytes)
            print(f"  - wrote {CT_PART}")
            print(f"  - wrote {CUSTOMUI_PART}")
            print(f"  - wrote {CUI_RELS_PART}")
            print(f"  - wrote {IMAGE_PART} (logo embedded)")

    print(f"Done    : {out_path}")
    return out_path


VBA_SNIPPET = '''
' ===== Paste into a MODULE inside the .xlam (Alt+F11 > Insert > Module) =====
Option Explicit

Public Sub RunBunickSolver(control As IRibbonControl)
    MsgBox "Bunick Quant Solver works!", vbInformation, "Bunick Quant Solver"
    ' TODO: pull workbook data, call engine, write results back
End Sub

Public Sub GetBunickLogo(imageID As String, ByRef returnedVal)
    ' Callback must exist for loadImage; embedded image is resolved by Excel.
End Sub
' ===========================================================================
'''


def main():
    if len(sys.argv) < 2:
        print("Usage: python inject_ribbon.py <path-to.xlam>")
        sys.exit(1)
    try:
        out = inject(sys.argv[1])
    except Exception as e:
        print(f"ERROR: {e}")
        sys.exit(1)

    print()
    print("Next steps:")
    print("  1. Open the add-in (Alt+F11), Insert > Module, paste this VBA:")
    print(VBA_SNIPPET)
    print(f"  2. Excel: File > Options > Add-ins > Manage: Excel Add-ins > Go")
    print(f"     > Browse > select '{os.path.basename(out)}' > OK")
    print("  3. The 'Bunick Quant Solver' tab with the bunny logo appears.")


if __name__ == "__main__":
    main()