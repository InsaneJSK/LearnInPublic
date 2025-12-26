#!/usr/bin/env python3
import sys
from pathlib import Path
from pypdf import PdfReader, PdfWriter

def update_pdf_title(pdf_path, new_title):
    pdf_path = Path(pdf_path)
    if not pdf_path.exists() or pdf_path.suffix.lower() != ".pdf":
        print(f"Error: '{pdf_path}' is not a valid PDF file.")
        return

    reader = PdfReader(str(pdf_path))
    writer = PdfWriter()

    # Copy all pages
    for page in reader.pages:
        writer.add_page(page)

    # Preserve existing metadata and update title
    metadata = reader.metadata or {}
    metadata.update({
        "/Title": new_title
    })
    writer.add_metadata(metadata)

    # Save to a new file (overwrite safe)
    output_file = pdf_path.with_name(pdf_path.stem + ".pdf")
    with open(output_file, "wb") as f:
        writer.write(f)

    print(f"Updated PDF saved as: {output_file}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python update_pdf_title.py <pdf_path> <new_title>")
        sys.exit(1)

    pdf_path, new_title = sys.argv[1], sys.argv[2]
    update_pdf_title(pdf_path, new_title)
