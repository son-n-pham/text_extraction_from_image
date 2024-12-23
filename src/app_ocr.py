from PIL import Image
import pytesseract
import easyocr
from doctr.io import DocumentFile
from doctr.models import ocr_predictor
import os
import sys


def extract_doctr_text(doctr_result):
    text = []
    for page in doctr_result['pages']:
        for block in page['blocks']:
            for line in block['lines']:
                line_text = []
                for word in line['words']:
                    line_text.append(word['value'])
                text.append(' '.join(line_text))
    return '\n'.join(text)


def process_image(image_path):
    results = {}

    # Tesseract OCR
    try:
        pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
        image = Image.open(image_path)
        results['tesseract'] = pytesseract.image_to_string(image)
    except Exception as e:
        print(f"Tesseract error: {e}")
        results['tesseract'] = None

    # EasyOCR
    try:
        reader = easyocr.Reader(['en'])
        text_easyocr = reader.readtext(image_path, detail=0)
        results['easyocr'] = " ".join(text_easyocr)
    except Exception as e:
        print(f"EasyOCR error: {e}")
        results['easyocr'] = None

    # Doctr
    try:
        doctr_model = ocr_predictor(
            det_arch='db_resnet50', reco_arch='crnn_vgg16_bn', pretrained=True)
        document = DocumentFile.from_images([image_path])
        result = doctr_model(document)
        doctr_json = result.export()
        results['doctr'] = extract_doctr_text(doctr_json)
    except Exception as e:
        print(f"Doctr error: {e}")
        results['doctr'] = None

    return results


if __name__ == "__main__":
    image_path = './img/mang.jpeg'
    if not os.path.exists(image_path):
        print(f"Error: Image not found at {image_path}")
        sys.exit(1)

    results = process_image(image_path)

    # Write results to file
    with open("./output/from_ocr.md", "w", encoding="utf-8") as f:
        for engine, text in results.items():
            f.write(f"## {engine.upper()} Results\n")
            f.write(text if text else "Processing failed")
            f.write("\n\n")

    # Print results
    for engine, text in results.items():
        print(f"\n=== {engine.upper()} Results ===")
        print(text if text else "Processing failed")
