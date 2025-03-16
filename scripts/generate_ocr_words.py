import argparse
import json
from pathlib import Path
import re
from tqdm import tqdm
from PIL import Image, ImageDraw, ImageFont
from pytesseract import pytesseract, image_to_data, Output


def tesseract_ocr(image_file_path):
    # tesseract OCR
    ocr_output = image_to_data(image_file_path, output_type=Output.DICT)

    # Column primary to row primary
    column_names = []
    column_values = []

    for key, value in ocr_output.items():
        column_names.append(key)
        column_values.append(value)

    rows = []

    for row_values in zip(*column_values):
        row = {column_name: value for column_name, value in zip(column_names, row_values)}
        rows.append(row)

    # Row to word
    words = []

    for row in rows:
        # Select word level
        if row["level"] != 5:
            continue

        # if not row["text"].strip():
        #     continue

        word = {
            "bbox": [row["left"], row["top"], row["left"] + row["width"], row["top"] + row["height"]],
            "text": row["text"],
            "block_num": row["block_num"],
            "line_num": row["line_num"],
            "span_num": row["word_num"],
        }
        words.append(word)

    return words


def generate_ocr_results(data_dir_path, output_dir_path):
    splits = ["train", "val", "test"]

    for split in splits:
        image_dir_path = data_dir_path / split
        image_file_paths = list(image_dir_path.glob("*.jpg"))
        ocr_results_dir_path = output_dir_path / "ocr_results" / split

        print(f"Generating OCR result files to {ocr_results_dir_path}")

        for image_file_path in tqdm(image_file_paths):
            words = tesseract_ocr(str(image_file_path))
            ocr_results_file_name = f"{image_file_path.stem}_words.json"
            ocr_results_file_path = ocr_results_dir_path / ocr_results_file_name
            ocr_results_file_path.parent.mkdir(parents=True, exist_ok=True)

            with open(ocr_results_file_path, "w") as file_stream:
                json.dump(words, file_stream, indent=4)


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--tesseract_cmd',
                        help="Tesseract command path")
    parser.add_argument('--data_dir',
                        help="Root directory for source data to process")
    parser.add_argument('--output_dir',
                        help="Root directory for output data")
    return parser.parse_args()


def main():
    args = get_args()
    pytesseract.tesseract_cmd = args.tesseract_cmd
    data_dir_path = Path(args.data_dir)
    output_dir_path = Path(args.output_dir)

    generate_ocr_results(data_dir_path, output_dir_path)
    print("Completed!")


if __name__ == '__main__':
    main()
