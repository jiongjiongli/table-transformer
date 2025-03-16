import json
import argparse
import os
from pathlib import Path
import sys
from PIL import Image
import zipfile

from pytesseract import pytesseract

import_script_path = (Path(__file__).resolve().parent / "../scripts").resolve()
print(f"Importing {import_script_path}")
sys.path.append(str(import_script_path))

from generate_ocr_words import tesseract_ocr
from inference import TableExtractionPipeline, output_result


def get_args(input_args):
    parser = argparse.ArgumentParser()

    parser.add_argument('--use_ocr', action='store_true',
                        help="Whether use OCR service or use ground truth texts")
    parser.add_argument('--image_dir',
                        help="Directory for input images")
    parser.add_argument('--page_words_dir',
                        help="Directory for page input words")
    parser.add_argument('--table_words_dir',
                        help="Directory for table input words")
    parser.add_argument('--ocr_words_dir',
                        help="Directory for ocr input words")
    parser.add_argument('--out_dir',
                        help="Output directory")
    parser.add_argument('--mode',
                        help="The processing to apply to the input image and tokens",
                        choices=['detect', 'recognize', 'extract'])
    parser.add_argument('--structure_config_path',
                        help="Filepath to the structure model config file")
    parser.add_argument('--structure_model_path', help="The path to the structure model")
    parser.add_argument('--detection_config_path',
                        help="Filepath to the detection model config file")
    parser.add_argument('--detection_model_path', help="The path to the detection model")
    parser.add_argument('--detection_device', default="cuda")
    parser.add_argument('--structure_device', default="cuda")
    parser.add_argument('--crops', '-p', action='store_true',
                        help='Output cropped data from table detections')
    parser.add_argument('--objects', '-o', action='store_true',
                        help='Output objects')
    parser.add_argument('--cells', '-l', action='store_true',
                        help='Output cells list')
    parser.add_argument('--html', '-m', action='store_true',
                        help='Output HTML')
    parser.add_argument('--csv', '-c', action='store_true',
                        help='Output CSV')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Verbose output')
    parser.add_argument('--visualize', '-z', action='store_true',
                        help='Visualize output')
    parser.add_argument('--crop_padding', type=int, default=10,
                        help="The amount of padding to add around a detected table when cropping.")

    parser.add_argument('--tesseract_cmd',
                        help="Tesseract command path")

    return parser.parse_args(input_args)

class TableExtractModel:
    def __init__(self, args):
        self.args = args
        self.pipe = self.load_model()

    def load_model(self):
        args = self.args

        pytesseract.tesseract_cmd = args.tesseract_cmd
        print("Creating inference pipeline")
        pipe = TableExtractionPipeline(det_device=args.detection_device,
                                       str_device=args.structure_device,
                                       det_config_path=args.detection_config_path,
                                       det_model_path=args.detection_model_path,
                                       str_config_path=args.structure_config_path,
                                       str_model_path=args.structure_model_path)
        return pipe

    def rm_files(self, file_paths):
        for file_path in file_paths:
            file_path.unlink()

    def process(self, image_file_path):
        args = self.args

        print(f"args: {args}")

        prev_output_file_paths = list(Path(args.out_dir).glob(f"{Path(image_file_path).stem}*.*"))
        self.rm_files(prev_output_file_paths)

        img = Image.open(image_file_path)
        print("Image loaded.")

        image_file_name = Path(image_file_path).name

        if not args.use_ocr:
            print("Using table ground truth")
            table_tokens_path = os.path.join(args.table_words_dir, image_file_name.replace(".jpg", "_words.json"))
            if os.path.exists(table_tokens_path):
                with open(table_tokens_path, 'r') as f:
                    tokens = json.load(f)
            else:
                print(f"{table_tokens_path} does not exist!")
                tokens = []

        elif args.ocr_words_dir is not None:
            tokens = tesseract_ocr(image_file_path)
            tokens_path = os.path.join(args.ocr_words_dir, image_file_name.replace(".jpg", "_words.json"))

            Path(tokens_path).parent.mkdir(parents=True, exist_ok=True)

            print(f"Writing OCR results to: {tokens_path}")

            with open(tokens_path, "w") as file_stream:
                json.dump(tokens, file_stream, indent=4)

            with open(tokens_path, 'r') as f:
                tokens = json.load(f)

                # Handle dictionary format
                if type(tokens) is dict and 'words' in tokens:
                    tokens = tokens['words']

                # 'tokens' is a list of tokens
                # Need to be in a relative reading order
                # If no order is provided, use current order
                for idx, token in enumerate(tokens):
                    if not 'span_num' in token:
                        token['span_num'] = idx
                    if not 'line_num' in token:
                        token['line_num'] = 0
                    if not 'block_num' in token:
                        token['block_num'] = 0
        else:
            tokens = []

        if args.mode == 'recognize':
            extracted_table = self.pipe.recognize(img, tokens, out_objects=args.objects, out_cells=args.csv,
                                out_html=args.html, out_csv=args.csv)
            print("Table(s) recognized.")

            for key, val in extracted_table.items():
                output_result(key, val, args, img, image_file_name)

        if args.mode == 'detect':
            detected_tables = self.pipe.detect(img, tokens, out_objects=args.objects, out_crops=args.crops)
            print("Table(s) detected.")

            for key, val in detected_tables.items():
                output_result(key, val, args, img, image_file_name)

        if args.mode == 'extract':
            extracted_tables = self.pipe.extract(img, tokens, out_objects=args.objects, out_cells=args.csv,
                                            out_html=args.html, out_csv=args.csv,
                                            crop_padding=args.crop_padding)
            print("Table(s) extracted.")

            for table_idx, extracted_table in enumerate(extracted_tables):
                for key, val in extracted_table.items():

                    try:
                        output_result(key, val, args, extracted_table['image'],
                                      image_file_name.replace('.jpg', '_{}.jpg'.format(table_idx)))
                    except Exception as e:
                        import traceback
                        error_details = traceback.format_exc()
                        print(f"An error occurred when processing {image_file_path}: {e} {error_details}")
                        raise

        output_file_paths = list(Path(args.out_dir).glob(f"{Path(image_file_path).stem}*.*"))
        print(f"output_file_paths: {[str(output_file_path) for output_file_path in output_file_paths]}")

        zip_file_path = Path(args.out_dir) / "zips" / f"download_{Path(image_file_path).stem}.zip"
        zip_file_path.parent.mkdir(parents=True, exist_ok=True)

        if zip_file_path.exists():
            zip_file_path.unlink()

        print(f"zip_file_path: {zip_file_path}")

        with zipfile.ZipFile(zip_file_path, 'w') as myzip:
            for output_file_path in output_file_paths:
                arcname = str(Path(zip_file_path.stem) / output_file_path.name)
                myzip.write(str(output_file_path),
                            arcname=arcname)

        return zip_file_path
