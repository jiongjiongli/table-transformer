import argparse
from pathlib import Path
from tqdm import tqdm
import re
import fitz
from process_fintabnet import create_document_page_image


def find_pdf_file_path(pdf_dir_path, ann_file_path):
    match_result = re.match("^(.+)_page_([0-9]+)_table_(.+).xml$", ann_file_path.name)
    pdf_dir_names = match_result.group(1).split("_")
    pdf_file_name = f"page_{match_result.group(2)}.pdf"

    current_pdf_dir_path = pdf_dir_path

    for pdf_dir_name in pdf_dir_names:
        current_pdf_dir_path = current_pdf_dir_path / pdf_dir_name
        assert current_pdf_dir_path.exists(), current_pdf_dir_path

    pdf_file_path = current_pdf_dir_path / pdf_file_name

    assert pdf_file_path.exists(), pdf_file_path

    return pdf_file_path


def generate_images(data_dir_path,
                    pdf_dir_path,
                    output_dir_path):
    splits = ["train", "val", "test"]

    for split in splits:
        ann_dir_path = data_dir_path / split
        ann_file_paths = list(ann_dir_path.glob("*.xml"))
        page_images_dir_path = output_dir_path / "page_images" / split

        print(f"Generating image files to {page_images_dir_path}")

        for ann_file_path in tqdm(ann_file_paths):
            pdf_file_path = find_pdf_file_path(pdf_dir_path, ann_file_path)

            doc = fitz.open(pdf_file_path)

            page_num = 0
            output_image_max_dim = 1000
            page_image = create_document_page_image(doc,
                                                    page_num,
                                                    output_image_max_dim=output_image_max_dim)

            page_image_file_path = page_images_dir_path / ann_file_path.with_suffix(".jpg").name
            page_image_file_path.parent.mkdir(parents=True, exist_ok=True)
            page_image.save(page_image_file_path)
            doc.close()


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_dir',
                        help="Root directory for source data to process")
    parser.add_argument('--pdf_dir',
                        help="Root directory for pdf files")
    parser.add_argument('--output_dir',
                        help="Root directory for output data")
    return parser.parse_args()


def main():
    args = get_args()
    data_dir_path = Path(args.data_dir)
    pdf_dir_path = Path(args.pdf_dir)
    output_dir_path = Path(args.output_dir)

    generate_images(data_dir_path, pdf_dir_path, output_dir_path)
    print("Completed!")


if __name__ == '__main__':
    main()
