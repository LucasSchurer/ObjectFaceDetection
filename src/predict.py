from model import Model
import argparse
import os
import cv2
from datetime import datetime
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("input", help="image (png, jpg) or video (mp4) to be processed")
    parser.add_argument(
        "-y",
        "--yolo_model",
        default="../models/yolov8n.pt",
        help="yolo model to be loaded",
    )
    parser.add_argument("-o", "--output", help="processed input save path")
    parser.add_argument(
        "-nf",
        "--n_frames",
        default=None,
        help="number of frames to be processed from video",
        type=int,
    )

    parser.add_argument(
        "-do", "--detect_objects", action="store_true", help="detect_objects"
    )
    parser.add_argument(
        "-df", "--detect_faces", action="store_true", help="detect faces"
    )
    parser.add_argument(
        "-rf", "--recognize_faces", action="store_true", help="recognize faces"
    )

    return parser.parse_args()


def main():
    args = parse_args()

    if args.output is None:
        output_path = get_output_path(args.input)
    else:
        output_path = args.output

    model = Model()
    model.load(yolo_model=args.yolo_model)

    input_type = os.path.basename(args.input).split(".")[1]
    if input_type in ["png", "jpg"]:
        img = cv2.imread(args.input)

        final_img = detect_image(
            model,
            img,
            detect_objects=args.detect_objects,
            detect_faces=args.detect_faces,
            recognize_faces=args.recognize_faces,
        )

        cv2.imwrite(output_path, final_img)
    elif input_type in ["mp4"]:
        reader = cv2.VideoCapture(args.input)

        detect_video(
            model,
            reader,
            output_path,
            n_frames=args.n_frames,
            detect_objects=args.detect_objects,
            detect_faces=args.detect_faces,
            recognize_faces=args.recognize_faces,
        )
    else:
        print("Unsupported input type")

    return


def detect_image(
    model: Model,
    img: np.ndarray,
    detect_objects: bool = True,
    detect_faces: bool = True,
    recognize_faces: bool = True,
) -> np.ndarray:
    final_img = img

    if detect_objects:
        final_img = model.detect_objects(final_img)

    if detect_faces or recognize_faces:
        df_img, faces, boxes, _ = model.detect_faces(final_img, draw_confidence=True)

        if detect_faces:
            final_img = df_img

        if recognize_faces:
            final_img, _, _ = model.recognize_faces(
                final_img, faces, boxes, match_type="all"
            )

    return final_img


def detect_video(
    model: Model,
    reader: cv2.VideoCapture,
    output_path: str,
    n_frames: int = None,
    detect_objects: bool = True,
    detect_faces: bool = True,
    recognize_faces: bool = True,
):
    w = int(reader.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(reader.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = reader.get(cv2.CAP_PROP_FPS)
    four_cc = cv2.VideoWriter_fourcc(*"mp4v")

    writer = cv2.VideoWriter(output_path, four_cc, fps, (w, h))

    ret, frame = reader.read()

    curr_frame = 0
    if n_frames is not None:
        total_frames = n_frames
    else:
        total_frames = reader.get(cv2.CAP_PROP_FRAME_COUNT)

    while ret:
        if curr_frame > total_frames:
            break

        print(f"Processing {curr_frame}/{total_frames}", end="\r")

        img = detect_image(
            model,
            frame,
            detect_objects=detect_objects,
            detect_faces=detect_faces,
            recognize_faces=recognize_faces,
        )

        writer.write(img)
        ret, frame = reader.read()
        curr_frame += 1


def get_output_path(filename: str):
    output_filename = datetime.now().strftime("%Y-%m-%d_%H-%M_") + os.path.basename(
        filename
    )
    output_directory = "../saved"
    output_path = os.path.join(output_directory, output_filename)

    return output_path


if __name__ == "__main__":
    main()
