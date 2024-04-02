from model import Model
import argparse
import os
import cv2
from datetime import datetime
import numpy as np
from dotenv import load_dotenv


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("input", help="image (png, jpg) or video (mp4) to be processed")
    parser.add_argument("-o", "--output", help="path to processed file")
    parser.add_argument(
        "-y",
        "--yolo_model",
        default="../models/yolov8n.pt",
        help="path or yolo model name used for object detection",
    )
    parser.add_argument(
        "-nf",
        "--n_frames",
        default=None,
        help="number of frames to process from video",
        type=int,
    )

    # Object Detection Args

    parser.add_argument(
        "-do",
        "--detect_objects",
        action="store_true",
        help="if object detection should be used",
    )
    parser.add_argument(
        "-dot",
        "--object_detection_threshold",
        type=float,
        default=0.7,
        help="won't use objects which prediction confidence score are below the threshold",
    )
    parser.add_argument(
        "-docls",
        "--object_detection_draw_classification",
        action="store_true",
        help="if object classification label should be drawn",
    )
    parser.add_argument(
        "-doconf",
        "--object_detection_draw_confidence",
        action="store_true",
        help="if object classification confidence should be drawn",
    )

    # Face Detection Args

    parser.add_argument(
        "-df",
        "--detect_faces",
        action="store_true",
        help="if face detection should be used",
    )

    parser.add_argument(
        "-dft",
        "--face_detection_threshold",
        type=float,
        default=0.7,
        help="won't use faces which prediction confidence score are below the treshhold",
    )

    parser.add_argument(
        "-dfconf",
        "--face_detection_draw_confidence",
        action="store_true",
        help="if face detection confidence should be drawn",
    )

    # Face Recognition Args

    parser.add_argument(
        "-rf",
        "--recognize_faces",
        action="store_true",
        help="if face recognition should be used",
    )

    parser.add_argument(
        "-rft",
        "--face_recognition_threshold",
        type=float,
        default=0.7,
        help="threshold used to determine if a face match another face. Higher values will produce less accurate matches.",
    )

    parser.add_argument(
        "-rfmt",
        "--face_recognition_match_type",
        type=str,
        choices=["all", "best", "first"],
        default="best",
        help="",
    )

    return parser.parse_args()


def main():
    load_dotenv()
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
            object_detection_threshold=args.object_detection_threshold,
            object_detection_draw_classification=args.object_detection_draw_classification,
            object_detection_draw_confidence=args.object_detection_draw_confidence,
            detect_faces=args.detect_faces,
            face_detection_threshold=args.face_detection_threshold,
            face_detection_draw_confidence=args.face_detection_draw_confidence,
            recognize_faces=args.recognize_faces,
            face_recognition_threshold=args.face_recognition_threshold,
            face_recognition_match_type=args.face_recognition_match_type,
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
    object_detection_threshold: float = 0.7,
    object_detection_draw_classification: bool = True,
    object_detection_draw_confidence: bool = True,
    detect_faces: bool = True,
    face_detection_threshold: float = 0.7,
    face_detection_draw_confidence: bool = True,
    recognize_faces: bool = True,
    face_recognition_match_type: str = "all",
    face_recognition_threshold: float = 0.7,
) -> np.ndarray:
    final_img = img

    if detect_objects:
        final_img = model.detect_objects(
            final_img,
            threshold=object_detection_threshold,
            draw_classification=object_detection_draw_classification,
            draw_confidence=object_detection_draw_confidence,
        )

    if detect_faces or recognize_faces:
        df_img, faces, boxes, _ = model.detect_faces(
            final_img,
            threshold=face_detection_threshold,
            draw_confidence=face_detection_draw_confidence,
        )

        if detect_faces:
            final_img = df_img

        if recognize_faces:
            final_img, _, _ = model.recognize_faces(
                final_img,
                faces,
                boxes,
                match_type=face_recognition_match_type,
                match_threshold=face_recognition_threshold,
            )

    return final_img


def detect_video(
    model: Model,
    reader: cv2.VideoCapture,
    output_path: str,
    n_frames: int = None,
    detect_objects: bool = True,
    object_detection_threshold: float = 0.7,
    object_detection_draw_classification: bool = True,
    object_detection_draw_confidence: bool = True,
    detect_faces: bool = True,
    face_detection_threshold: float = 0.7,
    face_detection_draw_confidence: bool = True,
    recognize_faces: bool = True,
    face_recognition_match_type: str = "all",
    face_recognition_threshold: float = 0.7,
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
            object_detection_threshold=object_detection_threshold,
            object_detection_draw_classification=object_detection_draw_classification,
            object_detection_draw_confidence=object_detection_draw_confidence,
            detect_faces=detect_faces,
            face_detection_threshold=face_detection_threshold,
            face_detection_draw_confidence=face_detection_draw_confidence,
            recognize_faces=recognize_faces,
            face_recognition_threshold=face_recognition_threshold,
            face_recognition_match_type=face_recognition_match_type,
        )

        writer.write(img)
        ret, frame = reader.read()
        curr_frame += 1


def get_output_path(filename: str):
    output_filename = datetime.now().strftime("%Y-%m-%d_%H-%M_") + os.path.basename(
        filename
    )
    output_directory = os.environ.get("PREDICTION_OUTPUT_DIR")
    output_path = os.path.join(output_directory, output_filename)

    return output_path


if __name__ == "__main__":
    main()
