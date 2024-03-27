from ultralytics import YOLO
from facenet_pytorch import InceptionResnetV1, MTCNN
from facenet_pytorch.models.utils.detect_face import extract_face
import os
import cv2
import numpy as np
from torch import Tensor


class Model:
    def __init__(self) -> None:
        self.yolo = None
        self.mtcnn = None
        self.resnet = None
        self.saved_faces = []

    def load(self, yolo_model="yolov8n.pt"):
        self.load_yolo(model=yolo_model)
        self.load_mtcnn()
        self.load_resnet()
        self.load_face_embeddings()

    def load_mtcnn(self):
        self.mtcnn = MTCNN()

    def load_resnet(self):
        self.resnet = InceptionResnetV1(pretrained="vggface2").eval()

    def load_yolo(self, model="yolov8n.pt"):
        self.yolo = YOLO(model)

    def load_face_embeddings(
        self, saved_faces_embeddings_dir="../saved_faces/embeddings"
    ):
        files = os.listdir(saved_faces_embeddings_dir)

        for file in files:
            name = [str.title(s) for s in file.split(".")[0].split("_")]

            name = " ".join(name)

            embedding = np.load(f"{saved_faces_embeddings_dir}/{file}")
            self.saved_faces.append((name, embedding))

    def recognize_save_face(
        self,
        name: str,
        img: np.ndarray,
        save_directory="../saved_faces",
        save_image=True,
    ):
        face, box, _, _ = self.detect_faces(img)

        if box is None or len(box) == 0:
            print("No face was detected.")
            return
        elif len(box) > 1:
            print("Won't save face because multiple faces were detected.")
            return

        face = extract_face(img, box[0], 160)

        embedding = self.to_embedding(face)
        self.saved_faces.append((name, embedding))

        np.save(f"{save_directory}/embeddings/{name}", embedding)

        cv2.imwrite(
            f"{save_directory}/images/{name}.png",
            self.extract_image(img, [int(coordinate) for coordinate in box[0]]),
        )

    def detect_objects(
        self,
        img: np.ndarray,
        confidence_threshold: float = 0.7,
        plot_classification: bool = True,
        plot_confidence: bool = True,
    ):
        result = self.yolo.predict(img)[0]

        final_img = img
        size = len(result.boxes)

        for i in range(size):
            box = [int(c) for c in result.boxes[i].xyxy[0].tolist()]
            conf = result.boxes.conf[i].item()
            cls_index = int(result.boxes.cls[i])
            cls_name = result.names[cls_index]

            if conf < confidence_threshold:
                continue

            if plot_classification:
                self.draw_label(final_img, cls_name, (box[0], box[1] - 5))

            if plot_confidence:
                self.draw_label(final_img, f"{conf:.3f}", (box[0], box[1] - 30))

            # Draw the bounding box
            self.draw_box(final_img, box)

        return final_img

    def detect_faces(
        self,
        img: np.ndarray,
        draw_confidence: bool = False,
        draw_landmark: bool = False,
        draw_offset: tuple[int, int, int, int] = (0, 0, 0, 0),
    ) -> tuple[
        np.ndarray, np.ndarray[np.ndarray], np.ndarray[float], np.ndarray[np.ndarray]
    ]:
        """Detects all faces in a given image.

        Args:
            img (np.ndarray): Image in which faces will be detected.
            draw_confidence (bool, optional): Draw the confidence score above the detected face bounding box. Defaults to False.
            draw_landmark (bool, optional): _description_. Defaults to False.
            draw_offset (tuple[int, int, int, int], optional): Offset applied when drawing all elements. Defaults to (0, 0, 0, 0).

        Returns:
            tuple[np.ndarray, np.ndarray[np.ndarray], np.ndarray[float], np.ndarray[np.ndarray]]: A tuple containing the image generated after face detection, the bounding boxes, confidence scores, and landmarks of each face detected.
        """
        conf_height_offset = -5

        boxes, confs, landmarks = self.mtcnn.detect(
            img,
            landmarks=True,
        )

        if boxes is None:
            return img, None, None, None

        final_img = img

        # Draw bounding boxes, confidence scores and landmarks
        for box, conf, landmark in list(zip(boxes, confs, landmarks)):
            box = [int(box[i] + draw_offset[i]) for i in range(len(box))]

            self.draw_box(final_img, box)

            if draw_confidence:
                self.draw_label(
                    final_img, f"{conf:.3f}", (box[0], box[1] + conf_height_offset)
                )

            # TODO
            if draw_landmark:
                break

        return final_img, boxes, confs, landmarks

    def recognize_faces(
        self,
        img: np.ndarray,
        face_bounding_boxes: np.ndarray[np.ndarray],
        match_type: str = "best",
        match_threshold: float = 0.7,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Recognizes faces contained in a given image using the bounding boxes provided.

        Args:
            img (np.ndarray): Image in which faces will be recognized.
            face_bounding_boxes (np.ndarray[np.ndarray]): Bounding boxes of the faces to be recognized.
            match_type (str, optional): Type of recognition match.
                first: Stop and return the first match found. It's dependent on the order in which the saved faces are loaded.
                best: Among all matches, return the one with the smallest distance.
                all: Return all matches.
            match_threshold (float, optional): Minimum distance required to consider a match between two faces. Defaults to 0.7.

        Returns:
            tuple[np.ndarray, np.ndarray, np.ndarray]: A tuple containing the image after recognition, recognized names, and confidence scores.
        """

        names = []
        confs = []
        final_image = img

        for face_bounding_box in face_bounding_boxes:
            face_bounding_box = list(map(int, face_bounding_box))

            extracted_face = extract_face(img, face_bounding_box, 160)
            matches = self.face_recognition(
                extracted_face, match_type=match_type, match_threshold=match_threshold
            )

            for i in range(len(matches)):

                names.append(matches[i][0])
                confs.append(matches[i][1])

                self.draw_label(
                    final_image,
                    f"{matches[i][0]} {matches[i][1]:.3f}",
                    (face_bounding_box[0], face_bounding_box[1] - 10 - (i * 20)),
                )

        return final_image, names, confs

    def face_recognition(
        self,
        face: np.ndarray | Tensor,
        match_type: str = "best",
        match_threshold: float = 0.7,
    ) -> np.ndarray[tuple[str, float]]:
        """Try to find a saved face that matches the passed face.

        Args:
            face (np.ndarray | Tensor): Face used to find a match.
            match_type (str, optional): Type of recognition match.
                first: Stop and return the first match found. It's dependent on the order in which the saved faces are loaded.
                best: Among all matches, return the one with the smallest distance.
                all: Return all matches.
            match_threshold (float, optional): Minimum distance required to consider a match between two faces. Defaults to 0.7.

        Returns:
            np.ndarray[tuple[str, float]]: Returns an array of (name, distance) for all matches. Returns ("Unknown", 0) if no match was found or ("NoFace", 0) if the face provided is None.
        """

        if face is None:
            return [("NoFace", None)]

        matches = []

        if type(face) == Tensor:
            x1_tensor = face
        else:
            x1_tensor = self.img_to_tensor(face)

        x1_embedding = self.to_embedding(x1_tensor)

        for x2_name, x2_embedding in self.saved_faces:
            distance = np.linalg.norm(x1_embedding - x2_embedding)

            if distance < match_threshold:
                if match_type == "first":
                    return [(x2_name, distance)]
                else:
                    matches.append((x2_name, distance))

        if matches:
            if match_type == "all":
                return matches
            else:
                best = matches[0]

                for match in matches:
                    if match[1] < best[1]:
                        best = match

                return [(best[0], best[1])]
        else:
            return [("Unknown", None)]

    def draw_label(
        self,
        img: np.ndarray,
        text: str,
        coordinates: tuple[int, int],
    ) -> None:
        padding = 2
        fontFace = cv2.FONT_HERSHEY_SIMPLEX
        fontScale = 0.6
        textColor = (255, 255, 255)
        backgroundColor = (0, 0, 0)
        thickness = 1

        (w, h), _ = cv2.getTextSize(text, fontFace, fontScale, thickness=thickness)

        cv2.rectangle(
            img,
            (coordinates[0] - padding, coordinates[1] + padding),
            (coordinates[0] + w + padding, coordinates[1] - h - padding),
            backgroundColor,
            -1,
        )

        cv2.putText(
            img,
            text,
            (coordinates[0], coordinates[1]),
            fontFace=fontFace,
            fontScale=fontScale,
            color=textColor,
            thickness=thickness,
            lineType=cv2.LINE_AA,
        )

    def draw_box(self, img: np.ndarray, box: np.ndarray) -> None:
        color = (0, 0, 255)
        thickness = 1

        cv2.rectangle(
            img,
            (box[0], box[1]),
            (box[2], box[3]),
            color=color,
            thickness=thickness,
        )

    def img_to_tensor(self, img: np.ndarray) -> Tensor:
        tensor = Tensor(img.transpose(2, 0, 1))

        return tensor

    def extract_image(self, img: np.ndarray, box: np.array) -> np.ndarray:
        extracted_image = img[box[1] : box[3], box[0] : box[2]]
        return extracted_image

    def to_embedding(self, img_tensor: Tensor) -> np.ndarray:
        embedding = self.resnet(img_tensor.unsqueeze(0)).detach().numpy()

        return embedding