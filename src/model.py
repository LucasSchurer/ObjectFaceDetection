from ultralytics import YOLO
from facenet_pytorch import InceptionResnetV1, MTCNN
from facenet_pytorch.models.utils.detect_face import extract_face
import os
import cv2
import numpy as np
from torch import Tensor
from torch.cuda import set_device


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
        self.mtcnn = MTCNN(thresholds=[0.3, 0.4, 0.5], keep_all=True)

    def load_resnet(self):
        self.resnet = InceptionResnetV1(pretrained="vggface2").eval()

    def load_yolo(self, model="yolov8n.pt"):
        self.yolo = YOLO(model)

    def load_face_embeddings(self):
        embeddings_dir = os.environ.get("SAVE_FACES_EMBEDDING_DIR")

        files = os.listdir(embeddings_dir)

        for file in files:
            name = [str.title(s) for s in file.split(".")[0].split("_")]

            name = " ".join(name)

            embedding = np.load(f"{embeddings_dir}/{file}")
            self.saved_faces.append((name, embedding))

    def recognize_save_face(self, name: str, img: np.ndarray) -> bool:
        faces = self.mtcnn(img)

        if faces is None:
            print("No face was detected.")
            return False

        if len(faces) > 1:
            print("Won't save face because multiple faces were detected.")
            return False

        boxes, _ = self.mtcnn.detect(img)

        embedding = self.to_embedding(faces[0])
        self.saved_faces.append((name, embedding))

        np.save(f"{os.environ.get('SAVE_FACES_EMBEDDING_DIR')}/{name}", embedding)

        cv2.imwrite(
            f"{os.environ.get('SAVE_FACES_IMAGES_DIR')}/{name}.png",
            self.extract_image(img, [int(coordinate) for coordinate in boxes[0]]),
        )

    def detect_objects(
        self,
        img: np.ndarray,
        threshold: float = 0.7,
        draw_classification: bool = True,
        draw_confidence: bool = True,
    ):
        result = self.yolo.predict(img, conf=threshold)[0]

        final_img = img.copy()
        size = len(result.boxes)

        for i in range(size):
            box = [int(c) for c in result.boxes[i].xyxy[0].tolist()]
            conf = result.boxes.conf[i].item()
            cls_index = int(result.boxes.cls[i])
            cls_name = result.names[cls_index]

            if draw_classification:
                self.draw_label(final_img, cls_name, (box[0], box[1] - 5))

            if draw_confidence:
                self.draw_label(final_img, f"{conf:.3f}", (box[0], box[1] - 30))

            # Draw the bounding box
            self.draw_box(final_img, box)

        return final_img

    def detect_faces(
        self,
        img: np.ndarray,
        threshold: float = 0.7,
        draw_confidence: bool = False,
    ) -> tuple[np.ndarray, np.ndarray[Tensor], np.ndarray, np.ndarray[float]]:
        conf_height_offset = -5

        faces, confs = self.mtcnn(img, return_prob=True)
        final_img = img.copy()
        boxes = []

        if faces is not None:
            boxes, _ = self.mtcnn.detect(img)
            boxes = list(map(lambda box: [int(c) for c in box], boxes))

            for i, conf in enumerate(confs):
                if conf < threshold:
                    continue

                box = boxes[i]

                self.draw_box(final_img, box)

                if draw_confidence:
                    self.draw_label(
                        final_img, f"{conf:.3f}", (box[0], box[1] + conf_height_offset)
                    )

        return final_img, faces, boxes, confs

    def recognize_faces(
        self,
        img: np.ndarray,
        faces: Tensor,
        boxes: np.ndarray,
        match_type: str = "best",
        match_threshold: float = 0.7,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if faces is None:
            return img, [], []

        final_image = img.copy()

        names = []
        confs = []

        for i, face in enumerate(faces):
            matches = self.face_recognition(
                face, match_type=match_type, match_threshold=match_threshold
            )

            box = boxes[i]

            for i in range(len(matches)):
                names.append(matches[i][0])
                confs.append(matches[i][1])

                label = matches[i][0]

                if matches[i][1] is not None:
                    label += f" {matches[i][1]:.3f}"

                self.draw_label(
                    final_image,
                    label,
                    (box[0], box[1] - 10 - (i * 20)),
                )

        return final_image, names, confs

    def face_recognition(
        self,
        face_tensor: Tensor,
        match_type: str = "best",
        match_threshold: float = 0.7,
    ) -> np.ndarray[tuple[str, float]]:
        if face_tensor is None:
            return [("NoFace", None)]

        matches = []

        x1_embedding = self.to_embedding(face_tensor)

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

    def to_embedding(self, tensor: Tensor) -> np.ndarray:
        embedding = self.resnet(tensor.unsqueeze(0)).detach().numpy()

        return embedding
