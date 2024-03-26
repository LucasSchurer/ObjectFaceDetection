from ultralytics import YOLO
from ultralytics.engine.results import Results
from facenet_pytorch import InceptionResnetV1, MTCNN
import os
import cv2
import numpy as np
from torch import Tensor

class Model() :
    def __init__(self) -> None:
        self.yolo = None
        self.mtcnn = None
        self.resnet = None
        self.saved_faces = []

    def load(self, yolo_model = 'yolov8n.pt') :
        self.load_yolo(model=yolo_model)
        self.load_mtcnn()
        self.load_resnet()
        self.load_face_embeddings()

    def load_mtcnn(self) :
        self.mtcnn = MTCNN()

    def load_resnet(self) :
        self.resnet = InceptionResnetV1(pretrained='vggface2').eval()

    def load_yolo(self, model = 'yolov8n.pt') :
        self.yolo = YOLO(model)

    def load_face_embeddings(self, saved_faces_embeddings_dir = '../saved_faces/embeddings') :
        files = os.listdir(saved_faces_embeddings_dir)

        for file in files :
            name = [str.title(s) for s in file.split('.')[0].split('_')]

            name = ' '.join(name)

            embedding = self.load_face_embedding(f'{saved_faces_embeddings_dir}/{file}')
            self.saved_faces.append((name, embedding))
    
    def load_face_embedding(self, file) -> np.ndarray :
        face_embedding = np.load(file)

        return face_embedding
    
    def detect_save_face_embedding(self, name : str, img: np.ndarray, save_directory = '../saved_faces', save_image = True) :
        face = self.mtcnn(img)
        embedding = self.to_embedding(face)
        self.saved_faces.append((name, embedding))

        np.save(f'{save_directory}/embeddings/{name}', embedding)
        
        if save_image : 
            cv2.imwrite(f'{save_directory}/images/{name}.png', img)        

    def detect_all(self, img: np.ndarray, face_match_threshold = 0.7, plot_classification = True, plot_object_confidence = True, plot_face_distance = True, stop_first_match = True, show_only_best_match = True, object_confidence_threshold = 0.0) -> np.ndarray :
        result = self.detect_objects(img)
        plotted_img = img
        size = len(result.boxes)

        for i in range(size) :
            xyxy = [int(c) for c in result.boxes[i].xyxy[0].tolist()]
            conf = result.boxes.conf[i].item()
            cls_index = int(result.boxes.cls[i])
            cls_name = result.names[cls_index]

            if conf < object_confidence_threshold :
                continue

            # If is a person try to find and plot the face
            if cls_index == 0 :
                plotted_img = self.find_plot_face(img=img, xyxy=xyxy, match_threshold=face_match_threshold, plot_face_distance=plot_face_distance, stop_first_match=stop_first_match, show_only_best_match=show_only_best_match)
            
            plotted_img = self.plot_object(img=img, xyxy=xyxy, cls_name=cls_name, conf=conf, plot_classification=plot_classification, plot_confidence=plot_object_confidence)

        return plotted_img 

    def find_plot_face(self, img : np.ndarray, xyxy : np.array, match_threshold=0.7, plot_face_distance = True, stop_first_match = True, show_only_best_match = True) -> np.ndarray :
        face = self.extract_image(img=img, xyxy=xyxy)

        matches = self.find_face_match_img(img=face, match_threshold=match_threshold, stop_first_match=stop_first_match)
        best_match = matches[0]

        if show_only_best_match and best_match[1] is not None :
            for match in matches :
                if match[1] < best_match[1] :
                    best_match = match
            
            matches = [best_match]

        boxes, _ = self.mtcnn.detect(img=face)
        
        plotted_image = img

        if boxes is not None :
            face_xyxy = boxes[0]
            face_xyxy = [int(c) for c in face_xyxy]

            # Adjust face_xyxy to be correctly positioned into the original image
            face_xyxy[0] += xyxy[0]
            face_xyxy[1] += xyxy[1]
            face_xyxy[2] += xyxy[0]
            face_xyxy[3] += xyxy[1]

            cv2.rectangle(plotted_image, 
                          (face_xyxy[0], face_xyxy[1]), (face_xyxy[2], face_xyxy[3]), 
                          color=(0, 255, 0), thickness=2)

            # Iterator to adjust face name/distance to prevent overlaps
            j = 0
            jx = 80
            for name, distance in matches :
                if plot_face_distance and distance is not None :
                    cv2.putText(plotted_image, f'{distance:.3f}', 
                                (face_xyxy[0] + (jx * j), face_xyxy[1] - 20), 
                                fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, 
                                color=(0, 0, 255), thickness=1, lineType=cv2.LINE_AA)

                cv2.putText(plotted_image, name, 
                            (face_xyxy[0] + (jx * j), face_xyxy[1] - 5), 
                            fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, 
                            color=(0, 0, 255), thickness=1, lineType=cv2.LINE_AA)
                
                j += 1

        return plotted_image
    
    def plot_object(self, img : np.ndarray, xyxy : np.array, cls_name: str, conf : float, plot_classification = True, plot_confidence = True) -> np.ndarray :
        plotted_img = img

        if plot_classification :
            cv2.putText(plotted_img, cls_name, (xyxy[0], xyxy[1] - 5), 
                            fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, 
                            color=(255, 0, 0), thickness=1, lineType=cv2.LINE_AA)
            
        if plot_confidence :
            cv2.putText(plotted_img, f'{conf:.3f}', (xyxy[0], xyxy[1] - 20), 
                            fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, 
                            color=(255, 0, 0), thickness=1, lineType=cv2.LINE_AA)
        
        cv2.rectangle(plotted_img, (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]), color=(255, 0, 0), thickness=2)

        return plotted_img

    def detect_objects(self, img : np.ndarray) -> Results :
        result = self.yolo.predict(img)[0]
        return result    

    def face_match_img(self, x1: np.ndarray, x2: np.ndarray, match_threshold = 0.7) :
        x1_tensor = self.img_to_tensor(x1)
        x2_tensor = self.img_to_tensor(x2)

        return self.face_match_tensor(x1=x1_tensor, x2=x2_tensor)

    def face_match_tensor(self, x1: Tensor, x2: Tensor, match_threshold = 0.7) :
        x1_embedding = self.to_embedding(x1)
        x2_embedding = self.to_embedding(x2)

        return self.face_match_embedding(x1=x1_embedding, x2=x2_embedding, match_threshold=match_threshold)
    
    def face_match_embedding(self, x1: np.ndarray, x2: np.ndarray, match_threshold = 0.7) -> tuple[bool, float] :
        distance = np.linalg.norm(x1 - x2)

        return distance < match_threshold, distance

    def find_face_match_img(self, img: np.ndarray, match_threshold = 0.7, stop_first_match = True) -> tuple[bool, float] :
        face = self.mtcnn(img)

        return self.find_face_match_tensor(img=face, match_threshold=match_threshold, stop_first_match=stop_first_match)

    def find_face_match_tensor(self, img: Tensor, match_threshold = 0.7, stop_first_match = True ) -> tuple[bool, float] :
        if img is None :
            return [('NoFace', None)]
        
        x1_embedding = self.to_embedding(img=img)

        matches = []

        for name, x2_embedding in self.saved_faces :
            result, distance = self.face_match_embedding(x1=x1_embedding, x2=x2_embedding, match_threshold=match_threshold)

            if result :
                if stop_first_match :
                    return [(name, distance)]
                else :
                    matches.append((name, distance))
        
        if matches :
            return matches
        else :
            return [('Unknown', None)]

    def extract_image(self, img: np.ndarray, xyxy: np.array) -> np.ndarray : 
        extracted_image = img[xyxy[1] : xyxy[3], xyxy[0] : xyxy[2]]
        return extracted_image

    def img_to_tensor(self, img : np.ndarray) -> Tensor :
        tensor = Tensor(img.transpose(2, 0, 1))
        
        return tensor

    def to_embedding(self, img: Tensor) -> np.ndarray : 
        embedding = self.resnet(img.unsqueeze(0)).detach().numpy()

        return embedding
        
        
