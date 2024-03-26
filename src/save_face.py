from model import Model
import argparse
import cv2

def parse_args() :
    parser = argparse.ArgumentParser()

    parser.add_argument('name')
    parser.add_argument('file')
    parser.add_argument('-d', '--saved_faces_dir', default='../saved_faces')
    parser.add_argument('-s', '--save_image', choices=[True, False], default=True)

    return parser.parse_args()

def main() :
    args = parse_args()

    model = Model()
    model.load_mtcnn()
    model.load_resnet()

    img = cv2.imread(args.file)
    model.detect_save_face_embedding(name=args.name, img=img, save_directory=args.saved_faces_dir, save_image=args.save_image)

if __name__ == '__main__' :
    main()