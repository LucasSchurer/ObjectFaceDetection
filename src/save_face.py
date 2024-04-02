from model import Model
import argparse
import cv2
from dotenv import load_dotenv


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("name")
    parser.add_argument("file")

    return parser.parse_args()


def main():
    load_dotenv()
    args = parse_args()

    model = Model()
    model.load_mtcnn()
    model.load_resnet()

    img = cv2.imread(args.file)
    model.recognize_save_face(name=args.name, img=img)


if __name__ == "__main__":
    main()
