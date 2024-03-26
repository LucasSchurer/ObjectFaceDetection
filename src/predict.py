from model import Model
import argparse
import os
import cv2
from datetime import datetime

def parse_args() :
    parser = argparse.ArgumentParser()

    parser.add_argument('input', help='image (png, jpg) or video (mp4) to be processed')
    parser.add_argument('-y', '--yolo_model', default='../models/yolov8n.pt', help='yolo model to be loaded')
    parser.add_argument('-o', '--output', default=None, help='processed input save path')
    parser.add_argument('-nf', '--n_frames', default=None, help='number of frames to be processed from video', type=int)

    parser.add_argument('-xf', '--dont_detect_faces', action='store_true', default=False, help='exclude face detection')

    return parser.parse_args()

def main() :
    args = parse_args()

    model = Model()

    model.load(yolo_model=args.yolo_model)

    input_type = os.path.basename(args.input).split('.')[1]

    if input_type in ['png', 'jpg'] :
        predict_image(model=model, img_path=args.input, output_path=args.output, detect_faces=not args.dont_detect_faces)
    elif input_type in ['mp4'] :
        predict_video(model=model, video_path=args.input, output_path=args.output, n_frames=args.n_frames, detect_faces=not args.dont_detect_faces)
    else :
        print('Unsupported input type')

    return

def predict_video(model : Model, video_path : str, output_path : str, n_frames : int = None, detect_faces : bool = True) :
    video = cv2.VideoCapture(video_path)

    x_shape = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    y_shape = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = video.get(cv2.CAP_PROP_FPS)
    four_cc = cv2.VideoWriter_fourcc(*"mp4v")
    
    if output_path is None :
        output_path = get_output_path(video_path)

    out = cv2.VideoWriter(output_path, four_cc, fps, (x_shape, y_shape)) 

    ret, frame = video.read()    

    i = 0
    while ret :
        if n_frames is not None :
            if i > n_frames :
                break

        if detect_faces :
            result = model.detect_all(frame)
        else :
            result = model.detect_objects(frame).plot()

        out.write(result)
        ret, frame = video.read()
        i += 1

def predict_image(model : Model, img_path : str, output_path : str, detect_faces : bool = True) :
    img = cv2.imread(img_path)

    if detect_faces :
        result = model.detect_all(img)
    else :
        result = model.detect_objects(img).plot()

    if output_path is None :
        output_path = get_output_path(img_path)

    cv2.imwrite(output_path, result)

def get_output_path(filename : str) :
    output_filename = datetime.now().strftime('%Y-%m-%d_%H-%M_') + os.path.basename(filename)
    output_directory = '../saved'
    output_path =  os.path.join(output_directory, output_filename)

    return output_path

if __name__ == '__main__' :
    main()