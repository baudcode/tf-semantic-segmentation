from PIL import Image, ImageFont, ImageDraw
from timeit import default_timer as timer

from ..visualizations import masks
from ..processing import dataset as pre_dataset
import cv2
import numpy as np
import tqdm


def predict_video(model, video_path, stream=True, output_path=None, resize_method='resize_with_pad'):

    size = tuple(model.input.shape[1:3])
    depth = model.input.shape[-1]
    color_mode = pre_dataset.ColorMode.GRAY if depth == 1 else pre_dataset.ColorMode.RGB

    vid = cv2.VideoCapture(video_path)
    if not vid.isOpened():
        raise IOError("Couldn't open webcam or video")

    if output_path:
        video_FourCC = cv2.VideoWriter_fourcc(*"mp4v")
        video_fps = vid.get(cv2.CAP_PROP_FPS)
        video_size = (int(vid.get(cv2.CAP_PROP_FRAME_WIDTH)),
                      int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT)))

        out = cv2.VideoWriter(output_path, video_FourCC, video_fps, video_size)

    accum_time = 0
    curr_fps = 0
    fps = "FPS: ??"
    prev_time = timer()
    with tqdm.tqdm(desc="Predict on Video") as tq:

        return_value, frame = vid.read()
        while return_value:
            # read the frame
            image = frame / 255.
            image, _ = pre_dataset.resize_and_change_color(image, None, size, color_mode, resize_method=resize_method)
            # get detections
            images = np.expand_dims(image, axis=0)
            p = model.predict(images)

            num_classes = p.shape[-1] if p.shape[-1] > 1 else 2
            result = masks.get_colored_segmentation_mask(p, num_classes, images=images)[0]

            # calc fps
            curr_time = timer()
            exec_time = curr_time - prev_time
            prev_time = curr_time
            accum_time = accum_time + exec_time
            curr_fps = curr_fps + 1

            if accum_time > 1:
                accum_time = accum_time - 1
                fps = "FPS: " + str(curr_fps)
                curr_fps = 0

            if output_path:
                out.write(result)

            if stream:
                # put fps
                cv2.putText(result, text=fps, org=(3, 15), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                            fontScale=0.50, color=(255, 0, 0), thickness=2)
                cv2.namedWindow("result", cv2.WINDOW_NORMAL)
                cv2.imshow("result", result)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            tq.update(1)
            return_value, frame = vid.read()


if __name__ == "__main__":
    from tensorflow.keras.models import load_model
    model_path = "/home/baudcode/Code/tf-semantic-segmentation/logs/unet-v2-tacobinary-generator-ranger-1e-4-bce_dice/model-best.h5"
    model = load_model(model_path, compile=False)

    video_path = '../../dwhelper/VideoClip.mp4'
    predict_video(model, 2, video_path, output_path="")
