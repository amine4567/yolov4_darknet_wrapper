from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from matplotlib.cm import get_cmap
from PIL import Image, ImageDraw, ImageFile, ImageFont

COLORMAP = get_cmap("rainbow")


def apply_nms(
    boxes: np.ndarray,
    confs: np.ndarray,
    nms_thresh: float = 0.5,
    min_mode: bool = False,
) -> np.ndarray:
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    areas = (x2 - x1) * (y2 - y1)
    order = confs.argsort()[::-1]

    keep = []
    while order.size > 0:
        idx_self = order[0]
        idx_other = order[1:]

        keep.append(idx_self)

        xx1 = np.maximum(x1[idx_self], x1[idx_other])
        yy1 = np.maximum(y1[idx_self], y1[idx_other])
        xx2 = np.minimum(x2[idx_self], x2[idx_other])
        yy2 = np.minimum(y2[idx_self], y2[idx_other])

        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h

        if min_mode:
            over = inter / np.minimum(areas[order[0]], areas[order[1:]])
        else:
            over = inter / (areas[order[0]] + areas[order[1:]] - inter)

        inds = np.where(over <= nms_thresh)[0]
        order = order[inds + 1]

    return np.array(keep)


def process_model_output(
    output: Tuple[np.ndarray, np.ndarray],
    conf_thresh: float,
    nms_thresh: float,
    classes_names: List[str],
) -> List[Dict]:
    box_array = output[0][:, :, 0]  # shape (batch_size, grid_cell_id, coordinate_id)
    confs = output[1]  # shape (batch_size, grid_cell_id, class_id)

    num_classes = confs.shape[2]

    max_conf = np.max(confs, axis=2)
    max_id = np.argmax(confs, axis=2)

    bboxes_batch = []
    for i in range(box_array.shape[0]):
        argwhere = max_conf[i] > conf_thresh
        l_box_array = box_array[i, argwhere, :]
        l_max_conf = max_conf[i, argwhere]
        l_max_id = max_id[i, argwhere]

        bboxes = []
        # nms (Non-maximum Suppression) is applied for each class
        for j in range(num_classes):
            cls_argwhere = l_max_id == j
            ll_box_array = l_box_array[cls_argwhere, :]
            ll_max_conf = l_max_conf[cls_argwhere]
            ll_max_id = l_max_id[cls_argwhere]

            keep = apply_nms(ll_box_array, ll_max_conf, nms_thresh)

            if keep.size > 0:
                ll_box_array = ll_box_array[keep, :]
                ll_max_conf = ll_max_conf[keep]
                ll_max_id = ll_max_id[keep]

                for k in range(ll_box_array.shape[0]):
                    bboxes.append(
                        [
                            ll_box_array[k, 0],
                            ll_box_array[k, 1],
                            ll_box_array[k, 2],
                            ll_box_array[k, 3],
                            ll_max_conf[k],
                            ll_max_conf[k],
                            ll_max_id[k],
                        ]
                    )

        bboxes_batch.append(bboxes)

    pred_results = [
        {
            "class_id": box[6],
            "normalized_class_id": box[6] / num_classes,
            "class_name": classes_names[box[6]],
            "confidence": box[5],
            "relative_coordinates": {
                "x1": box[0],
                "y1": box[1],
                "x2": box[2],
                "y2": box[3],
            },
        }
        for box in bboxes_batch[0]
    ]

    return pred_results


def add_bboxes_to_img(
    img_array: np.ndarray,
    pred_results: List[Dict],
    text_font: ImageFont = None,
    text_size: int = 12,
) -> ImageFile:
    if text_font is None:
        text_font = "DejaVuSansMono.ttf"

    font = ImageFont.truetype(text_font, text_size)

    img_pil = Image.fromarray(img_array)
    img_width, img_height = img_pil.size

    img_draw = ImageDraw.Draw(img_pil)

    for obj_data in pred_results:
        relative_coords = obj_data["relative_coordinates"]
        confidence = round(obj_data["confidence"], 2)
        label = f"{obj_data['class_name']}: {confidence:.2f}"

        x1 = round(relative_coords["x1"] * img_width)
        y1 = round(relative_coords["y1"] * img_height)
        x2 = round(relative_coords["x2"] * img_width)
        y2 = round(relative_coords["y2"] * img_height)

        color = np.rint(
            np.array(COLORMAP(obj_data["normalized_class_id"])) * 255
        ).astype(int)
        img_draw.rectangle(
            (x1, y1, x2, y2), outline=(color[2], color[1], color[0]), width=1
        )

        text_w, text_h = img_draw.textsize(label, font=font)

        if y1 < 0:
            text_y_high = 0
            text_y_low = text_h
        elif 0 <= y1 < text_h:
            text_y_high = y1
            text_y_low = y1 + text_h
        else:
            text_y_high = y1 - text_h
            text_y_low = y1

        img_draw.rectangle(
            (x1, text_y_high, x1 + text_w + 5, text_y_low),
            fill=(color[2], color[1], color[0], 200),
        )
        img_draw.text((x1 + 2, text_y_high), label, fill="black", font=font)

    return img_pil


def parse_metrics_results(metrics_filepath: str) -> Tuple[float, pd.DataFrame]:
    with open(metrics_filepath, "r") as fp:
        file_content = fp.readlines()
    file_content = [line.strip() for line in file_content]

    map_line = [
        line for line in file_content if line.startswith("mean average precision")
    ][0]
    ap_per_class_lines = [line for line in file_content if line.startswith("class_id")]

    map_score = float(map_line.split("=")[1].split(",")[0])

    ap_per_class_df = pd.DataFrame(
        [
            dict(
                [
                    list(map(lambda x: x.strip(), elt.split("=")))
                    for elt in line.replace("\t", ",")
                    .replace("(", "")
                    .replace(")", "")
                    .split(",")
                ]
            )
            for line in ap_per_class_lines
        ]
    )
    ap_per_class_df["class_id"] = ap_per_class_df["class_id"].astype(int)
    ap_per_class_df["TP"] = ap_per_class_df["TP"].astype(int)
    ap_per_class_df["FP"] = ap_per_class_df["FP"].astype(int)
    ap_per_class_df["ap"] = ap_per_class_df.ap.str.strip("%").astype(float) / 100

    ap_per_class_df.sort_values("class_id", ascending=True)
    return map_score, ap_per_class_df
