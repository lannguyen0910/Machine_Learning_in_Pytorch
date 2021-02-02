import numpy as np


def iou_compute(predict, truth):
    x1, y1, x_w1, y_h1 = predict[0], predict[1], predict[0] + \
        predict[2], predict[1] + predict[3]
    x2, y2, x_w2, y_h2 = truth[0], truth[1], truth[0] + \
        truth[2], truth[1] + truth[3]

    # overlap on Ox, Oy
    overlap_x = max(0, min(x_w1, x_w2) - max(x1, x2))
    overlap_y = max(0, min(y_h1, y_h2) - max(y2, y1))
    overlap_area = overlap_x * overlap_y

    total_area = predict[2] * predict[3] + \
        truth[2] * truth[3] - overlap_area + 1e-6

    return np.round(overlap_area / float(total_area), 3)
