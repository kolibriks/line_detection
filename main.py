"""Detect road lines using segmentation and Hough transformation"""


import cv2
import numpy as np
from scipy import stats


def hough_transform(canny_img,
                    rho=1,
                    theta=(np.pi / 180) * 1,
                    threshold=15,
                    min_line_len=20,
                    max_line_gap=10):
    return cv2.HoughLinesP(canny_img, rho, theta, threshold, np.array([]), minLineLength=min_line_len,
                           maxLineGap=max_line_gap)


def draw_lines(img, lines, color, thickness=5, make_copy=True):
    # copy the passed image
    img_copy = np.copy(img) if make_copy else img

    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(img_copy, (x1, y1), (x2, y2), color, thickness)
    return img_copy


def find_lane_lines_formula(lines):
    xs = []
    ys = []

    for line in lines:
        for x1, y1, x2, y2 in line:
            xs.append(x1)
            xs.append(x2)
            ys.append(y1)
            ys.append(y2)

    # calculate parameters of the lines using a linear least-squares regression
    A, b, r_value, p_value, std_err = stats.linregress(xs, ys)

    # y = Ax + b
    return A, b


def get_vertices_for_img():
    region_bottom_left = (200, 680)
    region_top_left = (600, 450)
    region_top_right = (750, 450)
    region_bottom_right = (1100, 650)
    vert = np.array([[region_bottom_left, region_top_left, region_top_right, region_bottom_right]], dtype=np.int32)

    return vert


def region_of_interest(img):
    # define a blank mask to start with
    mask = np.zeros_like(img)

    # define a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255

    vert = get_vertices_for_img()

    # fill pixels inside the polygon defined by "vertices" with the fill color
    cv2.fillPoly(mask, vert, ignore_mask_color)

    # return the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


def trace_lane_line(img, lines, top_y, make_copy=True):
    A, b = find_lane_lines_formula(lines)

    bottom_y = img.shape[0] - 1
    # y = Ax + b, therefore x = (y - b) / A
    x_to_bottom_y = (bottom_y - b) / A

    top_x_to_y = (top_y - b) / A

    new_lines = [[[int(x_to_bottom_y), int(bottom_y), int(top_x_to_y), int(top_y)]]]
    return draw_lines(img, new_lines, color=[0, 255, 0], make_copy=make_copy)


def trace_both_lane_lines(img, left_lane_lines, right_lane_lines):
    height = img.shape[0]

    # region_top_left[1]
    full_left_lane_img = trace_lane_line(img, left_lane_lines, height*10//17, make_copy=True)
    full_left_right_lanes_img = trace_lane_line(full_left_lane_img, right_lane_lines, height*10//17, make_copy=False)

    img_with_lane_weight = cv2.addWeighted(img, 0.7, full_left_right_lanes_img, 0.3, 0.0)

    return img_with_lane_weight


def separate_lines(lines, img):
    img_shape = img.shape

    middle_x = img_shape[1] / 2

    left_lane_lines = []
    right_lane_lines = []

    for line in lines:
        for x1, y1, x2, y2 in line:
            dx = x2 - x1
            dy = y2 - y1
            if dx == 0 or dy == 0:
                # gradient is undefined at this dx and dy
                continue

            slope = dy / dx
            # lines with a small slope are likely to be horizontal
            epsilon = 0.1
            if abs(slope) <= epsilon:
                continue
            if slope < 0 and x1 < middle_x and x2 < middle_x:
                # line should also be within the left hand side of region of interest
                left_lane_lines.append([[x1, y1, x2, y2]])
            elif x1 >= middle_x and x2 >= middle_x:
                # line should also be within the right hand side of region of interest
                right_lane_lines.append([[x1, y1, x2, y2]])

    return left_lane_lines, right_lane_lines


def do_segment(frame):
    height, width = frame.shape[0], frame.shape[1]
    # create a polygon for the mask defined by five coordinates
    polygon = np.array([[(0, height), (width, height), (width, height*10//17), (width*11//20, height*10//17),
                         (0, height*4//5)]])
    # create an image filled with zero intensities with the same dimensions as the frame
    mask = np.zeros_like(frame)
    # allow the mask to be filled with values of 1 and the other areas to be filled with values of 0
    cv2.fillPoly(mask, polygon, 255)
    # A bitwise and operation between the mask and frame keeps only the triangular area of the frame
    segment = cv2.bitwise_and(frame, mask)
    return segment


def main():
    # variables
    rgb_white_1 = (140, 140, 140)
    rgb_white_2 = (255, 255, 255)
    new_size_for_video = (720, 1280)

    frame_id = 0

    # initialize video
    cap = cv2.VideoCapture('4.mp4')
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    fourcc = cv2.VideoWriter_fourcc(*'DIVX')
    videoWriter = cv2.VideoWriter('Video_result.avi', fourcc, fps, new_size_for_video)

    dots = np.zeros((720, 1280, 3), np.uint8)
    dots[:] = (0, 0, 0)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        # resize frames and convert to rgb
        frame = cv2.resize(frame, new_size_for_video)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # create mask
        mask = cv2.inRange(frame, rgb_white_1, rgb_white_2)
        mask = do_segment(mask)

        # collect all lines using hough transformation
        hough_lines = hough_transform(mask)
        # separate lines
        left_lane_lines, right_lane_lines = separate_lines(hough_lines, frame)
        # find out parameters for the lines and draw it
        full_lane_drawn_img = trace_both_lane_lines(frame, left_lane_lines, right_lane_lines)

        videoWriter.write(full_lane_drawn_img)

        print(frame_id)
        frame_id += 1

    videoWriter.release()
    cap.release()


if __name__ == '__main__':
    main()
