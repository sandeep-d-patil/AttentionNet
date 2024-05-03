import cv2
import numpy as np
import math
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN, AgglomerativeClustering, KMeans
from sklearn.preprocessing import StandardScaler
from collections import Counter


# Parameters
min_area_threshold = 100

image = cv2.imread(
    "/home/sandeep/PycharmProjects/TuSimple_test_output/clips/0531/1492626394610203677/20pred.jpg",
    cv2.IMREAD_GRAYSCALE,
)
# image = cv2.resize(image, (512, 256))
source_image = cv2.imread(
    "/home/sandeep/PycharmProjects/TuSimple_test_output/clips/0531/1492626394610203677/20.jpg"
)
# source_image = cv2.resize(source_image, (512, 256))
fs = cv2.FileStorage(
    "/home/sandeep/PycharmProjects/LaneDetection/tusimple_ipm_remap.yml",
    cv2.FILE_STORAGE_READ,
)
remap_to_ipm_x = fs.getNode("remap_ipm_x").mat()
remap_to_ipm_y = fs.getNode("remap_ipm_y").mat()
ret = {
    "remap_to_ipm_x": remap_to_ipm_x,
    "remap_to_ipm_y": remap_to_ipm_y,
}

fs.release()

for x in range(250):
    for y in range(1280):
        image[x, y] = 0

image = cv2.erode(image, kernel=np.ones((3, 3), np.uint8), iterations=0)
image = cv2.dilate(image, kernel=np.ones((3, 3), np.uint8), iterations=0)
kernel = cv2.getStructuringElement(shape=cv2.MORPH_ELLIPSE, ksize=(5, 5))
morphological_ret = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel, iterations=1)
cv2.connectedComponentsWithStats(morphological_ret, connectivity=8, ltype=cv2.CV_32S)
connect_components_analysis_ret = cv2.connectedComponentsWithStats(
    morphological_ret, connectivity=8, ltype=cv2.CV_32S
)
labels = connect_components_analysis_ret[1]
stats = connect_components_analysis_ret[2]
plt.imshow(labels)
plt.show()

for index, stat in enumerate(stats):
    print(index)
    print(stat)
    if stat[4] <= min_area_threshold:
        idx = np.where(labels == index)
        morphological_ret[idx] = 0

plt.imshow(morphological_ret)
plt.show()
idx = np.where(morphological_ret == 255)
lane_coordinate = np.vstack((idx[1], idx[0])).transpose()

# db = KMeans(n_clusters=4, random_state=1000)
db = DBSCAN(eps=0.01, min_samples=2)
# db = AgglomerativeClustering(distance_threshold=None, n_clusters=10)

features = StandardScaler().fit_transform(lane_coordinate)
db.fit(features)

db_labels = db.labels_
unique_labels = np.unique(db_labels)
num_clusters = len(unique_labels)
# cluster_centers = db.components_
ret = {
    "origin_features": features,
    "cluster_nums": num_clusters,
    "db_labels": db_labels,
    "unique_labels": unique_labels,
    # 'cluster_center': cluster_centers
}
mask = np.zeros([720, 1280, 3], dtype=np.uint8)
coord = lane_coordinate
lane_coords = []
_color_map = [
    np.array([255, 0, 0]),
    np.array([0, 255, 0]),
    np.array([0, 0, 255]),
    np.array([125, 125, 0]),
    np.array([0, 125, 125]),
    np.array([125, 0, 125]),
    np.array([50, 100, 50]),
    np.array([100, 50, 100]),
    np.array([125, 125, 0]),
    np.array([0, 125, 125]),
    np.array([125, 0, 125]),
    np.array([50, 100, 50]),
]
counts = Counter(db_labels)

for index, label in enumerate(unique_labels.tolist()):
    if label == -1:
        continue
    if counts[label] > 1000:

        idx = np.where(db_labels == label)
        print(index)
        pix_coord_idx = tuple((coord[idx][:, 1], coord[idx][:, 0]))
        mask[pix_coord_idx] = _color_map[index]
        lane_coords.append(coord[idx])

plt.imshow(mask)
plt.show()
fit_params = []
src_lane_pts = []  # lane pts every single lane
for lane_index, coords in enumerate(lane_coords):
    # if data_source == 'tusimple':
    tmp_mask = np.zeros(shape=(720, 1280), dtype=np.uint8)
    tmp_mask[tuple((np.int_(coords[:, 1]), np.int_(coords[:, 0])))] = 255
    # else:
    #     raise ValueError('Wrong data source now only support tusimple')
    tmp_ipm_mask = cv2.remap(
        tmp_mask, remap_to_ipm_x, remap_to_ipm_y, interpolation=cv2.INTER_NEAREST
    )
    nonzero_y = np.array(tmp_ipm_mask.nonzero()[0])
    nonzero_x = np.array(tmp_ipm_mask.nonzero()[1])

    fit_param = np.polyfit(nonzero_y, nonzero_x, 2)
    fit_params.append(fit_param)

    [ipm_image_height, ipm_image_width] = tmp_ipm_mask.shape
    plot_y = np.linspace(10, ipm_image_height, ipm_image_height - 10)
    fit_x = fit_param[0] * plot_y**2 + fit_param[1] * plot_y + fit_param[2]
    # fit_x = fit_param[0] * plot_y ** 3 + fit_param[1] * plot_y ** 2 + fit_param[2] * plot_y + fit_param[3]

    lane_pts = []
    for index in range(0, plot_y.shape[0], 5):
        src_x = remap_to_ipm_x[
            int(plot_y[index]), int(np.clip(fit_x[index], 0, ipm_image_width - 1))
        ]
        if src_x <= 0:
            continue
        src_y = remap_to_ipm_y[
            int(plot_y[index]), int(np.clip(fit_x[index], 0, ipm_image_width - 1))
        ]
        src_y = src_y if src_y > 0 else 0

        lane_pts.append([src_x, src_y])

    src_lane_pts.append(lane_pts)

# tusimple test data sample point along y axis every 10 pixels
source_image_width = source_image.shape[1]
for index, single_lane_pts in enumerate(src_lane_pts):
    single_lane_pt_x = np.array(single_lane_pts, dtype=np.float32)[:, 0]
    single_lane_pt_y = np.array(single_lane_pts, dtype=np.float32)[:, 1]
    # if data_source == 'tusimple':
    start_plot_y = 240
    end_plot_y = 720
    # else:
    #     raise ValueError('Wrong data source now only support tusimple')
    step = int(math.floor((end_plot_y - start_plot_y) / 10))
    for plot_y in np.linspace(start_plot_y, end_plot_y, step):
        diff = single_lane_pt_y - plot_y
        fake_diff_bigger_than_zero = diff.copy()
        fake_diff_smaller_than_zero = diff.copy()
        fake_diff_bigger_than_zero[np.where(diff <= 0)] = float("inf")
        fake_diff_smaller_than_zero[np.where(diff > 0)] = float("-inf")
        idx_low = np.argmax(fake_diff_smaller_than_zero)
        idx_high = np.argmin(fake_diff_bigger_than_zero)

        previous_src_pt_x = single_lane_pt_x[idx_low]
        previous_src_pt_y = single_lane_pt_y[idx_low]
        last_src_pt_x = single_lane_pt_x[idx_high]
        last_src_pt_y = single_lane_pt_y[idx_high]

        if (
            previous_src_pt_y < start_plot_y
            or last_src_pt_y < start_plot_y
            or fake_diff_smaller_than_zero[idx_low] == float("-inf")
            or fake_diff_bigger_than_zero[idx_high] == float("inf")
        ):
            continue

        interpolation_src_pt_x = (
            abs(previous_src_pt_y - plot_y) * previous_src_pt_x
            + abs(last_src_pt_y - plot_y) * last_src_pt_x
        ) / (abs(previous_src_pt_y - plot_y) + abs(last_src_pt_y - plot_y))
        interpolation_src_pt_y = (
            abs(previous_src_pt_y - plot_y) * previous_src_pt_y
            + abs(last_src_pt_y - plot_y) * last_src_pt_y
        ) / (abs(previous_src_pt_y - plot_y) + abs(last_src_pt_y - plot_y))

        if interpolation_src_pt_x > source_image_width or interpolation_src_pt_x < 10:
            continue

        lane_color = _color_map[index].tolist()
        cv2.circle(
            source_image,
            (int(interpolation_src_pt_x), int(interpolation_src_pt_y)),
            5,
            lane_color,
            -1,
        )
        print("interpolation_src_pt_y", interpolation_src_pt_y)
        # print("interpolation_src_pt_x", interpolation_src_pt_x)

while True:
    cv2.imshow("label", label)
    cv2.imshow("morphological_ret", morphological_ret)
    cv2.imshow("mask", mask)
    cv2.imshow("source_image", source_image)

    c = cv2.waitKey(1000)
    if c == 27:
        break
