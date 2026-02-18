import cv2
import numpy as np
import matplotlib.pyplot as plt

# ===============================
# 0. 이미지 로드 + 현미경 원형 Crop
# ===============================
img_full = cv2.imread("Sample_3.png")
assert img_full is not None, "이미지를 불러올 수 없습니다."

H, W = img_full.shape[:2]
center_full = (W // 2, H // 2)
radius_full = min(center_full[0], center_full[1])

# 정사각형 crop
x0 = center_full[0] - radius_full
y0 = center_full[1] - radius_full
x1 = center_full[0] + radius_full
y1 = center_full[1] + radius_full

img = img_full[y0:y1, x0:x1].copy()
orig = img.copy()

h, w = img.shape[:2]
center = (w // 2, h // 2)
radius = min(center)

# 원형 mask
Y, X = np.ogrid[:h, :w]
circle_mask = (X - center[0])**2 + (Y - center[1])**2 <= radius**2

# ===============================
# 1. Grayscale
# ===============================
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# ===============================
# 2. k-means (원 내부만 사용)
# ===============================
Z = gray[circle_mask].reshape((-1, 1)).astype(np.float32)

K = 2
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 15, 0.5)

_, labels, centers = cv2.kmeans(
    Z, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS
)

centers = centers.flatten()

# 2D 매핑
segmented = np.zeros_like(gray)
segmented[circle_mask] = centers[labels.flatten()].astype(np.uint8)

# ===============================
# 3. Binary mask 생성
# ===============================
cell_cluster = np.argmin(centers)

binary = np.zeros_like(gray)
cell_pixels = (labels.flatten() == cell_cluster).astype(np.uint8) * 255
binary[circle_mask] = cell_pixels

# 원 바깥 0으로 설정
binary[~circle_mask] = 0

# ===============================
# 4. Hole Filling
# ===============================
mask_ff = np.zeros((h + 2, w + 2), np.uint8)
binary_filled = binary.copy()

for x in range(w):
    cv2.floodFill(binary_filled, mask_ff, (x, 0), 255)
    cv2.floodFill(binary_filled, mask_ff, (x, h - 1), 255)

for y in range(h):
    cv2.floodFill(binary_filled, mask_ff, (0, y), 255)
    cv2.floodFill(binary_filled, mask_ff, (w - 1, y), 255)

binary_filled_inv = cv2.bitwise_not(binary_filled)
binary = binary | binary_filled_inv

# ===============================
# 5. Morphology
# ===============================
kernel = np.ones((3, 3), np.uint8)
binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)

# 작은 잡음 제거 + 내부 채우기
binary_clean = np.zeros_like(binary)

cnts, _ = cv2.findContours(
    binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
)

for cnt in cnts:
    if cv2.contourArea(cnt) < 30:
        continue
    cv2.drawContours(binary_clean, [cnt], -1, 255, -1)

binary = binary_clean

# ===============================
# 6. Distance Transform
# ===============================
dist = cv2.distanceTransform(binary, cv2.DIST_L2, 5)


min_peak_ratio = 0.20
local_max = (cv2.dilate(dist, np.ones((3, 3))) == dist) & \
            (dist > min_peak_ratio * dist.max())

num_peaks, peak_labels = cv2.connectedComponents(local_max.astype(np.uint8))

peak_points = []
for i in range(1, num_peaks):
    ys, xs = np.where(peak_labels == i)
    peak_points.append((int(np.mean(ys)), int(np.mean(xs))))

# ===============================
# 7. NMS
# ===============================
min_peak_distance = 13
peak_points = sorted(peak_points, key=lambda p: dist[p[0], p[1]], reverse=True)

filtered_peaks = []
for (y, x) in peak_points:
    if all(np.hypot(y - fy, x - fx) >= min_peak_distance for (fy, fx) in filtered_peaks):
        filtered_peaks.append((y, x))

# ===============================
# 8. 감염 분석
# ===============================
ROI_RADIUS = 17
INFECTED_MEAN_THRESHOLD = 170

output = orig.copy()
infected_rois = []
rbc_count = 0
infected_count = 0

for (y, x) in filtered_peaks:

    # 원 바깥이면 스킵
    if not circle_mask[y, x]:
        continue

    y1, y2 = max(0, y-ROI_RADIUS), min(gray.shape[0], y+ROI_RADIUS+1)
    x1, x2 = max(0, x-ROI_RADIUS), min(gray.shape[1], x+ROI_RADIUS+1)

    roi = gray[y1:y2, x1:x2]
    if roi.size == 0:
        continue

    h_r, w_r = roi.shape
    roi_mask = np.zeros((h_r, w_r), dtype=np.uint8)
    cv2.circle(roi_mask, (x - x1, y - y1), ROI_RADIUS-2, 255, -1)

    mean_intensity = cv2.mean(roi, mask=roi_mask)[0]

    if mean_intensity <= INFECTED_MEAN_THRESHOLD:
        color = (0, 0, 255) # 감염 (Red)
        infected_count += 1
        roi_norm = cv2.normalize(roi, None, 0, 255, cv2.NORM_MINMAX)
        infected_rois.append(roi_norm)
    else:
        color = (0, 255, 0)  # 정상 (Green)

    cv2.circle(output, (x, y), ROI_RADIUS, color, 2)
    rbc_count += 1

# ===============================
# 9. 시각화
# ===============================
fig, ax = plt.subplots(1, 2, figsize=(16, 8))
ax[0].imshow(cv2.cvtColor(output, cv2.COLOR_BGR2RGB))
ax[0].set_title(f"Detection: Total={rbc_count}, Infected={infected_count}")
ax[1].imshow(dist, cmap="jet")
ax[1].set_title("Distance Transform (Peaks)")
for a in ax: a.axis("off")
plt.tight_layout()
plt.show()

# ===============================
# 10. 감염 히트맵
# ===============================
if infected_rois:
    n_show = min(len(infected_rois), 6)
    fig, ax = plt.subplots(2, n_show, figsize=(3 * n_show, 6))
    if n_show == 1:
        ax = ax.reshape(2, 1)

    DARK_CUT_OFF = 80

    for i in range(n_show):

        ax[0, i].imshow(infected_rois[i], cmap="gray")
        ax[0, i].set_title(f"Infected #{i+1}")

        roi_target = infected_rois[i].copy()
        mask = roi_target < DARK_CUT_OFF

        roi_refined = np.full_like(roi_target, 255)
        roi_refined[mask] = roi_target[mask]

        ax[1, i].imshow(roi_refined, cmap="magma")
        ax[1, i].set_title("Infected Cell Pattern")

    for a in ax.flatten():
        a.axis("off")

    plt.suptitle("Infected pattern", fontsize=16)
    plt.tight_layout()
    plt.show()
