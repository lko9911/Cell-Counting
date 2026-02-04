import cv2
import numpy as np
import matplotlib.pyplot as plt

# ===============================
# 0. 이미지 로드
# ===============================
img = cv2.imread("image2.png")
assert img is not None, "이미지를 불러올 수 없습니다."
orig = img.copy()

# ===============================
# 1. Grayscale
# ===============================
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# ===============================
# 2. k-means
# ===============================
Z = gray.reshape((-1, 1)).astype(np.float32)
K = 2
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)

_, labels, centers = cv2.kmeans(
    Z, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS
)
segmented = centers[labels.flatten()].reshape(gray.shape).astype(np.uint8)

# ===============================
# 3. Binary mask
# ===============================
#_, binary = cv2.threshold(
#    segmented, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
#)

cell_cluster = np.argmin(centers)  # 더 어두운 클러스터
binary = (labels.reshape(gray.shape) == cell_cluster).astype(np.uint8) * 255

# ===============================
# 3-1. Hole filling (적혈구 내부 구멍 제거)
# ===============================
h, w = binary.shape
mask = np.zeros((h + 2, w + 2), np.uint8)

binary_filled = binary.copy()
mask = np.zeros((h + 2, w + 2), np.uint8)

# 테두리 전체를 seed로 flood fill
for x in range(w):
    cv2.floodFill(binary_filled, mask, (x, 0), 255)
    cv2.floodFill(binary_filled, mask, (x, h-1), 255)

for y in range(h):
    cv2.floodFill(binary_filled, mask, (0, y), 255)
    cv2.floodFill(binary_filled, mask, (w-1, y), 255)

binary_filled_inv = cv2.bitwise_not(binary_filled)
binary = binary | binary_filled_inv


# ===============================
# 4. Morphology
# ===============================
kernel = np.ones((3, 3), np.uint8)
opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)

# ===============================
# 4-1. Contour-based hole filling (강제 채우기)
# ===============================
binary_filled = np.zeros_like(opening)

cnts, _ = cv2.findContours(
    opening, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
)

for cnt in cnts:
    area = cv2.contourArea(cnt)
    if area < 30:   # 기존 면적 필터와 맞추기
        continue
    cv2.drawContours(binary_filled, [cnt], -1, 255, -1)

binary = binary_filled

# ===============================
# 5. Distance Transform
# ===============================
dist = cv2.distanceTransform(opening, cv2.DIST_L2, 5)

# ===============================
# 6. DT Peak Detection
# ===============================
min_peak_ratio = 0.20
kernel_peak = np.ones((3, 3), np.uint8)

local_max = cv2.dilate(dist, kernel_peak) == dist
local_max &= (dist > min_peak_ratio * dist.max())

num_peaks, peak_labels = cv2.connectedComponents(
    local_max.astype(np.uint8)
)
num_peaks -= 1  # background 제거

# ===============================
# 7. Peak centroid 추출
# ===============================
peak_points = []

for i in range(1, num_peaks + 1):
    ys, xs = np.where(peak_labels == i)
    if len(xs) == 0:
        continue

    cy = int(np.mean(ys))
    cx = int(np.mean(xs))
    peak_points.append((cy, cx))

# ===============================
# 8. Peak NMS (겹침 제거)
# ===============================
min_peak_distance = 13  

# DT 값 기준으로 강한 peak부터
peak_points = sorted(
    peak_points,
    key=lambda p: dist[p[0], p[1]],
    reverse=True
)

filtered_peaks = []

for (y, x) in peak_points:
    keep = True
    for (fy, fx) in filtered_peaks:
        if np.hypot(y - fy, x - fx) < min_peak_distance:
            keep = False
            break
    if keep:
        filtered_peaks.append((y, x))

# ===============================
# 9. 결과 시각화
# ===============================

# ===============================
# 감염 판별 파라미터
# ===============================
ROI_RADIUS = 7                 # 파란 원 반지름 (DT peak와 동일)
INFECTED_MEAN_THRESHOLD = 180   # 어두울수록 감염 (히스토그램 보고 조절)

output = orig.copy()

rbc_count = 0
infected_count = 0
mean_values = []

for (y, x) in filtered_peaks:

    # ===============================
    # 원형 ROI 마스크 생성
    # ===============================
    roi_mask = np.zeros_like(gray, dtype=np.uint8)
    cv2.circle(roi_mask, (x, y), ROI_RADIUS, 255, -1)

    # 평균 그레이스케일 값
    mean_intensity = cv2.mean(gray, mask=roi_mask)[0]
    mean_values.append(mean_intensity)

    # 감염 판별 (어두울수록 감염)
    if mean_intensity <= INFECTED_MEAN_THRESHOLD:
        color = (0, 255, 0)   # 🔴 감염
        infected_count += 1
    else:
        color = (0, 255, 0)   # 🟢 정상

    cv2.circle(output, (x, y), ROI_RADIUS, color, 1)
    rbc_count += 1

print("적혈구 개수:", rbc_count)
print("감염된 적혈구 개수:", infected_count)

plt.figure(figsize=(5,4))
plt.hist(mean_values, bins=30)
plt.xlabel("Mean Grayscale Intensity (ROI)")
plt.ylabel("Count")
plt.title("RBC Intensity Distribution")
plt.show()

# ===============================
# 10. Visualization
# ===============================
fig, ax = plt.subplots(2, 3, figsize=(15, 10))

ax[0,0].imshow(cv2.cvtColor(orig, cv2.COLOR_BGR2RGB))
ax[0,0].set_title("Original")

ax[1,2].imshow(gray, cmap="gray")
ax[1,2].set_title("Grayscale")

binary_inv = 255 - binary
ax[0,2].imshow(segmented, cmap="gray")
ax[0,2].set_title("k-means result")

ax[1,0].imshow(binary, cmap="gray")
ax[1,0].set_title("Binary mask")

im = ax[1,1].imshow(dist, cmap="jet")
ax[1,1].set_title("Distance Transform")

plt.colorbar(im, ax=ax[1,1])

ax[0,1].imshow(cv2.cvtColor(output, cv2.COLOR_BGR2RGB))
ax[0,1].set_title(f"Count green-circle (Count = {rbc_count}) (Infected = {infected_count})")
#ax[0,1].set_title(f"Count green-circle (Count = {rbc_count})")

for a in ax.flatten():
    a.axis("off")

plt.tight_layout()
plt.show()
