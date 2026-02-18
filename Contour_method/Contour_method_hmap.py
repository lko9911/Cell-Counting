import cv2
import numpy as np
import matplotlib.pyplot as plt

# ===============================
# 0. 이미지 로드
# ===============================
img = cv2.imread("Sample_3.png")
assert img is not None, "이미지를 불러올 수 없습니다."

# ===============================
# 0-1. 현미경 원형 Crop 추가
# ===============================
H, W = img.shape[:2]
center_full = (W // 2, H // 2)
radius_full = min(center_full[0], center_full[1])

# 정사각형 crop
x0 = center_full[0] - radius_full
y0 = center_full[1] - radius_full
x1 = center_full[0] + radius_full
y1 = center_full[1] + radius_full

img = img[y0:y1, x0:x1].copy()
orig = img.copy()

h, w = img.shape[:2]
center = (w // 2, h // 2)
radius = min(center)

# 원형 mask 생성
Y, X = np.ogrid[:h, :w]
circle_mask = (X - center[0])**2 + (Y - center[1])**2 <= radius**2

# 원 밖 제거 (Watershed 왜곡 방지)
img[~circle_mask] = 255
orig = img.copy()

orig = img.copy()

# ===============================
# 1. Grayscale 
# ===============================
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# ===============================
# 2. K-means (배경 / RBC 분리)
# ===============================
Z = gray.reshape((-1, 1)).astype(np.float32)

K = 2
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)

_, labels, centers = cv2.kmeans(
    Z, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS
)

segmented = centers[labels.flatten()].reshape(gray.shape)
segmented = segmented.astype(np.uint8)

# ===============================
# 3. Binary mask (적혈구 = 흰색)
# ===============================
_, binary = cv2.threshold(
    segmented, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
)

# ===============================
# 3-1. Hole filling (적혈구 내부 구멍 제거 준비)
# ===============================
h, w = binary.shape
mask = np.zeros((h + 2, w + 2), np.uint8)

binary_filled = binary.copy()
mask = np.zeros((h + 2, w + 2), np.uint8)

# ===============================
# 4. Morphology + Contour 정제
# ===============================
kernel = np.ones((3, 3), np.uint8)
opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)

binary_final = np.zeros_like(opening)
cnts, _ = cv2.findContours(opening, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
for cnt in cnts:
    if cv2.contourArea(cnt) > 30:
        cv2.drawContours(binary_final, [cnt], -1, 255, -1)

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
# 4. Morphology (노이즈 제거)
# ===============================
kernel = np.ones((3, 3), np.uint8)

opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)
sure_bg = cv2.dilate(opening, kernel, iterations=2)

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
# 6. Marker 생성
# ===============================
_, sure_fg = cv2.threshold(
    dist, 0.4 * dist.max(), 255, 0
)
sure_fg = np.uint8(sure_fg)

unknown = cv2.subtract(sure_bg, sure_fg)

num_labels, markers = cv2.connectedComponents(sure_fg)
markers = markers + 1
markers[unknown == 255] = 0

# ===============================
# 7. Watershed
# ===============================
markers = cv2.watershed(img, markers)

# ===============================
# 8. Contour 분석 & 감염 판별
# ===============================
INFECTED_MEAN_THRESHOLD = 180

output = orig.copy()
rbc_count = 0
infected_count = 0
infected_rois = []

for label in np.unique(markers):
    if label <= 1:
        continue

    mask = np.zeros(gray.shape, dtype="uint8")
    mask[markers == label] = 255

    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        continue

    cnt = max(cnts, key=cv2.contourArea)
    area = cv2.contourArea(cnt)
    if area < 30:
        continue

    mean_intensity = cv2.mean(gray, mask=mask)[0]

    if mean_intensity <= INFECTED_MEAN_THRESHOLD:
        color = (255, 0, 0)  
        infected_count += 1

        x, y, w, h = cv2.boundingRect(cnt)
        infected_rois.append(gray[y:y+h, x:x+w])
    else:
        color = (0, 255, 0)   

    cv2.drawContours(output, [cnt], -1, color, 1)
    rbc_count += 1

print(f"Total RBC: {rbc_count}")
print(f"Infected RBC: {infected_count}")

# [1] 전체 분석 결과
fig, ax = plt.subplots(1, 2, figsize=(16, 8))
ax[0].imshow(cv2.cvtColor(output, cv2.COLOR_BGR2RGB))
ax[0].set_title(f"Detection: Total={rbc_count}, Infected={infected_count}")
ax[1].imshow(dist, cmap="jet")
ax[1].set_title("Distance Transform (Peaks)")
for a in ax: a.axis("off")
plt.tight_layout()
plt.show()

# ===============================
# 9. 감염 RBC 히트맵 시각화
# ===============================
if infected_rois:
    n_show = min(len(infected_rois), 6)
    fig, ax = plt.subplots(2, n_show, figsize=(3 * n_show, 6))
    if n_show == 1: ax = ax.reshape(2, 1)

    DARK_CUT_OFF = 180 

    for i in range(n_show):

        ax[0, i].imshow(infected_rois[i], cmap="gray")
        ax[0, i].set_title(f"Infected #{i+1}")
        roi_target = infected_rois[i].copy()
        
        mask = roi_target < DARK_CUT_OFF
        
        roi_refined = np.full_like(roi_target, 255) 
        roi_refined[mask] = roi_target[mask] 

        # 3. 히트맵 시각화
        im = ax[1, i].imshow(roi_refined, cmap="magma")
        ax[1, i].set_title(f"Infected Cell Pattern")
        
    for a in ax.flatten():
        a.axis("off")
        
    plt.suptitle("Infected pattern", fontsize=16)
    plt.tight_layout()
    plt.show()