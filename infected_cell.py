import cv2
import numpy as np
import matplotlib.pyplot as plt

# ===============================
# 0. 이미지 로드
# ===============================
img = cv2.imread("image2.png")
assert img is not None, "이미지를 불러올 수 없습니다."

img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# ===============================
# 1. Grayscale 변환
# ===============================
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

plt.figure(figsize=(4,4))
plt.imshow(gray, cmap='gray') 
plt.title("Grayscale")  
plt.axis('off')
plt.show()

# ===============================
# 2. Gaussian Blur (조명 보정)
# ===============================
gray_blur = cv2.GaussianBlur(gray, (5,5), 0)

# ===============================
# 3. Inverse Otsu Threshold
#    (어두운 영역 = 감염 세포)
# ===============================
_, mask = cv2.threshold(
    gray_blur,
    0,
    255,
    cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
)

plt.figure(figsize=(4,4))
plt.imshow(mask, cmap='gray')
plt.title("Inverse Otsu mask (dark regions)")
plt.axis('off')
plt.show()

# ===============================
# 4. Morphological 후처리
# ===============================
kernel = np.ones((3,3), np.uint8)

mask_clean = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
mask_clean = cv2.morphologyEx(mask_clean, cv2.MORPH_CLOSE, kernel, iterations=2)

plt.figure(figsize=(4,4))
plt.imshow(mask_clean, cmap='gray')
plt.title("Cleaned mask")
plt.axis('off')
plt.show()

# ===============================
# 5. 컨투어 검출
# ===============================
contours, _ = cv2.findContours(
    mask_clean,
    cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE
)

print(f"Detected regions (raw): {len(contours)}")

# ===============================
# 6. 면적 필터링
# ===============================
filtered = []
min_area = 50
max_area = 2000

for cnt in contours:
    area = cv2.contourArea(cnt)
    if min_area < area < max_area:
        filtered.append(cnt)

print(f"Filtered infected cell candidates: {len(filtered)}")

# ===============================
# 7. 결과 오버레이
# ===============================
overlay = img_rgb.copy()
for cnt in filtered:
    cv2.drawContours(overlay, [cnt], -1, (255, 0, 0), -1)

plt.figure(figsize=(6,6))
plt.imshow(overlay)
plt.title("Infected cells (Grayscale-based)")
plt.axis('off')
plt.show()
