import cv2
import numpy as np

# 1. 원본 이미지 로드
img = cv2.imread("sample3.png")
h, w = img.shape[:2]

# 2. K-means 입력 형태로 변환 (N x 3)
Z = img.reshape((-1, 3))
Z = np.float32(Z)

# 3. K-means 설정
K = 2  # 배경 / 검은 동그라미
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)

_, labels, centers = cv2.kmeans(
    Z,
    K,
    None,
    criteria,
    10,
    cv2.KMEANS_RANDOM_CENTERS
)

# 4. 결과 복원
labels = labels.reshape((h, w))
centers = np.uint8(centers)

# 5. 가장 어두운 클러스터 찾기 (검은 동그라미)
brightness = centers.mean(axis=1)
black_cluster = np.argmin(brightness)

# 6. 검은 영역 마스크 생성
mask = np.zeros((h, w), dtype=np.uint8)
mask[labels == black_cluster] = 255

# 7. Morphology (노이즈 제거)
kernel = np.ones((3, 3), np.uint8)
mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

# 8. 컨투어 추출
contours, _ = cv2.findContours(
    mask,
    cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE
)

# 9. 면적 필터링
min_area = 20
max_area = 2000
valid = [c for c in contours if min_area < cv2.contourArea(c) < max_area]

print("검은 동그라미 개수:", len(valid))

# 10. 결과 시각화
vis = img.copy()
cv2.drawContours(vis, valid, -1, (0, 0, 255), 2)

cv2.imwrite("kmeans_circle_count.png", vis)

cv2.imshow("Original", img)
cv2.imshow("Mask (Black Cluster)", mask)
cv2.imshow("Detected Circles", vis)
cv2.waitKey(0)
cv2.destroyAllWindows()
