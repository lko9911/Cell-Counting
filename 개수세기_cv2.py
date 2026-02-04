import cv2
import numpy as np

# 1. Binary 이미지 로드
binary = cv2.imread("output_binary.jpg", cv2.IMREAD_GRAYSCALE)

# 2. 검은 동그라미 → 흰색으로 반전
inv = cv2.bitwise_not(binary)

# 3. Morphology OPEN (작은 노이즈 제거)
kernel = np.ones((3, 3), np.uint8)
clean = cv2.morphologyEx(inv, cv2.MORPH_OPEN, kernel, iterations=1)

# 4. 컨투어 추출
contours, _ = cv2.findContours(
    clean,
    cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE
)

# 5. 면적 필터링 (동그라미 크기에 맞게 조절)
min_area = 1     # 너무 작으면 노이즈
max_area = 100000000   # 너무 크면 합쳐진 영역
valid = [
    c for c in contours
    if min_area < cv2.contourArea(c) < max_area
]

# 6. 개수
print("검은 동그라미 개수:", len(valid))

# 7. 시각화
vis = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
cv2.drawContours(vis, valid, -1, (0, 0, 255), 1)

cv2.imwrite("circle_count_result.png", vis)

cv2.imshow("Binary", binary)
cv2.imshow("Detected Circles", vis)
cv2.waitKey(0)
cv2.destroyAllWindows()
