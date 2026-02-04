import cv2
import numpy as np

# 1. 원본 이미지 로드 (컬러)
img = cv2.imread("image.png")

# 2. 그레이스케일
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 3. 노이즈 제거
blur = cv2.GaussianBlur(gray, (5, 5), 0)

# 4. Binary (검은 동그라미 → 흰색)
_, binary = cv2.threshold(
    blur,
    0,
    255,
    cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
)

# 5. Morphology (작은 노이즈 제거)
kernel = np.ones((3, 3), np.uint8)
binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)

# 6. 컨투어 추출
contours, _ = cv2.findContours(
    binary,
    cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE
)

# 7. 면적 필터링 (중요!)
min_area = 20
max_area = 2000
valid = [
    c for c in contours
    if min_area < cv2.contourArea(c) < max_area
]

print("검은 동그라미 개수:", len(valid))

# 8. 결과 시각화 (원본 이미지 위에 표시)
result = img.copy()
cv2.drawContours(
    result,
    valid,
    -1,
    (0, 0, 255),   # 빨간색
    2
)

# 9. 저장
cv2.imwrite("binary_circle_count_result.png", result)

# 10. 화면 출력
cv2.imshow("Original", img)
cv2.imshow("Binary Mask", binary)
cv2.imshow("Detected Circles", result)
cv2.waitKey(0)
cv2.destroyAllWindows()
