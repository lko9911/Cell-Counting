import cv2

# 1. 이미지 로드
img = cv2.imread("image3.png")

# 2. 그레이스케일 변환
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 3. 노이즈 제거 (Gaussian Blur)
blur = cv2.GaussianBlur(gray, (5, 5), 0)

# 4. 이진화 (Otsu Threshold)
_, binary = cv2.threshold(
    blur,
    0,                          # threshold 값 (Otsu 사용 시 0)
    255,
    cv2.THRESH_BINARY + cv2.THRESH_OTSU
)

# 5. 저장
cv2.imwrite("output_binary.jpg", binary)

# 6. 출력
cv2.imshow("Gray", gray)
cv2.imshow("Blur", blur)
cv2.imshow("Binary", binary)
cv2.waitKey(0)
cv2.destroyAllWindows()
