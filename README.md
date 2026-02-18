# 📌 Project Title
> Blood Cell Counting Method
<br>

![Python](https://img.shields.io/badge/Python-3.10-blue)
![Status](https://img.shields.io/badge/Status-In%20Progress-orange)

---

## 📖 Overview
Cell Counting 관련 코드 정리 

- 🔍 Problem: 적혈구 세기 
- 💡 Solution: Analysis, DT_peak_method, Countour_method

---

## ⭐ Key Features
- ✅ Analysis: 이미지의 색상 분포 분석용
<img width="4210" height="1209" alt="그림2" src="https://github.com/user-attachments/assets/553f1829-7957-4f49-96d8-fa98d183638e" /><br><br>

  
- ✅ DT_peak_method
<img width="1463" height="988" alt="스크린샷 2026-02-18 100800" src="https://github.com/user-attachments/assets/4aaa7985-0033-415d-8a30-c4a8d008acab" />
<img width="1147" height="586" alt="dt" src="https://github.com/user-attachments/assets/c9a4f432-428e-405a-946c-3174a35e39fb" />

<br><br>

  
- ✅ Countour_method
<img width="1533" height="976" alt="스크린샷 2026-02-18 100652" src="https://github.com/user-attachments/assets/549c2256-a93f-4a6d-880c-6ad4c320bf83" />
<img width="1122" height="590" alt="스크린샷 2026-02-18 101426" src="https://github.com/user-attachments/assets/9320acf4-4658-4630-9103-e609af6816ef" />
<br><br>

---

## 🏗 Project Structure
```bash
Project/
├── Analysis/              
│   ├── 3차원_시각화.py
│   └── 히스토그램.py           
├── Contour_method/               
│   ├── 개수세기_통합(kmeans-watershed).py
│   ├── 개수세기_통합(kmeans-watershed)_hmap.py
│   ├── Contour_method_microscope.py
│   └── Contour_method_hmap.py  
├── DT_peak_method/
│   ├── 개수세기_circle_감염_hmap.py
│   ├── 개수세기_통합(blue-circle)_GY_hyper.py  
│   ├── DT_peak_method_microscope.py
│   └── DT_peak_method_hmap.py
├── Dummy/ # 필요없는 파일 모음 (무시하기)    
├── requirements.txt
├──.gitignore
└── README.md
```

---

## ⚙️ Installation
```bash
git clone https://github.com/lko9911/Cell-Counting.git
cd Cell-Counting
pip install -r requirements.txt
```
