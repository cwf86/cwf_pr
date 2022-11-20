import numpy as np
import cv2
from matplotlib import pyplot as plt

# 加载图像并转为灰度图
#im = cv2.imread(r"D:\cwf_github\python\cwf_pr\cwf_python_tmp_test\cwf2.png")
im = cv2.imread(r"D:\cwf_github\python\cwf_pr\cwf_python_tmp_test\2457331_234414382000_2.png")
imGray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
imgGray_float32 = np.float32(imGray)

fft_result = cv2.dft(imgGray_float32, flags = cv2.DFT_COMPLEX_OUTPUT)
fft_result_shift = np.fft.fftshift(fft_result) # 调整为中心对齐的频谱图
fft_result_shift_img = 20*np.log(cv2.magnitude(fft_result_shift[:,:,0],fft_result_shift[:,:,1]))

plt.imshow(fft_result_shift_img, cmap = 'gray')
plt.show()
""" fft_result_shift_img = fft_result_shift_img.astype(np.uint8)
cv2.imshow("fimg",fft_result_shift_img) """
cv2.waitKey()

# 对频域图进行傅里叶逆变换，恢复成空域图
fft_result_ishift = np.fft.ifftshift(fft_result_shift)
img_ifft = cv2.idft(fft_result_ishift)
img_back = cv2.magnitude(img_ifft[:,:,0],img_ifft[:,:,1])

plt.imshow(img_back, cmap = 'gray')
plt.show()
""" img_back = img_back.astype(np.uint8)
cv2.imshow("img",img_back) """
cv2.waitKey()

