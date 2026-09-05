import cv2

image = cv2.imread("output/20260905_210750_raw.png")
cv2.imwrite("output/latest_wind_top_debug.png", image[0:500, 1600:2250])
cv2.imwrite("output/latest_wind_candidate_debug.png", image[300:420, 1840:2050])
