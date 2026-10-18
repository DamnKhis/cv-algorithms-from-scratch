import cv2
import numpy as np


def fast_hough_lines(edge_img, threshold):
    height, width = edge_img.shape
    diag_len = int(np.ceil(np.sqrt(height ** 2 + width ** 2)))
    rhos = np.linspace(-diag_len, diag_len, diag_len * 2)
    thetas = np.deg2rad(np.arange(0, 180))

    accumulator = np.zeros((len(rhos), len(thetas)), dtype=np.int32)
    y_idxs, x_idxs = np.nonzero(edge_img)

    cos_t = np.cos(thetas)
    sin_t = np.sin(thetas)

    for t_idx in range(len(thetas)):
        rho_values = x_idxs * cos_t[t_idx] + y_idxs * sin_t[t_idx]
        rho_indices = np.round(rho_values + diag_len).astype(int)
        np.add.at(accumulator[:, t_idx], rho_indices, 1)

    r_indices, t_indices = np.where(accumulator > threshold)
    indices = np.argsort(accumulator[r_indices, t_indices])[::-1]

    found_lines = []
    for idx in indices:
        r, t = rhos[r_indices[idx]], thetas[t_indices[idx]]
        if not any(abs(r - fr) < 25 and abs(t - ft) < 0.3 for fr, ft in found_lines):
            found_lines.append((r, t))
        if len(found_lines) > 15: break
    return found_lines


def fast_hough_circles(edge_img, min_r, max_r, threshold):
    height, width = edge_img.shape
    y_idxs, x_idxs = np.nonzero(edge_img)
    all_circles = []

    for r in range(min_r, max_r + 1, 2):
        acc = np.zeros((height, width), dtype=np.int32)
        for angle in range(0, 360, 15):
            theta = np.deg2rad(angle)
            a = (x_idxs - r * np.cos(theta)).astype(int)
            b = (y_idxs - r * np.sin(theta)).astype(int)
            valid = (a >= 0) & (a < width) & (b >= 0) & (b < height)
            np.add.at(acc, (b[valid], a[valid]), 1)

        y_c, x_c = np.where(acc > threshold)
        for i in range(len(x_c)):
            all_circles.append((x_c[i], y_c[i], r, acc[y_c[i], x_c[i]]))

    all_circles.sort(key=lambda x: x[3], reverse=True)
    final_circles = []
    for c in all_circles:
        x, y, r, score = c
        if not any(np.sqrt((x - fx) ** 2 + (y - fy) ** 2) < 40 for fx, fy, fr in final_circles):
            final_circles.append((x, y, r))
        if len(final_circles) > 10: break
    return final_circles


img = cv2.imread('small.jpg')

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (5, 5), 0)
edges = cv2.Canny(blurred, 50, 150)
canvas = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

print("началось...")

my_lines = fast_hough_lines(edges, threshold=100)
my_circles = fast_hough_circles(edges, 30, 120, threshold=1)

for rho, theta in my_lines:
    a, b = np.cos(theta), np.sin(theta)
    x0, y0 = a * rho, b * rho
    p1 = (int(x0 + 1000 * (-b)), int(y0 + 1000 * a))
    p2 = (int(x0 - 1000 * (-b)), int(y0 - 1000 * a))
    cv2.line(canvas, p1, p2, (0, 255, 0), 2)

for (x, y, r) in my_circles:
    cv2.circle(canvas, (x, y), r, (255, 0, 0), 2)

cv_canvas = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
cv_circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, 1, 50, param1=70, param2=70, minRadius=30, maxRadius=120)
if cv_circles is not None:
    for i in np.uint16(np.around(cv_circles[0, :])):
        cv2.circle(cv_canvas, (i[0], i[1]), i[2], (0, 0, 255), 2)

cv2.imshow('1. Original Edges (Canny)', edges)
cv2.imshow('2. My Hough on Edges (Lines=Green, Circles=Blue)', canvas)
cv2.imshow('3. OpenCV Result on Edges (Red)', cv_canvas)

print("Нажмите любую клавишу для выхода.")
cv2.waitKey(0)
cv2.destroyAllWindows()
