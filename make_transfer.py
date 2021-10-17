import cv2
import math
import dlib
import numpy as np

# Reading example and subject images
img_source1 = cv2.imread('img_makeup.jpg', cv2.IMREAD_UNCHANGED)
img_source2 = cv2.imread('subject_3.jpeg', cv2.IMREAD_UNCHANGED)
img_source2_copy = np.copy(img_source2)
img_source1_copy = img_source1.astype(float)
output = np.copy(img_source2).astype(np.float64)


coords = []

# function triangulation method for image warping


def triangulation(img_source):

    # gace detection
    img3 = np.copy(img_source)
    img = np.copy(img_source)
    img1 = np.copy(img_source)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
    faces = detector(img)
    for face in faces:

        # 68 points detection
        x1 = face.left()
        y1 = face.top()
        x2 = face.right()
        y2 = face.bottom()
        landmarks = predictor(img, face)
        for n in range(0, 68):
            x = landmarks.part(n).x
            y = landmarks.part(n).y
            coords.append((x, y))
            cv2.circle(img1, (x, y), 4, (0, 255, 0), -1)

    def draw_circle(event, x, y, flags, param):
        global coords

        # marking more points by hand on the face
        if event == cv2.EVENT_LBUTTONDBLCLK:
            coords.append((x, y))
            cv2.circle(img1, (x, y), 4, (0, 255, 0), -1)
            cv2.imshow("image", img1)
    cv2.imshow("image", img1)
    cv2.namedWindow('image')
    cv2.setMouseCallback('image', draw_circle)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    points = np.array(coords, np.int32)
    convexhull = cv2.convexHull(points)

    rect = cv2.boundingRect(convexhull)
    (x, y, w, h) = rect

    # marking triangles on the face using delaunay triangulation
    subdiv = cv2.Subdiv2D(rect)
    subdiv.insert(coords)
    triangles = subdiv.getTriangleList()
    triangles = np.array(triangles, dtype=np.int32)
    index_triangles = []
    for t in triangles:

        ram = np.array([[t[0], t[1]], [t[2], t[3]], [t[4], t[5]]])
        cv2.polylines(img3, [ram], True, (0, 255, 0), 1)
        pt1 = (t[0], t[1])
        pt2 = (t[2], t[3])
        pt3 = (t[4], t[5])
        index_pt1 = np.where((points == pt1).all(axis=1))
        index_pt1 = index_pt1[0][0]

        index_pt2 = np.where((points == pt2).all(axis=1))
        index_pt2 = index_pt2[0][0]

        index_pt3 = np.where((points == pt3).all(axis=1))
        index_pt3 = index_pt3[0][0]

        triangle = [index_pt1, index_pt2, index_pt3]
        index_triangles.append(triangle)

    cv2.imshow("image", img3)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return index_triangles


# triangulation for example image
store2 = triangulation(img_source1)
indices1 = coords.copy()
coords = []
# triangulation for target image
store1 = triangulation(img_source2)
indices2 = coords.copy()

# function for wrapping the source image


def wrap():
    for i in range(0, len(store2)):

        # Barycentric warping
        tr1_pt1 = indices1[store2[i][0]]
        y11 = float(tr1_pt1[1])
        x11 = float(tr1_pt1[0])
        tr1_pt2 = indices1[store2[i][1]]
        y12 = float(tr1_pt2[1])
        x12 = float(tr1_pt2[0])
        tr1_pt3 = indices1[store2[i][2]]
        y13 = float(tr1_pt3[1])
        x13 = float(tr1_pt3[0])
        triangle1 = np.array([tr1_pt1, tr1_pt2, tr1_pt3], np.int32)

        tr2_pt1 = indices2[store2[i][0]]
        y21 = float(tr2_pt1[1])
        x21 = float(tr2_pt1[0])
        tr2_pt2 = indices2[store2[i][1]]
        y22 = float(tr2_pt2[1])
        x22 = float(tr2_pt2[0])
        tr2_pt3 = indices2[store2[i][2]]
        y23 = float(tr2_pt3[1])
        x23 = float(tr2_pt3[0])

        triangle2 = np.array([tr2_pt1, tr2_pt2, tr2_pt3], np.int32)

        r1 = cv2.boundingRect(triangle1)
        r2 = cv2.boundingRect(triangle2)

        a1 = 1/2.0*(abs(x11*(y12-y13)+x12*(y13-y11)+x13*(y11-y12)))
        a2 = 1/2.0*(abs(x21*(y22-y23)+x22*(y23-y21)+x23*(y21-y22)))

        (x, y, w, h) = r2
        k = 0
        for j in range(y, y+h+1):
            for i in range(x, x+w+1):
                value = cv2.pointPolygonTest(triangle2, (i, j), False)
                if value != -1 and a2 != 0:
                    y1 = float(j)
                    x1 = float(i)

                    s1 = 1/2*(abs(x1*(y22-y23)+x22*(y23-y1)+x23*(y1-y22)))
                    s2 = 1/2*(abs(x21*(y1-y23)+x1*(y23-y21)+x23*(y21-y1)))
                    s3 = 1/2*(abs(x21*(y22-y1)+x22*(y1-y21)+x1*(y21-y22)))

                    a = s1/a2
                    b = s2/a2
                    c = s3/a2

                    x2 = a*x11+b*x12+c*x13
                    y2 = a*y11+b*y12+c*y13

                    y2f, y2i = math.modf(y2)
                    x2f, x2i = math.modf(x2)
                    x2i = int(x2i)
                    y2i = int(y2i)

                    output[j, i] = (1.0-x2f)*(1.0-y2f)*img_source1_copy[y2i,
                                                                        x2i]+x2f*(1.0-y2f)*img_source1_copy[y2i, x2i+1]
                    output[j, i] = output[j, i]+x2f*y2f*img_source1_copy[y2i +
                                                                         1, x2i+1]+(1.0-x2f)*y2f*img_source1_copy[y2i+1, x2i]


wrap()
output = output.astype(np.uint8)


cv2.imshow("image", output)
cv2.waitKey(0)
cv2.destroyAllWindows()


# string facual points in dictionary for easy access during makeup transfer
dict = {}
dict_lip = {}
indices = np.array(indices2, np.int32)
convex_face = cv2.convexHull(indices)
rect1 = cv2.boundingRect(convex_face)
(x, y, w, h) = rect1
convex_lips = indices[48:60]
convex_hole = indices[60:68]
convex_leye = indices[36:42]
covex_reye = indices[42:48]
convex_lbrow = indices[17:22]
convex_rbrow = indices[22:27]
test = []

for j in range(y, y+h+1):
    for i in range(x, x+w+1):
        c_value = -1
        b = -1
        if cv2.pointPolygonTest(convex_hole, (i, j), False) != -1 or cv2.pointPolygonTest(covex_reye, (i, j), False) != -1 or cv2.pointPolygonTest(convex_leye, (i, j), False) != -1:
            c_value = 3
            b = 0
        elif cv2.pointPolygonTest(convex_lips, (i, j), False) == 1:
            dict_lip[(j, i)] = 2
            test.append((i, j))
            c_value = 2
            b = 0
        elif cv2.pointPolygonTest(convex_lbrow, (i, j), False) == 1 or cv2.pointPolygonTest(convex_rbrow, (i, j), False) == 1:
            c_value = 1
            b = 0.3
        elif cv2.pointPolygonTest(convex_face, (i, j), False) == 1:
            c_value = 1
            b = 1

        if c_value != -1 and b != -1:
            dict[(j, i)] = (c_value, b)


def lab_conversion(img):
    image_LAB = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    return image_LAB


def layer_decomposition(img):
    lightness_layer = img[:, :, 0]
    face_structure_layer = cv2.bilateralFilter(lightness_layer, 9, 75, 75)
    detail_layer = lightness_layer - face_structure_layer
    color_layer_a = img[:, :, 1]
    color_layer_b = img[:, :, 2]
    return [face_structure_layer, detail_layer, color_layer_a, color_layer_b]


def sharpImage(img, sigma, k, p):
    sigma_large = sigma * k
    G_small = cv2.GaussianBlur(img, (0, 0), sigma)
    G_large = cv2.GaussianBlur(img, (0, 0), sigma_large)
    result = (1+p) * G_small - p * G_large
    return result


def softThreshold(img, epsilon, phi):
    res = np.zeros(img.shape)
    img_bright = img >= epsilon
    img_dark = img < epsilon
    res[img_bright] = 1.0
    res[img_dark] = 1.0 + np.tanh(phi * (img[img_dark] - epsilon))
    return res


def XDoG(img, sigma, k_sigma, p, epsilon, phi):
    S = sharpImage(img, sigma, k_sigma, p)
    SI = np.multiply(img, S)
    final_image = softThreshold(SI, epsilon, phi)
    return final_image


example_lab = lab_conversion(output)

subject_lab = lab_conversion(img_source2)


(example_face, example_detail, example_color_a,
 example_color_b) = layer_decomposition(example_lab)
example_face = example_face.astype(np.float64)
example_detail = example_detail.astype(np.float64)
example_color_a = example_color_a.astype(np.float64)
example_color_b = example_color_b.astype(np.float64)
(subject_face, subject_detail, subject_color_a,
 subject_color_b) = layer_decomposition(subject_lab)
subject_face = subject_face.astype(np.float64)
subject_detail = subject_detail.astype(np.float64)
subject_color_a = subject_color_a.astype(np.float64)
subject_color_b = subject_color_b.astype(np.float64)


def skin_detail_transfer(src_img, tar_img, delta_i, delta_e):
    result_detail = np.copy(tar_img)
    for key in dict.keys():
        if dict[key][0] == 1:
            result_detail[key[0]][key[1]] = delta_i * tar_img[key[0]
                                                              ][key[1]] + delta_e * src_img[key[0]][key[1]]
    return result_detail


def color_transfer(src_img, tar_img, gamma):
    result_color = np.copy(tar_img)
    for key, value in dict.items():
        if (value[0] != 3):
            result_color[key[0]][key[1]] = (
                1 - gamma) * tar_img[key[0]][key[1]] + gamma * src_img[key[0]][key[1]]
        else:
            result_color[key[0]][key[1]] = tar_img[key[0]][key[1]]

    return result_color


def highlight_and_shading_transfer(src_img, tar_img):
    src_img = src_img.astype(np.uint8)
    tar_img = tar_img.astype(np.uint8)
    derivative_src = cv2.Laplacian(src_img, cv2.CV_64F)
    derivative_tar = cv2.Laplacian(tar_img, cv2.CV_64F)
    derivative_res = np.copy(derivative_tar)
    l = 0
    for key, value in dict.items():
        if ((value[1] * abs(derivative_src[key[0]][key[1]])) > abs(derivative_tar[key[0]][key[1]])):
            derivative_res[key[0]][key[1]] = derivative_src[key[0]][key[1]]
        else:
            derivative_res[key[0]][key[1]] = derivative_tar[key[0]][key[1]]
    result_structure = np.copy(derivative_res).astype(np.uint8)
    result_structure = result_structure+cv2.GaussianBlur(tar_img, (5, 5), 0)

    return result_structure


def histogram_equalize(img):
    equalized = cv2.equalizeHist(img)
    return equalized


def gaussian(x, sigma):
    return (1.0 / (math.sqrt(2 * math.pi * (sigma * 2)))) * math.exp(- (x * 2) / (2 * (sigma ** 2)))


def dist_pixels(pi, pj, qi, qj):
    return math.sqrt(((pi - qi) ** 2) + ((pj - qj) ** 2))


def lip_transfer(src_img, tar_img, sigma):
    result_img = np.copy(tar_img)
    source_l = src_img[:, :, 0]
    target_l = tar_img[:, :, 0]
    source_a = src_img[:, :, 1]
    target_a = tar_img[:, :, 1]
    source_b = src_img[:, :, 2]
    target_b = tar_img[:, :, 2]

    source_l = source_l.astype(np.uint8)
    target_l = target_l.astype(np.uint8)
    source_l = cv2.equalizeHist(source_l)
    target_l = cv2.equalizeHist(target_l)

    source_l = source_l.astype(np.float64)
    target_l = target_l.astype(np.float64)
    for key1 in dict_lip.keys():
        i = key1[0]
        j = key1[1]
        intensity_tar = target_l[i][j]
        max_gaussian = -1
        max_i = i
        max_j = j

        for key2 in dict_lip.keys():
            x = key2[0]
            y = key2[1]
            intensity_src = source_l[x][y]
            intensity_diff = abs(intensity_src - intensity_tar)
            distance = dist_pixels(i, j, x, y)
            gaussian_val = gaussian(distance, sigma) * \
                gaussian(intensity_diff, sigma)
            if (gaussian_val > max_gaussian):
                max_gaussian = gaussian_val
                max_i = x
                max_j = y

        result_img[i][j][0] = source_l[max_i][max_j]
        result_img[i][j][1] = source_a[max_i][max_j]
        result_img[i][j][2] = source_b[max_i][max_j]

    return result_img


result_detail = skin_detail_transfer(example_detail, subject_detail, 0, 1)
result_color_a = color_transfer(example_color_a, subject_color_a, 0.8)
result_color_b = color_transfer(example_color_b, subject_color_b, 0.8)
result_structure = highlight_and_shading_transfer(example_face, subject_face)

result_l = result_detail+result_structure

result_l = result_l.reshape(result_l.shape[0], result_l.shape[1], 1)
result_color_a = result_color_a.reshape(
    result_color_a.shape[0], result_color_a.shape[1], 1)
result_color_b = result_color_b.reshape(
    result_color_b.shape[0], result_color_b.shape[1], 1)
result_lab = np.concatenate((result_l, result_color_a, result_color_b), axis=2)
result_lab = lip_transfer(example_lab, result_lab, 1)
result_lab = result_lab.astype(np.uint8)
result_img = cv2.cvtColor(result_lab, cv2.COLOR_LAB2BGR)


cv2.imshow("image", result_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
