from PIL import Image
import cv2
import numpy as np
import scipy.ndimage as inter

class FusionPreprocess():
    def __init__(self, image_path, norm = True, skew = True, scale = True, denoise = True, thin = True):
        self.image_path = image_path
        self.norm = norm
        self.skew = skew
        self.scale = scale
        self.denoise = denoise
        self.thin = thin

    def correct_skew(self, image, delta=1, limit=5):
        def determine_score(arr, angle):
            data = inter.rotate(arr, angle, reshape=False, order=0)
            histogram = np.sum(data, axis=1, dtype=float)
            score = np.sum((histogram[1:] - histogram[:-1]) ** 2, dtype=float)
            return score

        thresh = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

        scores = []
        angles = np.arange(-limit, limit + delta, delta)
        for angle in angles:
            score = determine_score(thresh, angle)
            scores.append(score)

        best_angle = angles[scores.index(max(scores))]

        (h, w) = image.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, best_angle, 1.0)
        corrected = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, 
                                   borderMode=cv2.BORDER_REPLICATE)

        return corrected


    def preprocess(self):
        image = cv2.imread(self.image_path, cv2.IMREAD_GRAYSCALE)
        if self.norm:
            norm_img = np.zeros((image.shape[0], image.shape[1]))
            image = cv2.normalize(image, norm_img, 0, 255, cv2.NORM_MINMAX)
        
        if self.skew:
            image = self.correct_skew(image)

        if self.scale:
            image = Image.fromarray(image)
            length_x, width_y = image.size
            factor = min(1, float(1024.0 / length_x))
            size = int(factor * length_x), int(factor * width_y)
            image_resized = image.resize(size, Image.Resampling.LANCZOS)
            image_resized.save("temp_image.jpg", dpi=(300, 300))
            PIL_image = Image.open("temp_image.jpg")
            image = np.array(PIL_image)

        if self.denoise:
            denoised_image = cv2.fastNlMeansDenoising(image, None, 30, 7, 21)
            image = cv2.bilateralFilter(denoised_image, 9, 75, 75)

        if self.thin:
            kernel = np.ones((5,5), np.uint8)
            image = cv2.erode(image, kernel, iterations=1)

        final_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        return final_image