import cv2
import numpy as np
import torch
import torchvision.transforms as transforms


class Image:
    def __init__(self):
        #img data
        self.img = cv2.imread("indoor/clear/1423.png")
        self.threshold = 50
        self.light = False
        self.collision = False

        #model to detect collision
        self.model_path = 'dpt_levit_224.pt'
        self.path = torch.load(self.model_path,map_location=torch.device('cpu'))
        self.midas = torch.hub.load("intel-isl/MiDaS", "MiDaS")
        self.midas.to(torch.device('cpu'))
        self.midas.eval()


    def update(self,img):
        self.img = img


    def check_luminosity(self):
        processed = cv2.cvtColor(self.img, cv2.COLOR_BGR2HSV)
        h,s,v = cv2.split(processed)
        avg_v = np.mean(v)
        percent = (avg_v/255)*100
        print(percent)
        print(avg_v)

        if percent < self.threshold:
            print("dark")
            self.light = True
            v = np.clip(v + 100, 0, 255)
            processed = cv2.merge((h,s,v))
            img = cv2.cvtColor(processed, cv2.COLOR_BGR2HSV)
            cv2.imshow("", img)

        elif percent >=self.threshold:
            print("light")
            self.light = False
            cv2.imshow("", self.img)

        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def check_collision(self):
        img_rgb = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img_rgb, (384, 384))
        img_tensor = transforms.ToTensor()(img_resized).unsqueeze(0)

        with torch.no_grad():
            prediction = self.midas(img_tensor.to(torch.device('cpu')))

        depth_map = prediction.squeeze().cpu().numpy()

        depth_map_resized = cv2.resize(depth_map, (self.img.shape[1], self.img.shape[0]))

        depth_map_normalized = (depth_map_resized - depth_map_resized.min()) / (
                    depth_map_resized.max() - depth_map_resized.min()) * 255

        depth_map_uint8 = depth_map_normalized.astype(np.uint8)

        depth_map_uint8_color = cv2.cvtColor(depth_map_uint8, cv2.COLOR_GRAY2BGR)

        img_with_depth = np.hstack((self.img, depth_map_uint8_color))
        img_with_depth = cv2.cvtColor(img_with_depth, cv2.COLOR_BGR2GRAY)
        threshold_distance = 1.0
        min_area_threshold = 10

        obstacle_mask = img_with_depth < threshold_distance
        contours, _ = cv2.findContours(obstacle_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        collision_risk = False
        self.collision = False
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > min_area_threshold:
                collision_risk = True
                self.collision = True

        print(collision_risk)

        cv2.imshow("Image with Depth Map", img_with_depth)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def process_image(self):
        # self.update(img)
        self.check_luminosity()
        self.check_collision()

img = Image()
img.process_image()