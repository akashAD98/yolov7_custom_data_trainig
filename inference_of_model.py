import numpy as np 
from time import sleep,time
import torch
import dxcam
from lib.SCgrab import grab_screen,monitorDimensions
from torchvision import transforms
from utils.datasets import letterbox
from utils.general import non_max_suppression
from utils.plots import output_to_keypoint, plot_skeleton_kpts

#Variables
total_fps = 0
frame_width = 416
frame_height = 416
NMS_conf = 0.3

#Loading model
device = torch.device("cuda:0")
weigths = torch.load('best.pt')
model = weigths['model']
model = model.half().to(device)
_ = model.eval()


#Get Monitor Dimensions
monitor = monitorDimensions(416)

# #start screencapture
# camera = dxcam.create(region=monitor)
# camera.start(target_fps=160, video_mode=True)


with torch.no_grad():
    while True:
        orig_image = np.array(grab_screen(region=monitor))
        image = letterbox(orig_image, (frame_width), stride=64, auto=True)[0]
        image_ = image.copy()
        image = transforms.ToTensor()(image)
        image = torch.tensor(np.array([image.numpy()]))
        image = image.to(device)
        image = image.half()


        # Get the start time.
        start_time = time()

        output = model(image)[0]
        print(output.shape[0])
        # Get the end time.
        end_time = time()
        # output = non_max_suppression(np.array(output), conf_thres=0.25, iou_thres=0.45, classes=0, agnostic=False, multi_label=False,labels=())
        # Get the fps.
        fps = 1 / (end_time - start_time)
        print(fps)
        
