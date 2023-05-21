import matplotlib.pyplot as plt
import matplotlib
import numpy as np

from utils import torch_utils
from utils.datasets import LoadImages

matplotlib.use('TkAgg')
import torch
from models.experimental import attempt_load
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
#model = attempt_load("./best.pt")  # load FP32 model
# print(model)
# print(model.model[4].cv3.conv)
# print(model.model[5].conv)
# print(model.model[6].cv3.conv)
# print(model.model[7].conv)
# print(model.model[9].cv2.conv)
def hook_fn(model,inout,output):
    feature_map = output[0].cpu().detach().numpy()
    #Obtain thermal map
    # min_value = np.min(feature_map)  # obtain feature map mim
    # offset = np.abs(min_value)  #count offset
    # adjusted_feature_map = feature_map + offset  # add offset
    # plt.imshow(adjusted_feature_map[0],cmap='jet')
    # plt.axis('off')
    # plt.colorbar()

    #obtain feature map
    plt.imshow(feature_map[0])
    plt.savefig('1.png')
# hook = model.model[4].cv3.conv.register_forward_hook(hook_fn)
# #inputs = torch.randn(1,3,640,640)
#
# output = model(img)


