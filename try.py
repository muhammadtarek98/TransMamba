import torch,torchvision,cv2
from PIL import Image
from collections import OrderedDict
from waternet.waternet.net import WaterNet
from waternet.waternet.data import transform
import numpy as np
def load_model(device:torch.device,
               ckpt_dir:str="/home/muahmmad/projects/Image_enhancement/waternet/weights/waternet_exported_state_dict-daa0ee.pt"):
    model=WaterNet()
    ckpt=torch.load(f=ckpt_dir,map_location=device,weights_only=True)
    print(ckpt.keys())
    model.load_state_dict(state_dict=ckpt)
    model=model.to(device=device)
    return model

def transform_array_to_image(arr):
    arr=np.clip(a=arr,a_min=0,a_max=1)
    arr=(arr*255.0).astype(np.uint8)
    return arr
def transform_image(img):
    trans=torchvision.transforms.Compose(transforms=[
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Resize(size=(720,720),interpolation=torchvision.transforms.InterpolationMode.NEAREST_EXACT),
        torchvision.transforms.Normalize(mean=[0,0,0],
                                         std=[1,1,1])
    ])
    raw_image_tensor=trans(img)
    wb, gc, he=transform(img)
    wb_tensor=trans(wb)
    gc_tensor=trans(gc)
    he_tensor=trans(he)
    return {"X":torch.unsqueeze(input=raw_image_tensor,dim=0),
            "wb":torch.unsqueeze(input=wb_tensor,dim=0),
            "gc":torch.unsqueeze(input=gc_tensor,dim=0),
            "he":torch.unsqueeze(input=he_tensor,dim=0)}
if __name__ =="__main__":
    image=