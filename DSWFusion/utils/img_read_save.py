import numpy as np
import cv2
import os
from skimage.io import imsave

import numpy as np
import os
from skimage.io import imsave


def image_read_cv2(path, mode='RGB'):
    img_BGR = cv2.imread(path).astype('float32')
    assert mode == 'RGB' or mode == 'GRAY' or mode == 'YCrCb', 'mode error'
    if mode == 'RGB':
        img = cv2.cvtColor(img_BGR, cv2.COLOR_BGR2RGB)
    elif mode == 'GRAY':  
        img = np.round(cv2.cvtColor(img_BGR, cv2.COLOR_BGR2GRAY))
    elif mode == 'YCrCb':
        img = cv2.cvtColor(img_BGR, cv2.COLOR_BGR2YCrCb)
    return img

def img_save(image,imagename,savepath):
    if not os.path.exists(savepath):
        os.makedirs(savepath)
    # Gray_pic
    # imsave(os.path.join(savepath, "{}.png".format(imagename)),image)

    imsave(os.path.join(savepath, "{}.png".format(imagename)), image.astype(np.uint8));

def img_save1(tensor_or_np, name, save_dir):
    """
        tensor_or_np: torch.Tensor 或 np.ndarray
                      形状可以是 (H,W), (H,W,C), (B,H,W,C), (C,H,W) 等
        name: 不含后缀的文件名
        save_dir: 保存目录
        """
    # 1) 统一转成 numpy
    if hasattr(tensor_or_np, 'detach'):  # torch tensor
        img = tensor_or_np.detach().cpu().numpy()
    else:
        img = np.asarray(tensor_or_np)

    # 2) 去掉 batch 维
    if img.ndim == 4 and img.shape[0] == 1:  # (1,H,W,C) or (1,C,H,W)
        img = img[0]

    # 3) 调整通道维到最后一维
    if img.ndim == 3 and img.shape[0] in (1, 3, 4):  # (C,H,W)
        img = np.transpose(img, (1, 2, 0))

    # 4) 去掉单通道的通道维：(H,W,1) -> (H,W)
    if img.ndim == 3 and img.shape[-1] == 1:
        img = img.squeeze(-1)

    # 5) 归一化到 0-255 并转 uint8
    if img.dtype != np.uint8:
        if img.max() <= 1.0:  # 0-1 float
            img = (img * 255).round()
        else:  # 0-255 float
            img = img.round()
        img = img.astype(np.uint8)

    # 6) 保存
    os.makedirs(save_dir, exist_ok=True)
    imsave(os.path.join(save_dir, f"{name}.png"), img)