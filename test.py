import os
from PIL import Image
from model import FCN
from config import config
from dataloader.dataloader import get_val_loader
from dataloader.RGBXdataset import RGBXDataset
import cv2
import torch
import numpy as np

model = 'FCN'
net = FCN(config.num_classes, config.hyper_parm)


#test
def evaluate(net, val_loader, val_dataset, show_image=True):

    dataloader = iter(val_loader)

    for idx in range(config.num_eval_imgs):
        data = dataloader.next()

        img = data['data']
        label = data['label']
        modal_x = data['modal_x']
        name = data['fn']
        name = "".join(name)

        net.cuda()
        out = net([img.float().cuda(), modal_x.float().cuda()])
        pred = out.argmax(dim=1).squeeze().data.cpu().numpy()

        if show_image:
            fn = name + '.png'
            # save colored result
            result_img = Image.fromarray(pred.astype(np.uint8), mode='P')
            class_colors = val_dataset.get_class_colors()
            palette_list = list(np.array(class_colors).flat)
            if len(palette_list) < 768:
                palette_list += [0] * (768 - len(palette_list))
            result_img.putpalette(palette_list)
            result_img.save(os.path.join(config.save_img_dir, fn))

            # save raw result
            # fn1 = name + '_raw.png'
            # cv2.imwrite(os.path.join(config.save_img_dir, fn1), pred)

if __name__ == "__main__":
    net.load_state_dict(torch.load(config.model_path))
    val_loader, val_dataset = get_val_loader(RGBXDataset)
    evaluate(net, val_loader, val_dataset, show_image=config.show_image)