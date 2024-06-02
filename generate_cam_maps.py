import argparse
import torch
import torch.nn.functional as F
import os, sys, copy, json, importlib, shutil, cv2, numpy as np
import torch, torchvision
import matplotlib.pyplot as plt
import helpers
from configs.arguments import parse_args
from configs import transforms
from models.model import Net
from PIL import Image

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_path', type=str, help='image path')
    parser.add_argument('--model_path', type=str, help='model path')
    parser.add_argument('--dataset_transform', default='cub_transform', type=str, help='dataset transform function')
    parser.add_argument('--image_size', type=int, default=448, help='image size')
    parser.add_argument('--save_dir', type=str, help='directory to save cam maps')
    parser.add_argument('--cam_maps', type=str, default='cape', choices=['cape', 'mu-cape', 'cam'], 
                    help='type of cam map type')
    
    args = parser.parse_args()
    
    return args

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    opts =  vars(args)
    # additional options
    opts['network'] = 'resnet50'
    opts['pretrained'] = True
    opts['num_class'] = 200 # for cub    
    opts['cls_cols_dict'] = {'orig': opts['num_class'], 'cape': opts['num_class']}
    
    # dataset
    dataset_transform_func = getattr(transforms, f"{opts['dataset_transform']}")
    _, val_transform = dataset_transform_func(opts['image_size'])
    
    # model
    model = Net(opts)
    if os.path.exists(opts['model_path']):
        model.load_state_dict(torch.load(opts['model_path']))        
    else:
        raise ValueError('File not exists in the reload path: {}'.format(opts['model_path']))
    
    model = model.to(device)
    model.eval()
    
    with torch.no_grad():
        image = val_transform(Image.open(opts['image_path'])).unsqueeze(0).to(device)
        img = helpers.inverse_normalize(tensor=copy.deepcopy(image), mean=transforms.mean, std=transforms.std) # channels, height, width
        img = img[0].cpu().detach().numpy().transpose(1, 2, 0) # height, width, channels
        w, h = image.size(-1), image.size(-2)
        outputs = model(image)
        
        if opts['cam_maps']=='cam':
            output = outputs['orig']['outcome']
            pred_val, pred = torch.max(output.softmax(dim=1), 1)
            cam_map = F.relu(outputs['orig']['weighted_contribution'][:, pred.item(),...])
            cam_values = helpers.normalize(cam_map.cpu().detach().numpy())
        elif opts['cam_maps']=='cape':
            output = outputs['cape']['outcome']
            pred_val, pred = torch.max(output, 1)
            cam_values = outputs['cape']['weighted_contribution'][:, pred.item(),...]
            cam_map = cam_values
        elif opts['cam_maps']=='mu-cape':
            output = outputs['cape']['outcome']
            # pred_val, pred = torch.topk(output, 2, -1)
            # pred_val, pred = pred_val[0][1], pred[0][1]
            pred_val, pred = torch.max(output, 1)
            cam_values = helpers.normalize(outputs['cape']['logcampe_clip0'][:, pred.item(),...].cpu().detach().numpy())
            cam_map = outputs['cape']['logcampe_clip0'][:, pred.item(),...]
        else:
            raise ValueError('Invalid settings!!!!')

        heatmap_cam = helpers.upsampling(cam_map.unsqueeze(0), (w, h))
        heatmap_cam = helpers.normalize(heatmap_cam) 
        if torch.is_tensor(cam_values):
            cam_values = cam_values.cpu().detach().numpy()
        
        cam_img = helpers.show_cam_on_image(img/img.max(), heatmap_cam[0], use_rgb=True)
        
        x_ = img.shape[0] // cam_values[0].shape[0]
        y_ = img.shape[1] // cam_values[0].shape[0]
        
        for x_index in range(cam_values[0].shape[0]):
            for y_index in range(cam_values[0].shape[1]):
                value = cam_values[0][y_index, x_index]
                if value > (5*cam_values[0].max()/100):
                    x = x_index * x_
                    y = y_index * y_
                    start_point = (x,y)
                    end_point = (x+x_,y+y_) 
                    cam_img = cv2.rectangle(cam_img, start_point, end_point, color=(0, 0, 0), thickness=1, lineType=cv2.LINE_AA)
                    if opts['cam_maps']=='cam':
                        text = str(int(round(100*value, 1)))
                        if len(text)==1:
                            location = ((start_point[0]+end_point[0])//2 - 7, (start_point[1]+end_point[1])//2)
                        else:
                            location = ((start_point[0]+end_point[0])//2 - 10, (start_point[1]+end_point[1])//2)
                    else:
                        text = str(round(100*value, 1) if opts['cam_maps']=='cape' else int(round(100*value, 1)))
                        location = ((start_point[0]+end_point[0])//2 - 7, (start_point[1]+end_point[1])//2)
                    cam_img = cv2.putText(cam_img, text, org=location,
                            fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.4,
                            color=(0, 0, 0), thickness=1, lineType=cv2.LINE_AA)
        
        cv2.imwrite(f"{opts['save_dir']}/original.jpg", cv2.cvtColor(np.uint8(255*img/img.max()), cv2.COLOR_RGB2BGR))
        cv2.imwrite(f"{opts['save_dir']}/{opts['cam_maps']}_pred_{pred.item()}_confidence_score_{round(pred_val.item(),3)}.jpg", cv2.cvtColor(np.uint8(cam_img), cv2.COLOR_RGB2BGR))
    

if __name__ == '__main__':
    torch.autograd.set_detect_anomaly(True)
    args = parse_args()
    main(args)
