import torch, numpy as np, random, os, string, secrets, torchvision, importlib, sys
from torchvision.datasets.folder import make_dataset, find_classes
from tqdm import tqdm
import cv2

IMG_EXTENSIONS = (
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP', '.tiff'
)

def merge_options(opts: dict, extra_opts: dict):
    return opts.update(extra_opts)



def import_module_from_file(file_path):
    # Get the directory path and file name
    directory = os.path.dirname(file_path)
    module_file = os.path.basename(file_path)
    module_name = os.path.splitext(os.path.basename(file_path))[0]

    # Add the directory containing the file to sys.path
    sys.path.append(directory)

    # Load the module dynamically
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    return module


def get_pretrained_model_params(network_name, dict_params_net):
    if 'resnet' in network_name:
        orig_model = getattr(torchvision.models, network_name)(weights='IMAGENET1K_V1')
        dict_params_model = dict(orig_model.state_dict())

        for name, name2 in zip(list(dict_params_net.keys())[:-8], list(dict_params_model.keys())[:-2]):
            dict_params_net[name].data.copy_(dict_params_model[name2].data)

        dict_params_net['classifiers.orig.2.weight'].data.copy_(dict_params_model['fc.weight'].data)
        dict_params_net['classifiers.orig.2.bias'].data.copy_(dict_params_model['fc.bias'].data)
        dict_params_net['classifiers.cape.2.weight'].data.copy_(dict_params_model['fc.weight'].data)
        dict_params_net['classifiers.cape.2.bias'].data.copy_(dict_params_model['fc.bias'].data)
    elif 'swin' in network_name:
        orig_model = getattr(torchvision.models, network_name)(weights='IMAGENET1K_V1')
        dict_params_model = dict(orig_model.state_dict())
        
        for name, name2 in zip(list(dict_params_net.keys())[:-7], list(dict_params_model.keys())[:-2]):
            dict_params_net[name].data.copy_(dict_params_model[name2].data)
        dict_params_net['classifiers.orig.2.weight'].data.copy_(dict_params_model['head.weight'].data)
        dict_params_net['classifiers.orig.2.bias'].data.copy_(dict_params_model['head.bias'].data)
        dict_params_net['classifiers.cape.2.weight'].data.copy_(dict_params_model['head.weight'].data)
        dict_params_net['classifiers.cape.2.bias'].data.copy_(dict_params_model['head.bias'].data)
        
    return dict_params_net


def set_random_seed(seed: int) -> None:
    print("Setting seeds ...... \n")
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic=  True
    
def worker_init_fn(worker_id):                                                        
    np.random.seed(np.random.get_state()[1][0] + worker_id)
    
def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")

def random_experiment_name(length=4):
    alphabet = string.ascii_letters + string.digits
    return ''.join(secrets.choice(alphabet) for i in range(length))

def upsampling(heatmap_campe, size_upsample):
    heatmap_campe = torch.nn.functional.interpolate(heatmap_campe, size_upsample, mode='bicubic', align_corners=True)
    return heatmap_campe.cpu().detach().numpy()[0]

def normalize(heatmap_campe):
    heatmap_campe = (heatmap_campe - np.min(heatmap_campe)) / (np.max(heatmap_campe) - np.min(heatmap_campe) + 1e-10) 
    return heatmap_campe

def show_cam_on_image(img: np.ndarray,
                      mask: np.ndarray,
                      use_rgb: bool = False,
                      colormap: int = cv2.COLORMAP_JET,
                      image_weight: float = 0.5) -> np.ndarray:
    """source: pytorch_grad_cam package"""
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), colormap)
    if use_rgb:
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    heatmap = np.float32(heatmap) / 255

    if np.max(img) > 1:
        raise Exception("The input image should np.float32 in the range [0, 1]")

    if image_weight < 0 or image_weight > 1:
        raise Exception(f"image_weight should be in the range [0, 1]. Got: {image_weight}")

    cam = (1 - image_weight) * heatmap + image_weight * img
    cam = cam / np.max(cam)
    return np.uint8(255 * cam)

def inverse_normalize(tensor, mean, std):
    '''
    does not work for batch of images, only works on single image 
    tensor shape: (1, nc, h, w)
    '''
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)
    return tensor