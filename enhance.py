import cv2, os, argparse
import glob, torch, shutil
import numpy as np


from PIL import Image
from os.path import dirname, join, basename, splitext
from tqdm import tqdm
from models.gfpgan import GFPGANer
from basicsr.utils import imwrite

import torchvision.transforms as transforms
from models.face_parsing import BiSeNet


def str2bool(v):
    """ NOTE: Type like bool lower become large for bash.
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def main():
    # ===> SR model configuration
    bg_upsampler = None  # use RealESRGAN for background upsampling not supported now
    arch = 'clean'
    channel_multiplier = 2
    model_name = 'GFPGANv1.3'
    url = 'https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth'

    model_path = os.path.join('models/gfpgan/weights', model_name + '.pth')
    restorer = GFPGANer(
        model_path=model_path,
        upscale=opt.upscale,
        arch=arch,
        channel_multiplier=channel_multiplier,
        bg_upsampler=bg_upsampler)
    
    # ===> parsing model configuration
    net = BiSeNet(n_classes=19)
    net.cuda()
    net.load_state_dict(torch.load('./models/face_parsing/weights/79999_iter.pth'))
    net.eval()

    to_tensor = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),])

    
    save_path = join(opt.save_path, basename(opt.video_path)[:-4])
    # ===> get input videos information
    cap = cv2.VideoCapture(str(opt.video_path))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # use mp4v coding
    output_path = join(save_path, basename(opt.video_path)[:-4] + '_teeth_enhanced.mp4')
    out_video = cv2.VideoWriter(output_path, fourcc, fps, (frame_height, frame_width))

    frame_idx = -1
    qbar = tqdm(total=frame_count, desc='processing frames')
    while cap.isOpened():
        still_reading, frame_rgb = cap.read()

        if not still_reading:
            cap.release()
            break

        frame_idx += 1
        qbar.update()
    
        # ===> restore faces and background if necessary
        cropped_faces, restored_faces, restored_img = restorer.enhance(
            frame_rgb,
            has_aligned=opt.aligned,
            only_center_face=opt.only_center_face,
            paste_back=True,
            weight=opt.weight)

        # ===> save restored img
        if restored_img is not None:
            if opt.suffix is not None:
                save_restore_path = os.path.join(save_path, 'restored_imgs', f'{str(frame_idx).zfill(5)}_{opt.suffix}.{opt.ext}')
            else:
                save_restore_path = os.path.join(save_path, 'restored_imgs', f'{str(frame_idx).zfill(5)}.{opt.ext}')
            
            if opt.save_restored_img:
                imwrite(restored_img, save_restore_path)
        
        # ===> parsing face segmentation using BiSeNet
        image = Image.fromarray(cv2.cvtColor(frame_rgb, cv2.COLOR_BGR2RGB))
        with torch.no_grad():
            img = to_tensor(image)
            img = torch.unsqueeze(img, 0)
            img = img.cuda()
            out = net(img)[0]
            vis_parsing = out.squeeze(0).cpu().numpy().argmax(0)
        
        lqfacemask = np.full_like(frame_rgb, 255)

        # ===> 11 is the label for teeth
        for pi in [11]:
            index = np.where(vis_parsing == pi)
            lqfacemask[index[0], index[1], :] = 0

        lqfacemask = cv2.GaussianBlur(lqfacemask, (7, 7), 5)
        lqfacemask = lqfacemask / 255.
        
        # ===> blending & saving the results
        teeth_enhanceimg = frame_rgb * lqfacemask + restored_img * (1 - lqfacemask)
        
        teeth_enhance_path = (join(save_path, 'teeth_enhance', f'{str(frame_idx).zfill(5)}.{opt.ext}'))
        imwrite(teeth_enhanceimg, teeth_enhance_path)
        
        out_video.write(np.uint8(teeth_enhanceimg))
    
    out_video.release()
    print(f'Successfully restored video: {basename(opt.video_path)}')
            


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Face Restoration using GFPGAN')
    parser.add_argument('--video_path', type=str, required=True, help='Input directory containing face images')
    parser.add_argument('--save_path', type=str, required=True, help='Output directory containing face images for teetch enhance')

    # we use version to select models, which is more user-friendly
    parser.add_argument(
        '--version', type=str, default='1.3', help='GFPGAN model version. Option: 1 | 1.2 | 1.3. Default: 1.3')
    parser.add_argument(
        '--upscale', type=int, default=1, help='The final upsampling scale of the image. Default: 2')

    parser.add_argument(
        '--bg_tile',
        type=int,
        default=400,
        help='Tile size for background sampler, 0 for no tile during testing. Default: 400')
    parser.add_argument('--suffix', type=str, default=None, help='Suffix of the restored faces')
    parser.add_argument('--only_center_face', action='store_true', help='Only restore the center face')
    parser.add_argument('--aligned', action='store_true', help='Input are aligned faces')
    parser.add_argument(
        '--ext',
        type=str,
        default='png',
        help='Image extension. Options: auto | jpg | png, auto means using the same extension as inputs. Default: auto')
    parser.add_argument('-w', '--weight', type=float, default=0.5, help='Adjustable weights.')
    parser.add_argument('--save_restored_img', type=str2bool, default=False, help='Save the SRImg (super-resolution image) for each face')
    
    opt = parser.parse_args()
    
    # ------------------------ input & output ------------------------
    main()