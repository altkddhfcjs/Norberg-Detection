import math
import argparse
import os, torch
import matplotlib.pyplot as plt
import PIL.Image as Image
from PIL import ImageDraw
from PIL import ImageFont
import numpy as np

import detectron2.utils.comm as comm
from detectron2.config import get_cfg
from detectron2.engine import default_argument_parser, default_setup, launch, DefaultPredictor
from detectron2.solver import build_lr_scheduler, build_optimizer
from detectron2.solver.build import get_default_optimizer_params, maybe_add_gradient_clipping
from detectron2 import model_zoo
from detectron2.modeling import build_model


#os.system("sudo conda install pytorch==1.8.0 torchvision==0.9.0 torchaudio==0.8.0 cudatoolkit=11.1 -c pytorch -c conda-forge")

parser = argparse.ArgumentParser(prog='Norberg Detection'
    , formatter_class=argparse.RawTextHelpFormatter
    , description=
        'Norberg Detection\n' 
        'output : norgerg info ( img, txt )'
)
parser.add_argument('--inpfolder', help="path norberg input directory") #, metavar="INPUTFOLDER", dest="INPUTROLDER")
parser.add_argument('--cpu', default=True, help="path norberg input directory") #, metavar="INPUTFOLDER", dest="INPUTROLDER")
parser.add_argument('-prePath',help='path of output file', metavar='PREPATH',dest='PREPATH')

args = parser.parse_args()

def get_config(config_file, is_cpu):
    print("CUDA: {} / torch: {} / cuda available: {}".format(torch.version.cuda, torch.__version__, torch.cuda.is_available()))
    cfg = get_cfg()
    cfg.merge_from_file(config_file)
    
    if is_cpu:
        print("Running CPU")
        cfg.MODEL.DEVICE = "cpu"
    else:
        if torch.cuda.is_available():
            print("Error: CUDA Failed")
            print("Running CPU")
            cfg.MODEL.DEVICE = "cpu"
        else:
            cfg.MODEL.DEVICE = "cuda"
            print("Running GPU!")
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.005
    cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = 0.005
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
    cfg.MODEL.ROI_KEYPOINT_HEAD.NUM_KEYPOINTS = 2
    cfg.MODEL.ROI_KEYPOINT_HEAD.MIN_KEYPOINTS_PER_IMAGE = 2
    
    # cfg.TEST.KEYPOINT_OKS_SIGMAS = [0.9, 0.9]
    # #cfg.MODEL.ROI_KEYPOINT_HEAD.POOLER_TYPE = "ROIAlignV2"
    # cfg.MODEL.ROI_KEYPOINT_HEAD.LOSS_WEIGHT = 1.0
    return cfg

def calc_angle(pt1, pt2, pt3):
    x1, y1 = pt1
    x2, y2 = pt2
    x3, y3 = pt3
    deg1 = (360 + math.degrees(math.atan2(x1 - x2, y1 - y2))) % 360
    deg2 = (360 + math.degrees(math.atan2(x3 - x2, y3 - y2))) % 360
    return deg2 - deg1 if deg1 <= deg2 else 360 - (deg1 - deg2)

def draw_result():
    return

def half_split(img, dis):
    half = np.array(img)
    right_leg = half[:, :dis]
    left_leg = half[:, dis:]
    return right_leg, left_leg

def demo(cfg, img_path, pre_path):
    try:
        if not(os.path.isdir(pre_path)):
            os.makedirs(os.path.join(pre_path))
    except OSError as e:
        if e.errno != errno.EEXIST:
            print("Failed to create directory!!!!!")
            raise
    
    # left, right mean leg direct
    resultTxtF = open(pre_path + "result.txt", "w")
    predictor = DefaultPredictor(cfg)
    img = Image.open(img_path)
    w, h = img.size
    
    right_leg, left_reg = half_split(img, w//2)
    
    right_result = predictor(right_leg)
    left_result = predictor(left_reg)

    right_leg_normal = True
    left_leg_normal = True
    ## right_leg prediction
    if len(right_result['instances'].pred_keypoints) <= 0:
        print("RIGHT LEG POINT ERROR!")
        right_leg_normal = False
    else:
        keypoint = right_result['instances'].pred_keypoints[0].cpu().detach().numpy()
        right_top = keypoint[0][:2]

        rt_x = right_top[0]
        rt_y = right_top[1]

        right_bottom = keypoint[1][:2]

        rb_x = right_bottom[0]
        rb_y = right_bottom[1]

    ## left_leg prediction
    if len(left_result['instances'].pred_keypoints) <= 0:
        print("LEFT LEG POINT ERROR!")
        left_leg_normal = False
    else:
        keypoint = left_result['instances'].pred_keypoints[0].cpu().detach().numpy()
        left_top = keypoint[0][:2]

        lt_x = left_top[0] + w//2
        lt_y = left_top[1]

        left_bottom = keypoint[1][:2]

        lb_x = left_bottom[0] + w//2
        lb_y = left_bottom[1]
    
    ## Result True
    if right_leg_normal and left_leg_normal:
        print("Calc norberg angle")
        right_leg_angle = calc_angle([lb_x, lb_y], [rb_x, rb_y], [rt_x, rt_y])
        left_leg_angle = calc_angle([lt_x, lt_y], [lb_x, lb_y], [rb_x, rb_y] )
        
        print("Right angle: {}".format(right_leg_angle))
        right_norberg_normal = "" #normal
        if float(right_leg_angle) >= 105.0:
            right_norberg_normal = "Normal"
        else:
            right_norberg_normal = "Hip dysplasia"
        
        print("Left angle: {}".format(left_leg_angle))
        left_norberg_normal = "" #normal
        if float(left_leg_angle) >= 105.0:
            left_norberg_normal = "Normal"
        else:
            left_norberg_normal = "Hip dysplasia"

        result_txt = "Right Leg: {:.2f}/{}, Left Leg: {:.2f}/{}".format(right_leg_angle, right_norberg_normal, left_leg_angle, left_norberg_normal)

        draw = ImageDraw.Draw(img)
        # Draw line
        # draw right top - right bottomd
        draw.line(((int(rt_x), int(rt_y)), (int(rb_x), int(rb_y))), fill=(0, 255, 0), width=4)
        # draw right bottom - left bottom
        draw.line(((int(rb_x), int(rb_y)), (int(lb_x), int(lb_y))), fill=(0, 0, 255), width=4)
        # draw left bottom - left top
        draw.line(((int(lb_x), int(lb_y)), (int(lt_x), int(lt_y))), fill=(255, 0, 0), width=4)

        # Draw scatter
        r = 4
        draw.ellipse((int(rt_x - r), int(rt_y - r), int(rt_x + r), int(rt_y + r)), fill=(255, 0, 0))
        draw.ellipse((int(rb_x - r), int(rb_y - r), int(rb_x + r), int(rb_y + r)), fill=(0, 255, 255))
        draw.ellipse((int(lb_x - r), int(lb_y - r), int(lb_x + r), int(lb_y + r)), fill=(255, 255, 0))
        draw.ellipse((int(lt_x - r), int(lt_y - r), int(lt_x + r), int(lt_y + r)), fill=(0, 255, 0))

        # Put angle value
        #font = ImageFont.truetype("arial.ttf", 50)
        #draw.text((10, 60), result_txt, fill=(255, 0, 0), font=font)
        resultTxtF.write(result_txt)
        img.save(pre_path + "result.png")
    elif right_leg_normal:
        draw = ImageDraw.Draw(img)
        # Draw line
        # draw right top - right bottom
        draw.line(((int(rt_x), int(rt_y)), (int(rb_x), int(rb_y))), fill=(0, 255, 0), width=4)

        # Draw scatter
        r = 4
        draw.ellipse((int(rt_x - r), int(rt_y - r), int(rt_x + r), int(rt_y + r)), fill=(255, 0, 0))
        draw.ellipse((int(rb_x - r), int(rb_y - r), int(rb_x + r), int(rb_y + r)), fill=(0, 255, 255))
        
        result_txt = "Error1: Right Leg: Nan, Left Leg: Nan"
        #font = ImageFont.truetype("arial.ttf", 50)
        #draw.text((10, 60), result_txt, fill=(255, 0, 0), font=font)
        resultTxtF.write(result_txt)
        img.save(pre_path + "result_r.png")

    elif left_leg_normal:
        draw = ImageDraw.Draw(img)
        # Draw line
        # draw left bottom - left top
        draw.line(((int(lb_x), int(lb_y)), (int(lt_x), int(lt_y))), fill=(255, 0, 0), width=4)
        
        # Draw scatter
        r = 4
        draw.ellipse((int(lb_x - r), int(lb_y - r), int(lb_x + r), int(lb_y + r)), fill=(255, 255, 0))
        draw.ellipse((int(lt_x - r), int(lt_y - r), int(lt_x + r), int(lt_y + r)), fill=(0, 255, 0))

        result_txt = "Error2: Right Leg: Nan, Left Leg: Nan"
        #font = ImageFont.truetype("arial.ttf", 50)
        #draw.text((10, 60), result_txt, fill=(255, 0, 0), font=font)
        resultTxtF.write(result_txt)
        img.save(pre_path + "result_l.png")
    else:
        draw = ImageDraw.Draw(img)
        result_txt = "Error3: Right Leg: Nan, Left Leg: Nan"
        #font = ImageFont.truetype("arial.ttf", 50)
        #draw.text((10, 60), result_txt, fill=(255, 0, 0), font=font)
        resultTxtF.write(result_txt)
        img.save(pre_path + "result_l.png")
    resultTxtF.close()

def main(argv):
    print("start norberg prediction")

    # img_path = "myCanvas/input.jpg"
    # config_path = "myCanvas/norberg_config.yaml"
    # model_path = "myCanvas/norberg_model.pth"
    # prePath = None
    
    img_path = argv.inpfolder+"/input.jpg"
    config_path = argv.inpfolder+"/norberg_config.yaml"
    model_path = argv.inpfolder+"/norberg_model.pth"
    prePath = argv.PREPATH # save path
    print(prePath)
    is_cpu = argv.cpu
    if is_cpu == True:
        print("CPU Running Setting")
    else:
        print("GPU Running Setting")
    cfg = get_config(config_path, is_cpu)
    cfg.MODEL.WEIGHTS = model_path
    
    demo(cfg, img_path, prePath)
    print("DONE")

if __name__ == "__main__":
    main(args)
