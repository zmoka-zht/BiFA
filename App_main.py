import numpy as np
import os
# os.system('nvidia-smi')
# os.system('ls /usr/local')
# os.system('pip install torch torchvision --extra-index-url https://download.pytorch.org/whl/cu116')
# os.system('pip install -U openmim')
# os.system('mim install mmcv-full')
import models
import gradio as gr
import torchvision
import torch
import core.metrics as Metrics


device = 'cuda:0' if torch.cuda.is_available() else 'cpu'




def transform_augment_cd(img, min_max=(0, 1)):
    totensor = torchvision.transforms.ToTensor()
    img = totensor(img)
    ret_img = img * (min_max[1] - min_max[0]) + min_max[0]
    return ret_img


def create_CD_model(checkpoint_path):
    from models.bifa_vis import Segformer_implict as bifavis
    from models.bifa import BiFA as bifa
    cd_model = bifa(backbone="mit_b0")
    cd_model.load_state_dict(torch.load(checkpoint_path), strict=False)
    cd_model.to(device)
    return cd_model

def bifa_inference(img1, img2, checkpoint_path):
    if checkpoint_path == 'WHU':
        checkpoint_path = 'experiments/pretrain/WHU.pth'
    elif checkpoint_path == 'LEVIR':
        checkpoint_path = 'experiments/pretrain/LEVIR.pth'
    elif checkpoint_path == 'LEVIR+':
        checkpoint_path = 'experiments/pretrain/LEVIR+.pth'
    elif checkpoint_path == 'SYSU':
        checkpoint_path = 'experiments/pretrain/SYSU.pth'
    elif checkpoint_path == 'DSIFN':
        checkpoint_path = 'experiments/pretrain/DSIFN.pth'
    elif checkpoint_path == 'CLCD':
        checkpoint_path = 'experiments/pretrain/CLCD.pth'
    else:
        raise NotImplementedError

    img1 = transform_augment_cd(img1, min_max=(-1, 1))
    img2 = transform_augment_cd(img2, min_max=(-1, 1))
    print('Use: ', device)
    cdmodel = create_CD_model(checkpoint_path)
    cdmodel.eval()
    img1 = img1.to(device)
    img1 = img1.unsqueeze(0)
    img2 = img2.to(device)
    img2 = img2.unsqueeze(0)

    with torch.no_grad():
        pred = cdmodel(img1, img2)
        G_pred = pred.detach()
        G_pred = torch.argmax(G_pred, dim=1)
        G_pred = G_pred * 2.0 - 1.0
        pred_cm = Metrics.tensor2img(G_pred.unsqueeze(1).repeat(1, 3, 1, 1),
                                     out_type=np.uint8, min_max=(0, 1))
    return pred_cm




title = "BiFA"
description = "Gradio demo for BiFA. Upload image from WHU or LEVIR or LEVIR+ or SYSU or DSIFN or CLCD Dataset or click any one of the examples, " \
              "Then click \"Submit\" and wait for the change detection result. \n" \
              "Paper: BiFA: Remote Sensing Image Change Detection with Bitemporal Feature Alignment"

article = "<p style='text-align: center'><a href='https://github.com/zmoka-zht' target='_blank'>BiFA Project " \
          "Page</a></p> "

examples = [
    ['examples/whu_A_123.png', 'examples/whu_B_123.png', 'WHU'],
    ['examples/whu_A_541.png', 'examples/whu_B_541.png', 'WHU'],
    ['examples/whu_A_28.png', 'examples/whu_B_28.png', 'WHU'],
    ['examples/whu_A_635.png', 'examples/whu_B_635.png', 'WHU'],

    ['examples/levir_A_776.png', 'examples/levir_B_776.png', 'LEVIR'],
    ['examples/levir_A_964.png', 'examples/levir_B_964.png', 'LEVIR'],
    ['examples/levir_A_1665.png', 'examples/levir_B_1665.png', 'LEVIR'],
    ['examples/levir_A_1856.png', 'examples/levir_B_1856.png', 'LEVIR'],

    ['examples/levir+_A_182.png', 'examples/levir+_B_182.png', 'LEVIR+'],
    ['examples/levir+_A_1646.png', 'examples/levir+_B_1646.png', 'LEVIR+'],
    ['examples/levir+_A_4875.png', 'examples/levir+_B_4875.png', 'LEVIR+'],
    ['examples/levir+_A_5301.png', 'examples/levir+_B_5301.png', 'LEVIR+'],

    ['examples/sysu_A_22.png', 'examples/sysu_B_22.png', 'SYSU'],
    ['examples/sysu_A_846.png', 'examples/sysu_B_846.png', 'SYSU'],
    ['examples/sysu_A_1847.png', 'examples/sysu_B_1847.png', 'SYSU'],
    ['examples/sysu_A_3208.png', 'examples/sysu_B_3208.png', 'SYSU'],

    ['examples/dsifn_A_10.png', 'examples/dsifn_B_10.png', 'DSIFN'],
    ['examples/dsifn_A_16.png', 'examples/dsifn_B_16.png', 'DSIFN'],
    ['examples/dsifn_A_24.png', 'examples/dsifn_B_24.png', 'DSIFN'],
    ['examples/dsifn_A_36.png', 'examples/dsifn_B_36.png', 'DSIFN'],

    ['examples/clcd_A_5.png', 'examples/clcd_B_5.png', 'CLCD'],
    ['examples/clcd_A_52.png', 'examples/clcd_B_52.png', 'CLCD'],
    ['examples/clcd_A_55.png', 'examples/clcd_B_55.png', 'CLCD'],
    ['examples/clcd_A_58.png', 'examples/clcd_B_58.png', 'CLCD'],
]

with gr.Blocks() as demo:
    image_input1 = gr.inputs.Image(type='pil', label='Input1 Img')
    image_input2 = gr.inputs.Image(type='pil', label='Input2 Img')
    # with gr.Row().style(equal_height=True):
    # image_LR_output = gr.outputs.Image(label='LR Img', type='numpy')
    image_output = gr.outputs.Image(label='CD Result', type='numpy')
    with gr.Row():
        checkpoint = gr.inputs.Radio(['WHU', 'LEVIR', 'LEVIR+', 'SYSU', 'DSIFN', 'CLCD'], label='Checkpoint')

io = gr.Interface(fn=bifa_inference,
                  inputs=[image_input1,
                          image_input2,
                          checkpoint,
                          ],
                  outputs=[
                      # image_LR_output,
                      image_output
                  ],
                  title=title,
                  description=description,
                  article=article,
                  allow_flagging='auto',
                  examples=examples,
                  cache_examples=False, #True
                  layout="grid"
                  )
io.launch()