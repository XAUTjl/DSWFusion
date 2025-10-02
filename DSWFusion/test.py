from net import Encoder_WT, Decoder, SFEM, BaseFeatureExtraction, DetailFeatureExtraction, Sep
import os
import numpy as np
import torch
import torch.nn as nn
from utils.img_read_save import img_save,image_read_cv2

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
ckpt_path=r"models/best_model.pth"
for dataset_name in ["TNO"]:
    print("The test result of "+dataset_name)
    test_folder=os.path.join('test_img',dataset_name)
    test_out_folder=os.path.join('test_result',dataset_name)


    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    Encoder = nn.DataParallel(Encoder_WT()).to(device)
    Decoder = nn.DataParallel(Decoder()).to(device)
    SFEM_S = nn.DataParallel(SFEM()).to(device)
    Sepnet = nn.DataParallel(Sep()).to(device)
    BaseFuseLayer = nn.DataParallel(BaseFeatureExtraction(dim=64, num_heads=8)).to(device)
    DetailFuseLayer = nn.DataParallel(DetailFeatureExtraction(num_layers=1)).to(device)


    Encoder.load_state_dict(torch.load(ckpt_path)['Encoder'])
    Decoder.load_state_dict(torch.load(ckpt_path)['Decoder'])
    SFEM_S.load_state_dict(torch.load(ckpt_path)['SFEM_S'])
    BaseFuseLayer.load_state_dict(torch.load(ckpt_path)['BaseFuseLayer'])
    DetailFuseLayer.load_state_dict(torch.load(ckpt_path)['DetailFuseLayer'])


    Encoder.eval()
    Decoder.eval()
    SFEM_S.eval()
    Sepnet.eval()
    BaseFuseLayer.eval()
    DetailFuseLayer.eval()


    with torch.no_grad():
        for img_name in os.listdir(os.path.join(test_folder,"ir")):

            IR = image_read_cv2(os.path.join(test_folder,"ir",img_name),mode='GRAY')[np.newaxis,np.newaxis, ...]/255.0
            VI = image_read_cv2(os.path.join(test_folder,"vi",img_name), mode='GRAY')[np.newaxis,np.newaxis, ...]/255.0


            IR,VI = torch.FloatTensor(IR),torch.FloatTensor(VI)
            VI, IR = VI.cuda(), IR.cuda()


            # Feature Interaction Extraction and Integration Process
            feature_V_B, feature_V_D = Encoder(VI)
            feature_I_B, feature_I_D = Encoder(IR)
            # Semantic Feature Enhancement Process
            feature_V_Dd = SFEM_S(feature_V_D)
            feature_I_bd = SFEM_S(feature_I_B)
            feature_F_B = BaseFuseLayer(feature_I_bd + feature_V_B)
            feature_F_D = DetailFuseLayer(feature_I_D + feature_V_Dd)
            # Feature Fusion Process
            FuseI= Decoder(VI, feature_F_B, feature_F_D)


            FuseI = (FuseI - torch.min(FuseI)) / (torch.max(FuseI) - torch.min(FuseI))
            fi = np.squeeze((FuseI * 255).cpu().numpy())
            img_save(fi, img_name.split(sep='.')[0], test_out_folder)
