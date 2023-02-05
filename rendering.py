from os.path import join
import os
import torch
import numpy as np
from NetWorks.HeadNeRFNet import HeadNeRFNet
from NetWorks.models import AudioNet, AudioAttNet
import cv2
from HeadNeRFOptions import BaseOptions
from Utils.HeadNeRFLossUtils import HeadNeRFLossUtils
from Utils.RenderUtils import RenderUtils
import pickle as pkl
import time
from glob import glob
from tqdm import tqdm
import imageio
import random
import argparse
from tool_funcs import put_text_alignmentcenter

class FittingImage(object):
    
    def __init__(self, model_path, save_root, gpu_id) -> None:
        super().__init__()
        self.model_path = model_path

        self.device = torch.device("cuda:%d" % gpu_id)
        self.save_root = save_root
        self.opt_cam = True
        self.view_num = 45
        self.duration = 3.0 / self.view_num
        self.model_name = os.path.basename(model_path)[:-4]

        self.build_info()
        self.build_tool_funcs()


    def build_info(self):

        check_dict = torch.load(self.model_path, map_location=torch.device("cpu"))
        para_dict = check_dict["para"]
        self.opt = BaseOptions(para_dict)
        self.featmap_size = self.opt.featmap_size
        self.pred_img_size = self.opt.pred_img_size
        if not os.path.exists(self.save_root): os.mkdir(self.save_root)

        check_dict_ours = torch.load('fitting_res/HeadNeRF__model_Reso32HR.pth', map_location=torch.device("cpu"))
        net = HeadNeRFNet(self.opt, include_vd=False, hier_sampling=False)        
        net.load_state_dict(check_dict_ours["headnerf"])
        self.net = net.to(self.device)
        self.net.eval()

        args.dim_aud = 64
        args.win_size=16
        self.AudNet = AudioNet(args.dim_aud, args.win_size).to(self.device)
        self.AudNet.load_state_dict(check_dict_ours["audnet"])
        self.AudNet.to(self.device)
        self.AudAttNet = AudioAttNet().to(self.device)
        self.AudAttNet.load_state_dict(check_dict_ours["audattnet"])
        self.AudAttNet.to(self.device)
        self.AudNet.eval()
        self.AudAttNet.eval()

        code_dict = torch.load('fitting_res/Code__model_Reso32HR.pth', map_location=torch.device("cpu"))
        self.iden_offset = code_dict["code"]["iden"].to(self.device)
        self.expr_offset = code_dict["code"]["expr"].to(self.device)
        self.appea_offset = code_dict["code"]["appea"].to(self.device)

        cam_dict = torch.load('fitting_res/Cam__model_Reso32HR.pth', map_location=torch.device("cpu"))
        self.delta_EulurAngles = cam_dict["cam"]["delta_eulur"].to(self.device)
        self.delta_Tvecs = cam_dict["cam"]["delta_tvec"].to(self.device)


    def build_tool_funcs(self):
        self.loss_utils = HeadNeRFLossUtils(device=self.device)
        self.render_utils = RenderUtils(view_num=45, device=self.device, opt=self.opt)
        
        self.xy = self.render_utils.ray_xy
        self.uv = self.render_utils.ray_uv
    

    def load_data(self, i, iter_, img_path, mask_path, para_3dmm_path, matrix):

        # == audio == #
        self.smo_size = 8
        aud_features = np.load(os.path.join('./train_data', 'aud.npy'))
        auds_ = torch.Tensor(aud_features).to(self.device) # [843,16,29]

        import json
        test_dir = '/local_datasets/LipSync/cnn/0/transforms_val.json'
        with open(os.path.join(test_dir)) as fp:
            meta = json.load(fp)
        start = meta['frames'][0]['img_id']
        end = meta['frames'][-1]['img_id']

        smo_half_win = int(self.smo_size / 2)
        left_i = iter_ - smo_half_win
        right_i = iter_ + smo_half_win
        pad_left, pad_right = 0, 0
        if left_i < 0:
            pad_left = -left_i
            left_i = 0
        if right_i > end:
            pad_right = right_i-end
            right_i = end
        auds_win = auds_[left_i:right_i]
        if pad_left > 0:
            auds_win = torch.cat(
                (torch.zeros_like(auds_win)[:pad_left], auds_win), dim=0)
        if pad_right > 0:
            auds_win = torch.cat(
                (auds_win, torch.zeros_like(auds_win)[:pad_right]), dim=0)
        auds_win = self.AudNet(auds_win)
        self.aud = self.AudAttNet(auds_win)
        # aud = auds_win[smo_half_win]

        # == process imgs == #
        # img_size = (self.pred_img_size, self.pred_img_size)

        # img_path = os.path.join(img_path, str(iter_)+'.png')
        # mask_path = os.path.join(mask_path, str(iter_)+'_mask.png')
        # img = cv2.imread(img_path)
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # img = img.astype(np.float32)/255.0
        
        # gt_img_size = img.shape[0]
        # if gt_img_size != self.pred_img_size:
        #     img = cv2.resize(img, dsize=img_size, fx=0, fy=0, interpolation=cv2.INTER_LINEAR)
        
        # mask_img = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED).astype(np.uint8)
        # if mask_img.shape[0] != self.pred_img_size:
        #     mask_img = cv2.resize(mask_img, dsize=img_size, fx=0, fy=0, interpolation=cv2.INTER_NEAREST)
        # img[mask_img < 0.5] = 1.0
        
        # self.img_tensor = (torch.from_numpy(img).permute(2, 0, 1)).unsqueeze(0).to(self.device)
        # self.mask_tensor = torch.from_numpy(mask_img[None, :, :]).unsqueeze(0).to(self.device)
        
       # load init codes from the results generated by solving 3DMM rendering opt.
        para_3dmm_path = os.path.join(para_3dmm_path, str(i)+'_nl3dmm.pkl')
        with open(para_3dmm_path, "rb") as f: nl3dmm_para_dict = pkl.load(f)
        base_code = nl3dmm_para_dict["code"].detach().unsqueeze(0).to(self.device)
        
        self.base_iden = base_code[:, :self.opt.iden_code_dims]
        self.base_expr = base_code[:, self.opt.iden_code_dims:self.opt.iden_code_dims + self.opt.expr_code_dims]
        self.base_text = base_code[:, self.opt.iden_code_dims + self.opt.expr_code_dims:self.opt.iden_code_dims 
                                                            + self.opt.expr_code_dims + self.opt.text_code_dims]
        self.base_illu = base_code[:, self.opt.iden_code_dims + self.opt.expr_code_dims + self.opt.text_code_dims:]
        
        self.base_c2w_Rmat = nl3dmm_para_dict["c2w_Rmat"].detach().unsqueeze(0)
        self.base_c2w_Tvec = nl3dmm_para_dict["c2w_Tvec"].detach().unsqueeze(0).unsqueeze(-1)
        self.base_w2c_Rmat = nl3dmm_para_dict["w2c_Rmat"].detach().unsqueeze(0)
        self.base_w2c_Tvec = nl3dmm_para_dict["w2c_Tvec"].detach().unsqueeze(0).unsqueeze(-1)

        temp_inmat = nl3dmm_para_dict["inmat"].detach().unsqueeze(0)
        gt_img_size=512
        temp_inmat[:, :2, :] *= (self.featmap_size / gt_img_size)
        
        temp_inv_inmat = torch.zeros_like(temp_inmat)
        temp_inv_inmat[:, 0, 0] = 1.0 / temp_inmat[:, 0, 0]
        temp_inv_inmat[:, 1, 1] = 1.0 / temp_inmat[:, 1, 1]
        temp_inv_inmat[:, 0, 2] = -(temp_inmat[:, 0, 2] / temp_inmat[:, 0, 0])
        temp_inv_inmat[:, 1, 2] = -(temp_inmat[:, 1, 2] / temp_inmat[:, 1, 1])
        temp_inv_inmat[:, 2, 2] = 1.0
        
        self.temp_inmat = temp_inmat
        self.temp_inv_inmat = temp_inv_inmat

        self.cam_info = {
            "batch_Rmats": self.base_c2w_Rmat.to(self.device), # [1,3,3]
            "batch_Tvecs": self.base_c2w_Tvec.to(self.device), # [1,3,1]
            "batch_inv_inmats": self.temp_inv_inmat.to(self.device) # [1,3,3]
        }
        

    @staticmethod
    def eulurangle2Rmat(angles):
        batch_size = angles.size(0)
        
        sinx = torch.sin(angles[:, 0])
        siny = torch.sin(angles[:, 1])
        sinz = torch.sin(angles[:, 2])
        cosx = torch.cos(angles[:, 0])
        cosy = torch.cos(angles[:, 1])
        cosz = torch.cos(angles[:, 2])

        rotXs = torch.eye(3, device=angles.device).view(1, 3, 3).repeat(batch_size, 1, 1)
        rotYs = rotXs.clone()
        rotZs = rotXs.clone()
        
        rotXs[:, 1, 1] = cosx
        rotXs[:, 1, 2] = -sinx
        rotXs[:, 2, 1] = sinx
        rotXs[:, 2, 2] = cosx
        
        rotYs[:, 0, 0] = cosy
        rotYs[:, 0, 2] = siny
        rotYs[:, 2, 0] = -siny
        rotYs[:, 2, 2] = cosy

        rotZs[:, 0, 0] = cosz
        rotZs[:, 0, 1] = -sinz
        rotZs[:, 1, 0] = sinz
        rotZs[:, 1, 1] = cosz
        
        res = rotZs.bmm(rotYs.bmm(rotXs))
        return res
    
    
    def build_code_and_cam(self):
        
        # code
        shape_code = torch.cat([self.base_iden + self.iden_offset, self.base_expr + self.expr_offset], dim=-1)
        appea_code = torch.cat([self.base_text, self.base_illu], dim=-1) + self.appea_offset
        
        opt_code_dict = {
            "bg":None,
            "iden":self.iden_offset,
            "expr":self.expr_offset,
            "appea":self.appea_offset
        }
        code_info = {
            "bg_code": None, 
            "shape_code":shape_code, 
            "appea_code":appea_code, 
        }

        #cam
        if self.opt_cam:
            delta_cam_info = {
                "delta_eulur": self.delta_EulurAngles, 
                "delta_tvec": self.delta_Tvecs
            }

            batch_delta_Rmats = self.eulurangle2Rmat(self.delta_EulurAngles)
            base_Rmats = self.cam_info["batch_Rmats"]
            base_Tvecs = self.cam_info["batch_Tvecs"]
            
            cur_Rmats = batch_delta_Rmats.bmm(base_Rmats)
            cur_Tvecs = batch_delta_Rmats.bmm(base_Tvecs) + self.delta_Tvecs
            
            batch_inv_inmat = self.cam_info["batch_inv_inmats"] #[N, 3, 3]    
            batch_cam_info = {
                "batch_Rmats": cur_Rmats,
                "batch_Tvecs": cur_Tvecs,
                "batch_inv_inmats": batch_inv_inmat
            }
            
        else:
            delta_cam_info = None
            batch_cam_info = self.cam_info


        return code_info, opt_code_dict, batch_cam_info, delta_cam_info
    
    
    @staticmethod
    def enable_gradient(tensor_list):
        for ele in tensor_list:
            ele.requires_grad = True


    def perform_rendering(self, img_path, mask_path, para_3dmm_path, base_name, save_root):
        
        import json
        test_dir = '/local_datasets/LipSync/cnn/0/transforms_val.json'
        with open(os.path.join(test_dir)) as fp:
            meta = json.load(fp)
        test_len = len(meta['frames']) # 421~842
        start = meta['frames'][0]['img_id']
        end = meta['frames'][-1]['img_id']
        res_img_list = []
        for i, iter_ in enumerate(range(start, end)):
            if i%50==0:
                print(i, "/", test_len)

            self.load_data(i, iter_, img_path, mask_path, para_3dmm_path, matrix=meta['frames'][i])
            code_info, opt_code_dict, cam_info, delta_cam_info = self.build_code_and_cam()
            pred_dict = self.net("test", self.xy, self.uv, self.aud, **code_info, **cam_info)

            coarse_fg_rgb = pred_dict["coarse_dict"]["merge_img"]
            coarse_fg_rgb = (coarse_fg_rgb[0].detach().cpu().permute(1, 2, 0).numpy()* 255).astype(np.uint8)
            res_img_list.append(coarse_fg_rgb)

        NVRes_save_path = "%s/Ours_%s.gif" % (save_root, base_name)
        imageio.mimsave(NVRes_save_path, res_img_list, 'GIF', duration=self.duration)
        print("Rendering Done")


    def fitting_single_images(self, img_path, mask_path, para_3dmm_path, tar_code_path, save_root):
        
        base_name = os.path.basename(img_path)[4:-4]
        # load tar code
        if tar_code_path is not None:
            temp_dict = torch.load(tar_code_path, map_location="cpu")
            tar_code_info = temp_dict["code"]
            for k, v in tar_code_info.items():
                if v is not None:
                    tar_code_info[k] = v.to(self.device)
            self.tar_code_info = tar_code_info
        else:
            self.tar_code_info = None

        self.perform_rendering(img_path, mask_path, para_3dmm_path, base_name, save_root)
        

if __name__ == "__main__":

    torch.manual_seed(45)  # cpu
    torch.cuda.manual_seed(55)  # gpu
    np.random.seed(65)  # numpy
    random.seed(75)  # random and transforms
    # torch.backends.cudnn.deterministic = True  # cudnn
    # torch.backends.cudnn.benchmark = True
    # torch.backends.cuda.matmul.allow_tf32 = False
    # torch.backends.cudnn.allow_tf32 = False

    parser = argparse.ArgumentParser(description='a framework for fitting a single image using HeadNeRF')
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--img", type=str, required=True)
    parser.add_argument("--mask", type=str, required=True)
    parser.add_argument("--para_3dmm", type=str, required=True)
    parser.add_argument("--save_root", type=str, required=True)
    parser.add_argument("--target_embedding", type=str, default="")
    args = parser.parse_args()

    model_path = args.model_path
    save_root = args.save_root
    
    img_path = args.img
    mask_path = args.mask
    para_3dmm_path = args.para_3dmm
    
    if len(args.target_embedding) == 0:
        target_embedding_path = None
    else:
        target_embedding_path = args.target_embedding
    
        temp_str_list = target_embedding_path.split("/")
        if temp_str_list[1] == "*":
            temp_str_list[1] = os.path.basename(model_path)[:-4]
            target_embedding_path = os.path.join(*temp_str_list)
        
        assert os.path.exists(target_embedding_path)
    tt = FittingImage(model_path, save_root, gpu_id=0)
    tt.fitting_single_images(img_path, mask_path, para_3dmm_path, target_embedding_path, save_root)
