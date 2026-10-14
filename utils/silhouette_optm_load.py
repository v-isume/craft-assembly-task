# Derived from textured-3d-gan (https://github.com/dariopavllo/textured-3d-gan)
# Copyright (c) 2020 Dario Pavllo. MIT License, see rendering/LICENSE.

import torch
import imageio
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math
from packaging import version
import threading
import os
import kaolin

from rendering.parallel_rasterizer import ParallelKaolinRasterizer
from rendering.mesh_template import MeshTemplate

import cv2
from PIL import Image, ImageOps

def qrot(q, v):
    """
    Quaternion-vector multiplication (rotation of a vector)
    """
    assert q.shape[-1] == 4
    assert v.shape[-1] == 3
    
    qvec = q[:, 1:].unsqueeze(1).expand(-1, v.shape[1], -1)
    uv = torch.cross(qvec, v, dim=2)
    uuv = torch.cross(qvec, uv, dim=2)
    return v + 2 * (q[:, :1].unsqueeze(1) * uv + uuv)

def qmul(q, r):
    """
    Quaternion-quaternion multiplication
    """
    assert q.shape[-1] == 4
    assert r.shape[-1] == 4
    
    original_shape = q.shape
    
    # Compute outer product
    terms = torch.bmm(r.view(-1, 4, 1), q.view(-1, 1, 4))

    w = terms[:, 0, 0] - terms[:, 1, 1] - terms[:, 2, 2] - terms[:, 3, 3]
    x = terms[:, 0, 1] + terms[:, 1, 0] - terms[:, 2, 3] + terms[:, 3, 2]
    y = terms[:, 0, 2] + terms[:, 1, 3] + terms[:, 2, 0] - terms[:, 3, 1]
    z = terms[:, 0, 3] - terms[:, 1, 2] + terms[:, 2, 1] + terms[:, 3, 0]
    return torch.stack((w, x, y, z), dim=1).view(original_shape)

def transform_vertices(vtx, w, scale, translation, rotation, z0):
    # vtx (bs, nt, v, 3)
    # w (bs, nt)
    vtx = (vtx * w.unsqueeze(-1).unsqueeze(-1)).sum(dim=1)
    translation = torch.cat((translation, torch.zeros_like(translation[..., :1])), dim=-1)
    vtx = vtx.expand(scale.shape[0], -1, -1)
    print(vtx[0])
    print(scale.unsqueeze(1)[0])

    vtx = qrot(rotation, scale.unsqueeze(1)*vtx) + translation.unsqueeze(1)
    
    z0 = 1 + z0.unsqueeze(-1).exp()
    z = vtx[..., 2:]
    factor = (z0 - z/2)/(z0 + z/2)
    vtx = torch.cat((vtx[..., :2]*factor, z), dim=-1)
    
    vtx = vtx * torch.Tensor([1, -1, -1]).to(vtx.device)
    return vtx

# Camera proposals initialization
def quantize_views(num_views_azimuth=5, num_views_elevation=8, num_views_roll=1):
    with torch.no_grad():
        views = []
        for azimuth in np.linspace(0, 90, num_views_elevation, endpoint=True):
            for elevation in np.linspace(0, 360, num_views_azimuth, endpoint=False):
                if num_views_roll == 1:
                    rolls = [180]
                else:
                    rolls = np.linspace(-90-30, -90+30, num_views_roll, endpoint=True)
                for roll in rolls:
                    rad = roll / 180 * np.pi
                    q0 = torch.Tensor([np.cos(-rad/2), 0, 0, np.sin(-rad/2)])
                    rad = elevation / 180 * np.pi
                    q1 = torch.Tensor([np.cos(-rad/2), 0, np.sin(-rad/2), 0])
                    q0 = qmul(q0, q1)
                    rad = azimuth / 180 * np.pi
                    q = torch.Tensor([np.cos(-rad/2), 0, 0, np.sin(-rad/2)])
                    q = qmul(q0, q)
                    views.append(q)
        views = torch.stack(views, dim=0)
    return views

# Adam with batched full-matrix preconditioning
class AdamFull(optim.Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        if not 0.0 <= weight_decay:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay)
        super(AdamFull, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(AdamFull, self).__setstate__(state)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            grads = []
            first = False
            state0 = None
            for p in group['params']:
                assert p.grad is not None
                state = self.state[p]
                if state0 is None:
                    state0 = state
                if len(state) == 0:
                    first = True
                    state['exp_avg'] = state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state['step'] = 0
                    
                grads.append(p.grad)

            grads = torch.cat(grads, dim=-1)
            grads_sq = grads.unsqueeze(-1) @ grads.unsqueeze(-2)

            
            if first:
                # Exponential moving average of squared gradient values
                state0['exp_avg_sq'] = torch.zeros_like(grads_sq, memory_format=torch.preserve_format)
            exp_avg_sq = state0['exp_avg_sq']
            beta1, beta2 = group['betas']
            exp_avg_sq.mul_(beta2).add_(grads_sq, alpha=1 - beta2)
            M = (exp_avg_sq + group['eps']*torch.eye(exp_avg_sq.shape[-1], device=exp_avg_sq.device)).inverse()
            if version.parse(torch.__version__) < version.parse('1.8'):
                # As of version 1.8, PyTorch switched to a faster CUDA SVD routine. (CUDA >= 10.1 required)
                # Prior to that, the CPU version was much faster
                M = M.cpu()
            _, e, V = M.svd()
            e = e.to(exp_avg_sq.device)
            V = V.to(exp_avg_sq.device)
            Msqrtinv = V*e.clamp(min=0).sqrt().unsqueeze(-2) @ V.transpose(-1, -2)
            
            exp_avg_all = []
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')

                state = self.state[p]

                exp_avg = state['exp_avg']
                beta1, beta2 = group['betas']

                state['step'] += 1
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']

                if group['weight_decay'] != 0:
                    grad = grad.add(p, alpha=group['weight_decay'])

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_all.append(exp_avg)
                
            exp_avg_all = torch.cat(exp_avg_all, dim=-1)
            delta = (Msqrtinv @ exp_avg_all.unsqueeze(-1)).squeeze(-1) * math.sqrt(bias_correction2)
            param_ptr = 0
            for p in group['params']:
                state = self.state[p]
                grad = p.grad
                bias_correction1 = 1 - beta1 ** state['step']
                step_size = group['lr'] / bias_correction1
                delta_ = delta[..., param_ptr:param_ptr+grad.shape[-1]]
                param_ptr += grad.shape[-1]
                p.add_(delta_, alpha=-step_size)
            assert param_ptr == delta.shape[-1]

        return loss  

class PruningWrapper:
    def __init__(self, optimizer):
        self.optimizer = optimizer
        
    def zero_grad(self):
        for group in self.optimizer.param_groups:
            for p in group['params']:
                p.grad = None
                
    def step(self, closure=None):
        return self.optimizer.step(closure)
        
    @torch.no_grad()
    def prune(self, kept_indices):
        new_param_list = []
        for group in self.optimizer.param_groups:
            group_list = []
            for p in group['params']:
                assert p.grad is None, 'zero_grad() must be called before pruning'
                state = self.optimizer.state[p]
                new_param = p[kept_indices].detach().requires_grad_()
                group_list.append(new_param)
                del self.optimizer.state[p]
                self.optimizer.state[new_param] = state
                for k, v in state.items():
                    if torch.is_tensor(v) and v.shape[0] == p.shape[0]:
                        state[k] = v[kept_indices]
                
            group['params'] = group_list
            new_param_list += group_list
        return new_param_list

def alpha_l1loss(rgb_pred, rgb_real):

    batches = rgb_pred.shape[0]
    camera = rgb_pred.shape[1]
    
    real_r = rgb_real[:, :, 0, :, :]
    real_g = rgb_real[:, :, 1, :, :]
    real_b = rgb_real[:, :, 2, :, :]

    pred_r = rgb_pred[:, :, 0, :, :]
    pred_g = rgb_pred[:, :, 1, :, :]
    pred_b = rgb_pred[:, :, 2, :, :]

    alpha_pr = torch.where((pred_r != 0) |
                           (pred_g != 0) |
                           (pred_b != 0),
                            1, 0)

    alpha_gt = torch.where((real_r != 0) |
                           (real_g != 0) |
                           (real_b != 0),
                            1, 0)

    alpha_gt = alpha_gt.unsqueeze(2)
    alpha_pr = alpha_pr.unsqueeze(2)

    alpha_loss = abs(alpha_gt - alpha_pr).sum() / (batches * camera)

    return alpha_loss

def custom_L1Loss(prediction, target):

    class_to_color = { 
    'truck_body':    [255, 255, 51, 255],       #truck_body
    'truck_cabin':   [255, 0, 255, 255],        #truck_cabin
    'wheel':         [0, 255, 255, 255],        #wheel
    'bus_body':      [255, 153, 51, 255],       #bus_body
    'furniture_leg': [255, 0, 0, 255],          #furniture_leg
    'chair_seat':    [0, 0, 255, 255],          #chair_seat
    'chair_back':    [0, 255, 0, 255],          #chair_back
    'chair_arm':     [0, 255, 128, 255],        #chair_arm
    'table_surface': [102, 102, 0, 255]         #table_surface
    }

    weight_list = [1, 1, 1, 1, 1, 1, 1, 1, 1]

    real_r = target[:, :, 0, :, :]
    real_g = target[:, :, 1, :, :]
    real_b = target[:, :, 2, :, :]

    pred_r = prediction[:, :, 0, :, :]
    pred_g = prediction[:, :, 1, :, :]
    pred_b = prediction[:, :, 2, :, :]

    l1_loss = torch.zeros(1).cuda()
    mask_count = 0

    counter = 0

    used_weight = []

    tol = 1

    for key in class_to_color:
        curr_color = torch.tensor((class_to_color[key])[:3])

        real_binary = torch.where((real_r <= curr_color[0] + tol) &
                                  (real_g <= curr_color[1] + tol) &
                                  (real_b <= curr_color[2] + tol) &
                                  (real_r >= curr_color[0] - tol) &
                                  (real_g >= curr_color[1] - tol) &
                                  (real_b >= curr_color[2] - tol),
                                  1, 0)
        
        pred_binary = torch.where((pred_r <= curr_color[0] + tol) &
                                  (pred_g <= curr_color[1] + tol) &
                                  (pred_b <= curr_color[2] + tol) &
                                  (pred_r >= curr_color[0] - tol) &
                                  (pred_g >= curr_color[1] - tol) &
                                  (pred_b >= curr_color[2] - tol),
                                  1, 0)

        mask_loss = abs(real_binary - pred_binary).sum()
        union = torch.count_nonzero(real_binary + pred_binary)
        
        if union != 0:
            mask_count += 1
            l1_loss += (mask_loss * weight_list[counter] / union)
            used_weight.append(weight_list[counter])
        counter += 1
    l1_loss = l1_loss/sum(used_weight)

    return l1_loss

def color_count_nonzero(prediction, target):

    class_to_color = { 
    'truck_body':    [255, 255, 51, 255],       #truck_body
    'truck_cabin':   [255, 0, 255, 255],        #truck_cabin
    'wheel':         [0, 255, 255, 255],        #wheel
    'bus_body':      [255, 153, 51, 255],       #bus_body
    'furniture_leg': [255, 0, 0, 255],          #furniture_leg
    'chair_seat':    [0, 0, 255, 255],          #chair_seat
    'chair_back':    [0, 255, 0, 255],          #chair_back
    'chair_arm':     [0, 255, 128, 255],        #chair_arm
    'table_surface': [102, 102, 0, 255]         #table_surface
    }

    real_r = target[:, :, 0, :, :]
    real_g = target[:, :, 1, :, :]
    real_b = target[:, :, 2, :, :]

    pred_r = prediction[:, :, 0, :, :]
    pred_g = prediction[:, :, 1, :, :]
    pred_b = prediction[:, :, 2, :, :]

    nonzero_loss = torch.zeros(1).cuda()
    mask_count = 0

    tol = 1

    for key in class_to_color:
        curr_color = torch.tensor((class_to_color[key])[:3])

        real_binary = torch.where((real_r <= curr_color[0] + tol) &
                                  (real_g <= curr_color[1] + tol) &
                                  (real_b <= curr_color[2] + tol) &
                                  (real_r >= curr_color[0] - tol) &
                                  (real_g >= curr_color[1] - tol) &
                                  (real_b >= curr_color[2] - tol),
                                1, 0)
        
        pred_binary = torch.where((pred_r <= curr_color[0] + tol) &
                                  (pred_g <= curr_color[1] + tol) &
                                  (pred_b <= curr_color[2] + tol) &
                                  (pred_r >= curr_color[0] - tol) &
                                  (pred_g >= curr_color[1] - tol) &
                                  (pred_b >= curr_color[2] - tol),
                                1, 0)

        any_pred = torch.count_nonzero(pred_binary)
        any_real = torch.count_nonzero(real_binary)

        if any_real != 0:
            mask_count += 1
            loss = abs(any_pred - any_real) / any_real
            nonzero_loss += (loss)

    return nonzero_loss / mask_count


def mask_center(prediction, target):

    class_to_color = { 
    'truck_body':    [255, 255, 51, 255],       #truck_body
    'truck_cabin':   [255, 0, 255, 255],        #truck_cabin
    'wheel':         [0, 255, 255, 255],        #wheel
    'bus_body':      [255, 153, 51, 255],       #bus_body
    'furniture_leg': [255, 0, 0, 255],          #furniture_leg
    'chair_seat':    [0, 0, 255, 255],          #chair_seat
    'chair_back':    [0, 255, 0, 255],          #chair_back
    'chair_arm':     [0, 255, 128, 255],        #chair_arm
    'table_surface': [102, 102, 0, 255]         #table_surface
    }

    real_r = target[0, :, :]
    real_g = target[1, :, :]
    real_b = target[2, :, :]

    pred_r = prediction[0, :, :]
    pred_g = prediction[1, :, :]
    pred_b = prediction[2, :, :]

    tol = 1

    counter_dist = 0
    max_dist = np.sqrt(target.shape[1] ** 2 + target.shape[2] ** 2)
    sum_dist = 0

    thresh = 1e-9

    for key in class_to_color:
        curr_color = torch.tensor((class_to_color[key])[:3])

        real_binary = torch.where((real_r <= curr_color[0] + tol) &
                                  (real_g <= curr_color[1] + tol) &
                                  (real_b <= curr_color[2] + tol) &
                                  (real_r >= curr_color[0] - tol) &
                                  (real_g >= curr_color[1] - tol) &
                                  (real_b >= curr_color[2] - tol),
                                1, 0)
        
        pred_binary = torch.where((pred_r <= curr_color[0] + tol) &
                                  (pred_g <= curr_color[1] + tol) &
                                  (pred_b <= curr_color[2] + tol) &
                                  (pred_r >= curr_color[0] - tol) &
                                  (pred_g >= curr_color[1] - tol) &
                                  (pred_b >= curr_color[2] - tol),
                                1, 0)

        
        any_real = torch.count_nonzero(real_binary)
        center_real_x = 0
        center_real_y = 0
        center_pred_x = 0
        center_pred_y = 0


        if any_real > 0:
            real_idx = torch.where(real_binary == 1)
            pred_idx = torch.where(pred_binary == 1)

            center_real_y = torch.sum(real_idx[0])/(real_idx[0].size(dim=0) + thresh)
            center_real_x = torch.sum(real_idx[1])/(real_idx[1].size(dim=0) + thresh)

            center_pred_y = torch.sum(pred_idx[0])/(pred_idx[0].size(dim=0) + thresh)
            center_pred_x = torch.sum(pred_idx[1])/(pred_idx[1].size(dim=0) + thresh)

            dist = torch.sqrt((center_pred_y - center_real_y) ** 2 + (center_pred_x - center_real_x) ** 2) / max_dist

            sum_dist += dist
            counter_dist += 1


    return sum_dist / counter_dist


def run_sil_optimization(mesh_path, input_img_path, save_name, texture, batch_size=-1, image_resolution=256, symmetric=True,
                         gpu_ids='0', camera_optim_steps=100, camera_lr_decay_after=80):
    """"Explanation for the inputs
    mesh_path: path for template mesh model
    input_img_path: path for the input img you want to optimize for
    save_name: desired name for the rendered best pose of the given model for target img
    texture: segmented vertices of the template mesh model, where each part has a different color
    batch_size: int
    image_resolution: int, assumed to be squared image
    symmetric: bool
    num_workers: int
    gpu_ids: str with all the gpu ids you wish to use, comma-separated
    no_prune: bool, if True, do not prune proposals for the multi-template mode (slow)
    camera_optim_steps: int, optimization iterations per image
    camera_lr_decay_after: int, decay learning rate after this step
    v_agr_threshold: float, agreement score cutoff
    """

    #Register GPUs
    gpu_ids = [int(x) for x in gpu_ids.split(',')]
    print('Using {} GPUs: {}'.format(len(gpu_ids), gpu_ids))
    torch.cuda.set_device(min(gpu_ids))

    #Define batch size
    if batch_size == -1:
        # Default: 2 images per GPU (it's usually enough to saturate it due to the multiple camera hypotheses)
        batch_size = 2*len(gpu_ids)

    assert batch_size % len(gpu_ids) == 0, 'Batch size must be divisible by # gpus'

    # #Define mesh paths
    # raw_mesh_path = mesh_path.replace('_colored.obj', '_colored_unit.obj')

    #Symmetric assumption
    mesh_template = MeshTemplate(mesh_path, is_symmetric=symmetric)
    
    #Load template model
    #remeshed_meshes = kaolin.io.obj.import_mesh(raw_mesh_path).vertices
    remeshed_meshes = kaolin.io.obj.import_mesh(mesh_path).vertices
    remeshed_meshes = remeshed_meshes.unsqueeze(0)

    #Renderer resolution
    renderer_res = image_resolution

    #Initialize Kaolin renderer
    renderer = nn.DataParallel(ParallelKaolinRasterizer(renderer_res), gpu_ids)

    #Load original image
    #im = imageio.imread(input_img_path)

    gt_original = cv2.imread(input_img_path)
    grayscale = cv2.cvtColor(gt_original, cv2.COLOR_BGR2GRAY)
    th, thresholded = cv2.threshold(grayscale, 0, 255, cv2.THRESH_BINARY)

    bbox = cv2.boundingRect(thresholded)
    x, y, w, h = bbox

    gt_original = Image.open(input_img_path)
    gt_crop = gt_original.crop((x, y, x+w, y+h))

    pr_resize = ImageOps.pad(gt_crop, (256, 256), Image.NEAREST)
    im = np.array(pr_resize)

    #Extract silhouette
    non_zero = np.where(
            (im[:, :, 0] != 0) |
            (im[:, :, 1] != 0) |
            (im[:, :, 2] != 0)
            )
    
    sil = np.zeros(im.shape[:2], np.uint8)
    sil[non_zero] = 1

    non_zero_pixels = np.shape(non_zero)
    non_zero_pixels = non_zero_pixels[1]
    non_zero_pixels = torch.tensor(non_zero_pixels).cuda()

    #Normalize RGB image and formating
    im = np.array(im) / 255.0
    if len(im.shape) == 2:
        im = np.repeat(np.expand_dims(im, 2), 3, axis=2)
    im = np.moveaxis(im, -1, 0)

    im *= sil[np.newaxis, :, :]

    im = torch.FloatTensor(im)
    sil = torch.FloatTensor(sil).unsqueeze(0)

    combined = torch.cat((im, sil), dim=0)

    def render_mesh_kaolin(vtx, tex, image_size, return_hardmask=False, **kwargs):
        if 'bs' in kwargs and bs % len(gpu_ids) != 0:
            # Last batch not divisible by num_gpus. Render one by one
            return mesh_template.forward_renderer(renderer.module, vtx, tex, num_gpus=1,
                                           return_hardmask=return_hardmask, image_size=image_size, **kwargs)
        else:
            return mesh_template.forward_renderer(renderer, vtx, tex, num_gpus=len(gpu_ids),
                                           return_hardmask=return_hardmask, image_size=image_size, **kwargs)

    def render_mesh(vtx, tex, image_size=None, **kwargs):
        return render_mesh_kaolin(vtx, tex, image_size, **kwargs)

    def save_obj_kaolin(vtx, tex, out_path, **kwargs):
        mesh_template.export_obj(out_path, vtx, tex, False)

    #Initialize views
    initial_views = quantize_views().cuda()
    #criterion = nn.MSELoss(reduction='sum')
    criterion = nn.L1Loss(reduction='sum')

    #Lists to store proposals
    all_R_proposal = []
    all_s_proposal = []
    all_t_proposal = []
    all_z0_proposal = []
    all_w_proposal = []
    all_loss = []

    renderer.module.set_mode('alpha')

    print('Optimizing poses...')
    schedule = [camera_optim_steps*3//10, camera_optim_steps*6//10]

    #Single image optimization
    num_hypotheses = initial_views.shape[0] # Number of camera hypotheses
    bs = batch_size # Batch size
    nt = remeshed_meshes.shape[0] # Number of mesh templates

    #Propagate according to number of batches
    texture_in = torch.FloatTensor(texture).repeat(1, 1, 1, 1)
    texture_input = texture_in.clone()
    texture_input = texture_input.repeat(int(bs * num_hypotheses), 1, 1, 1)

    with torch.no_grad():
        R_proposal = initial_views.clone().unsqueeze(0).expand(bs, -1, -1)
        s_proposal = torch.ones((bs, num_hypotheses, 1), device=R_proposal.device)
        t_proposal = torch.zeros((bs, num_hypotheses, 2), device=R_proposal.device)
        z0_proposal = torch.full((bs, num_hypotheses, 1), 2, device=R_proposal.device, dtype=torch.float32) # Initialize to a distant value
        w_proposal = torch.zeros((bs, num_hypotheses, nt), device=R_proposal.device)

        # Replicate hypotheses according to # of mesh templates
        R_proposal = R_proposal.unsqueeze(2).expand(-1, -1, nt, -1).flatten(1, 2)
        s_proposal = s_proposal.unsqueeze(2).expand(-1, -1, nt, -1).flatten(1, 2)
        t_proposal = t_proposal.unsqueeze(2).expand(-1, -1, nt, -1).flatten(1, 2)
        z0_proposal = z0_proposal.unsqueeze(2).expand(-1, -1, nt, -1).flatten(1, 2)
        w_proposal = w_proposal.unsqueeze(2).expand(-1, -1, nt, -1).contiguous()

        # One-hot vector
        for k in range(nt):
            w_proposal[:, :, k, k] = 1
        w_proposal = w_proposal.flatten(1, 2)
        num_hypotheses *= nt

        R_proposal = R_proposal.flatten(0, 1)
        s_proposal = s_proposal.flatten(0, 1)
        t_proposal = t_proposal.flatten(0, 1)
        z0_proposal = z0_proposal.flatten(0, 1)
        w_proposal = w_proposal.flatten(0, 1)

        raw_vtx_all = remeshed_meshes.unsqueeze(0).cuda()

        # Set initial scale to maximize camera area (this is done in vertex space without rendering)
        raw_vtx_nograd = raw_vtx_all.unsqueeze(1).expand(bs, num_hypotheses, nt, -1, -1).contiguous().flatten(0, 1)
        vtx = transform_vertices(raw_vtx_nograd, w_proposal, s_proposal, t_proposal, F.normalize(R_proposal, dim=-1), z0_proposal)
        scale_max = vtx[..., :2].abs().max(dim=-1).values.max(dim=-1).values
        #print(s_proposal.shape)
        #print(scale_max.shape)
        s_proposal.data[:] = 0.9 * (1/scale_max.view(s_proposal.shape)) # Matches cmr margin

    #Enable gradients
    R_proposal.requires_grad_()
    s_proposal.requires_grad_()
    t_proposal.requires_grad_()
    z0_proposal.requires_grad_()

    #Camera parameters
    camera_optim_lr = 0.1
    camera_optim_momentum = 0.9

    optim_params = [R_proposal, s_proposal, t_proposal, z0_proposal]
    camera_optim = AdamFull(optim_params, lr=camera_optim_lr, betas=(camera_optim_momentum, 0.95))
    camera_optim = PruningWrapper(camera_optim)

    num_camera_steps = camera_optim_steps
    render_size = renderer_res // 2

    target_cache = {} # Copy targets only in the first iteration, to avoid too much cross-GPU data movement
    target_cache_2 = {}
    target_cache_lock = threading.Lock()

    combined = combined[None, ...]

    torch.autograd.set_detect_anomaly(True)

    X_rgb =combined[:, 0:3].cuda()
    X_alpha = combined[:, 3:4].cuda()

    black_img = np.zeros(X_rgb.shape)
    black_img = torch.FloatTensor(black_img)
    black_img = black_img.to('cuda')

    black_img_2 = np.zeros(X_alpha.shape)
    black_img_2 = torch.FloatTensor(black_img_2)
    black_img_2 = black_img_2.to('cuda')

    non_zero_pixels = non_zero_pixels * bs * num_hypotheses
    
    output_path = os.path.join('.', input_img_path.split('/')[1], 'silhouette_optm', input_img_path.split('/')[-2])
    os.makedirs(output_path, exist_ok=True)
    
    #Optimization loop
    for k in range(num_camera_steps):
        renderer.module.set_sigma_mul(10**(-(k/(num_camera_steps-1))/num_camera_steps))
        raw_vtx_nograd = raw_vtx_all.unsqueeze(1).expand(bs, num_hypotheses, nt, -1, -1).contiguous().flatten(0, 1)
        vtx = transform_vertices(raw_vtx_nograd, w_proposal, s_proposal, t_proposal, F.normalize(R_proposal, dim=-1), z0_proposal)
        # Closure for parallel loss computation inside each GPU with nn.DataParallel (avoids unnecessary data aggregation/exchange)
        def closure(image_pred, alpha_pred, **kwargs):
            device = alpha_pred.device
            with target_cache_lock:
                if device not in target_cache:
                    rgb_target = kwargs['rgb_target']
                    alpha_target = kwargs['alpha_target']
                    target_cache[device] = rgb_target
                    target_cache_2[device] = alpha_target
                else:
                    assert kwargs['rgb_target'] is None
                    assert kwargs['alpha_target'] is None
                    rgb_target = target_cache[device]
                    alpha_target = target_cache_2[device]
            
            alpha_pred = alpha_pred.view(-1, num_hypotheses, *alpha_pred.shape[1:])
            # render_alpha = alpha_pred.detach().to('cpu').numpy()
            # print(np.shape(render_alpha))
            # print("!!!!")
            # for j in range(5):
            #     for i in range(40):
            #         os.makedirs(os.path.join(output_path, save_name, f'batch_{j}_png'), exist_ok=True)
            #         save_path = os.path.join(output_path, save_name, f'batch_{j}_png', f'pose_{i}_step_{k}.png')

            #         new_render = np.moveaxis(render_alpha[j][i], 0, -1)
            #         new_render = np.squeeze(new_render, -1)

            #         print(np.shape(new_render))

            #         imageio.imwrite(save_path, (new_render * 255).astype(np.uint8))

            alpha_target = alpha_target.unsqueeze(1).expand(bs, num_hypotheses, -1, -1, -1)
            
            camera_loss = criterion(alpha_pred, alpha_target) / (render_size * render_size * bs * num_hypotheses)

            # Aggregate using sum instead of mean -> independent of batch size (gradients are disjoint)
            camera_loss = camera_loss*alpha_pred.shape[0]

            print(camera_loss)
            print("-----------------------------------------------------------")

            return camera_loss.unsqueeze(0)

        if k == 0 or (k-1) in schedule:
            if render_size == renderer_res:
                X_rgb_target = X_rgb
                X_alpha_target = X_alpha
            else:
                X_rgb_target = F.interpolate(X_rgb, size=render_size, mode='bilinear', align_corners=False)
                X_alpha_target = F.interpolate(X_alpha, size=render_size, mode='bilinear', align_corners=False)
        else:
            # Already copied
            X_rgb_target = None
            X_alpha_target = None

        print(k)
        camera_loss, _ = render_mesh(vtx, texture_input, image_size=render_size,
                                                      closure=closure, rgb_target=X_rgb_target, alpha_target=X_alpha_target, bs=bs)

        # for j in range(bs):
        #     for i in range(40):
        #         os.makedirs(os.path.join(output_path, save_name, f'batch_{j}_obj'), exist_ok=True)
        #         save_path_obj = os.path.join(output_path, save_name, f'batch_{j}_obj', f'pose_{i}_step_{k}.obj')
        #         save_obj_kaolin(vtx[j][i].unsqueeze(0).unsqueeze(0), texture_input[0].unsqueeze(0), save_path_obj)

        # Add brick wall loss (prevent object from going outside the camera frustum)
        brick_wall_loss = F.relu(vtx[:, :, :2].abs() - 1)**2
        brick_wall_loss = brick_wall_loss.mean(dim=[1, 2])
        camera_loss = camera_loss.sum() + brick_wall_loss.sum()

        # Update cameras
        camera_loss.backward()
        camera_optim.step()
        camera_optim.zero_grad()

        R_proposal.data[:] = F.normalize(R_proposal.data, dim=-1) # Renormalize rotation
        s_proposal.data.abs_() # Scale is always positive
        z0_proposal.data.clamp_(-4, 4) # Avoid extreme values which might lead to numerical instability
        t_proposal.data.clamp_(-1, 1)

        with torch.no_grad():
            if k in schedule:
                if render_size < renderer_res:
                    render_size += 64 # 128 -> 192 -> 256
                    target_cache = {} # Reset target cach
    
        if k == camera_lr_decay_after: # Lower learning rate after this threshold
            for param_group in camera_optim.optimizer.param_groups:
                param_group['lr'] *= 0.1
            camera_optim_lr *= 0.1

    camera_optim = None # Delete optimizer


    renderer.module.set_mode('texture')
    # Evaluate and store final camera hypotheses
    with torch.no_grad():
        raw_vtx_nograd = raw_vtx_all.unsqueeze(1).expand(bs, num_hypotheses, nt, -1, -1).contiguous().flatten(0, 1)
        vtx = transform_vertices(raw_vtx_nograd, w_proposal, s_proposal, t_proposal, R_proposal, z0_proposal)

        def closure(image_pred, alpha_pred, **kwargs):
            alpha_pred = alpha_pred.view(-1, num_hypotheses, *alpha_pred.shape[1:])
            image_pred = image_pred.view(-1, num_hypotheses, *image_pred.shape[1:])
            device = alpha_pred.device
            rgb_target = target_cache[device].unsqueeze(1).expand(bs, num_hypotheses, -1, -1, -1)
            alpha_target = target_cache_2[device].unsqueeze(1).expand(bs, num_hypotheses, -1, -1, -1)

            # camera_loss = torch.zeros((batch_size, num_hypotheses))
            silhouette_loss_matrix = torch.zeros((batch_size, num_hypotheses))
            silhouette_loss2_matrix = torch.zeros((batch_size, num_hypotheses))
            silhouette_loss3_matrix = torch.zeros((batch_size, num_hypotheses))

            for i in range(batch_size):
                for j in range(num_hypotheses):
                    image_tensor = image_pred[i, j, :, :, :]
                    # image_np = image_pred[i, j, :, :, :].detach().cpu().numpy()
                    # image_np = np.moveaxis(image_np, 0, -1)
                    # image_np = np.round(image_np.astype(np.uint8), decimals=0)
                    # grayscale = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
                    # th, thresholded = cv2.threshold(grayscale, 0, 255, cv2.THRESH_BINARY)
                    # bbox = cv2.boundingRect(thresholded)
                    # x, y, w, h = bbox

                    # pr_original = Image.fromarray(image_np).convert('RGB')
                    # pr_crop = pr_original.crop((x, y, x+w, y+h))

                    # pr_resize = ImageOps.pad(pr_crop, (256, 256), Image.NEAREST)
                    # image_pred_test = np.array(pr_resize)
                    # image_pred_test = np.moveaxis(image_pred_test, 2, 0)
                    # image_tensor = torch.tensor(image_pred_test).cuda()
                    
                    silhouette_loss_matrix[i, j] =  custom_L1Loss((image_tensor.unsqueeze(0).unsqueeze(0)).int(), (rgb_target[i, j, :, :, :].unsqueeze(0).unsqueeze(0) * 255).int())
                    silhouette_loss2_matrix[i, j] = alpha_l1loss(image_tensor.unsqueeze(0).unsqueeze(0), rgb_target[i, j, :, :, :].unsqueeze(0).unsqueeze(0) * 255) / (render_size * render_size)
                    silhouette_loss3_matrix[i, j] = mask_center(image_tensor, rgb_target[i, j, :, :, :] * 255)
            camera_loss = (silhouette_loss_matrix * 0.7 + silhouette_loss2_matrix * 0.15 + silhouette_loss3_matrix * 0.15) # 0.45 0.1 0.45

            return camera_loss

        camera_loss, _ = render_mesh(vtx, texture_input,
                                            closure=closure,
                                            bs=bs)

        R_proposal = R_proposal.view(bs, num_hypotheses, 4)
        s_proposal = s_proposal.view(bs, num_hypotheses, 1)
        t_proposal = t_proposal.view(bs, num_hypotheses, 2)
        z0_proposal = z0_proposal.view(bs, num_hypotheses, 1)
        w_proposal = w_proposal.view(bs, num_hypotheses, nt)

    all_R_proposal.append(R_proposal.detach().cpu())
    all_s_proposal.append(s_proposal.detach().cpu())
    all_t_proposal.append(t_proposal.detach().cpu())
    all_z0_proposal.append(z0_proposal.detach().cpu())
    all_w_proposal.append(w_proposal.detach().cpu())
    all_loss.append(camera_loss.detach().cpu())

    all_R_proposal = torch.cat(all_R_proposal, dim=0)
    all_s_proposal = torch.cat(all_s_proposal, dim=0)
    all_t_proposal = torch.cat(all_t_proposal, dim=0)
    all_z0_proposal = torch.cat(all_z0_proposal, dim=0)
    all_w_proposal = torch.cat(all_w_proposal, dim=0)
    all_loss = torch.cat(all_loss, dim=0)

    min_list = []

    for a in range(bs):
        min_list.append(all_loss[a].min(axis=0).values)

    # Debugging: save all possibilities
    # for j in range(bs):
    #     for i in range(40):
    #         curr_w = all_w_proposal[j][i]
    #         curr_s = all_s_proposal[j][i]
    #         curr_t = all_t_proposal[j][i]
    #         curr_R = all_R_proposal[j][i]
    #         curr_z = all_z0_proposal[j][i]

    #         curr_w = curr_w[None, :]
    #         curr_s = curr_s[None, :]
    #         curr_t = curr_t[None, :]
    #         curr_R = curr_R[None, :]
    #         curr_z = curr_z[None, :]

    #         curr_raw_vtx_nograd = raw_vtx_all.detach().unsqueeze(1).expand(1, 1, nt, -1, -1).contiguous().flatten(0, 1).to('cpu')
    #         curr_vtx = transform_vertices(curr_raw_vtx_nograd, curr_w, curr_s, curr_t, F.normalize(curr_R, dim=-1), curr_z)


    #         os.makedirs(os.path.join(output_path, save_name, f'batch_{j}_best'), exist_ok=True)
    #         save_path = os.path.join(output_path, save_name, f'batch_{j}_best', f'pose_{i}.png')
    #         os.makedirs(os.path.join(output_path, save_name, f'batch_{j}_best_obj'), exist_ok=True)
    #         save_path_obj = os.path.join(output_path, save_name, f'batch_{j}_best_obj',f'pose{i}.obj')
    #         save_obj_kaolin(curr_vtx, texture_input[0].unsqueeze(0), save_path_obj)
    #         rgb, result_render, _ = render_mesh_kaolin(curr_vtx, texture_input[0].unsqueeze(0), 256, return_hardmask=True)
    #         result_render = result_render.squeeze(0)
    #         result_render = result_render.detach().squeeze(0).to('cpu').numpy()

    #         rgb = rgb.squeeze(0)
    #         rgb = rgb.detach().to('cpu').numpy()

    #         rgb = np.moveaxis(rgb, 0, -1)

    #         imageio.imwrite(save_path, (rgb).astype(np.uint8))

    best_batch = np.argmin(min_list)
    best_iter = np.argmin(all_loss[best_batch])

    best_w = all_w_proposal[best_batch][best_iter]
    best_s = all_s_proposal[best_batch][best_iter]
    best_t = all_t_proposal[best_batch][best_iter]
    best_R = all_R_proposal[best_batch][best_iter]
    best_z = all_z0_proposal[best_batch][best_iter]

    best_w = best_w[None, :]
    best_s = best_s[None, :]
    best_t = best_t[None, :]
    best_R = best_R[None, :]
    best_z = best_z[None, :]

    best_raw_vtx_nograd = raw_vtx_all.detach().unsqueeze(1).expand(1, 1, nt, -1, -1).contiguous().flatten(0, 1).to('cpu')
    best_vtx = transform_vertices(best_raw_vtx_nograd, best_w, best_s, best_t, F.normalize(best_R, dim=-1), best_z)

    save_path = os.path.join(output_path, save_name + f'_best.png')
    save_path_obj = os.path.join(output_path, save_name + f'_best')
    save_obj_kaolin(best_vtx, texture_input[0].unsqueeze(0), save_path_obj)
    rgb, result_render, faces_idx = render_mesh_kaolin(best_vtx, texture_input[0].unsqueeze(0), 256, return_hardmask=True)
    result_render = result_render.squeeze(0)
    result_render = result_render.detach().squeeze(0).to('cpu').numpy()

    rgb = rgb.squeeze(0)
    rgb = rgb.detach().to('cpu').numpy()

    rgb = np.moveaxis(rgb, 0, -1)
    print(rgb.shape)
    imageio.imwrite(save_path, (rgb).astype(np.uint8))

    faces_idx = faces_idx.detach().to('cpu').numpy() #face index to rgb x y pixel values
    best_transformation = [best_w, best_s, best_t, best_R, best_z]

    return min_list[best_batch], best_transformation, faces_idx