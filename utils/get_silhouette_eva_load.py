import os
import utils.silhouette_optm_load as sil_opt #changed 01/31
import pymeshlab
import json
import numpy as np
import torch
import cv2
from utils.dictionaries import class_to_color, id_to_color
import imageio

from rendering.mesh_template import MeshTemplate
from rendering.parallel_rasterizer import ParallelKaolinRasterizer
import torch.nn as nn
import torch.nn.functional as F
import kaolin

from PIL import Image

def get_best_model(class_name, all_models_par_folder, input_rgb):

    out_path = os.path.join('outputs', 'silhouette_optm', input_rgb.split('/')[-2])

    # load from template mesh dataset
    models_par_folder = os.path.join(all_models_par_folder, class_name)
    all_models = os.listdir(models_par_folder)
    class_models = [x.replace('_colored.obj', '') for x in all_models if (class_name in x) and x.endswith('.obj')]
    class_models.sort()

    # figure out from previous results, which is the best one
    best_rgb = Image.open(os.path.join(out_path, 'best_optimized_pose.png'))
    all_proposals = os.listdir(out_path)
    all_proposals.sort()
    all_proposals = [x for x in all_proposals if ('.png' in x) and (x != 'best_optimized_pose.png')]
    for i in range(len(all_proposals)):
        curr_rgb = Image.open(os.path.join(out_path, all_proposals[i]))
        if np.all((np.array(best_rgb) == np.array(curr_rgb))):
            min_idx = i
            break

    best_class = class_models[min_idx]
    best_file_name = f'iteration_{class_name}_{best_class}_best.png'

    mesh_path = os.path.join(out_path, best_file_name.replace('.png', '.obj'))

    # re-render the best result to get indices of visible faces
    mesh_template = MeshTemplate(os.path.join(out_path, f'{best_class}_colored_unit.obj'), is_symmetric=False)

    renderer_res = 256
    gpu_ids = [0]
    # initialize Kaolin renderer
    renderer = nn.DataParallel(ParallelKaolinRasterizer(renderer_res), gpu_ids)

    # load mesh using kaolin
    remeshed_meshes = kaolin.io.obj.import_mesh(mesh_path).vertices
    remeshed_meshes = remeshed_meshes.unsqueeze(0)
    raw_vtx_all = remeshed_meshes.unsqueeze(0).cuda()

    best_vtx = raw_vtx_all.detach().unsqueeze(1).expand(1, 1, 1, -1, -1).contiguous().flatten(0, 1).to('cpu')

    # texture
    texture = imageio.imread(os.path.join(models_par_folder, f'{best_class}_colored.png'))

    new_texture = np.zeros_like(texture)
    new_class_tex = np.zeros((9, 1024, 1024))

    #Part type to part color dictionary
    classname_to_color = { 
        'truck_body':       0, #truck_body
        'truck_cabin':      1,  #truck_cabin
        'wheel':            2,  #wheel
        'bus_body':         3, #bus_body
        'furniture_leg':    4,    #furniture_leg
        'chair_seat':       5,    #chair_seat
        'chair_back':       6,    #chair_back
        'chair_arm':        7,  #chair_arm
        'table_surface':    8   #table_surface
    }

    #2.2. Recolor according to dictionary so that the colors match the PointRend prediction
    with open(f'{models_par_folder}/{best_class}_dict.txt') as f:
        data = f.read()
    f.close()
    id_to_part = json.loads(data)
    for i in range(len(id_to_part)):
        target_color = class_to_color[id_to_part[str(i)]]
        lookup_color = id_to_color[i]

        for j in range(texture.shape[0]):
            for k in range(texture.shape[1]):
                if (texture[j][k] == lookup_color).all():
                    new_texture[j][k] = target_color
                    new_class_tex[classname_to_color[id_to_part[str(i)]]][j][k] = 1

    texture = np.array(new_texture)
    new_texture = np.moveaxis(texture[..., :3], -1, 0)

    texture_in = torch.FloatTensor(new_texture).repeat(1, 1, 1, 1)
    texture_input = texture_in.clone()
    texture_input = texture_input.repeat(int(1 * 40), 1, 1, 1)
    texture_input = texture_input.to('cpu')
    best_vtx = best_vtx.to('cpu')

    # render function
    def render_mesh_kaolin(vtx, tex, image_size, return_hardmask=False, **kwargs):
        return mesh_template.forward_renderer(renderer, vtx, tex, num_gpus=1,
                return_hardmask=return_hardmask, image_size=image_size, **kwargs)

    renderer.module.set_mode('texture')
    rgb, result_render, faces_idx = render_mesh_kaolin(best_vtx.squeeze(0), texture_input[0].unsqueeze(0), 256, return_hardmask=True)
    torch.cuda.empty_cache()

    rgb = rgb.squeeze(0)
    rgb = rgb.detach().to('cpu').numpy()

    rgb = np.moveaxis(rgb, 0, -1)
    imageio.imwrite(os.path.join(out_path, 're_render.png'), (rgb).astype(np.uint8))

    p_transform = np.load(os.path.join(out_path, 'full_transform.npz'))
    transform_list = []
    transform_list.append(torch.from_numpy(p_transform['w']))
    transform_list.append(torch.from_numpy(p_transform['s']))
    transform_list.append(torch.from_numpy(p_transform['t']))
    transform_list.append(torch.from_numpy(p_transform['R']))
    transform_list.append(torch.from_numpy(p_transform['z']))

    faces_idx = faces_idx.detach().to('cpu').numpy()

    return transform_list, best_class, faces_idx