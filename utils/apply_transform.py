import os
import pymeshlab
import torch
from utils.visible import qrot
import torch.nn.functional as F

def merge_and_rotate(img_name, transform, output_folder):
        primitives_folder = os.path.join(output_folder, 'primitives', img_name)
        primitives_files = os.listdir(primitives_folder)
        ms = pymeshlab.MeshSet()
        
        for primitive in primitives_files:
            primitive_path = os.path.join(primitives_folder, primitive)
            ms.load_new_mesh(primitive_path)
        
        ms.generate_by_merging_visible_meshes()
        face_matrix = ms.current_mesh().face_matrix()

        best_R = F.normalize(transform[3], dim=-1)
        best_T = transform[2]
    
        best_R = torch.tensor(best_R, dtype=float)
        best_T = torch.cat((best_T, torch.zeros_like(best_T[..., :1])), dim=-1)
    
        scale = torch.diag(torch.tensor([1, -1, -1])) #originally [1, -1, -1]
        scale = torch.tensor(scale, dtype=float)
    
        vtx_matrix = ms.current_mesh().vertex_matrix()
    
        vtx_matrix = torch.tensor(vtx_matrix, dtype=float)
    
        vtx = qrot(best_R, vtx_matrix.unsqueeze(0))
        vtx = torch.matmul(vtx, scale)
        vtx = vtx + best_T
    
        vtx = vtx.squeeze(0)
        vtx = vtx.clone().detach().numpy()

        ms.add_mesh(pymeshlab.Mesh(
                       vertex_matrix=vtx,
                       face_matrix=face_matrix))
        
        ms.save_current_mesh(os.path.join(primitives_folder, f'final.obj'))