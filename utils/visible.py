import torch
import numpy as np
import pymeshlab
import torch.nn.functional as F
import json
import cv2
import os
import imageio

from utils.dictionaries import id_to_color, class_to_color

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

def recolor_part(part, part_class):
    #part given as a mesh
    #part class given as string
    no_vtx = part.vertex_number()
    part_color = class_to_color[part_class]

    color_matrix = np.full((no_vtx, 4), np.asarray(part_color) / 255.0)
    
    m2 = pymeshlab.Mesh(
        vertex_matrix=part.vertex_matrix(),
        face_matrix=part.face_matrix(),
        v_color_matrix=color_matrix)
    
    return m2

def part_from_mesh(mesh_, part_id, part_class, color_matrix, mesh_path=True):
    #part_type: string

    part_color = np.asarray(id_to_color[part_id])
    ms = pymeshlab.MeshSet()

    if mesh_path:
        ms.load_new_mesh(mesh_)
    else:
        ms.add_mesh(mesh_)

    vtx_matrix = ms.current_mesh().vertex_matrix()
    face_matrix = ms.current_mesh().face_matrix()

    m1 = pymeshlab.Mesh(
        vertex_matrix=vtx_matrix,
        face_matrix=face_matrix,
        f_color_matrix=color_matrix / 255)
    
    ms = pymeshlab.MeshSet()
    ms.add_mesh(m1)

    ms.compute_selection_by_condition_per_face(condselect = f'( fr >= {part_color[0] - 1}) && (fr <= {part_color[0] + 1}) && (fg >= {part_color[1] - 1}) && (fg <= {part_color[1] + 1}) && (fb >= {part_color[2] - 1}) && (fb <= {part_color[2] + 1}) && (fa == {part_color[3]})')

    ms.apply_selection_inverse(invfaces=True, invverts=True)
    ms.meshing_remove_selected_vertices_and_faces()

    mesh = ms.current_mesh()
    
    m2 = recolor_part(mesh, part_class)
    return m2

def find_visible(mesh_template_path, transform, all_masks, points_2d, all_classes, img_name, output_folder):
    
    #Read dictionary
    dictionary_path = mesh_template_path.replace('colored.obj', 'dict.txt')
    with open(dictionary_path) as f:
        data = f.read()

    id_to_part = json.loads(data)

    print(id_to_part)
    print(dictionary_path)

    ms = pymeshlab.MeshSet()
    ms.load_new_mesh(mesh_template_path)
    ms.compute_matrix_from_translation(traslmethod = 'Center on Scene BBox')
    ms.compute_matrix_from_scaling_or_normalization(unitflag = True)

    best_R = F.normalize(transform[3], dim=-1)
    best_T = transform[2]

    best_R = torch.tensor(best_R, dtype=float)
    best_T = torch.cat((best_T, torch.zeros_like(best_T[..., :1])), dim=-1)

    scale = torch.diag(torch.tensor([1, -1, -1])) #originally [1, -1, -1]
    scale = torch.tensor(scale, dtype=float)

    vtx_matrix = ms.current_mesh().vertex_matrix()
    vtx_matrix = torch.tensor(vtx_matrix, dtype=float)
    
    vtx_scale = transform[1].unsqueeze(1) * vtx_matrix
    vtx_scale = torch.matmul(vtx_scale, scale)

    #Apply transform
    vtx = qrot(best_R, vtx_matrix.unsqueeze(0))
    vtx = torch.matmul(vtx, scale)
    vtx = vtx + best_T
    vtx = vtx.squeeze(0)
    vtx = vtx.clone().detach().numpy()

    list_centers = []

    # Get list of centers:
    for mask in all_masks:
        #mask is 0 on background, rest is ID
        binary_mask = np.zeros_like(mask)
        binary_mask[np.nonzero(mask)] = 1
        binary_mask = np.uint8(binary_mask * 255)

        # Find contours:
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        if len(contours) > 1:
            max_area = 0
            max_idx = 0
            for i in range(len(contours)):
                curr_area = cv2.contourArea(contours[i])
                if curr_area > max_area:
                    max_area = curr_area
                    max_idx = i

            contours = contours[max_idx]
        else:
            contours = contours[0]

        rect = cv2.minAreaRect(contours)
        box = cv2.boxPoints(rect)
        box = np.intp(box)

        # Destroy all created windows:
        center_x = 0
        center_y = 0
        for pt in box:
            center_x += pt[0]
            center_y += pt[1]

        center_x //= 4
        center_y //= 4
        list_centers.append([center_x, center_y])
    
    texture_img = imageio.imread(mesh_template_path.replace('.obj', '.png'))
    height = texture_img.shape[0]
    width = texture_img.shape[1]
    uv_coord = ms.current_mesh().wedge_tex_coord_matrix()
    face_mtx = ms.current_mesh().face_matrix()
    face_color_matrix = np.zeros((face_mtx.shape[0], 4))


    counter = 0
    for n in range(0, uv_coord.shape[0], 3):
        x_1 = int(uv_coord[n][0] * width)
        y_1 = int((1 - uv_coord[n][1]) * height)

        x_2 = int(uv_coord[n+1][0] * width)
        y_2 = int((1 - uv_coord[n+1][1]) * height)

        x_3 = int(uv_coord[n+2][0] * width)
        y_3 = int((1 - uv_coord[n+2][1]) * height)

        x_avg = int((x_1 + x_2 + x_3)/3)
        y_avg = int((y_1 + y_2 + y_3)/3)

        color_avg = texture_img[y_avg][x_avg]

        face_color_matrix[counter] = color_avg

        counter += 1
    
    m2 = pymeshlab.Mesh(
        vertex_matrix=vtx_scale.squeeze(0).detach().numpy(),
        face_matrix=face_mtx,
        f_color_matrix=face_color_matrix / 255.0)

    
    m1 = pymeshlab.Mesh(
        vertex_matrix=vtx,
        face_matrix=face_mtx,
        f_color_matrix=face_color_matrix / 255.0)
    
    ms = pymeshlab.MeshSet()
    ms.add_mesh(m1)

    parts_face = []
    # Separate the faces that belong to each part of the template mesh
    for key in id_to_color.keys():
        curr_color = id_to_color[key]
        single_part_face = []
        for i in range(face_color_matrix.shape[0]):
            if (curr_color == face_color_matrix[i]).all():
                single_part_face.append(i)
        parts_face.append(single_part_face)
                
    face_px = points_2d
    face_px = face_px.squeeze(0)

    # Create an image with the part idx instead of face id
    new_face_px = np.full(face_px.shape, -1)
    visible_parts = []
    for x in range(face_px.shape[0]):
        for y in range(face_px.shape[1]):
            for j in range(len(parts_face)):
                if face_px[x][y] in parts_face[j]:
                    new_face_px[x][y] = j
                    if j not in visible_parts:
                        visible_parts.append(j)

    parts_center_x = np.zeros_like(visible_parts)
    parts_center_y = np.zeros_like(visible_parts)
    counter_part = np.zeros_like(visible_parts)
    # Calculate the center of each visible part from the template
    for y in range(new_face_px.shape[0]):
        for x in range(new_face_px.shape[1]):
            if new_face_px[x][y] != -1:
                for k in range(len(visible_parts)):
                    if visible_parts[k] == new_face_px[x][y]:
                        parts_center_x[k] += x
                        parts_center_y[k] += y
                        counter_part[k] += 1
    
    parts_center_x = [x/y for x, y in zip(parts_center_x, counter_part)]
    parts_center_y = [x/y for x, y in zip(parts_center_y, counter_part)]

    class_to_label = ['truck_body', 'truck_cabin', 'wheel', 'bus_body', 'furniture_leg', 'chair_seat', 'chair_back', 'chair_arm', 'table_surface']

    best_parts = []
    best_masks = []

    for center, mask_class in zip(list_centers, all_classes):
        x_mask = center[1]
        y_mask = center[0]
        min_dist = 256 * 256
        best_idx = -1
        for j in range(len(parts_center_x)):
            x_part = parts_center_x[j]
            y_part = parts_center_y[j]
            dist = np.sqrt((x_mask - x_part) * (x_mask - x_part) +  (y_mask - y_part) * (y_mask - y_part))
            curr_class = class_to_label.index(id_to_part[str(visible_parts[j])])
            if (dist < min_dist) and (visible_parts[j] not in best_parts) and (mask_class == curr_class):
                min_dist = dist
                best_idx = visible_parts[j]

        if best_idx != -1:
            best_parts.append(best_idx)
            best_masks.append(class_to_label.index(id_to_part[str(best_idx)]))


    os.makedirs(os.path.join(output_folder, 'parts', img_name),exist_ok=True)

    mask_labels = [class_to_label[x] for x in best_masks]
    
    for part, label in zip(best_parts, mask_labels):
        
        part_mesh = part_from_mesh(m2, part, label, face_color_matrix, mesh_path=False)
        ms = pymeshlab.MeshSet()
        ms.add_mesh(part_mesh)
        if ms.current_mesh().vertex_number() == 0:
            input("Error, check")
        ms.meshing_remove_connected_component_by_face_number(mincomponentsize = 2)
        if ms.current_mesh().vertex_number() == 0:
            input("Error after filter, check")
        ms.save_current_mesh(os.path.join(output_folder, 'parts', img_name, f'part_{part}.ply'))

    return best_parts, best_masks

