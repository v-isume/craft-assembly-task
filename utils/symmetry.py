import numpy as np
import pymeshlab
import os

def findSymmetric(visible_parts_ids, img_name, output_folder):

    list_of_parts = visible_parts_ids

    files_folder = os.path.join(output_folder, 'parts', img_name)

    test_break = False
    list_clone = [] 

    output_folder = os.path.join(output_folder, 'symmetry', img_name)
    os.makedirs(output_folder, exist_ok=True)    

    x_position_list = []

    #First check the center of parts to see if there are already symmetric parts around XZ plane
    list_centers = []
    for part in list_of_parts:
        path = files_folder + f'/part_{part}.ply'

        ms = pymeshlab.MeshSet()
        ms.load_new_mesh(path)

        bbox = ms.current_mesh().bounding_box()
        vtx = ms.current_mesh().vertex_matrix()

        center = (bbox.max() + bbox.min())/2
        list_centers.append(center)

    has_sym = []
    for center in list_centers:
        mirror_center = [-center[0], center[1], center[2]]
        append = True
        for next_center in list_centers:
            if (center == next_center).all():
                continue
            dist = np.sqrt((mirror_center[0] - next_center[0]) ** 2 + (mirror_center[1] - next_center[1]) ** 2 + (mirror_center[2] - next_center[2]) ** 2)
            if dist < 0.05:
                has_sym.append(True)
                append = False
                break
        if append:
            has_sym.append(False)

    assert len(has_sym) == len(list_centers), f"Something went wrong, check: has_sym={has_sym}, list_centers={list_centers}"

    #Propose symmetric part
    for i in range(len(list_of_parts)):
        if has_sym[i] == True:
            continue
        
        part = list_of_parts[i]
        path = files_folder + f'/part_{part}.ply'
        ms = pymeshlab.MeshSet()
        ms.load_new_mesh(path)

        bbox = ms.current_mesh().bounding_box()
        vtx = ms.current_mesh().vertex_matrix()

        center = (bbox.max() + bbox.min())/2

        vtx_new = np.zeros(vtx.shape)

        #first test one point to determine if it is positive or negative
        positive = False
        if vtx[0][0] > 0:
            positive = True
            for i in range(len(vtx)):
                if vtx[i][0] < 0:
                    test_break = True
                    break
                else:
                    vtx_new[i][0] = -vtx[i][0]
                    vtx_new[i][1] = vtx[i][1]
                    vtx_new[i][2] = vtx[i][2]
        
        else:
            for i in range(len(vtx)):
                if vtx[i][0] > 0:
                    test_break = True
                    break
                else:
                    vtx_new[i][0] = -vtx[i][0]
                    vtx_new[i][1] = vtx[i][1]
                    vtx_new[i][2] = vtx[i][2]

        face_matrix = ms.current_mesh().face_matrix()

        if test_break:
            test_break = False
            continue
        
        ms = pymeshlab.MeshSet()
        ms.add_mesh(pymeshlab.Mesh(
                       vertex_matrix=vtx_new,
                       face_matrix=face_matrix))

        x_position_list.append(positive)
        ms.save_current_mesh(os.path.join(output_folder, f'cloned_{part}.obj'))
        list_clone.append(part)

    return list_clone, x_position_list