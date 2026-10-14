import os
import pymeshlab
import numpy as np


def check_overlap(box1, box2):

    box1_center = (box1.max() + box1.min())/2
    box2_center = (box2.max() + box2.min())/2

    overlap_x = abs(box1_center[0] - box2_center[0]) <= (box1.dim_x() + box2.dim_x()) / 2
    overlap_y = abs(box1_center[1] - box2_center[1]) <= (box1.dim_y() + box2.dim_y()) / 2
    overlap_z = abs(box1_center[2] - box2_center[2]) <= (box1.dim_z() + box2.dim_z()) / 2

    return overlap_x and overlap_y and overlap_z

def check_floating(box1, box2):

    box1_center = (box1.max() + box1.min())/2
    box2_center = (box2.max() + box2.min())/2

    float_x = abs(box1_center[0] - box2_center[0]) > (box1.dim_x() + box2.dim_x()) / 2
    float_y = abs(box1_center[1] - box2_center[1]) > (box1.dim_y() + box2.dim_y()) / 2
    float_z = abs(box1_center[2] - box2_center[2]) > (box1.dim_z() + box2.dim_z()) / 2

    return [float_x, float_y, float_z]

def repositionVisible(primitives, img_name, output_folder, class_name):
    #Create output folder
    reposition_folder = os.path.join(output_folder, 'parts_mod', img_name)
    os.makedirs(reposition_folder, exist_ok=True)

    #Files folder
    files_folder = os.path.join(output_folder, 'scene_proposal', img_name)

    if class_name in ["truck", "bus"]:
        for i in range(len(primitives)):
            if primitives[i].part_cat == "axle":
                connected_parts = primitives[i].connections
                axle_path = files_folder + f'/scene_object_{i}.obj'
                w1_path = files_folder + f'/scene_object_{connected_parts[0]}.obj'
                w2_path = files_folder + f'/scene_object_{connected_parts[1]}.obj'

                ms = pymeshlab.MeshSet()
                ms.load_new_mesh(axle_path)
                axle_bbox = ms.current_mesh().bounding_box()
                ms.load_new_mesh(w1_path)
                w1_bbox = ms.current_mesh().bounding_box()
                ms.load_new_mesh(w2_path)
                w2_bbox = ms.current_mesh().bounding_box()

                axle_center = (axle_bbox.max() + axle_bbox.min())/2
                w1_center = (w1_bbox.max() + w1_bbox.min())/2
                w2_center = (w2_bbox.max() + w2_bbox.min())/2

                overlap_x1 = (axle_bbox.dim_x() + w1_bbox.dim_x()) / 2 - abs(axle_center[0] - w1_center[0])
                overlap_x2 = (axle_bbox.dim_x() + w2_bbox.dim_x()) / 2 - abs(axle_center[0] - w2_center[0])

                direction1 = np.sign(w1_center[0] - axle_center[0])
                direction2 = np.sign(w2_center[0] - axle_center[0])

                overlap_x1 = direction1 * overlap_x1
                overlap_x2 = direction2 * overlap_x2

                ms.set_current_mesh(1)
                ms.compute_matrix_from_translation_rotation_scale(translationx = overlap_x1)
                save_name = f"primitive_{connected_parts[0]}.obj"
                ms.save_current_mesh(os.path.join(reposition_folder, save_name))
                
                ms.set_current_mesh(2)
                ms.compute_matrix_from_translation_rotation_scale(translationx = overlap_x2)

                save_name = f"primitive_{connected_parts[1]}.obj"
                ms.save_current_mesh(os.path.join(reposition_folder, save_name))
                
    else:
        list_parts = os.listdir(files_folder)
        list_parts = [x.replace("scene_object_", "").replace(".obj", "") for x in list_parts]
        largest_dim = 0
        for part in list_parts:

            path = files_folder + f'/scene_object_{part}.obj'

            ms = pymeshlab.MeshSet()
            ms.load_new_mesh(path)

            bbox = ms.current_mesh().bounding_box()

            #Since all dimensions are axis aligned, we can use the bbox dimensions to estimate which part is bigger
            longest_dim = np.max(bbox.max() - bbox.min())        

            if longest_dim > largest_dim:
                largest_dim = longest_dim
                largest_id = part

        #move all parts to avoid collistion
        #first load the largest part
        ms = pymeshlab.MeshSet()
        path_largest = files_folder + f'/scene_object_{largest_id}.obj'
        ms.load_new_mesh(path_largest)

        bbox_largest = ms.current_mesh().bounding_box()

        for part in list_parts:
            if part != largest_id:
                path = files_folder + f'/scene_object_{part}.obj'

                ms = pymeshlab.MeshSet()
                path_largest = files_folder + f'/scene_object_{largest_id}.obj'
                ms.load_new_mesh(path_largest)

                bbox_largest = ms.current_mesh().bounding_box()

                ms.load_new_mesh(path)
                bbox = ms.current_mesh().bounding_box()
                ms.set_current_mesh(1)

                overlap = check_overlap(bbox_largest, bbox)
                floating = check_floating(bbox_largest, bbox)

                print("-------------------")
                print(overlap)
                print(floating)

                bbox_largest_center = (bbox_largest.max() + bbox_largest.min())/2
                bbox_center = (bbox.max() + bbox.min())/2

                if overlap:
                    overlap_x = (bbox_largest.dim_x() + bbox.dim_x()) / 2 - abs(bbox_largest_center[0] - bbox_center[0])
                    overlap_y = (bbox_largest.dim_y() + bbox.dim_y()) / 2 - abs(bbox_largest_center[1] - bbox_center[1])
                    overlap_z = (bbox_largest.dim_z() + bbox.dim_z()) / 2 - abs(bbox_largest_center[2] - bbox_center[2])

                    mtv = np.array([overlap_x, overlap_y, overlap_z])
                    axis = np.argmin(mtv)

                    if mtv[axis] > 0:
                        direction = np.sign(bbox_center[axis] - bbox_largest_center[axis])
                        mtv[axis] = direction * mtv[axis]

                    final_mtv = [0, 0 ,0]
                    for i in range(3):
                        if i == axis:
                            final_mtv[i] = mtv[i]

                    ms.compute_matrix_from_translation_rotation_scale(translationx = final_mtv[0], translationy = final_mtv[1], translationz = final_mtv[2])

                elif any(floating):
                    print("hey")
                    floating_x = (bbox_largest_center[0] - bbox_center[0]) - (bbox_largest.dim_x() + bbox.dim_x()) / 2
                    floating_y = (bbox_largest_center[1] - bbox_center[1]) - (bbox_largest.dim_y() + bbox.dim_y()) / 2
                    floating_z = (bbox_largest_center[2] - bbox_center[2]) - (bbox_largest.dim_z() + bbox.dim_z()) / 2

                    mtv = np.array([floating_x, floating_y, floating_z])
                    print(mtv)

                    for i in range(3):
                        if floating[i]:
                            direction = np.sign(bbox_largest_center[i] - bbox_center[i])
                            mtv[i] = direction * mtv[i]
                        else:
                            mtv[i] = 0

                    ms.compute_matrix_from_translation_rotation_scale(translationx = mtv[0], translationy = mtv[1], translationz = mtv[2])


                save_name = f"primitive_{part.index}.obj"
                ms.save_current_mesh(os.path.join(reposition_folder, save_name))
                

if __name__ == "__main__":
    repositionVisible([0, 1, 2, 3, 4, 5], "0010", "../backup/results_silhouette_good_1216/")