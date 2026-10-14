# 1. Read parts
import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation as R
import os
from utils.dictionaries import primitive_dictionary

maskid_name = ['truck_body', 'truck_cabin', 'wheel', 'bus_body', 'furniture_leg', 'chair_seat', 'chair_back', 'chair_arm', 'table_surface']

class Primitive_img():
    def __init__(self, shape, dimensions, index, rotation=None, translation=None, center=None, part_cat=None, part_type=None, idx_s=None, connections=None):
        self.shape = shape
        self.dimensions = dimensions
        self.index = index
        self.rotation = rotation
        self.translation = translation
        self.center = center
        self.part_type = part_type
        self.part_cat = part_cat
        if part_type != None:
            assert part_type in ["visible", "hidden", "symmetric"], f'Part type is unknown: {part_type}. Should be one of [visible, hidden, symmetric].'
            if part_type == "symmetric":
                self.cloned_ref = idx_s
        if part_cat == "axle":
            self.connections = connections


def getPrimitive(list_idx, list_mask_ids, list_clone, img_name, output_folder):

    #list_idx = list of template mesh parts id
    #list_mask_ids = list of mask ids predictions from PointRend
    list_name = [maskid_name[x] for x in list_mask_ids]

    #find position of cloned part in the list_idx
    pos_clone = [list_idx.index(x) for x in list_clone]
    list_clone_mask_id = [list_mask_ids[x] for x in pos_clone]

    list_name_clone = [maskid_name[x] for x in list_clone_mask_id]

    output_path = os.path.join(output_folder, 'primitives', img_name)
    os.makedirs(output_path, exist_ok=True)

    out_primitives = []

    for idx, obj_type in zip(list_idx, list_name):
        file_name = os.path.join(output_folder, 'parts', img_name, f'part_{idx}.ply')
        mesh = o3d.io.read_triangle_mesh(file_name)

        aabb = mesh.get_axis_aligned_bounding_box()

        translation = aabb.get_center()

        aabb_max = aabb.max_bound
        aabb_min = aabb.min_bound

        dimensions = [aabb_max[0] - aabb_min[0], aabb_max[1] - aabb_min[1], aabb_max[2] - aabb_min[2]]

        for dim in dimensions:
            if dim <= 0:
                print(dimensions)
                print(aabb_max)
                print(aabb_min)
                print(file_name)
                print("STOP")

        primitive = primitive_dictionary[obj_type]

        if primitive == 'cylinder':
            primitive_mesh = o3d.geometry.TriangleMesh.create_cylinder(dimensions[1]/2, dimensions[0])
            rotation = R.from_euler('y', 90, degrees=True)
            primitive_mesh = primitive_mesh.rotate(rotation.as_matrix())
            primitive = primitive_mesh.translate(translation)
            o3d.io.write_triangle_mesh(os.path.join(output_path, f'primitive_{idx}.obj'), primitive)

            center = primitive.get_axis_aligned_bounding_box().get_center()
            
            out_primitives.append(Primitive_img('cylinder', [dimensions[1]/2, dimensions[1]/2, dimensions[0]], f'{idx}', rotation, translation, center, obj_type, "visible"))

        elif primitive == 'cuboid':
            primitive_mesh = o3d.geometry.TriangleMesh.create_box(dimensions[0], dimensions[1], dimensions[2])
            primitive_mesh = primitive_mesh.translate([-x/2 for x in dimensions])
            primitive = primitive_mesh.translate(translation)
            o3d.io.write_triangle_mesh(os.path.join(output_path, f'primitive_{idx}.obj'), primitive)

            center = primitive.get_axis_aligned_bounding_box().get_center()

            out_primitives.append(Primitive_img('cuboid', [dimensions[0], dimensions[1], dimensions[2]], f'{idx}', translation=translation, center=center, part_cat=obj_type, part_type="visible"))

    for idx, obj_type in zip(list_clone, list_name_clone):
        file_name = os.path.join(output_folder, 'symmetry', img_name, f'cloned_{idx}.obj')
        mesh = o3d.io.read_triangle_mesh(file_name)

        aabb = mesh.get_axis_aligned_bounding_box()

        translation = aabb.get_center()

        aabb_max = aabb.max_bound
        aabb_min = aabb.min_bound

        dimensions = [aabb_max[0] - aabb_min[0], aabb_max[1] - aabb_min[1], aabb_max[2] - aabb_min[2]]

        primitive = primitive_dictionary[obj_type]

        if primitive == 'cylinder':
            primitive_mesh = o3d.geometry.TriangleMesh.create_cylinder(dimensions[1]/2, dimensions[0])
            rotation = R.from_euler('y', 90, degrees=True)
            primitive_mesh = primitive_mesh.rotate(rotation.as_matrix())
            primitive = primitive_mesh.translate(translation)
            o3d.io.write_triangle_mesh(os.path.join(output_path, f'primitive_cloned_{idx}.obj'), primitive)
            
            center = primitive_mesh.get_axis_aligned_bounding_box().get_center()
            
            out_primitives.append(Primitive_img('cylinder', [dimensions[1]/2, dimensions[1]/2, dimensions[0]], f'cloned_{idx}', rotation, translation, center, part_cat=obj_type, part_type="symmetric", idx_s=idx))

        elif primitive == 'cuboid':
            primitive_mesh = o3d.geometry.TriangleMesh.create_box(dimensions[0], dimensions[1], dimensions[2])
            primitive_mesh = primitive_mesh.translate([-x/2 for x in dimensions])
            primitive = primitive_mesh.translate(translation)
            o3d.io.write_triangle_mesh(os.path.join(output_path, f'primitive_cloned_{idx}.obj'), primitive)
            
            center = primitive_mesh.get_axis_aligned_bounding_box().get_center()
            
            out_primitives.append(Primitive_img('cuboid', [dimensions[0], dimensions[1], dimensions[2]], f'cloned_{idx}', translation = translation, center=center, part_cat=obj_type, part_type="symmetric", idx_s=idx))

    all_list = list_name
    list_file_names = [f'primitive_{list_idx[x]}.obj' for x in range(len(all_list))]
    
    count = 0
    for i in list_name_clone:
        all_list.append(i)
        list_file_names.append(f'primitive_cloned_{list_clone[count]}.obj')
        count += 1

    threshold = 0.05
    
    for i in range(len(all_list)):
        for j in range(len(all_list)):
            if (i >= j):
                continue
            if (all_list[i] == 'wheel') and (all_list[j] == 'wheel'):
                part_1 = o3d.io.read_triangle_mesh(os.path.join(output_path, list_file_names[i]))
                part_2 = o3d.io.read_triangle_mesh(os.path.join(output_path, list_file_names[j]))

                aabb_1 = part_1.get_axis_aligned_bounding_box()
                aabb_2 = part_2.get_axis_aligned_bounding_box()

                center_1 = aabb_1.get_center()
                center_2 = aabb_2.get_center()
        
                #dist = np.sqrt((-center_1[0] - center_2[0]) ** 2 + (center_1[1] - center_2[1]) ** 2 + (center_1[2] - center_2[2]) ** 2)
                dist = np.sqrt((center_1[1] - center_2[1]) ** 2 + (center_1[2] - center_2[2]) ** 2) #distance in y and z coordinate must be 0 for them to be aligned

                if dist < threshold:
                    aabb_1_max = aabb_1.max_bound
                    aabb_1_min = aabb_1.min_bound

                    dimensions = [aabb_1_max[0] - aabb_1_min[0], aabb_1_max[1] - aabb_1_min[1], aabb_1_max[2] - aabb_1_min[2]]
                    primitive_mesh = o3d.geometry.TriangleMesh.create_cylinder(dimensions[1]*0.3/2, abs(center_2[0] - center_1[0]) - dimensions[0])
                    rotation = R.from_euler('y', 90, degrees=True)
                    primitive_mesh = primitive_mesh.rotate(rotation.as_matrix())

                    translation = [(center_1[0] + center_2[0])/2, center_1[1], center_1[2]]
                    primitive_mesh = primitive_mesh.translate(translation)

                    o3d.io.write_triangle_mesh(os.path.join(output_path, f'hidden_connection_{i}.obj'), primitive_mesh)

                    center = primitive_mesh.get_axis_aligned_bounding_box().get_center()

                    out_primitives.append(Primitive_img('cylinder', [dimensions[1]*0.3/2, dimensions[1]*0.3/2, abs(center_2[0] - center_1[0]) - dimensions[0]], f'hidden_connection_{i}', rotation, translation, center, "axle", "hidden", connections=[i, j]))
    return out_primitives