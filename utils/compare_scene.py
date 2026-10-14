import numpy as np
import open3d as o3d
import os
from utils.apply_transform import merge_and_rotate

from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment

def emd(X, Y):
    d = cdist(X, Y)
    assignment = linear_sum_assignment(d)
    return d[assignment].sum() / min(len(X), len(Y))

class Normal_part():
    def __init__(self, dimensions, shape, translation=None, rotation=None):
        self.dimensions = dimensions
        self.shape = shape
        self.translation = translation
        self.rotation = rotation

#TODO review sorted and index calls. Since same values are possible, it messes up when figuring out the order. Do test with truck03 only
def getScene(template_primitives, scene_primitives, transform, img_name, output_folder):

    #1. Get the longest dimension of any part from the template
    max_dim = 0
    for i in range(len(template_primitives)):
        dimensions = template_primitives[i].dimensions
        max_dim_part = max(dimensions)
            
        if max_dim_part > max_dim:
            max_dim = max_dim_part
            max_tmp_index = i
    
    n_template = []
    #2. Normalize all template parts according to this maximum dimension
    for part in template_primitives:
        n_dimensions = [x/max_dim for x in part.dimensions]
        n_part = Normal_part(n_dimensions, part.shape)
        n_template.append(n_part)

    #3. Rank objects in the scene for the part with the longest dimension
    ref = n_template[max_tmp_index].dimensions
    ref.sort()
    #Ranking using msqe
    msqe_ranking = []
    for j in range(len(scene_primitives)):
        max_curr_dim = max(scene_primitives[j].dimensions)
        n_scene_dims = [x/max_curr_dim for x in scene_primitives[j].dimensions]
        n_scene_dims.sort()
        msqe = 0
        for k in range(len(n_scene_dims)):
            msqe += np.sqrt((n_scene_dims[k] - ref[k]) * (n_scene_dims[k] - ref[k]))
        msqe = msqe/len(n_scene_dims)
        msqe_ranking.append((j, msqe))
    
    msqe_ranking = sorted(msqe_ranking, key=lambda a:a[1])

    vol_ranking = []
    #Ranking using volume
    for l in range(len(scene_primitives)):
        dimensions = scene_primitives[l].dimensions
        if scene_primitives[l].shape == 'cuboid':
            curr_vol = dimensions[0] * dimensions[1] * dimensions[2]
        elif scene_primitives[l].shape == 'cylinder':
            curr_vol = dimensions[0] * dimensions[1] * dimensions[2] * np.pi
        vol_ranking.append((l, curr_vol))
    
    vol_ranking = sorted(vol_ranking, key=lambda a:a[1], reverse=True)

    avg_ranking = []
    for m in range(len(msqe_ranking)):
        curr_idx = msqe_ranking[m][0]
        for n in range(len(vol_ranking)):
            if curr_idx == vol_ranking[n][0]:
                med_ranking = ((m + 1) + (n + 1))/2
                avg_ranking.append((curr_idx, med_ranking))
    
    avg_ranking = sorted(avg_ranking, key=lambda a:a[1])

    max_sce_idx = avg_ranking[0][0]

    #Choose the highest ranking one for the part from 1
    max_dim_sce = max(scene_primitives[avg_ranking[0][0]].dimensions)
    n_scene = []
    counter_test = 0
    for part in scene_primitives:
        n_dimensions = [x/max_dim_sce for x in part.dimensions]
        for dim in n_dimensions:
            if dim > 1:
                counter_test += 1
        n_part = Normal_part(n_dimensions, part.shape)
        n_scene.append(n_part)

    best_parts_idx = [-1 for x in range(len(n_template))]
    best_parts_idx[max_tmp_index] = max_sce_idx

    all_std = []
    for n in range(len(n_template)):
        min_std = 1e8
        if n == max_tmp_index:
            continue
        best_sce_part = -1
        for o in range(len(n_scene)):
            if n_template[n].shape == n_scene[o].shape:
                n_dim = n_template[n].dimensions
                n_sce = n_scene[o].dimensions
                max_a = max(n_dim)
                max_b = max(n_sce)
                if n_template[n].shape == 'cuboid':
                    n_dim.sort()
                    n_sce.sort()
                    #diff = [abs(x - y) for x,y in zip(n_dim, n_sce)]
                    ratio = [abs(x - y) for x,y in zip(n_dim, n_sce)]
                else:
                    #diff = [abs(x - y) for x,y in zip(n_dim, n_sce)]
                    ratio = [abs(x - y) for x,y in zip(n_dim, n_sce)]
                ratio2 = [abs(x/max_a - y/max_b) for x,y in zip(n_dim, n_sce)]
                #avg_diff = np.average(diff)
                #std_diff = np.std(diff)
                avg_diff = np.average(ratio)
                std_diff = np.average(ratio2)
                #std_diff = np.std(ratio)
                rmse = np.sqrt(avg_diff * avg_diff + std_diff * std_diff)
                if (rmse < min_std) and (o not in best_parts_idx):
                    min_std = rmse
                    best_sce_part = o
            #print(n_scene[o].dimensions)
            #print(scene_primitives[o].dimensions)
            #print("-------------------------")
        #     if n == 1:
        #         print("-----------------------")
        #         print(n_dim)
        #         print(n_sce)
        #         print(ratio)
        #         print(avg_diff)
        #         print(std_diff)
        #         input("check")
        # print(n_dim)
        if best_sce_part == -1:
            print(n_template[n].dimensions)
            print(n_scene[o].dimensions)
            print(len(n_scene))
            print(counter_test)
            input("here")      
        best_parts_idx[n] = best_sce_part
        all_std.append(min_std)

    #4. Create a mockup of the assembled object using the selected SCENE primitives
    output_path = os.path.join(output_folder, 'scene_proposal', img_name)
    os.makedirs(output_path, exist_ok=True)

    #Save primitives from scene
    norm_dist_list = []
    max_tmp_primitive = template_primitives[max_tmp_index]

    max_tmp_dim = max_tmp_primitive.dimensions
    max_tmp_center = max_tmp_primitive.center
        
    list_index = []
    list_index.append(0)

    #Should be normalized in terms of the normalized dimensions of the corresponding part

    #9.1 Calculate distance between centers of primitives in the template mesh
    for n in range(len(template_primitives)):
        if n != max_tmp_index:
            primitive = template_primitives[n]
            center = primitive.center
            dist = [center[0] - max_tmp_center[0], center[1] - max_tmp_center[1], center[2] - max_tmp_center[2]]
            #dist = [center[0] - max_tmp_center[0], center[1] - max_tmp_center[1], center[2] - max_tmp_center[2]]
            # Calculate normalized distance in relation to dimensions of the part
            norm_dist = [x/y for x, y in zip(dist, max_tmp_dim)]
            #norm_dist = [x/y for x, y in zip(dist, primitive.dimensions)]
        else:
            norm_dist = [0, 0, 0]
            first_dist = [x/y for x, y in zip(max_tmp_center, max_tmp_dim)]
        
        norm_dist_list.append(norm_dist)

    max_sce_dim = scene_primitives[avg_ranking[0][0]].dimensions

    max_tmp_pair = [(x, max_tmp_dim[x]) for x in range(len(max_tmp_dim))]

    max_tmp_pair = sorted(max_tmp_pair, key=lambda x:x[1])
    max_sce_dim.sort()

    max_sce_re = [0 for _ in range(len(max_sce_dim))]

    for a in range(len(max_sce_re)):
        axis = max_tmp_pair[a][0]
        max_sce_re[axis] = max_sce_dim[a]

    #9.2. Iterate through proposed SCENE primitives and reposition them to match the template
    for o in range(len(template_primitives)):
        primitive_og = scene_primitives[best_parts_idx[o]]
        if primitive_og.shape == 'cuboid':
            dim_pair = [(x, template_primitives[o].dimensions[x]) for x in range(len(template_primitives[o].dimensions))]
            dim_pair = sorted(dim_pair, key=lambda x:x[1])
            dim_og = primitive_og.dimensions
            dim_og.sort()
            dim_re = [0 for _ in range(3)]
            for a in range(len(dim_pair)):
                axis = dim_pair[a][0]
                dim_re[axis] = dim_og[a]
            primitive = o3d.geometry.TriangleMesh.create_box(dim_re[0], dim_re[1], dim_re[2])
        else:
            primitive = o3d.geometry.TriangleMesh.create_cylinder(primitive_og.dimensions[0], primitive_og.dimensions[2])
        primitive = primitive.translate(-primitive.get_center())
        scaled_first = [x * y for x, y in zip(first_dist, max_sce_re)]
        primitive = primitive.translate(scaled_first)

        aabb = primitive.get_axis_aligned_bounding_box()
        aabb_dim = aabb.get_extent()

        if template_primitives[o].rotation != None:
            primitive = primitive.rotate(template_primitives[o].rotation.as_matrix())
        
        if best_parts_idx[o] == max_sce_idx:
            translation = [0, 0, 0]
        else:
            translation = [x * y for x, y in zip(norm_dist_list[o], max_sce_re)]
        #translation = [x * y for x, y in zip(norm_dist_list[o], (dim_re))]
        primitive_i = primitive.translate(np.asarray(translation))
        o3d.io.write_triangle_mesh(os.path.join(output_path, f'scene_object_{o}.obj'), primitive_i)

        print(o)
        print(translation)
        print(primitive_i.get_center())
        print(aabb_dim)
        print("-------------------------------")

    #9.3. Apply transform to the whole object (Detected pose)
    merge_and_rotate(img_name, transform, output_folder)

    return best_parts_idx