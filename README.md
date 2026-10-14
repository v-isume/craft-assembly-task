# Component Selection for Craft Assembly Tasks

Code for the paper **Component Selection for Craft Assembly Tasks**, published in IEEE Robotics and Automation Letters (RA-L) and presented at ICRA 2025.

Vitor Hideyo Isume, Takuya Kiyokawa, Natsuki Yamanobe, Yukiyasu Domae, Weiwei Wan, Kensuke Harada

[Project page](https://v-isume.github.io/craft-assembly-task/) | [Paper](https://ieeexplore.ieee.org/document/10631671) | [arXiv](https://arxiv.org/abs/2407.14001)

Given an RGB image of a target object, the pipeline picks which objects from a scene to use to build a craft version of that object. It runs in five stages:

1. **Part segmentation:** EVA-02 (Cascade Mask R-CNN, ViTDet-L) detects the object class and segments its visible parts.
2. **Template retrieval and pose optimization:** finds the labeled template mesh that best matches the part masks and estimates its pose, using differentiable silhouette rendering with Kaolin.
3. **Primitive simplification:** reduces the visible parts of the posed template to primitive shapes (cuboids and cylinders), using symmetry to complete parts that aren't visible.
4. **Scene matching:** matches each primitive to an object in the scene based on local and global proportions.
5. **Repositioning:** adjusts the positions of the selected objects to remove collisions.

Supported object categories are `truck`, `bus`, `chair` and `table`.

## Repository structure

```
main.py                 # Runs stages 2–5 on every image in input/
cfg/
  folders_paths.yaml    # Input, output, template and scene paths
  eva2_*.py, ...        # EVA-02 / detectron2 model configs
utils/
  run_eva02.py          # Stage 1: EVA-02 segmentation (writes outputs/eva_predictions/)
  get_obj_from_eva.py   # Loads EVA-02 predictions
  get_silhouette_eva_load.py, silhouette_optm_load.py   # Template retrieval and pose optimization
  visible.py, symmetry.py, mesh_toprimitive.py          # Primitive generation
  compare_scene.py, apply_transform.py                  # Scene matching
  collision_check_new.py                                # Repositioning
  dictionaries.py       # Class, color and primitive definitions
rendering/              # Kaolin-based differentiable renderer
scene/scene_1221.npz    # Primitives of the available objects in the scene
input/                  # Put input RGB images (.jpg) here
```

## Installation

You need a CUDA-capable GPU.

1. **EVA-02 detectron2.** Clone [EVA](https://github.com/baaivision/EVA) and build its detectron2 fork from source, following the [EVA-02 det instructions](https://github.com/baaivision/EVA/tree/master/EVA-02/det):

   ```bash
   git clone https://github.com/baaivision/EVA.git
   cd EVA/EVA-02/det
   python -m pip install -e .
   ```

2. **Other dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

   Install the [PyTorch](https://pytorch.org/get-started/locally/) and [Kaolin](https://kaolin.readthedocs.io/en/latest/notes/installation.html) builds that match your CUDA version before running this.

## Data and weights

These files must be downloaded separately into the repository root:

| Path | Contents | Source |
|---|---|---|
| `weights/eva02_L_coco_det_sys_o365.pth` | EVA-02-L detector trained on Objects365 and COCO (finds the object class) | [EVA-02 det model zoo](https://github.com/baaivision/EVA/tree/master/EVA-02/det) |
| `weights/model_final.pth` | EVA-02-L fine-tuned on craft parts | [model_final.pth](https://drive.google.com/file/d/1gLZgCvnx8mtHaLkX5HzV13eA0NpjatX2/view?usp=sharing)|
| `template_db/` | Labeled template meshes for each category | TODO: add link |

## Usage

Run all commands from the repository root.

**1. Segment the input images with EVA-02:**

```bash
python -m utils.run_eva02
```

Put your `.jpg` images in `input/` first. The script runs on every `.jpg` there, and the results for each image go to `outputs/eva_predictions/<image name>/`:

- `prediction.npz`
- `all_combined.png`
- one mask per part

**2. Run component selection:**

```bash
python main.py
```

`main.py` processes every `.jpg` in `input/` that has EVA-02 predictions. For each image, it prints the matching between template parts and scene objects. All results are saved to `outputs/all_results.npz`.

Paths are set in `cfg/folders_paths.yaml`. Keep `input_folder: './input'` and `output_folder: './outputs'`, because `utils/get_obj_from_eva.py` looks for predictions using those folder names.

## Citation

```bibtex
@ARTICLE{10631671,
  author={Isume, Vitor H. and Kiyokawa, Takuya and Yamanobe, Natsuki and Domae, Yukiyasu and Wan, Weiwei and Harada, Kensuke},
  journal={IEEE Robotics and Automation Letters},
  title={Component Selection for Craft Assembly Tasks},
  year={2024},
  volume={9},
  number={9},
  pages={8122-8129},
  doi={10.1109/LRA.2024.3440847}}
```

## License

This project is released under the [MIT License](LICENSE).

## Acknowledgments

The segmentation stage is built on [EVA-02](https://github.com/baaivision/EVA) and [detectron2](https://github.com/facebookresearch/detectron2).

The code for the pose optimization step, including the differentiable renderer in `rendering/`, is derived from [textured-3d-gan](https://github.com/dariopavllo/textured-3d-gan), released under the MIT License (see [rendering/LICENSE](rendering/LICENSE)).
