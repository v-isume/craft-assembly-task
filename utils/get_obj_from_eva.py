
# get results from the prediction of EVA
import os
import numpy as np
from PIL import Image, ImageOps

def get_list_class_name(batch):
    #Batch should be a list of paths to the images

    list_masks = []
    list_class_id = []
    list_class_name = []
    
    #3.1 loop
    for img in batch:

        curr_mask_list = []
        
        pred_path = img.replace('input/', 'outputs/eva_predictions/').replace('.jpg', '')
        #read predictions
        pred_results = np.load(os.path.join(pred_path, 'prediction.npz'))
        mask_pred = pred_results['all_masks']
        class_pred = pred_results['all_classes']
        major_class = pred_results['major_class'].item()

        keep = np.ones(len(mask_pred), dtype=bool)
        # verify if mask is empty
        for i in range(len(mask_pred)):
            if np.max(mask_pred[i]) == 0:
                keep[i] = False
        
        new_masks = mask_pred[keep]
        for new in new_masks:
            curr_img = Image.fromarray(new, 'L')
            curr_img = ImageOps.pad(curr_img, (256, 256))
            curr_mask_list.append(np.array(curr_img))
        

        
        # attach to lists
        list_masks.append(curr_mask_list)
        #list_masks.append(mask_pred[keep])
        list_class_id.append(class_pred[keep])
        list_class_name.append(major_class)

    return list_masks, list_class_id, list_class_name