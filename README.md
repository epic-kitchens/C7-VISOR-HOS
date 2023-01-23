# EPIC-KITCHENS Hand Object Segmentation (HOS) Challenge

## HOS Challenge on Codalab platform
To participate and submit to this HOS challenge, register at the [HOS Codalab Challenge](https://codalab.lisn.upsaclay.fr/competitions/9910?secret_key=4ddd83a9-9d78-4c4d-b03f-7f5f4e32434f).


## Data Download
Please Go to [EPIC-KITCHENS VISOR](https://epic-kitchens.github.io/VISOR/) official webpage to download the whole dataset, providing RGB frames, masks and hand-object relations in train/val/test splits. If you are interested in our data generation pipeline, please also check our [VISOR paper](https://arxiv.org/abs/2209.13064). 


## Data Loader and Environments
- Refer to [VISOR-HOS](https://github.com/epic-kitchens/VISOR-HOS) repository for how to load the data and generate COCO format annotation from the raw format in the download. You can also use other ways to load the data and other data formats. 
- Refer to [VISOR-HOS Environment](https://github.com/epic-kitchens/VISOR-HOS#environment) session for environment setup. 
 

## Evaluation
- FYI, we use the COCO Mask AP metric implemented in [Detectron2](https://github.com/facebookresearch/detectron2) to get the numbers in our paper and in our challenge.
- Plese refer to [VISOR-HOS](https://github.com/epic-kitchens/VISOR-HOS) to check how we get COCO Mask AP evaluation in our baseline method. 
- For the HOS Codalab Challenge evaluation, with the 4 prediction PTH files prepared as instructed in the Codalab challenge, you can run the commend below to get the scores as in the table.
- For more details about the prediction PTH file and its format, if you follow the [Detectron2 COCOEvaluator](https://detectron2.readthedocs.io/en/latest/modules/evaluation.html#detectron2.evaluation.COCOEvaluator), there will be `instances_predictions.pth` automatically generated and saved in your output folder, `/path/to/your/ouputs/inference`. So anytime you are confused or uncertain about the PTH format, please check the 
 
Add [VISOR-HOS](https://github.com/epic-kitchens/VISOR-HOS) as submodule of this repository.
```
git submodule add https://github.com/epic-kitchens/VISOR-HOS
```
Make sure your `input_dir` folder structure is as below:
```
/path/to/your/inputs
|--- ref
|    |--- epick_visor_coco_hos
|    |--- epick_visor_coco_handside
|    |--- epick_visor_coco_contact
|    |--- epick_visor_coco_combineHO
|--- res
|    |--- instances_predictions_hand_obj.pth
|    |--- instances_predictions_handside.pth
|    |--- instances_predictions_contact.pth
|    |--- instances_predictions_combineHO.pth
```

Then, simply evaluate with the command below:
```
python evaluate_hos.py input_dir output_dir
```


