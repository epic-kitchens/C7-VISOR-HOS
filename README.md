# EPIC-KITCHENS Hand Object Segmentation (HOS) Challenge

## HOS Challenge on Codalab platform
To participate and submit to this HOS challenge, register at the [HOS Codalab Challenge](https://codalab.lisn.upsaclay.fr/competitions/9969?secret_key=415101af-8490-404b-a28c-d6a4fbad4bcd).


## Data Download
Please Go to [EPIC-KITCHENS VISOR](https://epic-kitchens.github.io/VISOR/) official webpage to download the whole dataset, providing RGB frames, masks and hand-object relations in train/val/test splits. If you are interested in our data generation pipeline, please also check our [VISOR paper](https://arxiv.org/abs/2209.13064). 


## Data Loader and Environments
- Refer to [VISOR-HOS](https://github.com/epic-kitchens/VISOR-HOS) repository for how to load the data and generate COCO format annotation from the raw format in the download. You can also use other ways to load the data and other data formats too. 
- Refer to [VISOR-HOS Environment](https://github.com/epic-kitchens/VISOR-HOS#environment) session for environment setup. 
 

## Evaluation
- FYI, we use the COCO Mask AP metric implemented in [Detectron2](https://github.com/facebookresearch/detectron2) to get the numbers in our paper and in our challenge.
- Plese refer to [VISOR-HOS](https://github.com/epic-kitchens/VISOR-HOS) to check how we get COCO Mask AP evaluation in our baseline method. 
- For the HOS Codalab Challenge evaluation, with the 4 prediction PTH files prepared as instructed in the Codalab challenge, you can run the commend below to get the scores as in the table.
- For more details about the prediction PTH file and its format, if you use the [Detectron2 COCOEvaluator](https://detectron2.readthedocs.io/en/latest/modules/evaluation.html#detectron2.evaluation.COCOEvaluator), there will be a `instances_predictions.pth` file automatically generated and saved in the output folder, `/path/to/your/outputs/inference`. So anytime you are confused or uncertain about the PTH format, you can check that too.
 

Get [VISOR-HOS](https://github.com/epic-kitchens/VISOR-HOS) as submodule of this repository:
```
git submodule update --init
```

Make sure your `input_dir` folder structure is as below. Under the `input_dir`, there will be a `ref` sub-folder containing the COCO format annotations and a `res` sub-folder containing the predictions of your method in the required format.

```
/path/to/your/inputs
|--- ref
|    |--- epick_visor_coco_hos
|         |---annotations
|             |---val.json
|    |--- epick_visor_coco_handside
|         |---annotations
|             |---val.json
|    |--- epick_visor_coco_contact
|         |---annotations
|             |---val.json
|    |--- epick_visor_coco_combineHO
|         |---annotations
|             |---val.json
|--- res
|    |--- instances_predictions_hand_obj.pth
|    |--- instances_predictions_handside.pth
|    |--- instances_predictions_contact.pth
|    |--- instances_predictions_combineHO.pth
```

Then, simply evaluate with the command below:
```
python evaluate_hos.py --input_dir=/path/to/your/inputs --output_dir=/path/to/your/outputs
```


