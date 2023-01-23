'''
EPIC-KITCHENS Hand Object Segmentation Challenge Evaluation.
'''
import os, sys, pdb, argparse
sys.path.append('VISOR-HOS')
import torch
from pathlib import Path

# register dataset
from detectron2.data import MetadataCatalog
from hos.data.datasets.epick import register_epick_instances
from hos.evaluation.epick_evaluation import EPICKEvaluator

# __here__ = Path(__file__).absolute().parent
# sys.path.append(str(__here__.parent))



def main(args):
    #********** check and make directory **********#
    input_dir, output_dir = args.input_dir, args.output_dir
    assert input_dir.exists(), "Expected input folder {} to exist".format(input_dir)
    results_dir = input_dir / "res"
    reference_dir = input_dir / "ref"
    assert results_dir.exists(), "Expected results folder {} to exist".format(input_dir)
    assert reference_dir.exists(), "Expected reference folder {} to exist".format(
        input_dir
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    
    
    
    #********** register data **********#ß
    version = reference_dir / 'epick_visor_coco_hos'
    register_epick_instances("epick_visor_2022_test_hos", {}, f"{version}/annotations/test.json", f"{version}/test")
    MetadataCatalog.get("epick_visor_2022_test_hos").thing_classes = ["hand", "object"]

    version = reference_dir / 'epick_visor_coco_contact'
    register_epick_instances("epick_visor_2022_test_contact", {}, f"{version}/annotations/test.json", f"{version}/test")
    MetadataCatalog.get("epick_visor_2022_test_contact").thing_classes = ["not_incontact", 'incontact']

    version = reference_dir / 'epick_visor_coco_handside'
    register_epick_instances("epick_visor_2022_test_handside", {}, f"{version}/annotations/test.json", f"{version}/test")
    MetadataCatalog.get("epick_visor_2022_test_handside").thing_classes = ["left", "right"]

    version = reference_dir / 'epick_visor_coco_combineHO'
    register_epick_instances("epick_visor_2022_test_combineHO", {}, f"{version}/annotations/test.json", f"{version}/test")
    MetadataCatalog.get("epick_visor_2022_test_combineHO").thing_classes = ["combineHandObj"]
    
    
    
    #********** evaluation **********#
    task_ls = ['hand_obj', 'handside', 'contact', 'combineHO']
    eval_task_ls = ['hand_obj', 'handside', 'contact', 'combineHO']
    dataset_ls = ['epick_visor_2022_test_hos', 'epick_visor_2022_test_handside', 'epick_visor_2022_test_contact', 'epick_visor_2022_test_combineHO']
    scores = {}
    for i, task in enumerate(task_ls):
        
        # ****** path ******#
        pred_path = results_dir / f'instances_predictions_{task}.pth'
        dataset_name = dataset_ls[i]
        eval_task = eval_task_ls[i]
        
        result_dir = os.path.join('./d2_outputs/', dataset_name)
        os.makedirs(result_dir, exist_ok=True)
        print('result dir = ', result_dir)

        
        # ****** evaluator ******#
        evaluator = EPICKEvaluator(dataset_name, output_dir=result_dir, eval_task=eval_task, tasks=['segm'])
        evaluator.reset()


        
        # ****** load preds ******#
        print("loading predictions ...")
        f = open(pred_path, 'rb')
        predictions = torch.load(f) # make sure the format is correct
        for i, item in enumerate(predictions):
            id = item['image_id']
            for j, ins in enumerate(item['instances']):
                predictions[i]['instances'][j]['category_id']  = predictions[i]['instances'][j]['category_id'] + 1 

        
        print("feeding predictions...")
        evaluator._predictions = predictions # list(itertools.chain(*predictions))
        
        
        # ****** evaluating ******#
        print("evaluating with [detectron2 evaluation]")
        results_i = evaluator.evaluate()
        
        
        # ****** store scores ******#
        if task == 'hand_obj':
            scores['hand']   = f"{results_i['segm']['AP-hand']}"
            scores['object'] = f"{results_i['segm']['AP-object']}"
            
        elif task == 'handside':
            scores['handside']   = f"{results_i['segm']['AP']}"
            
        elif task == 'contact':
            scores['contact']   = f"{results_i['segm']['AP']}"
            
        elif task == 'combineHO':
            scores['combineHO']   = f"{results_i['segm']['AP']}"
            
            
    # ****** save ******#
    score_path = output_dir / 'scores.txt'
    with open(score_path, 'w') as f:
        for name, score in scores.items():
            f.write(f'{name}: {score}\n')

        
    
    print(f'scores = {scores}')
    

if __name__ == '__main__':
    
    
    parser = argparse.ArgumentParser(
        description="Evaluate EPIC-Kitchens Hand Object Segmentation challenge results",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "input_dir",
        type=Path,
        help="Directory containing subfolders `res`, "
        "the submitted results, and `ref`, the ground"
        "truth reference to evaluate against",
    )
    parser.add_argument(
        "output_dir", type=Path, help="Directory in which to write `scores.txt`"
    )
    args = parser.parse_args()
    
    main(args)