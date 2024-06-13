# VisionQA-Llama2-OWLViT

## Introduce
This is a multimodal model design for the Vision Question Answering (VQA) task. It integrates the Llama2 13B, OWL-ViT, and YOLOv8 models, utilizing hard prompt tuning.
### features:
1. Llama2 13B handles language understanding and generation.
2. OWL-ViT identifies objects in the image relevant to the question.
3. YOLOv8 efficiently detects and annotates objects within the image <br>

Combining these models leverages their strengths for precise and efficient VQA, ensuring accurate object recognition and context understanding from both language and visual inputs.
## Requirement
```markdown
pip install requirements.txt
```
## Data
I evaluate the testing data from the [GQA](https://cs.stanford.edu/people/dorarad/gqa/download.html) dataset.
## Eval
```markdown
python val_zero_shot.py 
```
--imgs_path: The path of the GQA data image file <br>
--dataroot: The path of the GQA data <br>
--mode: ['testdev', 'val', 'train'] <br>

## Run
```markdown
python zero_shot.py
```
--img_path: The path of the question image <br>
--yolo_weight: The pre-train yolov8 weight <br>

## Predict result
1. The resutl of GQA accuracy score is 0.52.

![image](https://github.com/ycchen218/VisionQA-Llama2-OWLViT/blob/main/git_image/QA.png)
