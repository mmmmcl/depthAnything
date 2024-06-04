## DepthAnything
```
    torch==2.1.2
    cuda121
```
1.Train teahcer with all labled imgs to generate psedulabels
``` 
python train_teacher.py
```
2.Train student with  labled and unlabeled imgs
```
python train_student.py
```
### train_student:
- Mini dataset lead to overfitting
![train_student](readme_img\image.png)
![train_student](readme_img\image1.png)