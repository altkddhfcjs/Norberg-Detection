# Norberg-Detection
## v.1.

## Dataon의 workflow용 software 설명

### 해당 글은 https://dataon.kisti.re.kr/ 의 CANVAS에 등록된 Software 메뉴얼이다.

### Overview

![참고용](https://user-images.githubusercontent.com/14813266/142758591-964597ab-558e-4885-8c36-827262a173fa.png)

#### 그림과 같이 2개의 입력이 필요하다. 
#### -inpfolder는 폴더를 입력받는다. 
##### 폴더는 내부는 input.jpg, norberg_config.yaml, norberg_model.pth 다음과 같은 이름으로 저장되어야 한다.
##### input.jpg: 애완견의 다리 x-ray 사진으로 모델의 입력으로 사용됨
##### norberg_config.yaml: Detectron2의 모델 설정파일
##### norberg_model.pth: 모델 weight 파일
#### 
#### -cpu는 bool형태로 True/False로 입력을 받는다.
##### 아무것도 입력 안하는 경우 Default인 True로 진행된다.
##### False로 설정할 경우 GPU 환경에서 모델이 동작하고, True의 경우 CPU환경에서 모델이 동작한다.

###  Detectron2
### https://github.com/facebookresearch/detectron2
### https://detectron2.readthedocs.io/en/latest/tutorials/deployment.html

#### Norberg-Detection은 Keypoint R-CNN의 모델이다.
##### 다음 모델은 Detectron2 기반으로 작성되었다. 
##### Detectron2는 Computer vision task 중 Object detection, Segmentation, Keypoint estimation 등을 지원하는 플랫폼으로
##### 제공되는 모델을 config 파일만 불러와서 간편하게 사용할 수 있으며, 모듈화되어 있는 API를 이용하여 사용자가 편하게 모델을 만들 수도 있다.
##### Norberg-Detection에서 사용하는 config파일과 model.pth는 아래의 링크에서 다운받을 수 있다.
##### https://drive.google.com/drive/folders/1ta3su6ZDObIXC024DpV1DvQkz3sQZ6Ad?usp=sharing

### Detectron2 customizing

#### 사용자는 본인이 학습하고 제작한 Detectron2 기반의 모델을 불러와서 수정하여 software를 사용할 수 있다.

##### 문의 mail: cv.hyunseop.kim@gmail.com 
##### 데이터셋에 대한 정보는 추후 기술
