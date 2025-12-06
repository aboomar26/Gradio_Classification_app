import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import cv2
import os



class CustomCNNClassifier(nn.Module):
  def __init__ (self,input_dim , num_classes):
    super(CustomCNNClassifier,self).__init__()
    self.input_dim = input_dim
    self.num_classes = num_classes


    self.conv_layer = nn.Sequential(
        
                                            
        nn.Conv2d(3, 32, 3, padding=1),
        nn.BatchNorm2d(32),
        nn.ReLU(),
        nn.Conv2d(32, 32, 3, padding=1),   
        nn.BatchNorm2d(32),
        nn.ReLU(),
        nn.MaxPool2d(2, 2),

        # Block 2
        nn.Conv2d(32, 64, 3, padding=1),
        nn.BatchNorm2d(64),
        nn.ReLU(),
        nn.Conv2d(64, 64, 3, padding=1),
        nn.BatchNorm2d(64),
        nn.ReLU(),
        nn.MaxPool2d(2, 2),

        # Block 3
        nn.Conv2d(64, 128, 3, padding=1),
        nn.BatchNorm2d(128),
        nn.ReLU(),
        nn.Conv2d(128, 128, 3, padding=1),
        nn.BatchNorm2d(128),
        nn.ReLU(),
        nn.MaxPool2d(2, 2),

        # Block 4
        nn.Conv2d(128, 256, 3, padding=1),
        nn.BatchNorm2d(256),
        nn.ReLU(),
        nn.Conv2d(256, 256, 3, padding=1),
        nn.BatchNorm2d(256),
        nn.ReLU(),
        nn.MaxPool2d(2, 2),



    )

    self.to_linear = None
    self._get_conv_output(self.input_dim)


    self.linear_layer = nn.Sequential(
    
          nn.Linear(self.to_linear, 256),
          nn.ReLU(),
          nn.Dropout(0.4),

          nn.Linear(256, 128),
          nn.ReLU(),
          nn.Dropout(0.3),

          nn.Linear(128, num_classes),

    )



  def _get_conv_output(self,input_dim):
    with torch.no_grad():
      dummy_input = torch.zeros(1,3,input_dim,input_dim)
      output = self.conv_layer(dummy_input)
      self.to_linear = output.view(1,-1).size(1)

  def forward(self,x):
    x = self.conv_layer(x)
    x = x.view(x.size(0),-1)
    x = self.linear_layer(x)

    return x




class ImageClassifier:
    def __init__(self,model_path,class_name = None):
        
        self.model_path = model_path
        self.device = ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = CustomCNNClassifier(input_dim = 128, num_classes = 3).to(self.device)
        self.model.load_state_dict(torch.load(self.model_path,map_location = self.device))
        self.model.eval()


        if class_name is None:
            self.class_name = {0: 'dog', 1: 'snake', 2: 'cat'}
        else:
           self.class_name = class_name


        self.transform = transforms.Compose([
        transforms.Resize((128,128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                                            ])
        
       
           
        
    def Predict(self,image_path):
        self.image_path =image_path

        image = Image.open(self.image_path)
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
           output = self.model(image_tensor)
           _,predict = torch.max(output,1)

           label = self.class_name[predict.item()]
           img = cv2.imread(self.image_path)

           cv2.putText(img , label , (10,30) , cv2.FONT_HERSHEY_SIMPLEX , 1, (255,0,0) ,3)
           output_path = 'labels_image.jpg'
           cv2.imwrite(output_path , img)


           cwd = os.getcwd()
           output_path = os.path.join(cwd,output_path)

           return label ,output_path








