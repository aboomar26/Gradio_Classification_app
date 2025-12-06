import gradio as gr
from core.predict import ImageClassifier
import os
from PIL import Image

cwd = os.getcwd()
model_path = os.path.join(cwd,'model','cnn_model.pth')
class_name = {0: 'dog', 1: 'snake', 2: 'cat'}
classifier = ImageClassifier(model_path=model_path , class_name=class_name)

def classify_image(image):
    image_path = 'uploaded_image.jpg'
    image.save(image_path)

    label , output_path = classifier.Predict(image_path)

    return label ,Image.open(output_path)



demo = gr.Interface(
    fn = classify_image,
    inputs = gr.Image(type='pil'),
    outputs = [gr.Textbox(label='predixtion') , gr.Image(label = 'labeled image')],
    title = 'image Classification Gradio app',
    description = 'upload an image to classify it as dog ,cat,or snake'

    )

if __name__ == "__main__":
    demo.launch()
    



