import gradio as gr
import tensorflow as tf
from PIL import Image
import numpy as np

# Laden des Modells
model_path = "brain_classification.keras"
model = tf.keras.models.load_model(model_path)

# Klassenlabels
labels = ['glioma_tumor', 'meningioma_tumor', 'no_tumor', 'pituitary_tumor']

def predict_image(image):
    image = Image.fromarray(image.astype('uint8'), 'RGB')
    image = image.resize((224, 224)) 
    image = np.array(image)

    prediction = model.predict(np.expand_dims(image, axis=0))
    confidences = {labels[i]: float(prediction[0][i]) for i in range(len(labels))}
    return confidences

# Gradio interface
iface = gr.Interface(
    fn=predict_image,
    inputs=gr.Image(),
    outputs=gr.Label(num_top_classes=4),
    title="MRI Tumor Classifier",
    description="Upload your MRI image and the model will predict, if there's a tumor present"

)

iface.launch(share=True)
