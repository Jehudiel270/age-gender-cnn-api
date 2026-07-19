import tensorflow as tf

# Charge ton modèle Keras existant
model = tf.keras.models.load_model("model/AgeGenderCNN_v5.keras")

# Convertit en TFLite avec optimisation de taille (quantification)
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]

tflite_model = converter.convert()

with open("model/AgeGenderCNN_v5.tflite", "wb") as f:
    f.write(tflite_model)

print("Conversion terminée : model/AgeGenderCNN_v5.tflite")