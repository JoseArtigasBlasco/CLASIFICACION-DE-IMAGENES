import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
import numpy as np




# Cargar y preprocesar el conjunto de datos CIFAR-10
(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()


# Normalizar los valores de píxeles entre 0 y 1
train_images, test_images = train_images / 255.0, test_images / 255.0

# Define la arquitectura de la red convolucional
model = models.Sequential([
    # Primera capa convolucional
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    layers.MaxPooling2D((2, 2)),


    # Segunda capa convolucional
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.25),

    # Tercera capa convolucional
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.Dropout(0.25),

    # Aplanar las salidas y agregar una capa densa
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.25),
    layers.Dense(10)  # 10 salidas para las 10 clases
])

# Muestra la arquitectura del modelo
model.summary()

# Compilar el modelo
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# Entrenar el modelo
history = model.fit(train_images, train_labels, epochs=10,
                    validation_data=(test_images, test_labels))

# Evaluar el modelo en el conjunto de prueba
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
print(f'\nPrecisión en el conjunto de prueba: {test_acc}')

# Graficar la precisión y la pérdida a lo largo del tiempo
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Precisión en entrenamiento')
plt.plot(history.history['val_accuracy'], label='Precisión en validación')
plt.xlabel('Época')
plt.ylabel('Precisión')
plt.legend(loc='lower right')
plt.title('Precisión en entrenamiento y validación')

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Pérdida en entrenamiento')
plt.plot(history.history['val_loss'], label='Pérdida en validación')
plt.xlabel('Época')
plt.ylabel('Pérdida')
plt.legend(loc='upper right')
plt.title('Pérdida en entrenamiento y validación')

plt.show()

predictions = model.predict(test_images)

# Obtener la clase predicha para la primera imagen en el conjunto de prueba
predicted_label = np.argmax(predictions[0])

# Obtener la etiqueta verdadera para la primera imagen en el conjunto de prueba
true_label = test_labels[0][0]

print(f'Predicción: {predicted_label}, Etiqueta verdadera: {true_label}')

plt.figure(figsize=(10, 10))

for i in range(25):
    plt.subplot(5, 5, i + 1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)

    # Mostrar la imagen
    plt.imshow(test_images[i])

    # Mostrar la predicción y la etiqueta verdadera
    predicted_label = np.argmax(predictions[i])
    true_label = test_labels[i][0]
    color = 'blue' if predicted_label == true_label else 'red'

    plt.xlabel(f'Predicción: {predicted_label}\nEtiqueta: {true_label}', color=color) # 0.7210000157356262

plt.show()


#------





