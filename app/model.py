from tensorflow.keras.applications import DenseNet121
from tensorflow.keras import layers, models

class TransferLearningModel:
    def __init__(self, n_classes=1, input_shape=(640, 640, 3)):
        self.n_classes = n_classes
        self.input_shape = input_shape

    def build_model(self):
        # Load pre-trained DenseNet121 model with weights from ImageNet
        base_model = DenseNet121(weights='imagenet', include_top=False, input_shape=self.input_shape)

        # Freeze the layers of the base model
        for layer in base_model.layers:
            layer.trainable = False

        # Build the model
        model = models.Sequential()
        model.add(base_model)
        model.add(layers.GlobalAveragePooling2D())
        model.add(layers.Dense(512, activation='relu'))
        model.add(layers.BatchNormalization())
        model.add(layers.Dropout(0.2))
        model.add(layers.Dense(256, activation='relu'))
        model.add(layers.BatchNormalization())
        model.add(layers.Dropout(0.2))
        model.add(layers.Dense(self.n_classes, activation='sigmoid'))

        # Compile the model
        model.compile(optimizer='adam',
                      loss='binary_crossentropy',
                      metrics=['accuracy'])

        return model

