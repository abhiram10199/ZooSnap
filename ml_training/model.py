import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from dataclasses import dataclass
from typing import Tuple, List

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
global dataset_path
dataset_path = r"./ZooAnimals"
 
@dataclass
class ModelConfig:
    image_size: Tuple[int, int, int] = (224, 224, 3)
    batch_size: int = 32
    epochs: int = 15
    learning_rate: float = 1e-3
    num_classes: int = 19
    early_stopping_patience: int = 5
    reduce_lr_patience: int = 3
    model_checkpoint_path: str = "./best_model.keras"


class DataAugmentation:
    """Handles data augmentation layers."""
    @staticmethod
    def create_augmentation_layers() -> keras.Sequential:
        return keras.Sequential([
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.2),
            layers.RandomZoom(0.2),
            layers.RandomContrast(0.2),
        ])


class AnimalClassificationModel:
    """Builds and manages the animal classification CNN model."""
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self.model = None
        self.class_names = None
    
    def build_model(self) -> keras.Model:
        data_augmentation = DataAugmentation.create_augmentation_layers()
    
        base_model = keras.applications.EfficientNetB0(
            include_top=False,
            input_shape=self.config.image_size,
            weights="imagenet"
        )
        base_model.trainable = False
    
        inputs = keras.Input(shape=self.config.image_size)
        x = data_augmentation(inputs)
        x = keras.applications.efficientnet.preprocess_input(x)
        x = base_model(x, training=False)
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dense(256, activation="relu")(x)
        x = layers.Dropout(0.4)(x)
    
        outputs = layers.Dense(
            self.config.num_classes,
            activation="softmax"
        )(x)
    
        self.model = keras.Model(inputs, outputs)
        return self.model

    @staticmethod
    def _conv_block(filters: int, kernel_size: int) -> List[layers.Layer]:
        """Creates a convolutional block with pooling and batch normalisation."""
        return [
            layers.Conv2D(filters, kernel_size, activation="relu"),
            layers.MaxPooling2D(),
            layers.BatchNormalization(),
        ]
    
    @staticmethod
    def _dense_block(units: int, dropout: float = 0.0) -> List[layers.Layer]:
        """Creates a dense block with batch normalisation and optional dropout."""
        block = [
            layers.Dense(units, activation="relu"),
            layers.BatchNormalization(),
        ]
        if dropout > 0:
            block.append(layers.Dropout(dropout))
        return block
    
    def compile_model(self):
        self.model.compile(
            optimizer=keras.optimizers.Adam(self.config.learning_rate),
            loss=keras.losses.CategoricalCrossentropy(),
            metrics=["accuracy"]
        )
    
    def get_callbacks(self) -> List:
        """Returns training callbacks."""
        return [
            EarlyStopping(
                patience=self.config.early_stopping_patience,
                monitor='val_loss',
                restore_best_weights=True
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                min_lr=1e-7,
                patience=self.config.reduce_lr_patience,
                mode='min',
                verbose=1,
                factor=0.5
            ),
            ModelCheckpoint(
                monitor='val_loss',
                filepath=self.config.model_checkpoint_path,
                save_best_only=True
            )
        ]
    
    def train(self, train_dataset, validation_dataset):
        """Trains the model."""
        if self.model is None:
            raise ValueError("Model must be built and compiled before training")
        
        history = self.model.fit(
            train_dataset,
            epochs=self.config.epochs,
            callbacks=self.get_callbacks(),
            validation_data=validation_dataset
        )
        return history
    
    def predict_image(self, image_path: str) -> List[float]:
        """Predicts the class probabilities for a single image."""
        if self.model is None:
            raise ValueError("Model must be built before prediction")
        
        img = tf.keras.utils.load_img(
            image_path,
            target_size=self.config.image_size[:2]
        )
        img_array = tf.keras.utils.img_to_array(img)
        img_batch = tf.expand_dims(img_array, 0)
        
        predictions = self.model.predict(img_batch)
        return predictions[0]


class DatasetLoader:
    """Handles dataset loading and preprocessing."""

    @staticmethod
    def load_datasets(config: ModelConfig) -> Tuple:
        train_ds = tf.keras.utils.image_dataset_from_directory(
            "{dataset_path}",
            validation_split=0.2,
            subset="training",
            seed=123,
            image_size=config.image_size[:2],
            batch_size=config.batch_size,
            label_mode="categorical"
        )

        val_ds = tf.keras.utils.image_dataset_from_directory(
            "/kaggle/input/zoo-animals/",
            validation_split=0.2,
            subset="validation",
            seed=123,
            image_size=config.image_size[:2],
            batch_size=config.batch_size,
            label_mode="categorical"
        )

        class_names = train_ds.class_names

        train_ds = train_ds.cache().prefetch(tf.data.AUTOTUNE)
        val_ds = val_ds.cache().prefetch(tf.data.AUTOTUNE)

        return train_ds, val_ds, class_names
    
    @staticmethod
    def _optimise_dataset(dataset):
        """Applies caching and prefetching optimisations."""
        return dataset.cache().prefetch(tf.data.AUTOTUNE)


def main():
    print('---------------------------------------------')
    
    config = ModelConfig()
    
    train_dataset, validation_dataset, class_names = DatasetLoader.load_datasets(config)
    
    # Build and train model
    classifier = AnimalClassificationModel(config)
    classifier.class_names = class_names
    classifier.build_model()
    classifier.compile_model()
    
    print(f"Training model with {len(class_names)} classes: {class_names}")
    classifier.train(train_dataset, validation_dataset)
    
    # Test prediction - use a sample image from one of the class folders
    test_image_path = r".\ZooAnimals\tiger"
    if os.path.exists(test_image_path):
        # Get first image from tiger folder
        images = [f for f in os.listdir(test_image_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        if images:
            test_img = os.path.join(test_image_path, images[0])
            predictions = classifier.predict_image(test_img)
            print("\nPrediction results:")
            for i, class_name in enumerate(class_names):
                print(f"{class_name}: {100 * predictions[i]:.2f}%")


if __name__ == "__main__":
    main()
