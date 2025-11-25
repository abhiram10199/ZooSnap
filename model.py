import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from dataclasses import dataclass
from typing import Tuple, List
from dataset_loader import load_coco_dataset

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


@dataclass
class ModelConfig:
    """Configuration for model training."""
    image_size: Tuple[int, int, int] = (512, 512, 3)
    batch_size: int = 32
    epochs: int = 15
    learning_rate: float = 0.01
    num_classes: int = 15
    early_stopping_patience: int = 15
    reduce_lr_patience: int = 5
    model_checkpoint_path: str = './best_model.h5'


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
        
        model = keras.Sequential([
            layers.Input(self.config.image_size),
            data_augmentation,
            layers.Rescaling(scale=1./255),
            
            # Convolutional blocks
            *self._conv_block(32, 5),
            *self._conv_block(64, 3),
            *self._conv_block(128, 3),
            *self._conv_block(256, 3),
            
            # Dense layers
            layers.Flatten(),
            *self._dense_block(256, dropout=0.4),
            *self._dense_block(128, dropout=0.3),
            *self._dense_block(64, dropout=0.0),
            
            layers.Dense(self.config.num_classes, activation="softmax")
        ])
        
        self.model = model
        return model
    
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
        """Compiles the model with optimiser, loss, and metrics."""
        if self.model is None:
            raise ValueError("Model must be built before compiling")
        
        self.model.compile(
            optimizer = keras.optimizers.Adam(learning_rate=self.config.learning_rate),
            loss = keras.losses.CategoricalCrossentropy(),
            metrics = ["accuracy"]
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
        train_dataset, class_names = load_coco_dataset("ZooAnimals/train")
        validation_dataset, _ = load_coco_dataset("ZooAnimals/valid")
        testing_dataset, _ = load_coco_dataset("ZooAnimals/test")
        
        # Optimise dataset performance
        train_dataset = DatasetLoader._optimize_dataset(train_dataset)
        validation_dataset = DatasetLoader._optimize_dataset(validation_dataset)
        
        return train_dataset, validation_dataset, testing_dataset, class_names
    
    @staticmethod
    def _optimise_dataset(dataset):
        """Applies caching and prefetching optimisations."""
        return dataset.cache().prefetch(tf.data.AUTOTUNE)


def main():
    print('---------------------------------------------')
    
    config = ModelConfig()
    
    train_dataset, validation_dataset, testing_dataset, class_names = DatasetLoader.load_datasets(config)
    
    # Build and train model
    classifier = AnimalClassificationModel(config)
    classifier.class_names = class_names
    classifier.build_model()
    classifier.compile_model()
    
    print(f"Training model with {len(class_names)} classes: {class_names}")
    classifier.train(train_dataset, validation_dataset)
    
    # Test prediction
    test_image_path = r".\ZooAnimals\train\0a052663-dd0a-4faf-81d3-fb7dc9ae975e_jpg.rf.6797f74037925743dea2924ac8dd81b8.jpg"
    if os.path.exists(test_image_path):
        predictions = classifier.predict_image(test_image_path)
        print("\nPrediction results:")
        for i, class_name in enumerate(class_names):
            print(f"{class_name}: {100 * predictions[i]:.2f}%")


if __name__ == "__main__":
    main()
