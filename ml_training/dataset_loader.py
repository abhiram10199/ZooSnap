import os
import tensorflow as tf
from sklearn.model_selection import train_test_split

def load_image_dataset(data_dir, image_size=(512, 512), batch_size=32, validation_split=0.2):
    """
    Load dataset from directory structure where each subdirectory is a class.
    
    Args:
        data_dir: Root directory containing subdirectories for each animal class
        image_size: Target size for images (height, width)
        batch_size: Batch size for training
        validation_split: Fraction of data to use for validation
    
    Returns:
        train_dataset, validation_dataset, class_names
    """
    # Get all subdirectories (class names)
    class_names = sorted([d for d in os.listdir(data_dir) 
                         if os.path.isdir(os.path.join(data_dir, d))])
    
    if not class_names:
        raise ValueError(f"No subdirectories found in {data_dir}")
    
    # Collect all image paths and labels
    image_paths = []
    labels = []
    
    for class_idx, class_name in enumerate(class_names):
        class_dir = os.path.join(data_dir, class_name)
        for img_file in os.listdir(class_dir):
            if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                image_paths.append(os.path.join(class_dir, img_file))
                labels.append(class_idx)
    
    # Split into train and validation
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        image_paths, labels, test_size=validation_split, random_state=42, stratify=labels
    )
    
    num_classes = len(class_names)
    
    # Convert labels to categorical
    train_labels_cat = tf.keras.utils.to_categorical(train_labels, num_classes=num_classes)
    val_labels_cat = tf.keras.utils.to_categorical(val_labels, num_classes=num_classes)
    
    def load_and_preprocess(path, label):
        img = tf.io.read_file(path)
        img = tf.image.decode_image(img, channels=3, expand_animations=False)
        img = tf.image.resize(img, image_size)
        img = img / 255.0  # Normalize to [0, 1]
        return img, label
    
    # Create training dataset
    train_dataset = tf.data.Dataset.from_tensor_slices((train_paths, train_labels_cat))
    train_dataset = train_dataset.shuffle(buffer_size=len(train_paths))
    train_dataset = train_dataset.map(load_and_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    train_dataset = train_dataset.batch(batch_size)
    train_dataset = train_dataset.prefetch(tf.data.AUTOTUNE)
    
    # Create validation dataset
    val_dataset = tf.data.Dataset.from_tensor_slices((val_paths, val_labels_cat))
    val_dataset = val_dataset.map(load_and_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    val_dataset = val_dataset.batch(batch_size)
    val_dataset = val_dataset.prefetch(tf.data.AUTOTUNE)
    
    return train_dataset, val_dataset, class_names


