import json
import os
import tensorflow as tf

def load_coco_dataset(data_dir, image_size=(512, 512), batch_size=32):
    with open(os.path.join(data_dir, '_annotations.coco.json'), 'r') as f:
        coco_data = json.load(f)
    
    # Extract category mapping
    categories = {cat['id']: cat['name'] for cat in coco_data['categories']}
    
    # Create image_id to category mapping
    image_to_category = {}
    for ann in coco_data['annotations']:
        image_id = ann['image_id']
        category_id = ann['category_id']
        image_to_category[image_id] = category_id
    
    # Prepare file paths and labels
    image_paths = []
    labels = []
    for img in coco_data['images']:
        img_path = os.path.join(data_dir, img['file_name'])
        if os.path.exists(img_path) and img['id'] in image_to_category:
            image_paths.append(img_path)
            labels.append(image_to_category[img['id']])
    
    # Convert to dataset
    num_classes = len(categories)
    labels = tf.keras.utils.to_categorical([l for l in labels], num_classes=num_classes)
    
    def load_and_preprocess(path, label):
        img = tf.io.read_file(path)
        img = tf.image.decode_image(img, channels=3, expand_animations=False)
        img = tf.image.resize(img, image_size)
        return img, label
    
    dataset = tf.data.Dataset.from_tensor_slices((image_paths, labels))
    dataset = dataset.map(load_and_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    
    return dataset, list(categories.values())


