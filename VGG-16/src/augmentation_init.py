from augmentation import augment

augment(n_images_per_class=1500,
        resolution=64,
        seed=1,
        output_path="augmented-db")
