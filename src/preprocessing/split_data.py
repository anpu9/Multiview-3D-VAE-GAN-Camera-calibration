import os
import random
import shutil

def organize_datasets(directory_path):
    """
    Organizes datasets in the given directory into train, test, and validation sets.

    Args:
        directory_path (str): The path to the directory containing the dataset folders.
    """
    if not os.path.isdir(directory_path):
        print(f"Error: Directory not found at {directory_path}")
        return

    all_items = os.listdir(directory_path)
    instance_folders = sorted([item for item in all_items if os.path.isdir(os.path.join(directory_path, item)) and item.endswith("_dataset_instance")])
    processed_folders = sorted([item for item in all_items if os.path.isdir(os.path.join(directory_path, item)) and item.endswith("_processed")])

    num_instances = len(instance_folders)
    num_processed = len(processed_folders)

    if num_instances != num_processed:
        print("Error: The number of *_dataset_instance and *_processed folders is not equal.")
        return

    if num_instances == 0:
        print("No dataset folders found.")
        return

    total_datasets = num_instances
    print(f"Total number of datasets found: {total_datasets}")

    num_train = int(total_datasets * 0.7)
    num_test = int(total_datasets * 0.2)
    num_val = total_datasets - num_train - num_test

    print(f"Number of datasets for training: {num_train}")
    print(f"Number of datasets for testing: {num_test}")
    print(f"Number of datasets for validation: {num_val}")

    datasets = []
    for instance in instance_folders:
        prefix = instance.replace("_dataset_instance", "")
        processed_match = f"{prefix}_processed"
        if processed_match in processed_folders:
            datasets.append(prefix)
        else:
            print(f"Warning: Could not find a matching *_processed folder for {instance}")
            return

    random.shuffle(datasets)

    train_datasets = datasets[:num_train]
    test_datasets = datasets[num_train:num_train + num_test]
    val_datasets = datasets[num_train + num_test:]

    # Create train, test, and validation folders
    train_dir = os.path.join(directory_path, "train")
    test_dir = os.path.join(directory_path, "test")
    val_dir = os.path.join(directory_path, "validation")

    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)

    # Move datasets to respective folders
    def move_dataset(dataset_prefix, source_dir, target_dir):
        instance_name = f"{dataset_prefix}_dataset_instance"
        processed_name = f"{dataset_prefix}_processed"
        source_instance_path = os.path.join(source_dir, instance_name)
        source_processed_path = os.path.join(source_dir, processed_name)
        target_instance_path = os.path.join(target_dir, instance_name)
        target_processed_path = os.path.join(target_dir, processed_name)

        try:
            shutil.move(source_instance_path, target_instance_path)
            shutil.move(source_processed_path, target_processed_path)
            print(f"Moved dataset '{dataset_prefix}' to '{os.path.basename(target_dir)}'")
        except FileNotFoundError:
            print(f"Error: Could not find instance or processed folder for '{dataset_prefix}' in '{source_dir}'")

    print("\nMoving datasets:")
    for dataset in train_datasets:
        move_dataset(dataset, directory_path, train_dir)

    for dataset in test_datasets:
        move_dataset(dataset, directory_path, test_dir)

    for dataset in val_datasets:
        move_dataset(dataset, directory_path, val_dir)

    print("\nDataset organization complete.")

if __name__ == "__main__":
    target_directory = input("Enter the path to the directory containing the dataset folders: ")
    organize_datasets(target_directory)