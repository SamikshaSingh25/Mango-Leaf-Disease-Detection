# Mango-Leaf-Disease-Detection-Research
This repository contains two Jupyter notebooks that train image classification models to identify leaf diseases from a local dataset.

## Team
Samiksha Singh - 2210992238
Kavya Arora - 2210991776
Pranya Khattar - 2210990670
Ankita - 2210991280

## Project Structure

- `CustomCNN.ipynb` - A custom convolutional neural network built with TensorFlow/Keras.
- `MobileNetV2.ipynb` - A transfer learning model using MobileNetV2 with data augmentation.
- `1/` - Dataset folder containing subfolders for each class:
  - `Anthracnose`
  - `Bacterial Canker`
  - `Cutting Weevil`
  - `Die Back`
  - `Gall Midge`
  - `Healthy`
  - `Powdery Mildew`
  - `Sooty Mould`


## Requirements

Recommended Python packages:

- tensorflow
- numpy
- matplotlib
- scikit-learn
- seaborn
- pillow

If you want to install them manually, run:

```bash
pip install tensorflow numpy matplotlib scikit-learn seaborn pillow
```

## Running the Notebook

1. Open the workspace in VS Code or Jupyter Notebook.
2. Open either `CustomCNN.ipynb` or `MobileNetV2.ipynb`.
3. Make sure the dataset path is correct in the notebook.

### Dataset path settings

In both notebooks, update the dataset path to the location of the `1/` folder. For example:

```python
dataset_path = r"C:\Users\singh\Desktop\8th_project\1"
```

If you move the dataset, update that path before running the notebook.

## Notebook Details

### `CustomCNN.ipynb`

- Loads the dataset using `tf.keras.utils.image_dataset_from_directory`.
- Applies normalization and dataset performance optimizations.
- Builds a custom CNN with several convolutional and pooling layers.
- Trains for 30 epochs.
- Computes and displays:
  - classification report
  - F1 score chart
  - confusion matrix
  - accuracy and loss curves
  - sample predictions with true vs predicted labels

### `MobileNetV2.ipynb`

- Loads the dataset with an 80/10/10 train/validation/test split.
- Uses MobileNetV2 pre-trained on ImageNet as a base model.
- Applies data augmentation and feature extraction.
- Trains with early stopping.
- Evaluates test accuracy and displays:
  - classification report
  - confusion matrix
  - sample predictions
  - training and validation curves

## Notes

- Both notebooks assume the dataset folder is organized by class subfolders.
- If you run the notebooks on a machine without a GPU, training may take longer.
- Use the `validation_split` and `seed` settings to keep results reproducible.
- If you want to save trained models, add `model.save('model_name.h5')` after training.

## Tips

- Use `Runtime > Restart & Run All` in Jupyter after installing packages.
- If you see GPU-related warnings, confirm TensorFlow is installed correctly for your environment.
- You can change `IMG_HEIGHT`, `IMG_WIDTH`, and `BATCH_SIZE` in the notebooks if needed.


