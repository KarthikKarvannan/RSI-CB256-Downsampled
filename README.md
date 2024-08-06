# RSI-CB256 Downsampled - Remote Sensing Image Classification
Remote Sensing Image Classification Using Down Sampled RSI-CB256 

## Project Description
This project focuses on developing and training deep learning models to classify remote sensing images using the downsampled RSI-CB256 dataset. We aim to leverage convolutional neural networks (CNNs) and compare the performance of a custom CNN model with a pre-trained ResNet-18 model. The dataset is reduced to 15 classes and approximately 12,500 images to facilitate ease of implementation and accommodate available compute power.

## Objectives
- **Develop and Train Models**: Create deep learning models for remote sensing image classification.
- **Compare Models**: Evaluate the performance of a custom CNN against ResNet-18.
- **Dataset Utilization**: Use a downsampled version of the RSI-CB256 dataset.

## Dataset
- **Source**: RSI-CB256 dataset
- **Classes**: 15 (after downsampling)
- **Images**: 12,500 high-resolution images (256x256 pixels)

### Original Dataset Classes
The original dataset contains multiple land cover classes, including but not limited to:
- Airplane
- Airport
- Baseball field
- Basketball court
- Beach
- Bridge
- Cemetery
- Chaparral
- Christmas tree farm
- Closed road
- Coastal industrial area
- Coffee plantation
- Commercial area
- Construction site
- Crosswalk
- Dense residential area
- Desert
- Desert shrub
- Dry river
- Farmland
- Forest
- Golf course
- Greenhouse
- Ground track field
- Harbor
- Intersection
- Island
- Lake
- Meadow
- Mobile home park
- Mountain
- Mountain snow
- Oil refinery
- Orchard
- Overpass
- Parking lot
- Park
- Pond
- Port
- Quarry
- River
- River island
- Runway
- School
## Dataset Sample Images
![image](https://github.com/user-attachments/assets/4e5c952f-86c8-44d2-b312-393713fa9ebc)

## Data Preprocessing
- **Resizing**: All images were resized to 256x256 pixels.
- **Normalization**: Pixel values were normalized using mean [0.485, 0.456, 0.406] and standard deviation [0.229, 0.224, 0.225].
- **Data Augmentation**: Applied random horizontal flip, random rotation, and random resized crop to increase data diversity.
- **Splitting**: The dataset was split into training (80%), validation (10%), and test (10%) sets.

## Model Architectures
### Custom CNN
- **Convolutional Layers**: Three layers with filter sizes of 64, 128, and 256, using a 3x3 kernel, stride of 1, and padding of 1.
- **Activation**: ReLU activation function after each convolutional layer.
- **Pooling**: Max pooling with a 2x2 window and stride of 2.
- **Fully Connected Layers**: Two layers with 1024 neurons in the first layer and 15 neurons in the output layer, using dropout for regularization.

### ResNet-18
- **Architecture**: Pre-trained ResNet-18 model with the final fully connected layer modified to match the number of classes (15).

## Training Process
- **Custom CNN and ResNet-18**: Both models were trained using various hyperparameters to find the optimal configuration. Regularization techniques like dropout were used to prevent overfitting.

## Hyperparameter Tuning
- **Learning Rates**: 0.001, 0.0005
- **Batch Sizes**: 64, 128

## Evaluation
- **Metrics**: Accuracy, precision, recall, and F1-score.
- **Visualizations**: Model predictions and misclassifications were visualized to assess performance.

## Results
### Custom CNN
- **Validation Accuracy**: 96.24%
- **Test Accuracy**: 95.52%
- **Precision**: 96.23%
- **Recall**: 96.24%
- **F1-Score**: 96.21%

### ResNet-18
- **Validation Accuracy**: 99.12%
- **Test Accuracy**: 98.80%
- **Precision**: 99.13%
- **Recall**: 99.12%
- **F1-Score**: 99.12%

## Results & Evaluation Visuals
- Predicted Image Custom CNN:
![image](https://github.com/user-attachments/assets/12af7f0f-58ae-476e-9078-894b9095d54d)

- Predicted Image ResNet-18:
![image](https://github.com/user-attachments/assets/2611ca49-7d09-4466-9cab-c73936be6e7a)

- Model Train & Validation Charts
![image](https://github.com/user-attachments/assets/e1b63190-bbe1-42e0-a6cc-4017bd24c1e5)

- Heatmaps
![image](https://github.com/user-attachments/assets/df3d6c73-0f41-45be-b508-ffe36d7a1da8)
![image](https://github.com/user-attachments/assets/fc987573-1c05-4769-9ce0-c97fd9282366)

## Conclusion
This project successfully demonstrated the application of deep learning techniques in remote sensing image classification. The comparison highlighted the superior performance of the ResNet-18 model over the custom CNN. Future work includes exploring more advanced architectures, incorporating additional data augmentation techniques, and experimenting with other remote sensing datasets.

## Repository Contents
- **Dataset**: (https://figshare.com/articles/dataset/RSI-CB256/22139921?file=39360821)
- **Notebook**: PyTorch Capstone Project KarthikManivannan 1229717.ipynb
- **Project Proposal**: project_proposal.pdf
- **Project Report**: project_report.pdf
- **Project Presentation**: project_presentation.pptx

## How to Use
1. Clone the repository:
   ```bash
   git clone https://github.com/KarthikKarvannan/RSI-CB256-Downsampled.git
   ```
2. Navigate to the project directory:
   ```bash
   cd RSI-CB256-Downsampled
   ```
3. Open and run the Jupyter notebook:
   ```bash
   jupyter notebook "PyTorch Capstone Project KarthikManivannan 1229717.ipynb"
   ```
