
# Computer Vision Models for Object Detection and Segmentation

This repository contains implementations of various object detection and segmentation models using PyTorch, including FCNN (Fully Convolutional Neural Networks), SSD (Single Shot Multibox Detector), and YOLOv5 (You Only Look Once). These models have been trained and evaluated on popular datasets to explore their performance on image and video data.

## Project Highlights

- **FCNN for Image Segmentation**: 
  - Implements the FCN-ResNet50 architecture for semantic segmentation of images.
  - Provides a complete pipeline from loading an image, preprocessing, running inference, and visualizing segmented outputs.
  - Performance-optimized for GPU usage, with support for fast processing on CUDA-enabled devices.
  - Demonstrates pixel-wise classification by outputting colored masks over input images, with a detailed focus on performance and class palette mapping.
  
- **SSD for Object Detection**: 
  - Utilizes NVIDIA's pre-trained SSD model for object detection.
  - Efficiently processes multiple images in a batch using SSD, generating bounding boxes and classifying objects in real-time.
  - Implements post-processing utilities to decode model outputs and visualize results, highlighting detected objects with bounding boxes and class labels.
  - The project also benchmarks performance using confidence scores and frame processing times.

- **YOLOv5 for Real-Time Object Detection in Video**: 
  - Integrates the lightweight YOLOv5 architecture to detect objects in real-time from a video stream.
  - Tracks frame-by-frame detection results and visualizes bounding boxes, object names, and confidence levels on the output video.
  - Captures performance metrics like frames per second (FPS) and detection confidence, exporting results to both graphical visualizations and CSV for further analysis.
  - Provides outputs of video detection with overlays and also exports performance graphs (confidence per frame, FPS over time) to analyze model behavior.
  
## Project Structure

- `pytorchFCNN.py`: Implements FCN-ResNet50 and FCN-ResNet101 for image segmentation, including preprocessing, inference, and visualization.
- `pytorchSSD.py`: Uses NVIDIA's SSD model for object detection in images, showcasing bounding box predictions and class labels.
- `pytorchYOLOV5.py`: Leverages YOLOv5 for object detection in video streams with real-time performance tracking, including frame processing times and detection confidence levels.
- `requirements.txt`: Lists all the dependencies required to run the models, including PyTorch, OpenCV, and additional tools for visualization and performance tracking.

## Performance Metrics

- **FCNN (Segmentation)**: Processes images with pixel-level classification in ~X seconds per image on CUDA, demonstrating real-time viability with color-coded output visualizations.
- **SSD (Object Detection)**: Handles object detection across multiple image inputs, achieving bounding box predictions with a threshold confidence of 40%.
- **YOLOv5 (Video Detection)**: Capable of real-time object detection on video streams, processing at ~X FPS, and tracking performance over multiple frames.

## Technologies Used

- **PyTorch**: Core deep learning library used for loading pre-trained models, handling inference, and leveraging GPU acceleration.
- **OpenCV**: For image and video processing, including reading from video streams and overlaying detection results onto video frames.
- **Matplotlib**: For visualizing results, including segmentation maps, bounding box predictions, and performance graphs.
- **CSV**: For exporting performance data (FPS, confidence scores) in a structured format for further analysis.

## Usage

1. Install the required dependencies using the `requirements.txt`:
   ```bash
   pip install -r requirements.txt
   ```

2. Run the FCNN for image segmentation:
   ```bash
   python pytorchFCNN.py
   ```

3. Perform object detection using SSD:
   ```bash
   python pytorchSSD.py
   ```

4. Run YOLOv5 for real-time object detection on a video stream:
   ```bash
   python pytorchYOLOV5.py
   ```

## Future Improvements

- Exploring model optimization techniques like quantization for faster inference times.
- Extending support to other models such as Faster R-CNN or Mask R-CNN for instance segmentation tasks.
- Implementing additional video analysis metrics for YOLOv5 such as object tracking across frames.
