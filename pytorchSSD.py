import torch

import cv2 as cv
import numpy as np
import time
import matplotlib.pyplot as plt
import csv



# Model
ssd_model = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_ssd')
utils = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_ssd_processing_utils')

ssd_model.to('cuda')
ssd_model.eval()

elapsedTime = 0


uris = [
    'http://images.cocodataset.org/val2017/000000397133.jpg',
]

inputs = [utils.prepare_input(uri) for uri in uris]
tensor = utils.prepare_tensor(inputs)

t0 = time.time()
with torch.no_grad():
    detections_batch = ssd_model(tensor)
t = time.time() - t0

results_per_input = utils.decode_results(detections_batch)
best_results_per_input = [utils.pick_best(results, 0.40) for results in results_per_input]

classes_to_labels = utils.get_coco_object_dictionary()

from matplotlib import pyplot as plt
import matplotlib.patches as patches

for image_idx in range(len(best_results_per_input)):
    fig, ax = plt.subplots(1)
    # Show original, denormalized image...
    image = inputs[image_idx] / 2 + 0.5
    ax.imshow(image)
    # ...with detections
    bboxes, classes, confidences = best_results_per_input[image_idx]
    for idx in range(len(bboxes)):
        left, bot, right, top = bboxes[idx]
        x, y, w, h = [val * 300 for val in [left, bot, right - left, top - bot]]
        rect = patches.Rectangle((x, y), w, h, linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
        ax.text(x, y, "{} {:.0f}%".format(classes_to_labels[classes[idx] - 1], confidences[idx]*100), bbox=dict(facecolor='white', alpha=0.5))
plt.show()


# plt.plot(range(0,len(confList)), confList)
# plt.xlabel('Frame')
# plt.ylabel('Confidence')
# plt.title('Frame to Confidence - YOLOV5')
# plt.savefig('YOLOV5/FCGraph.png')
# plt.clf()
# plt.plot(range(0,len(fpsList)), fpsList)
# plt.xlabel('Frame')
# plt.ylabel('FPS')
# plt.title('Frame to FPS - YOLOV5 - Total Time: '+ str(elapsedTime) + 'ms')
# plt.savefig('YOLOV5/FFGraph.png')

# with open('YOLOV5/output.csv', 'w', newline='') as f:
#     writer = csv.writer(f)
#     for i in range(0,len(confList)):
#         writer.writerow([i, confList[i], fpsList[i]])
