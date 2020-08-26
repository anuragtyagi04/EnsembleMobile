import argparse
import glob
import os
import numpy as np
from utils import infer, get_ground_truth, draw_heatmap


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--debug', default=False)
    args = parser.parse_args()

    directory = os.getcwd()
    model_name = args.model
    model_path = os.path.join(directory, 'model', model_name + '.tflite')
    debug = args.debug
    # Load all annotation files in .txt format
    ground_truth_files = os.path.join(directory, 'dataset') + '/ground_truths'
    ground_truths = get_ground_truth(ground_truth_files)

    images = glob.glob(os.path.join(directory,  'dataset') + '/images/*')

    total_predictions = []
    total_ground_truths = []
    ignored_predictions = []
    ignored_ground_truths = []
    count = 0

    if len(ground_truths) == len(images):
        for image, gt in zip(images, ground_truths):
            count += 1
            print(f'Image number : {count}/{len(images)}')
            debug and print("Current Image :  ", image)
            predicted_classes = []
            prediction1 = infer(model_path, image)  # inference happens here

            # Extract classes only (already sorted by x coordinate)
            for key in prediction1:
                predicted_classes.append(key[0])

            debug and print("Predicted Classes: ", predicted_classes)
            debug and print("Ground Truth:      ", gt)

            # Filter out all incomplete predictions
            if len(predicted_classes) == 9:
                total_predictions.append(predicted_classes)
                total_ground_truths.append(gt)
                debug and print("===Sample Added=== \n")
            else:
                ignored_predictions.append(predicted_classes)
                ignored_ground_truths.append(gt)
                debug and print("===Sample Ignored=== \n")

    else:
        print("Unequal number of images and their GT annotations, please verify the 100 files")

    # Flatten list of all predictions
    total_predictions = [item for sublist in total_predictions for item in sublist]
    total_ground_truths = [item for sublist in total_ground_truths for item in sublist]

    # Dummy data for testing
    # total_predictions = [0, 5, 4, 9, 0, 0, 8, 1, 8, 0, 5, 5, 5, 8, 0, 6, 1, 2]
    # total_ground_truths = [0, 5, 4, 9, 0, 3, 8, 1, 8, 0, 5, 5, 5, 8, 0, 6, 1, 3]

    draw_heatmap(total_predictions, total_ground_truths)


if __name__ == '__main__':
    main()
