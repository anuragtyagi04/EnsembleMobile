import argparse
import sys
import glob
import os
from utils import infer, get_ground_truth, draw_heatmap, calculate_vote
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--models', nargs='+', help='<Required> Set flag', required=True)
    parser.add_argument('--debug', default=False)

    directory = os.getcwd()
    args = parser.parse_args()
    debug = args.debug
    model_paths = []
    models = args.models
    for model in models:
        model_paths.append(os.path.join(directory, 'model', model + '.tflite'))

    # Load all annotation files in .txt format
    ground_truth_files = os.path.join(directory, 'dataset') + '/ground_truths'
    ground_truths = get_ground_truth(ground_truth_files)

    # Load all 100 test images from hold out set T
    images = glob.glob(os.path.join(directory, 'dataset') + '/images/*')

    total_predictions = []
    total_ground_truths = []
    ignored_predictions = []
    ignored_ground_truths = []
    count = 0

    if len(ground_truths) == len(images):
        for image, gt in zip(images, ground_truths):
            net_prediction = []
            count += 1
            print(f'Image number : {count}/{len(images)}')
            debug and print("Current Image :  ", image)
            for model_path in model_paths:  # Iterate over all models
                predictions = []
                predicted_classes = []

                results = infer(model_path, image)  # Infer from the current model

                # Get classes and confidences as well in case Soft Voting is required (already sorted by x coordinate)
                for key in results:
                    predictions.append([key[0], round(key[1], 3)])
                    predicted_classes.append((key[0]))

                # Filter out all incomplete predictions
                if len(predictions) == 9:
                    net_prediction.append(predictions)
                    debug and print("===Sample Added===\n")
                else:
                    debug and print("===Sample Ignored===\n")

            # Send all three model predictions for voting
            # (P01, P02,...,P09) + (P11, P12,...,P19) + (P21, P22,...,P29)
            final_vote = calculate_vote(net_prediction, debug)
            debug and print("Ground Truth:      ", gt)
            debug and print("Predicted Classes: ", final_vote)
            debug and print("\n\n\n\n")
            # Add the voting result, i.e, the final prediction for one image to the total predictions
            total_predictions.append(final_vote)
            total_ground_truths.append(gt)
    else:
        print("Unequal number of images and their GT annotations, please verify the 100 files")

    # Flatten list of all predictions
    total_predictions = [item for sublist in total_predictions for item in sublist]
    total_ground_truths = [item for sublist in total_ground_truths for item in sublist]

    draw_heatmap(total_predictions, total_ground_truths)


if __name__ == '__main__':
    main()
