import cv2
import numpy as np
from tensorflow.lite.python.interpreter import Interpreter
import os
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
from pandas_ml import ConfusionMatrix


def infer(model_path, image):
    # Load the TensorFlow Lite model.
    interpreter = Interpreter(model_path=model_path)

    interpreter.allocate_tensors()

    # Get model details
    input_details = interpreter.get_input_details()  # Dimensions of image the model expects
    output_details = interpreter.get_output_details()  # Output Tensors
    input_height = input_details[0]['shape'][1]
    input_width = input_details[0]['shape'][2]
    floating_model = (input_details[0]['dtype'] == np.float32)  # Checks if model is quantised

    # Load image and resize to expected shape [1xHxWx3]
    image = cv2.imread(image)
    image_height, image_width, _ = image.shape
    new_image = cv2.resize(image, (input_width, input_height))
    input_data = np.expand_dims(new_image, axis=0)

    # Normalize pixel values if using a floating model (i.e. if model is non-quantized)
    input_mean = 127.5
    input_std = 127.5
    if floating_model:
        input_data = (np.float32(input_data) - input_mean) / input_std

    # Perform the actual detection by running the model with the image as input
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()

    # Retrieve detection results
    bboxes = (interpreter.get_tensor(output_details[0]['index'])[0]).tolist()  # Get all 4 bbox coordinates
    x_bboxes = [round(x[1], 2) for x in bboxes]  # Extract x-coordinates

    classes = (interpreter.get_tensor(output_details[1]['index'])[0]).tolist()
    classes = [int(x) for x in classes]
    confidences = (interpreter.get_tensor(output_details[2]['index'])[0]).tolist()
    confidences = [round(x, 2) for x in confidences]

    prediction = sorted(zip(classes, confidences, x_bboxes), key=lambda t: t[1], reverse=True)

    # check for more than 9 detections
    if len(prediction) != 9:
        prediction = prediction[:-1]

    prediction = sorted(prediction, key=lambda t: t[2])  # sort by x-coordinate

    return prediction


def get_ground_truth(ground_truth_files):
    all_files = os.listdir(ground_truth_files)
    true1 = []
    temp1 = []
    for file in all_files:
        file1 = open(ground_truth_files + '/' + file, 'r', )
        lines = file1.readlines()
        for line in lines:
            line = line.split()
            temp1.append(int(line[0]))
        true1.append(temp1)
        temp1 = []

    return true1


def calculate_vote(net_prediction, debug):
    # net_prediction = [
    #     [[0, 0.945], [5, 0.463], [5, 0.93], [3, 0.596], [3, 0.955], [7, 0.892], [6, 0.444], [9, 0.736], [6, 0.63]],
    #     [[5, 0.287], [2, 0.217], [5, 0.35], [7, 0.783], [6, 0.479], [9, 0.461], [6, 0.763], [9, 0.143], [9, 0.116]],
    #     [[5, 0.229], [5, 0.512], [5, 0.555], [4, 0.416], [6, 0.784], [9, 0.495], [0, 0.23], [9, 0.719], [6, 0.678]]
    # ]

    element1 = net_prediction[0]
    debug and print("Model 1 prediction :", element1)
    element2 = net_prediction[1]
    debug and print("Model 2 prediction :", element2)
    element3 = net_prediction[2]
    debug and print("Model 3 prediction :", element3)

    final = []
    if 1==1:
        for i in range(9):
            debug and print("\n\n")
            debug and print(element1[i][0])
            debug and print(element2[i][0])
            debug and print(element3[i][0])
            # Check if all 3 models predict the same class for a digit
            if element1[i][0] == element2[i][0] == element3[i][0]:
                final.append(element1[i][0])
                debug and print("Case 1: chose first element because all three matched")

            # Check if any two of the predictions match : Majority Voting

            elif element1[i][0] == element2[i][0] != element3[i][0]:
                final.append(element1[i][0])
                debug and print("Case 2.1: chose first element because 1 and 2 matched")

            elif element2[i][0] == element3[i][0] != element1[i][0]:
                final.append(element2[i][0])
                debug and print("Case 2.2: chose second element because 2 and 3 matched")

            elif element1[i][0] == element3[i][0] != element2[i][0]:
                final.append(element1[i][0])
                debug and print("Case 2.3: chose first element because 1 and 3 matched")

            # If none of the predicted classes for a digit match, use highest confidence : Soft Voting
            elif element1[i][0] != element2[i][0] != element3[i][0]:
                debug and print("Case 3: All three different")

                conf_array = [element1[i][1], element2[i][1], element3[i][1]]
                index = conf_array.index(max(conf_array))
                if index == 0:
                    final.append(element1[i][0])
                    debug and print("Case 3.1: added first with highest confidence")
                elif index == 1:
                    final.append(element2[i][0])
                    debug and print("Case 3.2: added second with highest confidence")

                elif index == 2:
                    final.append(element3[i][0])
                    debug and print("Case 3.3: Added third with highest confidence")
                else:
                    print("Unexpected and unhandled case, this should not have been printed")

            else:
                print("Unexpected and unhandled case, this should not have been printed")
    return final


def draw_heatmap(total_predictions, total_ground_truths):

    if len(total_predictions) == len(total_ground_truths):
        data = {'y_Actual': total_ground_truths,
                'y_Predicted': total_predictions
                }

        df = pd.DataFrame(data, columns=['y_Actual', 'y_Predicted'])

        confusion_matrix = pd.crosstab(df['y_Actual'], df['y_Predicted'], rownames=['Actual'], colnames=['Predicted'])
        sn.heatmap(confusion_matrix, annot=True, fmt='g', cmap='YlGnBu')
        Confusion_Matrix = ConfusionMatrix(df['y_Actual'], df['y_Predicted'])
        Confusion_Matrix.print_stats()
        plt.show()
