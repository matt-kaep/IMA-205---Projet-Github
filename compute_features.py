import numpy as np
import cv2
from skimage.color import rgb2gray
from skimage import feature, img_as_ubyte
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score, precision_score, \
                            recall_score, accuracy_score, classification_report
import cv2 as cv
import os
import DarkArtefactRemoval as dca
import dullrazor as dr
import segmentation_and_preprocessing as sp
from tqdm import tqdm
from skimage.feature import graycomatrix, graycoprops



def calculate_features(images, masks, lesions, plot_limit=200, affichage=False):
    feature_list = []

    for idx, (image, mask, lesion) in enumerate(zip(images, masks, lesions)):
        # Load the image and mask
        image = cv2.imread(image)
        mask = cv2.imread(mask, cv2.IMREAD_GRAYSCALE)
        lesion = cv2.imread(lesion)

        # Calculate the total area of the lesion
        area_total = np.sum(mask)

        # Calculate the perimeter of the lesion
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        index = np.argmax([len(c) for c in contours])
        lesion_region = contours[index]
        perimeter = cv2.arcLength(lesion_region, True)
        # Calculate the compact index
        compact_index = (perimeter ** 2) / (4 * np.pi * area_total)
        if affichage:
            if idx < plot_limit:
                print('\n-- BORDER IRREGULARITY --')
                print('Compact Index: {:.3f}'.format(compact_index))

        # Calculate color variegation
        lesion_r = lesion[:, :, 0]
        lesion_g = lesion[:, :, 1]
        lesion_b = lesion[:, :, 2]

        C_r = np.std(lesion_r) / np.max(lesion_r)
        C_g = np.std(lesion_g) / np.max(lesion_g)
        C_b = np.std(lesion_b) / np.max(lesion_b)

        if affichage:
            print('\n-- COLOR VARIEGATION --')
            print('Red Std Deviation: {:.3f}'.format(C_r))
            print('Green Std Deviation: {:.3f}'.format(C_g))
            print('Blue Std Deviation: {:.3f}'.format(C_b))

        # Calculate the diameter of the lesion
        x, y, w, h = cv2.boundingRect(lesion_region)
        diameter = max(w, h)

        if affichage:
            print('\n-- DIAMETER --')
            print('Diameter: {:.3f}'.format(diameter))

        # Convert the lesion region to grayscale
        gray_lesion = rgb2gray(lesion)

        # Compute the grey-level co-occurrence matrix
        glcm = feature.graycomatrix(image=img_as_ubyte(gray_lesion), distances=[1],
                                    angles=[0, np.pi/4, np.pi/2, np.pi * 3/2],
                                    symmetric=True, normed=True)

        # Compute texture features
        correlation = np.mean(feature.graycoprops(glcm, prop='correlation'))
        homogeneity = np.mean(feature.graycoprops(glcm, prop='homogeneity'))
        energy = np.mean(feature.graycoprops(glcm, prop='energy'))
        contrast = np.mean(feature.graycoprops(glcm, prop='contrast'))

        if affichage:
            if idx < plot_limit:
                print('\n-- TEXTURE --')
                print('Correlation: {:.3f}'.format(correlation))
                print('Homogeneity: {:.3f}'.format(homogeneity))
                print('Energy: {:.3f}'.format(energy))
                print('Contrast: {:.3f}'.format(contrast))

        if affichage:
            # Draw the contour on the image and display it
            cv2.drawContours(image, contours, -1, (0, 255, 0), 2)  # Draw all contours in green
            plt.imshow(image)
            plt.axis('off')
            plt.show()

        # Store features in a dictionary
        features = {
            'name': os.path.basename(image),
            'compact_index': compact_index,
            'C_r': C_r,
            'C_g': C_g,
            'C_b': C_b,
            'diameter': diameter,
            'correlation': correlation,
            'homogeneity': homogeneity,
            'energy': energy,
            'contrast': contrast
        }

        feature_list.append(features)

    # Convert the list of dictionaries to a DataFrame
    df = pd.DataFrame(feature_list)

    return df
