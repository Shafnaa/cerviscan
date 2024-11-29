from model.preprocessing.rgb_to_gray import __main__ as rgb_to_gray
from model.segmentation.multiotsu_segmentation import __main__ as multiotsu_segmentation
from model.segmentation.bitwise_operation import __main__ as bitwise_operation

from model.extraction.main import __main__ as feature_extraction

import os
import pickle
import xgboost
import pandas as pd
import cv2
# import matplotlib.pyplot as plt 


def main(image: cv2.Mat):
    # image = plt.imread(image_path)
    
    # Convert RBG Image to Grayscale Image
    gray_image = rgb_to_gray(image)
    # output_path = os.path.join(f'{image_output}', f"{os.path.basename(image_path)[:-4]}_gray.jpg")
    # cv2.imwrite(output_path, gray_image)
    # plt.imsave(output_path, gray_image, cmap="gray")
    
    # Get Image Masking
    mask_image = multiotsu_segmentation(gray_image)
    # output_path = os.path.join(f'{image_output}', f"{os.path.basename(image_path)[:-4]}_mask.jpg")
    # cv2.imwrite(output_path, mask_image)
    # plt.imsave(output_path, mask_image, cmap="gray")
    
    # Segment Image Using Image Masking
    segmented_image = bitwise_operation(image, mask_image)
    
    # output_path = os.path.join(f'{image_output}', f"{os.path.basename(image_path)[:-4]}_segmented.jpg")
    # plt.imsave(output_path, segmented_image, cmap="gray")
    
    features = feature_extraction(segmented_image)
    
    # df_features = pd.DataFrame.from_dict(features, orient='index').T.convert_dtypes(convert_floating=True).drop(labels=['LRLGLE_deg135', 'LRLGLE_deg90', 'LRLGLE_deg45', 'LRLGLE_deg0'], axis=1)
    
    # print(features.info())
    # print(features.head())
    
    model: xgboost.XGBClassifier = pickle.load(open('./model/xgb_best', 'rb'))
    # model = xgboost.XGBClassifier()
    # result = model.predict(df_features)
    result = model.predict(features)
    
    return {
        'features': features.to_dict(),
        'result': "Abnormal" if result[0] else "Normal"
    }
    
    
    