from model.extraction.lbp_extraction import get_lbp_features
from model.extraction.tamura_features import get_tamura
from model.extraction.glrlm_extraction import get_glrlm
from model.extraction.lab_color_moment import get_lab_color_moment
from model.extraction.yuv_color_moment import get_yuv_color_moment_features

import cv2
import numpy as np
import pandas as pd

# def __main__(image: cv2.Mat) -> dict[str, np.float64]:
def __main__(image: cv2.Mat) -> pd.DataFrame:
    
    yuv_features = get_yuv_color_moment_features(image)
    # lab_features_name = lab_color_moment_name()
    
    lbp_features = get_lbp_features(image)
    # lbp_features_name = get_lbp_name()
    
    glrlm_features = get_glrlm(image)
    # glrlm_features_name = get_glrlm_name()
    
    tamura_features = get_tamura(image)
    # tamura_features_name = get_tamura_name()
    
    features = {**yuv_features, **lbp_features, **glrlm_features, **tamura_features}
    
    # return lab_features | lbp_features | glrlm_features | tamura_features
    
    return pd.DataFrame.from_dict(features, orient='index').T.convert_dtypes(convert_floating=True).drop(labels=['LRLGLE_deg135', 'LRLGLE_deg90', 'LRLGLE_deg45', 'LRLGLE_deg0'], axis=1)
    
    # return pd.DataFrame.from_dict(features, orient='index').T.convert_dtypes(convert_floating=True)
    
    # features.extend(lab_features)
    # features.extend(lbp_features)
    # features.extend(glrlm_features)
    # features.extend(tamura_features)
    
    # features_name.extend(lab_features_name)
    # features_name.extend(lbp_features_name)
    # features_name.extend(glrlm_features_name)   
    # features_name.extend(tamura_features_name)
    
    # df_features = pd.DataFrame([features], columns=features_name)
    # df_features = df_features.loc[:, (df_features != 1).any()]
    
    # return df_features