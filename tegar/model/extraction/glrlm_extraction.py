import numpy as np
import cv2
import warnings
from model.extraction.GrayRumatrix import getGrayRumatrix

warnings.filterwarnings("ignore")

def get_glrlm_features_name(features, degs):
    glrlm_features_name = []
    for deg in degs:
        for feature in features:
            glrlm_features_name.append(f"{feature}_{deg[0]}")
    return glrlm_features_name

def get_glrlm_name():
    # GLRLM
    glrlm_features = ['SRE', 'LRE', 'GLN', 'RLN', 'RP', 'LGLRE', 'HGL', 'SRLGLE', 'SRHGLE', 'LRLGLE', 'LRHGLE']
    glrlm_degs = [['deg0'], ['deg45'], ['deg90'], ['deg135']]
    glrlm_features_name = get_glrlm_features_name(glrlm_features, glrlm_degs)
    
    return glrlm_features_name # 44

def get_glrlm(image: cv2.Mat, lbp: bool=False) -> dict[str, np.float64]:
    
    test = getGrayRumatrix()
    # test.read_img(path, lbp)
    test.set_img(image, lbp)

    DEG = [['deg0'], ['deg45'], ['deg90'], ['deg135']]

    glrlm_features_value = []

    for deg in DEG:
        test_data = test.getGrayLevelRumatrix(test.data,deg)
        
        #1
        SRE = test.getShortRunEmphasis(test_data) 
        SRE = np.squeeze(SRE)
        SRE = float(SRE)
        
        #2
        LRE = test.getLongRunEmphasis(test_data)
        LRE = np.squeeze(LRE)
        LRE = float(LRE)
        
        #3
        GLN = test.getGrayLevelNonUniformity(test_data)
        GLN = np.squeeze(GLN)
        GLN = float(GLN)
        
        #4
        RLN = test.getRunLengthNonUniformity(test_data)
        RLN = np.squeeze(RLN)
        RLN = float(RLN)

        #5
        RP = test.getRunPercentage(test_data)
        RP = np.squeeze(RP)
        RP = float(RP)
        
        #6
        LGLRE = test.getLowGrayLevelRunEmphasis(test_data)
        LGLRE = np.squeeze(LGLRE)
        LGLRE = float(LGLRE)
        
        #7
        HGL = test.getHighGrayLevelRunEmphais(test_data)
        HGL = np.squeeze(HGL)
        HGL = float(HGL)
        
        #8
        SRLGLE = test.getShortRunLowGrayLevelEmphasis(test_data)
        SRLGLE = np.squeeze(SRLGLE)
        SRLGLE = float(SRLGLE)
        
        #9
        SRHGLE = test.getShortRunHighGrayLevelEmphasis(test_data)
        SRHGLE = np.squeeze(SRHGLE)
        SRHGLE = float(SRHGLE)
        
        #10
        LRLGLE = test.getLongRunLow(test_data)
        LRLGLE = np.squeeze(LRLGLE)
        LRLGLE = float(LRLGLE)
        
        #11
        LRHGLE = test.getLongRunHighGrayLevelEmphais(test_data)
        LRHGLE = np.squeeze(LRHGLE)
        LRHGLE = float(LRHGLE)

        glrlm_features_value_per_deg = [SRE, LRE, GLN, RLN, RP, LGLRE, HGL, SRLGLE, SRHGLE, LRLGLE, LRHGLE]
        
        for value in glrlm_features_value_per_deg:
            glrlm_features_value.append(value)

    return {
        "SRE_deg0": glrlm_features_value[0],
        "LRE_deg0": glrlm_features_value[1],
        "GLN_deg0": glrlm_features_value[2],
        "RLN_deg0": glrlm_features_value[3],
        "RP_deg0": glrlm_features_value[4],
        "LGLRE_deg0": glrlm_features_value[5],
        "HGL_deg0": glrlm_features_value[6],
        "SRLGLE_deg0": glrlm_features_value[7],
        "SRHGLE_deg0": glrlm_features_value[8],
        "LRLGLE_deg0": glrlm_features_value[9],
        "LRHGLE_deg0": glrlm_features_value[10],
        "SRE_deg45": glrlm_features_value[11],
        "LRE_deg45": glrlm_features_value[12],
        "GLN_deg45": glrlm_features_value[13],
        "RLN_deg45": glrlm_features_value[14],
        "RP_deg45": glrlm_features_value[15],
        "LGLRE_deg45": glrlm_features_value[16],
        "HGL_deg45": glrlm_features_value[17],
        "SRLGLE_deg45": glrlm_features_value[18],
        "SRHGLE_deg45": glrlm_features_value[19],
        "LRLGLE_deg45": glrlm_features_value[20],
        "LRHGLE_deg45": glrlm_features_value[21],
        "SRE_deg90": glrlm_features_value[22],
        "LRE_deg90": glrlm_features_value[23],
        "GLN_deg90": glrlm_features_value[24],
        "RLN_deg90": glrlm_features_value[25],
        "RP_deg90": glrlm_features_value[26],
        "LGLRE_deg90": glrlm_features_value[27],
        "HGL_deg90": glrlm_features_value[28],
        "SRLGLE_deg90": glrlm_features_value[29],
        "SRHGLE_deg90": glrlm_features_value[30],
        "LRLGLE_deg90": glrlm_features_value[31],
        "LRHGLE_deg90": glrlm_features_value[32],
        "SRE_deg135": glrlm_features_value[33],
        "LRE_deg135": glrlm_features_value[34],
        "GLN_deg135": glrlm_features_value[35],
        "RLN_deg135": glrlm_features_value[36],
        "RP_deg135": glrlm_features_value[37],
        "LGLRE_deg135": glrlm_features_value[38],
        "HGL_deg135": glrlm_features_value[39],
        "SRLGLE_deg135": glrlm_features_value[40],
        "SRHGLE_deg135": glrlm_features_value[41],
        "LRLGLE_deg135": glrlm_features_value[42],
        "LRHGLE_deg135": glrlm_features_value[43]
    }

def get_glrlm_on(path):
    return get_glrlm(path, lbp='on')