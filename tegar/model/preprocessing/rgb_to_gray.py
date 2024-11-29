import cv2

def __main__(image: cv2.Mat):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    return gray_image
