import cv2
import numpy as np
import matplotlib.pyplot as plt

def apply_image_processing_operations(image_path):
    # Load the image
    original_image = cv2.imread(image_path)

    # Image Resizing
    resized_image = cv2.resize(original_image, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)

    # Image Rotation
    rows, cols, _ = original_image.shape
    rotation_matrix = cv2.getRotationMatrix2D((cols / 2, rows / 2), 30, 1)
    rotated_image = cv2.warpAffine(original_image, rotation_matrix, (cols, rows))

    # Image Grayscale Conversion
    grayscale_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)

    # Image Blurring
    blurred_image = cv2.GaussianBlur(original_image, (15, 15), 0)

    # Image Edge Detection
    edges = cv2.Canny(original_image, 100, 200)

    # Save the processed images
    cv2.imwrite('resized_image.jpg', resized_image)
    cv2.imwrite('rotated_image.jpg', rotated_image)
    cv2.imwrite('grayscale_image.jpg', grayscale_image)
    cv2.imwrite('blurred_image.jpg', blurred_image)
    cv2.imwrite('edges.jpg', edges)

    # Display all the processed images together with headers
    plt.figure(figsize=(12, 8))

    plt.subplot(231)
    plt.imshow(cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB))
    plt.title('Resized Image')
    plt.axis('off')

    plt.subplot(232)
    plt.imshow(cv2.cvtColor(rotated_image, cv2.COLOR_BGR2RGB))
    plt.title('Rotated Image')
    plt.axis('off')

    plt.subplot(233)
    plt.imshow(grayscale_image, cmap='gray')
    plt.title('Grayscale Image')
    plt.axis('off')

    plt.subplot(234)
    plt.imshow(cv2.cvtColor(blurred_image, cv2.COLOR_BGR2RGB))
    plt.title('Blurred Image')
    plt.axis('off')

    plt.subplot(235)
    plt.imshow(edges, cmap='gray')
    plt.title('Edges')
    plt.axis('off')

    plt.tight_layout()
    plt.show()


    image_path = r"C:/Users/asus/OneDrive - Indian Institute of Technology Bombay/Desktop/IIT B/Semester-4/Summer Projects/Cv for Dv/Git/CV_for_driveless/Image 1.jpg"  
    apply_image_processing_operations(image_path)