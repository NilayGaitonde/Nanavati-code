import os
import pydicom
import matplotlib.pyplot as plt
import numpy as np
import cv2
import re

st0_path = r"C:\\Users\\Anuj\\Downloads\\DICOM\\PA0\\ST0"

quit_program = False

def on_key(event):
    global quit_program
    if event.key == 'q':
        quit_program = True
        plt.close('all')  

# Traverse through each "SE" folder in the "ST0" folder
for se_folder in sorted(os.listdir(st0_path)):
    if quit_program:
        break  

    # Construct the path to each "SE" folder
    se_path = os.path.join(st0_path, se_folder)

    if os.path.isdir(se_path) and se_folder.startswith("SE"):
        print(f"Processing folder: {se_folder}")

        # Gather and sort DICOM files in numerical order based on filename 
        dicom_files = sorted(
            [f for f in os.listdir(se_path) if f.startswith("IM")],
            key=lambda x: int(re.search(r'\d+', x).group())
        )

        for filename in dicom_files:
            if quit_program:
                break  
            file_path = os.path.join(se_path, filename)
            
            try:
                dicom_file = pydicom.dcmread(file_path)
                
                if hasattr(dicom_file, 'pixel_array'):
                    original_image = dicom_file.pixel_array

                    # Normalize the image
                    normalized_image = (original_image - np.min(original_image)) / (np.max(original_image) - np.min(original_image))
                    normalized_image = (normalized_image * 255).astype(np.uint8)  # Scale to 8-bit
                    
                    # Apply Gaussian smoothing
                    smoothed_image = cv2.GaussianBlur(normalized_image, (5, 5), 0)
                    
                    # Apply adaptive histogram equalization
                    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                    equalized_image = clahe.apply(smoothed_image)
                    
                    # Resize the image for consistency (optional)
                    resized_image = cv2.resize(equalized_image, (256, 256))
                    
                    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
                    fig.canvas.mpl_connect('key_press_event', on_key)  
                    
                    axs[0].imshow(original_image, cmap='gray')
                    axs[0].set_title(f"Original Image - {filename} in {se_folder}")
                    axs[0].axis('off')
                    
                    axs[1].imshow(resized_image, cmap='gray')
                    axs[1].set_title("Processed Image")
                    axs[1].axis('off')
                    
                    plt.show()  
                    if quit_program:
                        break  # Exit the loop if "q" was pressed

                    plt.close(fig)  

                else:
                    print(f"{filename} in {se_folder} has no pixel data.")
                    
            except Exception as e:
                print("Could not read file:", filename, "| Error:", e)