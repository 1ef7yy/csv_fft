import cv2
import numpy as np
import csv


def jpeg_to_csv(image_path: str, csv_path: str) -> None:
    image = cv2.imread(image_path)

    # Convert the image to a NumPy array
    image_array = np.array(image)

    # Save the pixel values as a CSV file
    with open(csv_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        for row in image_array:
            print(type(row[0][0]))
            writer.writerow(row.tolist())

