import numpy as np
import os

def visualize_point_cloud():

    file_path = input("Enter the path to your .txt file (e.g., data.txt): ")
    output_folder_name = input("Enter the name for the output folder: ")

    try:
        a = float(input("Enter the lower displacement threshold 'a': "))
        b = float(input("Enter the upper displacement threshold 'b': "))
    except ValueError:
        print("\nInvalid input. Please enter numeric values for 'a' and 'b'.")
        return

    if a >= b:
        print("\nError: Threshold 'a' must be less than 'b'. Please try again.")
        return

    try:
        data = np.loadtxt(file_path)
    except FileNotFoundError:
        print(f"\nError: The file '{file_path}' was not found. Please check the path and try again.")
        return
    except Exception as e:
        print(f"\nAn error occurred while loading the file: {e}")
        return

    if data.shape[1] != 4:
        print("\nError: The data file must contain exactly 4 columns (x, y, z, displacement).")
        return

    x = data[:, 0]
    y = data[:, 1]
    z = data[:, 2]
    displacement = data[:, 3]

    colors = []

    for d in displacement:
        if d <= a:
            colors.append([0, 255, 0])
        elif d > b:
            colors.append([255, 0, 0]) 
        else:
            normalized_d = (d - a) / (b - a)

            green = np.array([0, 255, 0])
            yellow = np.array([255, 255, 0])
            red = np.array([255, 0, 0])

            if normalized_d <= 255/2:
                color = green + (yellow - green) * (normalized_d * 2)
            else:
                color = yellow + (red - yellow) * ((normalized_d - 255/2) * 2)
            
            colors.append(color)

    try:
        os.makedirs(output_folder_name, exist_ok=True)
    except OSError as e:
        print(f"\nError: Failed to create the directory '{output_folder_name}'. {e}")
        return

    combined_data = np.hstack((data, colors))

    output_data = combined_data[:, [0, 1, 2, 4, 5, 6, 3]]
    
    output_file_path = os.path.join(output_folder_name, "thresh_vis_point_cloud.txt")

    np.savetxt(output_file_path, output_data, fmt='%.6f', delimiter=' ', comments='')

    print("point cloud data has been saved")

if __name__ == "__main__":
    visualize_point_cloud()
