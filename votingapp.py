from flask import Flask, request, render_template, redirect, url_for, send_from_directory
import cv2
import numpy as np
import csv
import os

app = Flask(__name__)
UPLOAD_FOLDER = 'images/upload'
RESULT_FOLDER = 'images/read'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_files():
    files = request.files.getlist('files')
    if not files:
        return redirect(request.url)

    for file in files:
        if file.filename == '':
            continue

        file_path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(file_path)
        process_image(file_path)

    return redirect(url_for('results'))



@app.route('/static/read/<filename>')
def serve_image(filename):
    return send_from_directory(RESULT_FOLDER, filename)


@app.route('/results')
def results():
    processed_images = os.listdir(RESULT_FOLDER)

    # Initialize a dictionary to count black classifications for each block
    black_counts = {i: 0 for i in range(1, 19)}  # Blocks 1 to 18
    symbols = {
        1: 'chair.png', 
        2: 'bat.jpg',
        3: 'bat.jpg',
        4: 'bat.jpg', 
        5: 'bat.jpg',
        6: 'bat.jpg',
        7: 'bat.jpg', 
        8: 'bat.jpg',
        9: 'bat.jpg',
        10: 'bat.jpg', 
        11: 'bat.jpg',
        12: 'bat.jpg',
        13: 'bat.jpg', 
        14: 'bat.jpg',
        15: 'bat.jpg',
        16: 'bat.jpg', 
        17: 'bat.jpg',
        18: 'bat.jpg',
               
               } 

    # Read the CSV file and store the data
    csv_data = []
    csv_path = 'block_classification_results.csv'
    
    with open(csv_path, 'r') as csvfile:
        reader = csv.reader(csvfile)
        header = next(reader)  # Skip header
        
        for row in reader:
            csv_data.append(row)
            if len(row) < 3:  # Ensure there are enough columns
                continue
            
            # Attempt to convert block number to int
            try:
                block_number = int(row[1])  # Assuming the second column is the block number
            except ValueError:
                continue  # Skip if conversion fails
            
            # Increment the count for the specific block if the classification is "Black"
            if row[2] == 'Black' and block_number in black_counts:
                black_counts[block_number] += 1

    return render_template('results.html', images=processed_images, csv_data=csv_data,   black_counts=black_counts, symbols=symbols)



# without reject vote condition 
# def process_image(image_path):
#     block_height = 24
#     num_blocks = 18
#     block_gap = 20
#     margin_top = 300
#     margin_right = 90
#     threshold_value = 250

#     csv_path = 'block_classification_results.csv'
#     with open(csv_path, 'a', newline='') as csvfile:
#         writer = csv.writer(csvfile)
#         writer.writerow(['Image Name', 'Block Number', 'Classification'])

#         image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

#         if image is None:
#             print(f"Error: Image {image_path} not loaded. Check the file path.")
#             return

#         height, width = image.shape
#         block_width = width // num_blocks
#         color_image = cv2.imread(image_path)

#         for i in range(num_blocks):
#             x_start = width - block_width - margin_right
#             x_end = (width - 40)
#             y_start = margin_top + i * (block_height + block_gap)
#             y_end = min(y_start + block_height, height)

#             if y_start >= height:
#                 break

#             block = image[y_start:y_end, x_start:x_end]
#             avg_pixel_value = np.mean(block)

#             if avg_pixel_value < threshold_value:
#                 classification = 'Black'
#                 cv2.rectangle(color_image, (x_start, y_start), (x_end, y_end), (0, 0, 255), 3)
#             else:
#                 classification = 'White'
#                 cv2.rectangle(color_image, (x_start, y_start), (x_end, y_end), (0, 255, 0), 3)

#             writer.writerow([os.path.basename(image_path), i + 1, classification])
#             cv2.putText(color_image, f"{i + 1}: {classification}", 
#                         (x_start + 5, y_start + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

#         result_image_path = os.path.join(RESULT_FOLDER, f'highlighted_{os.path.basename(image_path)}')
#         cv2.imwrite(result_image_path, color_image)
#         print(f"Processed {os.path.basename(image_path)}: Saved highlighted image as '{result_image_path}'")


# with reject condition 

def process_image(image_path):
    block_height = 24
    num_blocks = 18
    block_gap = 20
    margin_top = 300
    margin_right = 90
    threshold_value = 250

    csv_path = 'block_classification_results.csv'
    with open(csv_path, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Image Name', 'Block Number', 'Classification'])

        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

        if image is None:
            print(f"Error: Image {image_path} not loaded. Check the file path.")
            return

        height, width = image.shape
        block_width = width // num_blocks
        color_image = cv2.imread(image_path)

        block_classifications = {}  # Track classification for each block

        for i in range(num_blocks):
            x_start = width - block_width - margin_right
            x_end = (width - 40)
            y_start = margin_top + i * (block_height + block_gap)
            y_end = min(y_start + block_height, height)

            if y_start >= height:
                break

            block = image[y_start:y_end, x_start:x_end]
            avg_pixel_value = np.mean(block)

            if avg_pixel_value < threshold_value:
                classification = 'Black'
                cv2.rectangle(color_image, (x_start, y_start), (x_end, y_end), (0, 0, 255), 3)
            else:
                classification = 'White'
                cv2.rectangle(color_image, (x_start, y_start), (x_end, y_end), (0, 255, 0), 3)

            writer.writerow([os.path.basename(image_path), i + 1, classification])
            cv2.putText(color_image, f"{i + 1}: {classification}", 
                        (x_start + 5, y_start + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Store the classification result
            block_classifications[i + 1] = classification

        # Apply yellow classification rules based on groups
        yellow_groups = [(1, 4, 2), (5, 6, 2), (7, 10, 2), (11, 14, 3), (15, 16, 2), (17, 18, 2)]
        for start, end, threshold in yellow_groups:
            black_count = sum(1 for block in range(start, end + 1) if block_classifications.get(block) == 'Black')
            if black_count >= threshold:
                for block in range(start, end + 1):
                    y_start = margin_top + (block - 1) * (block_height + block_gap)
                    y_end = min(y_start + block_height, height)
                    cv2.rectangle(color_image, (x_start, y_start), (x_end, y_end), (0, 255, 255), 3)  # Yellow
                    cv2.putText(color_image, f"{block}: Yellow", 
                                (x_start + 5, y_start + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

        result_image_path = os.path.join(RESULT_FOLDER, f'highlighted_{os.path.basename(image_path)}')
        cv2.imwrite(result_image_path, color_image)
        print(f"Processed {os.path.basename(image_path)}: Saved highlighted image as '{result_image_path}'")



if __name__ == '__main__':
    app.run(debug=True)
