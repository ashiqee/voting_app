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
    black_counts = {i: 0 for i in range(1, 7)}  # Blocks 1 to 6
    symbols = {
        1: '1.jpg',
        2: '2.jpg',
        3: '3.jpg',
        4: '4.jpg',
        5: '5.jpg',
        6: '6.jpg',
    }

    # Read the CSV file and store the data
    csv_data = []
    csv_path = 'block_classification_results.csv'

    # Group classifications by ballot paper (columns 1 to 6 form one ballot paper)
    ballot_paper_data = []

    with open(csv_path, 'r') as csvfile:
        reader = csv.reader(csvfile)
        header = next(reader)  # Skip header

        current_ballot = []
        for row in reader:
            csv_data.append(row)
            if len(row) < 3:  # Ensure there are enough columns
                continue

            # Attempt to convert block number to int
            try:
                block_number = int(row[1])  # Assuming the second column is the block number
            except ValueError:
                continue  # Skip if conversion fails

            # Add current row to the current ballot paper
            current_ballot.append((block_number, row[2]))

            # Check if we've processed all 6 columns of a ballot paper
            if block_number == 6:
                ballot_paper_data.append(current_ballot)
                current_ballot = []  # Reset for the next ballot paper

        # Process each ballot paper independently
        for ballot in ballot_paper_data:
            block_classifications = {block: classification for block, classification in ballot}

            # Check pairs for "Black" and exclude them from counts
            skipped_pairs = [(1, 2), (3, 4), (5, 6)]
            for block1, block2 in skipped_pairs:
                if (block_classifications.get(block1) == 'Black' and
                    block_classifications.get(block2) == 'Black'):
                    # Mark both blocks as excluded
                    block_classifications[block1] = 'Yellow'
                    block_classifications[block2] = 'Yellow'

            # Increment the count for each block unless it's part of a skipped pair
            for block_number, classification in block_classifications.items():
                if classification == 'Black' and block_number in black_counts:
                    black_counts[block_number] += 1

    return render_template(
        'results.html',
        images=processed_images,
        csv_data=csv_data,
        black_counts=black_counts,
        symbols=symbols
    )

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
    block_height = 75
    num_blocks = 6  # Total blocks
    block_gap = 22
    margin_top = 188
    margin_right = 20
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

        # Store the classification of each block
        block_classifications = {}

        for i in range(num_blocks):
            x_start = width - block_width - margin_right
            x_end = width - 63
            y_start = margin_top + i * (block_height + block_gap)
            y_end = min(y_start + block_height, height)

            if y_start >= height:
                break

            block = image[y_start:y_end, x_start:x_end]
            avg_pixel_value = np.mean(block)

            if avg_pixel_value < threshold_value:
                classification = 'Black'
                cv2.rectangle(color_image, (x_start, y_start), (x_end, y_end), (0, 0, 255), 3)  # Red
            else:
                classification = 'White'
                cv2.rectangle(color_image, (x_start, y_start), (x_end, y_end), (0, 255, 0), 3)  # Green

            writer.writerow([os.path.basename(image_path), i + 1, classification])
            block_classifications[i + 1] = classification

            cv2.putText(color_image, f"{i + 1}: {classification}",
                        (x_start + 5, y_start + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Check pairs for yellow marking
        pairs = [(1, 2), (3, 4), (5, 6)]
        for block1, block2 in pairs:
            if (block_classifications.get(block1) == 'Black' and 
                block_classifications.get(block2) == 'Black'):
                # Highlight both blocks in yellow
                for block in (block1, block2):
                    y_start = margin_top + (block - 1) * (block_height + block_gap)
                    y_end = min(y_start + block_height, height)
                    cv2.rectangle(color_image, (x_start, y_start), (x_end, y_end), (0, 255, 255), 3)  # Yellow
                    cv2.putText(color_image, f"{block}: Yellow",
                                (x_start + 5, y_start + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

                # Remove the blocks from the Black classification count
                block_classifications[block1] = 'Yellow'
                block_classifications[block2] = 'Yellow'

        # Save the processed image
        result_image_path = os.path.join(RESULT_FOLDER, f'highlighted_{os.path.basename(image_path)}')
        cv2.imwrite(result_image_path, color_image)
        print(f"Processed {os.path.basename(image_path)}: Saved highlighted image as '{result_image_path}'")

if __name__ == '__main__':
    app.run(debug=True)
