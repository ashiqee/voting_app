from flask import Flask, request, render_template, redirect, url_for, send_from_directory, jsonify
import cv2
import numpy as np
import csv
import sys
import os

app = Flask(__name__)

# Function to get the base path
def get_base_path():
    if getattr(sys, 'frozen', False):  # Check if running as a PyInstaller .exe
        return os.path.dirname(sys.executable)  # Directory of the .exe
    return os.path.abspath(os.path.dirname(__file__))  # Directory of the .py script

BASE_PATH = get_base_path()

# Update folders
UPLOAD_FOLDER = os.path.join(BASE_PATH, 'images/upload')
RESULT_FOLDER = os.path.join(BASE_PATH, 'images/read')

# Ensure directories exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

# Keep track of processed images
processed_images = set()  # Use a set for faster lookups

# Home route
@app.route('/')
def index():
    return render_template('index.html')

# File upload route with progress feedback
@app.route('/upload', methods=['POST'])
def upload_files():
    files = request.files.getlist('files')  # Get all uploaded files
    if not files:
        return redirect(request.url)

    already_processed = []  # List to hold already processed files
    for file in files:
        if file.filename == '':
            continue

        # Check if the file has already been processed
        if file.filename in processed_images:
            already_processed.append(file.filename)
            continue

          # Normalize filename
        normalized_filename = file.filename.replace(" ", "_")
        file_path = os.path.join(UPLOAD_FOLDER, normalized_filename)

        # Check if the file has already been processed
        if normalized_filename in processed_images:
            already_processed.append(normalized_filename)
            continue

        # Save the file
        file.save(file_path)

        # Process the image
        process_image(file_path)
        processed_images.add(normalized_filename)

    # Show an alert if some files were already processed
    if already_processed:
        return render_template('index.html', already_processed=already_processed)

    return redirect(url_for('results'))

# Serve processed images
@app.route('/read/<filename>')
def serve_image(filename):
    try:
        file_path = os.path.join(RESULT_FOLDER, filename)
     
        return send_from_directory(RESULT_FOLDER, filename)
    except FileNotFoundError:
        return "File not found!", 404

# Results route
@app.route('/results')
def results():
    # Get list of processed images
    processed_images_list = os.listdir(RESULT_FOLDER)

    # Pagination logic
    per_page = 50  # Default images per page
    page = int(request.args.get('page', 1))  # Current page, default to 1
    total_images = len(processed_images_list)
    total_pages = (total_images + per_page - 1) // per_page  # Calculate total pages

    # Get the images for the current page
    start = (page - 1) * per_page
    end = start + per_page
    paginated_images = processed_images_list[start:end]

    # Initialize dictionary for block counts
    black_counts = {i: 0 for i in range(1, 7)}
    symbols = {
        1: '1.jpg',
        2: '2.jpg',
        3: '3.jpg',
        4: '4.jpg',
        5: '5.jpg',
        6: '6.jpg',
    }

    # Read classification results from CSV
    csv_data = []
    csv_path = 'vote_results.csv'
    ballot_paper_data = []  # For grouping by ballot paper

    with open(csv_path, 'r') as csvfile:
        reader = csv.reader(csvfile)
        header = next(reader)  # Skip the header row

        current_ballot = []
        for row in reader:
            csv_data.append(row)

            # Ensure the row has enough columns
            if len(row) < 3:
                continue

            # Parse block number and classification
            try:
                block_number = int(row[1])
                classification = row[2]
            except ValueError:
                continue  # Skip invalid rows

            # Append to the current ballot
            current_ballot.append((block_number, classification))

            # Group blocks into ballot papers (6 blocks per ballot paper)
            if block_number == 6:
                ballot_paper_data.append(current_ballot)
                current_ballot = []

        # Process each ballot paper
        for ballot in ballot_paper_data:
            block_classifications = {block: classification for block, classification in ballot}

            # Handle reject conditions for specific pairs
            pairs = [(1, 2), (3, 4), (5, 6)]
            for block1, block2 in pairs:
                if block_classifications.get(block1) == 'Black' and block_classifications.get(block2) == 'Black':
                    # Mark as Yellow
                    block_classifications[block1] = 'Yellow'
                    block_classifications[block2] = 'Yellow'

            # Count valid Black classifications
            for block, classification in block_classifications.items():
                if classification == 'Black' and block in black_counts:
                    black_counts[block] += 1

    return render_template(
        'results.html',
        images=paginated_images,  # Pass only the images for the current page
        csv_data=csv_data,
        black_counts=black_counts,
        symbols=symbols,
        total_pages=total_pages,
        current_page=page,
    )

# Image processing logic
def process_image(image_path):
    # Block detection parameters (in mm to pixel mapping)
    block_height = 160  # Height of each block in pixels
    block_gap = 85  # Vertical gap between blocks in pixels
    num_blocks = 6  # Number of blocks
    threshold_value = 250  # Pixel intensity threshold for classification

    # Define the fixed image dimensions in mm (width ~222mm, height ~220mm)
    image_width_mm = 222
    image_height_mm = 220

    # Assuming a fixed DPI (Dots Per Inch) conversion, you may adjust the value based on your image resolution
    dpi = 96  # Example DPI; for a higher DPI, adjust accordingly
    image_width_pixels = int(image_width_mm / 25.4 * dpi)  # Convert mm to pixels using DPI
    image_height_pixels = int(image_height_mm / 25.4 * dpi)  # Convert mm to pixels using DPI

    # Calculate the starting margins dynamically based on the image size
    margin_top = 520  # Adjust based on the image size if necessary
    margin_right = 20  # Adjust right margin

    csv_path = 'vote_results.csv'
    # Write CSV header if it doesn't exist
    if not os.path.exists(csv_path):
        with open(csv_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Image Name', 'Block Number', 'Classification'])

    with open(csv_path, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

        if image is None:
            print(f"Error: Could not load image {image_path}")
            return

        height, width = image.shape
        color_image = cv2.imread(image_path)

        # Classify blocks
        block_classifications = {}
        for i in range(num_blocks):
            x_start = width - (width // num_blocks) - margin_right
            x_end = width - 150
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

            # Annotate block
            cv2.putText(color_image, f"{i + 1}: {classification}",
                        (x_start + 5, y_start + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Highlight reject conditions
        pairs = [(1, 2), (3, 4), (5, 6)]
        for block1, block2 in pairs:
            if block_classifications.get(block1) == 'Black' and block_classifications.get(block2) == 'Black':
                for block in (block1, block2):
                    y_start = margin_top + (block - 1) * (block_height + block_gap)
                    y_end = min(y_start + block_height, height)
                    cv2.rectangle(color_image, (x_start, y_start), (x_end, y_end), (0, 255, 255), 3)  # Yellow
                    cv2.putText(color_image, f"{block}: Yellow",
                                (x_start + 5, y_start + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

        # Save the highlighted image
        result_filename = os.path.basename(image_path).replace(" ", "_")
        result_image_path = os.path.join(RESULT_FOLDER, f'read_{result_filename}')
        cv2.imwrite(result_image_path, color_image)
        print(f"Processed {os.path.basename(image_path)}: Saved highlighted image as '{result_image_path}'")

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
