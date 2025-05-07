import nbformat
import base64
import os

# Load the notebook
with open("IMDB.ipynb", "r", encoding="utf-8") as f:
    nb = nbformat.read(f, as_version=4)

# Make a folder for images
os.makedirs("extracted_images", exist_ok=True)

image_count = 0

for cell in nb.cells:
    if cell.cell_type == "code":
        for output in cell.get("outputs", []):
            if "data" in output and "image/png" in output["data"]:
                image_data = output["data"]["image/png"]
                image_bytes = base64.b64decode(image_data)
                filename = f"extracted_images/image_{image_count:03d}.png"
                with open(filename, "wb") as img_file:
                    img_file.write(image_bytes)
                image_count += 1

print(f"Extracted {image_count} image(s).")
