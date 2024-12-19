from ollama_ocr import OCRProcessor

# Initialize OCR processor
ocr = OCRProcessor(model_name='llama3.2-vision:11b')  # You can use any vision model available on Ollama

# Process an image
result = ocr.process_image(
    image_path="path/to/your/image.png",
    format_type="markdown"  # Options: markdown, text, json, structured, key_value
)
print(result)