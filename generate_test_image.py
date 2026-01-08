from PIL import Image, ImageDraw, ImageFont
import os

img = Image.new('RGB', (800, 600), color=(255, 255, 255))
d = ImageDraw.Draw(img)
text = """
Brand: GeminiTest
Product: API Key Verifier
Net Quantity: 1 Unit
MRP: Rs 100.00
MFG Date: 01/01/2026
"""
d.text((50, 50), text, fill=(0, 0, 0))
img.save("test_image.jpg")
print("test_image.jpg created")
