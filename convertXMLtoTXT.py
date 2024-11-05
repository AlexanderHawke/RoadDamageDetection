# Before running this, you may need to install these extensions by running this code:
# pip install ultralytics lxml

import os
import xml.etree.ElementTree as ET

def convert_to_yolo_format(bbox, img_width, img_height):
    x_center = (bbox['xmin'] + bbox['xmax']) / 2 / img_width
    y_center = (bbox['ymin'] + bbox['ymax']) / 2 / img_height
    width = (bbox['xmax'] - bbox['xmin']) / img_width
    height = (bbox['ymax'] - bbox['ymin']) / img_height
    return x_center, y_center, width, height

def parse_annotation(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()
    
    # Parse image size with error handling for missing depth tags
    img_width = int(root.find('size/width').text)
    img_height = int(root.find('size/height').text)
    
    annotations = []
    
    for obj in root.findall('object'):
        label = obj.find('name').text
        
        bbox = {
            'xmin': float(obj.find('bndbox/xmin').text),
            'ymin': float(obj.find('bndbox/ymin').text),
            'xmax': float(obj.find('bndbox/xmax').text),
            'ymax': float(obj.find('bndbox/ymax').text),
        }
        
        yolo_bbox = convert_to_yolo_format(bbox, img_width, img_height)
        annotations.append((label, *yolo_bbox))
    
    return annotations

def process_annotations(data_folder):
    for root, _, files in os.walk(data_folder):
        if root.endswith("xmls"):
            for file in files:
                if file.endswith(".xml"):
                    xml_file = os.path.join(root, file)
                    txt_file = os.path.splitext(xml_file)[0] + ".txt"
                    annotations = parse_annotation(xml_file)
                    
                    with open(txt_file, "w") as f:
                        for label, x_center, y_center, width, height in annotations:
                            f.write(f"{label} {x_center} {y_center} {width} {height}\n")
                    print(f"Converted {xml_file} to {txt_file}")

# Convert all XML files in the annotations folder of each country
process_annotations(r"C:\Users\alexa\Desktop\Honours Project\RDD2022") # Replace this with your own file path to the folder with all .

# for root, _, files in os.walk(r"C:\Users\alexa\Desktop\Honours Project\RDD2022"):
#     print(f"Root: {root}, Subdirs: {_}, Files: {files}")