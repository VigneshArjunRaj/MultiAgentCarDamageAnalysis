from ultralytics import YOLO
import cv2
import yaml
from PIL import Image, ImageDraw, ImageFont
import numpy as np

config = yaml.safe_load(open("config.yaml"))

partModel = YOLO(config["carPartsArtifactPath"],verbose= False)
damageModel = YOLO(config["carDamageArtifactPath"], verbose= False)
generalModel = YOLO(config["generalArtifactPath"], verbose= False)

damageClasses = config["damageClasses"]
partsClasses = config["partsClasses"]

carcococlassid = config["cococlassidforCar"]

def checkForCar(results):
    """
    To check if there is a car in the results or not
    Args:
        results (dict): A dictionary containing YOLO detection results, typically with keys such as 'categories', 'boxes', and 'scores'.

    Returns:
        bool: True if a car is detected in the results, False otherwise.

    """
    for box in results[0].boxes:
            # Extract class name and confidence
            class_id = int(box.cls)
            if class_id == 2:
                 return True
    return False 


def mapDamagesParts(damages, parts, threshold=0.3):
    """
    Map detected damages to identified parts based on bounding box overlap.
    
    Args:
        damages (dict): Dictionary containing damages with bounding boxes, categories, and confidence scores.
        parts (dict): Dictionary containing parts with bounding boxes, categories, and confidence scores.
        threshold (float): Minimum overlap ratio to consider a part as affected by a damage.
    
    Returns:
        list: A mapping of damages to affected parts.
    """
    damlen = len(damages)
    damagesmap = []
    #print(damages)
    for damage in range(damlen):
        damage_boxes = damages[damage]['bounding boxes']
       
        
        part_boxes = parts[damage]['bounding boxes']
        part_categories = parts[damage]['Categories']

        
        # Iterate through each damage
        for i, damage_box in enumerate(damage_boxes):
            damage_box = damage_box[0]  # Extract box array from the nested structure
            x_min_d, y_min_d, x_max_d, y_max_d = damage_box
            
            # Iterate through each part
            for j, part_box in enumerate(part_boxes):
                part_box = part_box[0]  # Extract box array from the nested structure
                x_min_p, y_min_p, x_max_p, y_max_p = part_box
                
                # Calculate intersection
                x_min_i = max(x_min_d, x_min_p)
                y_min_i = max(y_min_d, y_min_p)
                x_max_i = min(x_max_d, x_max_p)
                y_max_i = min(y_max_d, y_max_p)
                
                # Calculate intersection area
                if x_min_i < x_max_i and y_min_i < y_max_i:
                    intersection_area = (x_max_i - x_min_i) * (y_max_i - y_min_i)
                else:
                    intersection_area = 0
                
                # Calculate areas of damage and part
                damage_area = (x_max_d - x_min_d) * (y_max_d - y_min_d)

                
                # Calculate overlap ratios
                overlap_ratio = intersection_area / damage_area if damage_area > 0 else 0
                
                # Check if overlap exceeds threshold
                if overlap_ratio > threshold:
                    damagesmap.append(part_categories[j])
            

        
    return damagesmap


def extract_color_region(image, mask_points):
    """
   
    Extracts the color region from the given image using the mask defined by [x, y] points.

    Args:
        image : The source image from which the region will be extracted.
        mask_points: A list of (x, y) coordinates defining the polygon mask for the region.

    Returns:
        np.ndarray: An array of pixel values from the extracted region.
    
    """
    # Create a mask image that is all transparent
    mask_image = Image.new("RGBA", image.size, (0, 0, 0, 0))
    mask_draw = ImageDraw.Draw(mask_image)

    # Draw the polygon mask on the transparent image
    mask_draw.polygon(mask_points, fill=(255, 255, 255, 255))

    # Extract the region from the original image using the mask
    region = Image.composite(image, Image.new("RGBA", image.size, (0, 0, 0, 0)), mask_image.convert("1"))
    return region


def map_damages_to_parts(damages, parts, threshold=0.2):
    """
    Map detected damages to identified parts based on bounding box overlap.
    
    Args:
        damages (dict): Dictionary containing damages with bounding boxes, categories, and confidence scores.
        parts (dict): Dictionary containing parts with bounding boxes, categories, and confidence scores.
        threshold (float): Minimum overlap ratio to consider a part as affected by a damage.
    
    Returns:
        list: A mapping of damages to affected parts.
    """
    damlen = len(damages)
    #print(damlen)
    damagesmap = []
    for damage in range(damlen):
        damage_boxes = damages[damage]['bounding boxes']
        damage_categories = damages[damage]['Categories']
        damage_confidences = damages[damage]['Confidence scores']
        
        part_boxes = parts[damage]['bounding boxes']
        part_categories = parts[damage]['Categories']
        part_confidences = parts[damage]['Confidence scores']
        
        damage_to_parts_mapping = []
        
        # Iterate through each damage
        for i, damage_box in enumerate(damage_boxes):
            damage_box = damage_box[0]  # Extract box array from the nested structure
            x_min_d, y_min_d, x_max_d, y_max_d = damage_box
            
            affected_parts = []
            
            # Iterate through each part
            for j, part_box in enumerate(part_boxes):
                part_box = part_box[0]  # Extract box array from the nested structure
                x_min_p, y_min_p, x_max_p, y_max_p = part_box
                
                # Calculate intersection
                x_min_i = max(x_min_d, x_min_p)
                y_min_i = max(y_min_d, y_min_p)
                x_max_i = min(x_max_d, x_max_p)
                y_max_i = min(y_max_d, y_max_p)
                
                # Calculate intersection area
                if x_min_i < x_max_i and y_min_i < y_max_i:
                    intersection_area = (x_max_i - x_min_i) * (y_max_i - y_min_i)
                else:
                    intersection_area = 0
                
                # Calculate areas of damage and part
                damage_area = (x_max_d - x_min_d) * (y_max_d - y_min_d)
                part_area = (x_max_p - x_min_p) * (y_max_p - y_min_p)
                
                # Calculate overlap ratios
                overlap_ratio = intersection_area / damage_area if damage_area > 0 else 0
                
                # Check if overlap exceeds threshold
                if overlap_ratio > threshold:
                    affected_parts.append({
                        "part_category": part_categories[j],
                        "part_confidence": part_confidences[j],
                        "overlap_ratio": overlap_ratio
                    })
            
            # Append the damage and its affected parts
            damage_to_parts_mapping.append({
                "damage_category": damage_categories[i],
                "damage_confidence": damage_confidences[i],
                "affected_parts": affected_parts
            })
        damagesmap.append(damage_to_parts_mapping)
        
    return damagesmap


def draw_boxes_with_hover(image, damage_data, part_data):
    """
    Overlay bounding boxes for damages and parts, with hover feature.
    Args:
        image: Original image (NumPy array).
        damage_data: Damage detection data (list of dicts).
        part_data: Parts detection data (list of dicts).
    Returns:
        Annotated image (NumPy array).
    """
    
    annotated_image = image.copy()
    categories = mapDamagesParts([damage_data,],[part_data,])


    
    # Convert the image to a format compatible with ImageDraw
    annotated_image = image.copy().convert("L").convert('RGBA')
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    overlay_draw = ImageDraw.Draw(overlay)
 

    # Font setup (ensure the font file path is correct, or use default)
    try:
        font = ImageFont.truetype("arial.ttf", size=15)
    except IOError:
        font = ImageFont.load_default()

            # Draw parts
    for mask, category, confidence in zip(
        part_data['masks'], part_data['Categories'], part_data['Confidence scores']
    ):
        if category in categories:
            color = "#fff"  # Yellow with transparency for parts

            mask_points = [(int(x), int(y)) for x, y in mask[0]]
            
            color_region = extract_color_region(image, mask_points)
        
            # Paste the color region onto the grayscale image
            annotated_image.paste(color_region, mask=color_region.split()[3]) 
        
            
            overlay_draw.polygon(mask_points, outline=color,width=2)
        
            # Draw category text
            text_pos = mask_points[0]  # Position of the first vertex
            
            bbox = overlay_draw.textbbox(text_pos, f"{category}", font=font)
            overlay_draw.rectangle(bbox, fill="#000c")
            overlay_draw.text(text_pos, f"{category}", font=font, fill="white")

    # Draw damage bounding boxes
    for box, category, confidence in zip(
        damage_data['bounding boxes'], damage_data['Categories'], damage_data['Confidence scores']
    ):
        x_min, y_min, x_max, y_max = map(int, box[0])
        color = (255, 255, 255)  # Red for damages
        
        # # Draw rectangle
        overlay_draw.rectangle([x_min, y_min, x_max, y_max], outline=color, width=1)

        # Get the exact colors from the original image for this mask

        
        # Draw the category text above the box
        text = f"{category}"
        text_pos = (x_min, y_min - 20)  # Slightly above the box

        bbox = overlay_draw.textbbox(text_pos, text, font=font)
        overlay_draw.rectangle(bbox, fill="#000c")
        overlay_draw.text(text_pos, text, font=font, fill="white")
        
        
        
        #overlay_draw.text(text_pos, text, fill=color, font=font,align= "center")
    annotated_image = Image.alpha_composite(annotated_image, overlay)

    return annotated_image


def grabDataFromResults(results, names, m = False):
    """
    Extracts information from YOLO results for specified categories.

    Args:
        results (dict): A dictionary containing YOLO detection results, with keys as 'categories', 'boxes', 'masks', and 'scores'.
        names (list of str): A list of category names to filter from the results.
        m (bool, optional): Indicates whether to include mask information in the output. Defaults to False.

    Returns:
        list: A list of extracted data for the specified categories, optionally including masks.
    """
    # Initialize lists to store extracted data
    #print(results)
    class_names = []
    confidences = []
    bboxes = []
    masks = []

    for box in results.boxes:
            # Extract class name and confidence
            class_id = int(box.cls)
            confidence = box.conf
            bbox = box.xyxy.cpu().numpy()  # Get bounding box coordinates (x1, y1, x2, y2)

            # Append data to lists
           
            class_names.append(names[class_id])
            confidences.append(confidence.item())
            bboxes.append(bbox)
    
    if m:
        for mask in results.masks:
            xy_coords = mask.xy
            masks.append(xy_coords)

   
    return class_names,confidences,bboxes, masks, results.orig_img


def inferParts(image):
    """
    Infers Parts segmentation from provided Image

    Args:
        image: Original image (NumPy array).
    Returns:
        dict: A dict of extracted data for categories, confidence scores, bounding boxes and masks
    """
    result = partModel(image)
    class_names,confidences,bboxes, masks, orig_img = grabDataFromResults(result[0],partsClasses,m=True)
    return {"Categories":class_names,"Confidence scores":confidences,"bounding boxes":bboxes,"masks": masks}


def inferdamage(image):
    """
    Infers damage detection from provided Image

    Args:
        image: Original image (NumPy array).
    Returns:
        dict: A dict of extracted data for categories, confidence scores, bounding boxes
    """
    result = damageModel(image)
    
    class_names,confidences,bboxes, masks, orig_img = grabDataFromResults(result[0],damageClasses)
    return {"Categories":class_names,"Confidence scores":confidences,"bounding boxes":bboxes}
    


def inferCar(image):
    """
    Infers Parts segmentation from provided Image

    Args:
        image: Original image (NumPy array).
    Returns:
        results (dict): A dictionary containing YOLO detection results, with keys as 'categories', 'boxes', 'masks', and 'scores'.
    """
    result = generalModel(image)
    print("------------------ In inferCar -----------------------")

    result = checkForCar(result)
    print(result)
    return result

