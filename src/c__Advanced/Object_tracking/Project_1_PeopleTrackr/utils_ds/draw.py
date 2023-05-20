import numpy as np
import cv2

palette = (2 ** 11 - 1, 2 ** 15 - 1, 2 ** 20 - 1)


import csv
import datetime
import atexit
import os

def write_to_csv(file_path, bbox, class_name):
    # Create header if file is new
    is_new_file = not os.path.isfile(file_path)
    with open(file_path, mode='a', newline='') as csv_file:
        writer = csv.writer(csv_file)
        if is_new_file:
            writer.writerow(['Timestamp', 'BBox', 'Class'])
        
        # Write data to file
        current_time = datetime.datetime.now()
        writer.writerow([current_time.strftime("%Y-%m-%d %H:%M:%S"), bbox, class_name])
    
    # Close file when program terminates
    @atexit.register
    def close_csv_file():
        csv_file.close()


def compute_color_for_labels(label):
    """
    Simple function that adds fixed color depending on the class
    """
    color = [int((p * (label ** 2 - label + 1)) % 255) for p in palette]
    return tuple(color)

def overlay_on_image(image,clr_trails):
    h, w = clr_trails.shape[:2]
    overlay = np.zeros((h, w, 3), dtype=np.uint8)
    
    gray = cv2.cvtColor(clr_trails,cv2.COLOR_BGR2GRAY)
    mask = cv2.threshold(gray,0,255,cv2.THRESH_BINARY)[1]
    inverted_mask = cv2.bitwise_not(mask)
    image = cv2.bitwise_and(image, image, mask=inverted_mask)
    overlay = cv2.bitwise_and(clr_trails, clr_trails, mask=mask)
    image = cv2.bitwise_or(overlay, image)
    return image

def inc_int(color,factor = 1.4):
    b,g,r = color
    new_b = b*factor if b*factor < 255 else 255
    new_g = g*factor if g*factor < 255 else 255
    new_r = r*factor if r*factor < 255 else 255
    return (new_b,new_g,new_r)


def draw_boxes(img, bbox, identities=None, offset=(0,0),mask = None, trajectories = None,id_to_track = None,t_classes = None,categories = None):
    for i,box in enumerate(bbox):
        x1,y1,x2,y2 = [int(i) for i in box]
        x1 += offset[0]
        x2 += offset[0]
        y1 += offset[1]
        y2 += offset[1]
        # box text and bar
        id = int(identities[i]) if identities is not None else 0    
        color = compute_color_for_labels(id)
        
        bbox_thickness = 3
        draw_trajectory = True
        if id_to_track and id == id_to_track:
            color = (0,255,0)
            write_to_csv("john_doe.csv",box,t_classes[i])
        elif id_to_track:
            color = (0,0,40)
            bbox_thickness = 1
            draw_trajectory = False
        
        label = '{}{:d}'.format("", id)
        label_fscale = 2
        if categories:
            # If categories are avaible display them along the tracked object
            label = categories[t_classes[i]]
            label_fscale = 1.5
        t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, label_fscale , 2)[0]
        cv2.rectangle(img,(x1, y1),(x2,y2),color,bbox_thickness)
        cv2.rectangle(img,(x1, y1),(x1+t_size[0]+3,y1+t_size[1]+4), color,-1)
        cv2.putText(img,label,(x1,y1+t_size[1]+4), cv2.FONT_HERSHEY_PLAIN, label_fscale, [255,255,255], 2)
        
        if draw_trajectory and trajectories and len(trajectories[id])==2:
            cv2.line(mask,trajectories[id][0],trajectories[id][1],inc_int(color),3)
            
        if trajectories:
            img = overlay_on_image(img,mask)
        
    return img


if __name__ == '__main__':
    for i in range(82):
        print(compute_color_for_labels(i))
