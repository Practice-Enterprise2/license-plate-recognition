import json
import os
import xml.etree.ElementTree as xml

import numpy
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.structures import BoxMode


def GetImageData(ImageFolder):
    ImageData = []
    # String containing the JSON file's location.
    FileJSON = os.path.join(ImageFolder, ".json")
    # Loading json file containing information about the images.
    with open(FileJSON) as File:
        ImageInformationJSON = json.load(File)
    DatasetImageDictionaries = []
    # Looping over each image using an index obtained enumerating the values of the previus read/loaded JSON file.
    for ImageID, Values in enumerate(ImageInformationJSON.values()):
        # Dictionary for containing individual image data.
        ImageData = {}
        # String containing the file path by concatenating the passed directory & the filename value.
        Filename = os.path.join(ImageFolder, Values["filename"])
        # Reading the height & width of the image. Currently using dummby values!
        Height, Width = cv2.imread(Filename).shape[:2]
        # Adding fields with image data.
        ImageData["file_name"] = Filename
        ImageData["image_id"] = ImageID
        ImageData["height"] = Height
        ImageData["width"] = Width
        # Reading the region values & putting them in a seperate object for later use.
        RegionAnnotations = Values["regions"]
        # Container for later on holding the box objects.
        BoxObjects = []
        # Loop for goining through the individual annotations. Underscore symbol is for ignoring the first
        for Annotation in RegionAnnotations:
            Annotation = Annotation["shape_attributes"]
            # Saving all X & Y point of the attributes.
            AllPointsX = Annotation["all_points_x"]
            AllPointsY = Annotation["all_points_y"]
            # Loops for creating polynomials/contours.
            OutlineObject = [(X+0.5, Y+0.5)
                             for X, Y in zip(AllPointsX, AllPointsY)]
            OutlineObject = [
                P for X in OutlineObject for P in X]
            # Saving the specific classes regarding the different region's attribute.
            RegionAttributesClass = Annotation["region_attributes"]["class"]
            if "car" in RegionAttributesClass:
                CategoryID = 0
            elif "license plate" in RegionAttributesClass:
                CategoryID = 1
            else:
                CategoryID = 2
            # Object for hold the box values.
            BoxObject = {
                "bbox": [numpy.min(AllPointsX), numpy.min(AllPointsY), numpy.max(AllPointsX), numpy.max(AllPointsY)],
                "bbox_mode": BoxMode.XYXY_ABS,
                "segmentation": OutlineObject,
                "category_id": CategoryID
            }
            # Adding current box object to the list.
            BoxObjects.append(BoxObject)
        # Saving the list of box objects as annotations.
        ImageData["annotations"] = BoxObjects
        # Adding the current image data to the list.
        DatasetImageDictionaries.append(ImageData)
    return DatasetImageDictionaries


# Loop for registering the datasets.
for D in ["training", "value"]:
    DatasetCatalog.register(
        "license_plate_"+D, lambda D=D: GetImageData("/images/"+D))
    MetadataCatalog.get(
        "license_plate_"+D).thing_classes = ["car", "license plate"]
# Getting & saving the metadata.
Metadata = MetadataCatalog.get("license_plate_training")
Dictionaries = GetImageData("/images/")
