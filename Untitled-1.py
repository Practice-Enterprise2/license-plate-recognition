import json
import os
from detectron2.structures import BoxMode
import numpy


def GetImageDataset(ImageDirectory):
    # Generating a string containing the JSON file's location.
    FileJSON = os.path.join(ImageDirectory, ".json")
    # Opening & loading the file containing the region data.
    with open(FileJSON) as File:
        ImageAnnotations = json.load(File)
    #  A collection where each dictionary contains information about a single image.
    DatasetDictionaries = []
    # Looping through the entire JSON file.
    for Index, Value in enumerate(ImageAnnotations.values()):
        # Container for holding the records, fields of the dataset.
        Record = {}
        # Generating a string containing the file's location.
        Filename = os.path.join(ImageDirectory, Value["filename"])
        # Reading of the height & width regarding the specific image.
        Height, Width = 0, 0  # cv2.imread(Filename).shape[:2]
        Record["file_name"] = Filename
        Record["image_id"] = Index
        Record["height"] = Height
        Record["width"] = Width
        # Saving data regarding the regions of the image.
        Annotations = Value["regions"]
        # Container of holding objects later on.
        Objects = []
        for _, Annotation in Annotations.items():
            # If false statement.
            assert not Annotation["region_attributes"]
            # Saving shape attributes.
            Annotation = Annotation["shape_attributes"]
            # Saving X points.
            PointsX = Annotation["all_points_x"]
            # Saving Y points.
            PointsY = Annotation["all_points_y"]
            # Generating poly based. (X & Y coordinates)
            Poly = [(X + 0.5, Y + 0.5) for X, Y in zip(PointsX, PointsY)]
            Poly = [P for X in Poly for P in X]
            # Generating an object with a bbox & poly.
            Object = {
                # Bounding box for object in question.
                "bbox": [numpy.min(PointsX), numpy.min(PointsY), numpy.max(PointsX), numpy.max(PointsY)],
                # Specific mode for the bounding box.
                "bbox_mode": BoxMode.XYXY_ABS,
                "segmentation": [Poly],
                "category_id": 0
            }
            # Appending the current object to the objects container.
            Objects.append(Object)
        # Adding of generated objects as annotations.
        Record["annotations"] = Objects
        # Adding records to the dataset dictionary.
        DatasetDictionaries.append(Record)
    return DatasetDictionaries
