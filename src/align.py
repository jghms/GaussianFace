import dlib

class GFAlign:
    """
    Face detection and alignment class.

    This class provides the tools to preprocess images such they are in a
    similar format. This happens by extracting face landmarks and transforming
    faces for the landmarks to be in the same place.
    """
    def __init__(self, predictor):
        """
        Initializes the face detector and landmakr predictor.

        Args:
            predictor: Path to the landmark specification file.
        """
        self.detector = dlib.get_frontal_face_detector()
        # self.predictor = predictor = dlib.shape_predictor(predictor)

    def detectAll(self, rgbImg):
        """
        Detects all faces in an image using the Dlib frontal face detector.

        Args:
            rgbImg: Image in the RGB color space. (opencv standard is BGR)
        Returns:
            [rects]: Array of rectangles of detected faces in image coordinates.
        """
        try:
            return self.detector(rgbImg, 1)
        except Exception as e:
            print("Warning: {}".format(e))
            return []

    def rect2BoundingBox(self, rect):
        """
        Converts the dlib rectangles to opencv bounding boxes.

        Args:
            rect: Dlib rectangle
        Returns:
            (x, y, w, h): Tuple-style bounding box like OpenCV would return it.
        """
    	x = rect.left()
    	y = rect.top()
    	w = rect.right() - x
    	h = rect.bottom() - y

    	# return a tuple of (x, y, w, h)
    	return (x, y, w, h)
