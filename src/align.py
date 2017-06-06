import dlib
import numpy as np

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
        self.predictor = None
        if predictor is not None:
            self.predictor = dlib.shape_predictor(predictor)

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

    def shape_to_np(self, shape, dtype="int"):
        """
        Converts the extracted dlib shapes to numpy arrays.

        Args:
            shape: Landmark coordinates as returned by dlib.

        Returns:
            np.array: 68x2 Numpy array containing the coordinates.
        """
    	coords = np.zeros((68, 2), dtype=dtype)
    	for i in range(0, 68):
    		coords[i] = (shape.part(i).x, shape.part(i).y)

    	return coords

    def landmarkPrediction(self, rgbImg, rect):
        """
        Predicts the facial landmarks in a given area of an image.

        This requires the predictor variable to be initialized when constructing
        the object.

        Raises:
            if self.predictor is None raises an exception.

        Args:
            rgbImg: Image containing the face at coords of rect.
            rect: Rectangle surrounding the face, this is where the algorithm
                looks at.

        Returns:
            np.array: numpy array of size 68x2 containing the landmark coordinates.
        """

        if self.predictor == None:
            raise Error("""
                GFAlign predictor was not initialized. Pass the filename of the
                landmark definitions during constructing the object.
            """)

        shape = self.predictor(rgbImg, rect)
        shape = self.shape_to_np(shape)

        return shape
