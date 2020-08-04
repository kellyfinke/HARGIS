########################################################################
#
# File:   thresholder.py
# Modified from Matt Zucker's pick_points.py by Kelly Finke and Quinn Freedman
# Date:   February 2019
#
# This file allows the user to easily choose the optimal parameters for the
# given input and saves the parameters for use in later analyses
########################################################################


from __future__ import print_function # Do Python 3-style printing
import cv2
import numpy as np
import sys
import pickle
import time
from skimage.morphology import skeletonize

# Max dims of image on screen, change these to get bigger/smaller window
#TODO for laptop 1024 , 600
MAX_WIDTH = 2048
MAX_HEIGHT = 900

# Factors for zooming in/out
ZOOM_IN_FACTOR = np.sqrt(2)
ZOOM_OUT_FACTOR = np.sqrt(2)/2

###############################################################################
# Helper Methods
###############################################################################

#def convert_and_scale(image):
#    """Converts image to u8 grayscale and scales it so that it so that its
#    maximum value is 255 and its min is 0
#    """
#    gr_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#    gr_min = np.min(gr_image)
#    gr_range = np.max(gr_image) - gr_min
#    gr_scaled = ((gr_image - gr_min) * 1.0 / gr_range * 255).astype("uint8")
#    return gr_scaled

def normalize_img(image, params):
    # Increases image contrast (for easier separation of foreground/background)
    # by zeroing out all pixels with intensity outside of the range
    # min_sat-max_sat and normalizing the remaining intensities

    #TODO ensure image is grayscale
    _, high_pass = cv2.threshold(image, params.min_sat, True, cv2.THRESH_BINARY)
    _, lo_pass = cv2.threshold(image, params.max_sat, True, cv2.THRESH_BINARY_INV)

    mask = high_pass * lo_pass
    norm = cv2.normalize(image, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, mask=mask)

    #assert(sum(norm[mask]) == 0) #if not, then mask isnt being set to zero
    #cv2.imwrite("normalizedd.png", norm)
    return norm


def get_aspect_ratio(contour):
    # Gets the height of the contour devited by its width
    _, _, w, h = cv2.boundingRect(contour)
    return float(h)/float(w)

def is_edge_blob(contour,shape):
    # Returns True if contour is boarding the edge of the image
    x,y,w,h = cv2.boundingRect(contour)
    return x <= 0 or y <= 0 or x + w >= shape[1] or y + h >= shape[0]


def prune_skeleton(skele):
    # Removes extra branches of skeletonized root, keeping only the most direct
    # path from the highest tip to the lowest tip

    test_int = skele.astype(np.uint8)

    int_skel = np.zeros_like(skele,dtype=np.uint8)
    int_skel[skele] = 1

    #assert(test_int == int_skel)

    # See docs #TODO for explaination #TODO
    kernel = np.ones((3,3))
    conv_skel = cv2.filter2D(int_skel, -1, kernel=kernel)

    # Same as skele but instead of all 1's:
    # 0=background, 2=tip, 3=part of branch, 4+ =junction (branches meeting)
    neighbor_skel = np.zeros_like(skele,dtype=np.uint8)
    neighbor_skel[skele] = conv_skel[skele]

    tips = np.where(neighbor_skel == 2)
    tips = zip(tips[0], tips[1])
    if len(tips) <1:
        # if skeleton is somehow a circle?
        return skele

    #TODO ensure first and last tips are always highest and lowest (argmax, argmin)
    #TODO sort by y value?

    # Save highest and lowest tips and remove from list
    start = tips[0]
    end = tips[-1]
    bad_tips = tips[1:-1]

    # Remove all branches that connect to tips that aren't start or end
    for tip in bad_tips:
        x = tip[0]
        y = tip[1]
        neighbor_skel[x,y] = 0

        # Remove branch connecting tip and nearest junction
        next = np.where(neighbor_skel[x-1:x+2, y-1:y+2] == 3)
        while len(next[0]) == 1:
            x, y = x + next[0][0] - 1, y + next[1][0] - 1
            neighbor_skel[x,y] = 0
            next = np.where(neighbor_skel[x-1:x+2, y-1:y+2] == 3)

    junctions = np.where(neighbor_skel > 3)
    junctions = zip(junctions[0], junctions[1])

    # Only keep junction points bordering 3's (remaining branches connecting
    # start and end). This occasionally leads to gaps in the skeleton but they
    # are small and are accounted for when calculating length
    for jun in junctions:
        x = jun[0]
        y = jun[1]
        threes = np.where(neighbor_skel[x-1:x+2, y-1:y+2] == 3)
        if len(threes[0]) == 0:
            neighbor_skel[x,y] = 0

    #TODO bubbles?
    return neighbor_skel >0


def apply_mask(image, params, to_skeletonize=False):
    """
    Takes an image and some parameters and applies image processing
    operations to detect objects in the image.

    If image_output_dir is specified, versions of the image will be saved
    to that directory before and after applying operations

    Returns a list of all the base points of the objects in the frame
    along with the marked-up image.

    :param image_output_dir: (str, int)
    :param image: np.ndarray
    :param params: ThresholdParameters
    :return: (np.ndarray, np.ndarray)
    """
    threshold = params.threshold
    morph_x = params.morph_x
    morph_y = params.morph_y
    min_area = params.min_contour_area
    max_area = params.max_contour_area
    contour_aspect = params.min_contour_aspect
    erased_mask = params.erased_mask

    _, mask = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY)
    if morph_x and morph_y:
        B = cv2.getStructuringElement(cv2.MORPH_RECT, (morph_x, morph_y))
        morphed_mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, B)
    else:
        morphed_mask = mask

    #abc = np.copy(morphed_mask)

    morphed_mask *= erased_mask
    #if image_output_dir:
    #    cv2.imwrite("%s/originals/frame_%d.png" % image_output_dir, image)
    #    cv2.imwrite("%s/thresholded/frame_%d.png" % image_output_dir, mask)
    #    cv2.imwrite("%s/after_morph/frame_%d.png" % image_output_dir, morphed_mask)

    _, contours, _ = cv2.findContours(morphed_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    contours = [c for c in contours if cv2.contourArea(c) > min_area
                and cv2.contourArea(c) < max_area
                and get_aspect_ratio(c) > contour_aspect
                and not is_edge_blob(c,image.shape)]

    bottom_points = [tuple(cnt[cnt[:, :, 1].argmax()][0])
                     for cnt in contours]

    #cv2.imwrite("mask.png", morphed_mask)
    display = cv2.cvtColor(morphed_mask, cv2.COLOR_GRAY2BGR)
    display = cv2.drawContours(display, contours, -1, (0, 255, 0), -1)
    #cv2.imwrite("threshed.png", display)


    if to_skeletonize: #TODO add toggle
        #skeletonize
        timea = time.time()
        to_skele = np.zeros_like(morphed_mask)
        skele = np.zeros_like(morphed_mask, dtype=bool)
        to_skele = cv2.drawContours(to_skele, contours, -1, 1, -1)
        kernel = np.ones((9,9))


        skeles = []
        for contour in contours:
            rect =cv2.boundingRect(contour)
            x_start = rect[0]
            x_end = rect[0]+rect[2]
            y_start = rect[1]
            y_end = rect[1]+rect[3]
            #cv2.imwrite("issue_plant.png", to_skele[y_start:y_end,x_start:x_end]*200)
            to_skele[y_start:y_end,x_start:x_end] = cv2.dilate(to_skele[y_start:y_end,x_start:x_end],kernel, iterations=1)
            #skele[y_start:y_end,x_start:x_end]= skeletonize(to_skele[y_start:y_end,x_start:x_end])

            temp_skele = skeletonize(to_skele[y_start:y_end,x_start:x_end])
            pruned_skele = prune_skeleton(temp_skele)
            #empty= np.zeros_like(skele, dtype=bool)
            #empty[y_start:y_end,x_start:x_end] = pruned_skele
            skeles.append((pruned_skele, rect))

        timeb = time.time()
        #print("skele time: ", timeb-timea)
        display[skele] = (0,0,255)
        return bottom_points, display, skeles

    """
    #naieve skeleton
    timec = time.time()
    pts = []
    lengths = []
    for contour in contours:
        lengths.append([])
        contour = sorted(contour[:,0,:], key=lambda tup: tup[1])
        for y_val in range(contour[-1][1], contour[0][1], -1):
            pts = []
            while(contour[-1][1] == y_val):
                pts.append(contour[-1][0])
                contour.pop()
            #print("y: ",y_val)
            #print("x's: ",pts)
            if(len(pts)>=2):
                x_center= (min(pts)+max(pts))/2
                lengths[-1].append((x_center,y_val))
            #print("point: ",lengths[-1][-1])
    timed = time.time()
    print("line time: ", timed -timec)
    for length in lengths:
        for pt in length:
            #print(pt)
            #print("point to draw: ", pt[0], " ", pt[1])
            display[pt[1], pt[0]] = (0,0,255)

    #moments skeleton
    timee = time.time()
    pts = []
    lengths = []
    just_contours = np.zeros((display.shape[0], display.shape[1]), dtype=np.float64)
    just_contours = cv2.drawContours(just_contours, contours, -1, 255, -1)
    for contour in contours:
        pts = []
        rect =cv2.boundingRect(contour)
        #print(rect)
        x_start = rect[0]
        x_end = rect[0]+rect[2]
        y_start = rect[1]
        y_end = rect[1]+rect[3]
        #display = cv2.rectangle(display, (x_start, y_start), (x_end,y_end), (0,0,255), 3)
        #just_contours = cv2.rectangle(just_contours, (x_start, y_start), (x_end,y_end), 255, 3)
        #cv2.imwrite("boundingrect.jpg", just_contours)
        #cv2.imwrite("specific_rct.jpg", just_contours[y_start:y_end,x_start:x_end])
        kernel = np.ones((5,5),np.float64)
        just_contours[y_start:y_end,x_start:x_end] = cv2.dilate(just_contours[y_start:y_end,x_start:x_end],kernel, iterations=2)
        #cv2.imwrite("dilated_rct.jpg", just_contours[y_start:y_end,x_start:x_end])
        for y_val in range(rect[1], rect[1]+rect[3]):
            #print(np.arange(x_start,x_end,1).shape)
            #print(just_contours[y_val,x_start:x_end].shape)
            #print(np.arange(x_start,x_end,1))

            #print(just_contours[y_val,x_start:x_end])

            center = np.average(np.arange(x_start,x_end,1), weights=just_contours[y_val,x_start:x_end])
            #print(center)
            pts.append((int(center),y_val))
            #print(y_val)
            #print(just_contours[y_val:y_val+2,x_start:x_end,])
            #m=cv2.moments(just_contours[x_start:x_end,y_val:y_val+1])
            #print(m)
            #if m["m00"] != 0:
            #    print("****")
            #    cx=int(m["m01"]/m["m00"])
            #    print(cx)
        lengths.append(pts)

    timef = time.time()
    print("moment time: ", timef -timee)
    for length in lengths:
        print(len(length))
        for pt in length:
            #print(pt)
            #print("point to draw: ", pt[0], " ", pt[1])
            display[pt[1], pt[0]] = (0,0,255)
    """

    #example_rect = cv2.boundingRect(contours[0])
    #x_start = example_rect[0]
    #x_end = example_rect[0]+example_rect[2]
    #y_start = example_rect[1]
    #y_end = example_rect[1]+example_rect[3]
    #cv2.imwrite("averaging_example_end.jpg", display[y_start:y_end,x_start:x_end])



    #TODO different color for mask in display
    #print(display.shape)
    #print(erased_mask.shape)
    #np.broadcast_to(erased_mask, (erased_mask.shape[0], erased_mask.shape[1], 3))
    #print(erased_mask.shape)

    #for i in range(erased_mask.shape[0]):
    #    for j in range(erased_mask.shape[1]):

    #        if erased_mask[i,j] == 0:
    #            display[i,j,:] = [0,0,255]
    #threeD_erased = np.dstack((erased_mask, erased_mask, erased_mask))
    #print(threeD_erased.shape)
    #print(display[erased_mask])

    #display[erased_mask,:] = (0,0,255)

    """
    blob_detector_params = cv2.SimpleBlobDetector_Params()
    blob_detector_params.filterByArea = True
    blob_detector_params.minArea = min_area
    blob_detector_params.maxArea = max_area

    blob_detector = cv2.SimpleBlobDetector_create(blob_detector_params)
    """

    return bottom_points, display


###############################################################################
# Classes
###############################################################################

class ThresholdParams:
    """
    A data class to store the parameters for the object detection process
    """
    def __init__(self,
                 max_sat=255,
                 min_sat=0,
                 threshold=65,
                 morph_x=1,
                 morph_y=3,
                 min_contour_area=1000,
                 max_contour_area=12000,
                 min_contour_aspect=1.5,
                 erased_mask=1 #TODO is this good #TODO no (not for new inputs)
                 ):
        self.max_sat = max_sat
        self.min_sat = min_sat
        self.threshold = threshold
        self.morph_x = morph_x
        self.morph_y = morph_y
        self.min_contour_area = min_contour_area
        self.max_contour_area = max_contour_area
        self.min_contour_aspect = min_contour_aspect
        self.erased_mask = erased_mask


class Thresholder(object):
    """
    Class for dynamically observing, adjusting, and saving threshold parameters #TODO new name?
    """


    def __init__(self, image, params_filename):
        # Takes a single image and a .pkl file of parameters

        # Save a copy of the original image for display
        self.orig_image = image
        self.image = image

        #print("orig ", image.shape)
        # Stash dimensions of original image
        h, w = image.shape[:2]
        self.orig_image_size = (w, h)

        # Stash the filename where we will save parameters
        self.params_filename = params_filename

        # Convert to greyscale for thresholding
        self.gr_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        #print("grey ", self.gr_image.shape)
        # Blur to reduce effect of noise
        self.gr_image = cv2.GaussianBlur(self.gr_image, (9,9),0)
        #print("blur", self.gr_image.shape)
        #cv2.imwrite("gr_only.png", self.gr_image)

        # See if we can retrieve presets from prexisting file
        # Otherwise, use default parameters
        try:
            with open(self.params_filename, "rb") as f:
                self.params = pickle.load(f)
            #print(self.params_filename)
            #print(self.params)
        except:
            self.params = ThresholdParams(erased_mask=np.ones_like(self.gr_image))
            #print('erased ', np.ones_like(self.gr_image).shape)
        #print("here")

        self.gr_image_scaled = normalize_img(self.gr_image, self.params)
        _, self.mask_image = apply_mask(self.gr_image_scaled, self.params)
        self.update_mask = False

        image_area = self.orig_image_size[0] * self.orig_image_size[1]
        self.trackbar_window = 'Parameters'
        cv2.namedWindow(self.trackbar_window)
        cv2.resizeWindow(self.trackbar_window, 600, 280)
        cv2.createTrackbar("Max Saturation", self.trackbar_window,
                           self.params.max_sat, 255, self.on_sat_change)
        cv2.createTrackbar("Min Saturation", self.trackbar_window,
                           self.params.min_sat, 255, self.on_sat_change) #TODO what if min>max
        cv2.createTrackbar("threshold", self.trackbar_window,
                           self.params.threshold, 255, self.on_slider_change)
        cv2.createTrackbar("morph_x", self.trackbar_window,
                           self.params.morph_x, 25, self.on_slider_change)
        cv2.createTrackbar("morph_y", self.trackbar_window,
                           self.params.morph_y, 25, self.on_slider_change)
        cv2.createTrackbar("Min Size", self.trackbar_window,
                           self.params.min_contour_area, 5000,
                           self.on_slider_change)
        cv2.createTrackbar("Max Size", self.trackbar_window,
                           self.params.max_contour_area, 20000,
                           self.on_slider_change)
        cv2.createTrackbar("contour_aspect", self.trackbar_window,
                           int(self.params.min_contour_aspect * 100), 300,
                           self.on_slider_change)

        # Determine minimum scaling factor of image
        fy = MAX_HEIGHT / float(h)
        fx = MAX_WIDTH / float(w)

        self.zoom = min(1.0, min(fx, fy))
        self.min_zoom = self.zoom

        # Set display size to scaled dimensions
        self.display_size = (int(round(w*self.zoom)), int(round(h*self.zoom)))

        # Initialize current scroll offset to 0, 0
        self.scroll_offset = np.array([0., 0.])

        # Transform the image to get ready for display
        self.transform_image()

        # Set up some internal flags for mouse interactions
        self.is_modified = False
        self.mouse_active = True
        self.dragging = False
        self.brush_size = 200

        # Create a window with OpenCV
        self.window = 'Visualize Mask'
        cv2.namedWindow(self.window)

        # Tell OpenCV we want mouse events
        cv2.setMouseCallback(self.window, self.mouse, self)

    # Create an affine transformation to display part or all of the
    # original image, then warp the image through the transform.
    def transform_image(self):

        # Get some short variable names to make a concise matrix definition
        z = self.zoom
        sx, sy = self.scroll_offset
        dw, dh = self.display_size
        ow, oh = self.orig_image_size

        # Define the forward transformation as a 3x3 matrix
        # in homogeneous coordinates (homography)
        self.M_forward = np.array([
            [z, 0, z*sx + 0.5*(dw - z*ow)],
            [0, z, z*sy + 0.5*(dh - z*oh)],
            [0, 0, 1]])

        # Define the inverse transformation as a 3x3 matrix
        self.M_inverse = np.linalg.inv(self.M_forward)

        # If zooming in, use box filter; otherwise use nearest neighbor.
        if self.zoom < 1:
            flags = cv2.INTER_AREA
        else:
            flags = cv2.INTER_NEAREST

        # Rescale and translate the image itself
        self.transformed_image = cv2.warpAffine(
            self.image, self.M_forward[:2],
            self.display_size, flags=flags)

        self.display_image = self.transformed_image.copy()

    def on_slider_change(self, _, data=None):
        self.is_modified = True

        #TODO change each time? or just self.params.threshold = ... etc.?

        self.params.threshold=cv2.getTrackbarPos("threshold", self.trackbar_window)
        self.params.morph_x=cv2.getTrackbarPos("morph_x", self.trackbar_window)
        self.params.morph_y=cv2.getTrackbarPos("morph_y", self.trackbar_window)
        self.params.min_contour_area=cv2.getTrackbarPos("Min Size", self.trackbar_window)
        self.params.max_contour_area=cv2.getTrackbarPos("Max Size", self.trackbar_window)
        self.params.min_contour_aspect=cv2.getTrackbarPos("contour_aspect", self.trackbar_window) / 100.0

        _, display = apply_mask(self.gr_image_scaled, self.params)
        self.update_mask = False

        self.image = display
        self.mask_image = display
        self.transform_image()

    def on_sat_change(self, _, data=None):
        self.is_modified = True
        self.update_mask = True

        self.params.max_sat=cv2.getTrackbarPos("Max Saturation", self.trackbar_window)
        self.params.min_sat=cv2.getTrackbarPos("Min Saturation", self.trackbar_window)

        norm = normalize_img(self.gr_image, self.params)
        #TODO change gr_image_scaled to gr_image

        self.image = norm
        self.gr_image_scaled = norm
        self.transform_image()



    # OpenCV mouse callback for window.

    def mouse(self, event, x, y, flag, param):

        # Do nothing if mouse ignored.
        if not self.mouse_active:
            return

        # Point in window coords
        _, _, w, h = cv2.getWindowImageRect('Visualize Mask')
        p_window = np.array((x, y))

        # Map through inverse of homography to get image coords
        #src = p_window.astype(float).reshape(-1, 1, 2)
        #dst = cv2.perspectiveTransform(src, self.M_inverse)
        #p_image = dst.reshape(-1, 2)

        # Was left mouse button depressed?
        if event == cv2.EVENT_LBUTTONDOWN:

            # Modify mouse flags
            self.dragging = True
            self.mouse_point = p_window

            #self.is_modified = True
            #self.cur_index = idx
            y_scale = int(1.0*y/h * len(self.params.erased_mask))
            x_scale = int(1.0*x/w * len(self.params.erased_mask[0]))
            #self.erased_mask[y_scale-self.brush_size:y_scale+self.brush_size, x_scale-self.brush_size:x_scale+self.brush_size] = 0
            self.clicked_point = (x_scale, y_scale)
            """
            # Update the display
            _, display = apply_mask(self.gr_image_scaled, self.params, self.erased_mask)

            self.image = display
            self.transform_image()
            """

        elif self.dragging and event == cv2.EVENT_MOUSEMOVE:
            """
            # Move the current point by the mouse displacement
            mouse_displacement = (p_window - self.mouse_point) / self.zoom
            #self.orig_points[self.cur_index] += mouse_displacement

            # Update flags
            self.mouse_point = p_window
            self.is_modified = True
            y_scale = int(1.0*y/h * len(self.erased_mask))
            x_scale = int(1.0*x/w * len(self.erased_mask[0]))
            self.erased_mask[y_scale-self.brush_size:y_scale+self.brush_size, x_scale-self.brush_size:x_scale+self.brush_size] = 0

            # Update the display
            _, display = apply_mask(self.gr_image_scaled, self.params, self.erased_mask)

            self.image = display
            self.transform_image()
            """
        elif self.dragging and event == cv2.EVENT_LBUTTONUP:
            # Left mouse button up, we are no longer dragging.
            self.dragging = False
            self.is_modified = True

            y_scale = int(1.0*y/h * len(self.params.erased_mask))
            x_scale = int(1.0*x/w * len(self.params.erased_mask[0]))
            #print("here: ", self.clicked_point, " ", x_scale,", ",y_scale)
            self.params.erased_mask[max(0,min(y_scale,self.clicked_point[1])):max(y_scale,self.clicked_point[1]), max(0,min(x_scale,self.clicked_point[0])):max(x_scale,self.clicked_point[0])] = 0

            _, display = apply_mask(self.gr_image_scaled, self.params)
            self.update_mask = False

            self.image = display
            self.transform_image()


    # Reset the view
    def reset_view(self):
        self.scroll_offset = np.array([0., 0.])
        self.zoom = self.min_zoom
        self.transform_image()

    # Show a bunch of strings on some grayed-out text.
    def show_strings(self, strlist):

        # Darken image by scaling down to 25% intensity
        dimmed_image = self.display_image/4

        # For each line in the list
        for i, line in enumerate(strlist):

            # For each pair of strings in the line
            for j, text in enumerate(line):

                # Display the text in white.
                cv2.putText(dimmed_image, text, (20+150*j, i*20+40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (255, 255, 255), 1, cv2.LINE_AA)

        # Show the dimmed image
        cv2.imshow(self.window, dimmed_image)

    # Show the help screen
    def help(self):

        # List of pairs of strings for display
        helpstr = [
            ('Use the slider bars (other window) to find', ''),
            ('the best parameters to mask your image.', ''),
            ('', ''),
            ('Keys:', ''),
            ('', ''),
            ('+ or -', 'zoom image'),
            ('W', 'scroll up'),
            ('S', 'scroll down'),
            ('A', 'scroll left'),
            ('D', 'scroll right'),
            ('', ''),
            ('O', 'display original image'),
            ('M', 'display masked image'),
            ('', '(automatic when parameters change)'),
            ('K', 'display skeletonized image'),
            ('', ''),
            ('SPACE', 'reset view'),
            ('ESC', 'exit and save parameters'),
            ('', ''),
            ('?', 'show this help screen'),
            ('', ''),
            ('Press any key to dismiss this help screen.', '')
        ]

        # Disable mouse
        self.mouse_active = False

        # Show text
        self.show_strings(helpstr)

        # Wait for any key
        while cv2.waitKey(5) < 0:
            pass

        # Re-enable mouse
        self.mouse_active = True

    # Show the screen to prompt save before quitting
    def prompt_quit(self):
        if not self.params_filename:
            return True
        # Quit immediately if no parameters modified.
        if not self.is_modified:
            return True

        # Compose text to show

        prompt = 'Do you want to save these parameters?'

        strings = [(prompt, ''),
                   ('', ''),
                   ('Y', 'Yes, save to ' + self.params_filename),
                   ('N', 'No, move on but don\'t save'),
                   ('', ''),
                   ('Any other key resumes interactive mode.', '')]

        # Disable mouse
        self.mouse_active = False

        # Show text
        self.show_strings(strings)

        # Wait for any key
        while True:
            key = cv2.waitKey(5)
            if key >= 0:
                break

        # See what key was
        c = chr(key & 0xff).lower()

        if c == 'y':
            print('Saving to {}'.format(self.params_filename))
            with open(self.params_filename, 'wb') as f:
                pickle.dump(self.params, f)
            done = True
        elif c == 'n':
            print('Quitting without saving!')
            done = True
        else:
            done = False

        # Re-enable mouse
        self.mouse_active = True

        # Return True if time to quit
        return done

    # Relative scrolling of window
    def scroll_window(self, sx, sy):
        self.scroll_offset += (sx, sy)
        self.transform_image()

    # Zoom image
    def zoom_image(self, factor):
        self.zoom = min(4.0, max(self.min_zoom, self.zoom * factor))
        self.transform_image()

    def set_display(self, image):
        self.image = image
        self.transform_image()

    # The main loop for this object
    def run(self):

        # Start by showing help screen
        self.help()

        # Forever loop
        while True:

            # Show display image
            cv2.imshow(self.window, self.display_image)

            # Get the key
            key = cv2.waitKey(5)

            if key < 0:
                # No key pressed, just loop
                continue

            # Get the ASCII character from the integer keycode
            c = chr(key & 0xff)

            # Figure out how much to scroll based on zoom
            scroll_amount = 32.0*self.min_zoom / self.zoom

            # Scroll more if CAPS
            if c.isupper():
                c = c.lower()
                scroll_amount *= 8

            # Handle keys
            if c in '+=':
                self.zoom_image(ZOOM_IN_FACTOR)
            elif c == '-':
                self.zoom_image(ZOOM_OUT_FACTOR)
            elif c == 'a':
                self.scroll_window(scroll_amount, 0)
            elif c == 'd':
                self.scroll_window(-scroll_amount, 0)
            elif c == 'w':
                self.scroll_window(0, scroll_amount)
            elif c == 's':
                self.scroll_window(0, -scroll_amount)
            elif c == 'o':
                self.set_display(self.orig_image)
            elif c == 'm':
                if self.update_mask:
                    _, self.mask_image = apply_mask(self.gr_image_scaled, self.params)
                    self.update_mask = False
                self.set_display(self.mask_image)
            elif c == "c":
                self.set_display(self.gr_image_scaled)
            elif c == ' ':
                self.reset_view()
            elif c == '?':
                self.help()
            elif key == 27:
                if self.prompt_quit():
                    break


######################################################################
# Our main function

def main():

    # Collect command-line args
    if len(sys.argv) != 3:
        print('usage: python', sys.argv[0], 'IMAGEFILE OUTPUTFILE')
        sys.exit(1)

    image_file = sys.argv[1]
    params_filename = sys.argv[2]

    # Load image file
    orig = cv2.imread(image_file)

    if orig is None:
        print('image not found:', image_file)
        sys.exit(1)

    # Create a Thresholder object and run it
    p = Thresholder(orig, params_filename)
    p.run()


if __name__ == '__main__':
    main()
