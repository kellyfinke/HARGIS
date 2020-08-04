from __future__ import print_function
import argparse
import cv2
import csv
import sys
import numpy as np
import time
import os
import errno
import pandas as pd
from tqdm import tqdm
from matplotlib import pyplot as plt
from thresholder import normalize_img, apply_mask, ThresholdParams, Thresholder

TARGET_FPS = 10
MAX_DELTA = 15

# Max dims of image on screen, change these to get bigger/smaller window
MAX_WIDTH = 1500
MAX_HEIGHT = 800

def mkdirs_if_not_exists(path):
    """Makes all directories in the path if they do not already exist.
    """
    try:
        os.makedirs(path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

 #TODO come up with scale
def listdir_nohidden(path):
    for file in os.listdir(path):
        if not file.startswith('.'):
            yield file

def main(folder, output_name=None, param_file=None, skeletonize=True):

    #Windows uses \ mac uses / #TODO
    slash = "/"
    if os.name == "nt":
        slash = "\\"
    #open images
    img_names = listdir_nohidden(folder)
    img_names = sorted(img_names) #TODO what if in date order

    first_frame = cv2.imread(folder+ slash+ img_names[0])

    #if first_frame == None:
    #    img_names.pop(0)
    #    first_frame = cv2.imread(folder+ "/"+ img_names[0]) #TODO whats up with this

    #param file to pass to thresholder (default if none given: folder name + .pkl)
    mkdirs_if_not_exists("parameters")
    mkdirs_if_not_exists("output")
    if not param_file:
        param_file = "parameters"+slash + folder.split(slash)[-1] + ".pkl" #TODO check that file doesn't end with '/'
        if param_file == "parameters"+slash+".pkl":
            #TODO better
            print("Please ensure that your folder name input is not empty and does not end with '/' or '\\'")

    # Start the parameter picker using just the first frame of the video
    # When they are finished picking parameters, we will continue
    param_picker = Thresholder(first_frame, param_file)
    param_picker.run()
    threshold_params = param_picker.params
    cv2.destroyAllWindows()

    root_tips = []
    #frame = 0
    print("Processing...")
    timing = {'gr': 0.0, "norm":0.0, "mask": 0.0, "match points": 0.0, "draw and track": 0.0, "skel where":0.0, "high low": 0.0, "rest of skel": 0.0, "draw start lines": 0.0, "all skel": 0.0}
    for frame_count, frame_file in enumerate(tqdm(img_names)):
        frame_start = time.time()
        # Get the frame.
        #ok, frame = capture.read(frame)
        frame = cv2.imread(folder+ slash+ frame_file)

        #print(frame_file)
        #print(frame.shape)
        # Bail if none.
        #if not ok or frame is None:
        #    break
        h = frame.shape[0]
        w = frame.shape[1]

        #######################################################################
        # Ensure images are the same size
        #TODO make method and apply before init call in thresholder
        if frame.shape != threshold_params.erased_mask.shape:
            dif_0 = h - threshold_params.erased_mask.shape[0]
            dif_1 = w - threshold_params.erased_mask.shape[1]

            if dif_0 == 1:
                threshold_params.erased_mask.append(threshold_params.erased_mask[-1,:])
            elif dif_0 == -1:
                threshold_params.erased_mask = threshold_params.erased_mask[0:-1,:]

            if dif_1 == 1:
                threshold_params.erased_mask.append(threshold_params.erased_mask[:,-1])
            elif dif_1 == -1:
                threshold_params.erased_mask = threshold_params.erased_mask[:,0:-1]

            if np.abs(dif_0) > 1 or np.abs(dif_1) > 1:
                print("Images are different sizes. Please ensure all images have the same dimensions")
        #######################################################################
        a = time.time()
        grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) #convert_and_scale(image)
        grayscale_blur = cv2.GaussianBlur(grayscale, (9,9), 0)
        b = time.time()
        gr_norm = normalize_img(grayscale_blur, threshold_params)
        c = time.time()
        if skeletonize:
            bottom_points, display_image, skeles = apply_mask(gr_norm, threshold_params, to_skeletonize=True)
        else:
            bottom_points, display_image = apply_mask(gr_norm, threshold_params)
        d = time.time()
        timing["gr"] += b-a
        timing["norm"] += c-b
        timing["mask"] += d-c
        #for b in bottom_points:
        #    cv2.drawMarker(display_image, b, (255,0,0),
        #                cv2.MARKER_CROSS, 5, 3)

        if len(root_tips) == 0: #Initialize important lists if first frame
            # Initialize the root_tips to just the plants that we see in the
            # first frame of the video
            root_tips = [[p] for p in bottom_points]
            #scale based average size of plants #TODO do better
            #TODO
            #show starting point in video
            if skeletonize:
                start_lines = [[(p[0]-MAX_DELTA,p[1]-15), (p[0]+MAX_DELTA,p[1]-15)] for p in bottom_points]
            else:
                start_lines = [[(p[0]-h/35,p[1]), (p[0]+h/35,p[1])] for p in bottom_points]
            #place for tracking % of frames plants can be tracked
            lost_points = np.zeros(len(root_tips))
            #skeleton sizes
            skel_sizes = [[] for p in root_tips]
            #convert between returned shapes and stored shapes
            index_conversion = np.arange(len(root_tips), dtype=int)
        else:
            # Track points
            e = time.time()
            for i, root in enumerate(root_tips):
                last_known_pos = root[-1]
                distances = np.sum((np.array(bottom_points) - last_known_pos) ** 2, axis=1)
                closest_point_index = np.argmin(distances)
                #print(i, " dis: ",distances[closest_point_index])
                if distances[closest_point_index] <= MAX_DELTA**2:


                    index_conversion[i] = closest_point_index

                    new_loc = bottom_points[closest_point_index]
                    #if the plant has been missing for a few frames, get it back and average new growth over lost frames
                    if len(root) < frame_count - 1:
                        old_x = root[-1][0]
                        old_y = root[-1][1]
                        dif_x = new_loc[0] - old_x
                        dif_y = new_loc[1] - old_y
                        frames_off = frame_count - len(root)
                        for i in range(1,frames_off + 1):
                            root.append((old_x + i/frames_off * dif_x, old_y + i/frames_off *dif_y))
                        assert(len(root) == frame_count)

                    else:
                        root.append(new_loc)
                else:

                    ######
                    # whats going on here
                    #print("too far")
                    #print(last_known_pos)
                    #cv2.imwrite("toofar"+str(i)+" "+str(frame_count)+".png", display_image[last_known_pos[1]-200:last_known_pos[1]+50,last_known_pos[0]-50:last_known_pos[0]+50])
                    #cv2.imwrite("toofar"+str(i)+" "+str(frame_count-1)+".png", last_display[last_known_pos[1]-200:last_known_pos[1]+50,last_known_pos[0]-50:last_known_pos[0]+50])
                    ######

                    lost_points[i] +=1
                    index_conversion[i] = -1
                f = time.time()
                timing["match points"] += f-e
            #    obj.append(last_known_pos)

        #for b in bottom_points:
        #    cv2.drawMarker(display_image, b, (0, 0, 255),
        #                cv2.MARKER_CROSS, 5, 3)
        colors = [(0,0,255),(255,0,0),(100,100,100),(255,255,0),(255,0,255),(255,0,150), (150,0,255), (150,150,150),(0,110,255)]
        g= time.time()
        for root, conv in enumerate(index_conversion):
            color = colors[root%len(colors)]
            if conv != -1:
                #cv2.drawMarker(display_image, bottom_points[conv], color,#(255, 255, 0),
                #            cv2.MARKER_CROSS, MAX_DELTA, 3)

                """
                if skeletonize:
                    skele_pts = np.where(skeles[conv] == True)
                    skele_indices = np.where(skele_pts[0] > start_lines[root][1][1])[0]
                    if len(skele_indices > 1):
                        skele_pts = zip(skele_pts[1], skele_pts[0])
                        skele_pts = skele_pts[skele_indices[0]:skele_indices[-1]]
                        sk_len = 0
                        for i in range(len(skele_pts) - 1):
                            p1 = tuple(skele_pts[i])
                            p2 = tuple(skele_pts[i + 1])
                            cv2.line(display_image, p1, p2, color,3)#(100, 100, 255), 3)
                            sk_len += np.sqrt(((p1[0]-p2[0])**2)+((p1[1]-p2[1])**2))
                        skel_sizes[root].append(sk_len)
                    else:
                        print("no skele points yet")

                """

                if skeletonize:
                    tz = time.time()
                    skele = skeles[conv][0]
                    rect = skeles[conv][1]
                    x_start = rect[0]
                    y_start = rect[1]

                    skele_pts = np.where(skele == True)
                    #print(bottom_points[conv])
                    ta = time.time()
                    low_index = np.where(skele_pts[0] + y_start > start_lines[root][1][1])[0]
                    high_index = np.where(skele_pts[0] + y_start < bottom_points[conv][1]-10)[0]
                    tb = time.time()
                    if len(low_index) > 0 and len(high_index) > 0 and low_index[0] < high_index[-1]:#len(skele_indices > 1):
                        skele_pts = zip(skele_pts[1], skele_pts[0])
                        skele_pts = skele_pts[low_index[0]:high_index[-1]]
                        sk_len = 0
                        for i in range(len(skele_pts) - 1):
                            p1 = (skele_pts[i][0] + x_start, skele_pts[i][1] + y_start)
                            p2 = (skele_pts[i + 1][0] + x_start, skele_pts[i + 1][1] + y_start)
                            cv2.line(display_image, p1, p2, color,3)#(100, 100, 255), 3)
                            sk_len += np.sqrt(((p1[0]-p2[0])**2)+((p1[1]-p2[1])**2))
                        if len(skel_sizes[root]) < frame_count - 1:
                            if len(skel_sizes[root]) == 0:
                                while len(skel_sizes[root]) < frame_count -1:
                                    skel_sizes[root].append(0)
                                skel_sizes[root].append(sk_len)
                            else:
                                old_size = skel_sizes[root][-1]
                                dif_size = sk_len - old_size
                                frames_off = frame_count - len(skel_sizes[root])
                                for i in range(1,frames_off + 1):
                                    skel_sizes[root].append(old_size + i/frames_off * dif_size)
                            assert(len(skel_sizes[root]) == frame_count)
                        else:

                            skel_sizes[root].append(sk_len)
                    else:
                        #print("no skele points yet ", root)
                        lost_points[root] +=1


                    tc = time.time()

                    timing["skel where"] += ta-tz
                    timing["high low"] += tb-ta
                    timing["rest of skel"] += tc-tb
                    timing["all skel"] += tc-tz
                else:
                    #TODO do correctly
                    print("hereeeeeee")
                    cv2.drawMarker(display_image, bottom_points[conv], color,#(255, 255, 0),
                                   cv2.MARKER_CROSS, h/35, 5)
                    for i, obj in enumerate(root_tips):
                        for i in range(len(obj) - 1):
                            p1 = tuple(obj[i])
                            p2 = tuple(obj[i + 1])
                            cv2.line(display_image, p1, p2, color,3)#(100, 100, 255), 3)


            tx = time.time()
            cv2.line(display_image, start_lines[root][0], start_lines[root][1], color,3)#(255,255,0),3)
            td =time.time()
            #if root == 1:
            #    print(display_image.shape)
            #    print(x_start)
            #    print(y_start-20)
            #    print(y_start+500)
            #    cv2.imwrite("tip_example_root_"+str(frame_count)+".png", display_image[y_start-20:y_start+470,x_start-20:x_start+100])
            #for i in range(len(root_tips[root]) - 1):
                #p1 = tuple(root_tips[root][i])
                #p2 = tuple(root_tips[root][i + 1])
                #cv2.line(display_image, p1, p2, (255, 255, 255), 2)
            q = time.time()
            timing["draw and track"] += q-g
            timing["draw start lines"] += td-tx

        """
        # Draw tracking points on video frame
        for i, point in enumerate(bottom_points):
            color = colors[i%len(colors)]
            cv2.drawMarker(display_image, point, color,#(255, 255, 0),
                           cv2.MARKER_CROSS, h/35, 5)
        for i,line in enumerate(start_lines):
            color = colors[i%len(colors)]
            cv2.line(display_image, line[0], line[1], color,3)#(255,255,0),3)
        for i, obj in enumerate(root_tips):
            color = colors[i%len(colors)]
            for i in range(len(obj) - 1):
                p1 = tuple(obj[i])
                p2 = tuple(obj[i + 1])
                cv2.line(display_image, p1, p2, color,3)#(100, 100, 255), 3)
        for i, skele in enumerate(skeles):
            color = colors[i%len(colors)]
            #skele.dtype = np.dtype('bool')
            #skele = cv2.dilate(skele, np.ones((2,2), dtype=bool), iterations=1)
            display_image[skele] = color
            skele_pts = np.where(skele == True)
            skele_pts = zip(skele_pts[1], skele_pts[0])
            for i in range(len(skele_pts) - 1):
                p1 = tuple(skele_pts[i])
                p2 = tuple(skele_pts[i + 1])
                cv2.line(display_image, p1, p2, color,3)#(100, 100, 255), 3)

        """


        # Save video frame
        #if writer:
        #    writer.write(display_image)

        # Figure out how long to wait before showing the next frame
        frame_end = time.time()
        elapsed_time = frame_end - frame_start
        target_delay = 1.0 / TARGET_FPS
        wait_time = target_delay - elapsed_time
        wait_time_ms = wait_time / 1000

        # Determine minimum scaling factor of image
        fy = MAX_HEIGHT / float(h)
        fx = MAX_WIDTH / float(w)
        scale = min(fx,fy)
        # Show the frame and wait until next frame
        #display_image = cv2.
        #cv2.imwrite("mask.png", display_image)
        last_display = display_image #TODO delete
        display_image = cv2.resize(display_image, (0,0), fx=scale, fy=scale)

        cv2.imshow("Progress", display_image)
        k = cv2.waitKey(max(1, wait_time_ms))

        # Check for ESC hit:
        if k & 0xFF == 27:
            break

    for obj in root_tips:
        if len(obj) <= frame_count:
            last_known_pos = obj[-1]
            for _ in range(frame_count - len(obj)+1):
                obj.append(last_known_pos)

    for plant in skel_sizes:
        if len(plant) <= frame_count:
            last_known_pos = plant[-1] if len(plant) >0 else 0
            for _ in range(frame_count - len(plant)+1):
                plant.append(last_known_pos)
    #capture.release()
    if not output_name:
        output_name = folder.split(slash)[-1]

    if not os.path.exists("output"):
        os.mkdir("output")
    mkdirs_if_not_exists("output"+slash+output_name)

    # Graph root movement
    #for r in root_tips:
    #    print(len(r))
    points = np.array(root_tips)

    miss_print = ""
    for i, misses in enumerate(lost_points):
        lost_points[i] = 1-1.0*misses/frame_count
        miss_print = miss_print + str(i)+ ":" + str(1.0*misses/frame_count)+ ", "
    #print(miss_print)
    pd.DataFrame(lost_points).to_csv("output"+slash+output_name+slash+output_name+"_quality.csv")


    # TODO: how to do this as a numpy operation?
    #necessary for easily plotting root growth locations
    max_y = np.amax(points[:, :, 1])
    for obj in points:
        for point in obj:
            point[1] = max_y - point[1]

    #Show root growth in x,y plane
    for plant_num, obj in enumerate(list(points)):
        plt.plot(*zip(*obj))
        plt.text(obj[0][0], obj[0][1]+10, plant_num)
    plt.title("Root Growth in Place")
    plt.savefig("output"+slash+output_name+slash+output_name+"_locations.png") #TODO superimpose id over image?
    plt.show()
    """
    for obj in list(points):
        distances = np.zeros(len(obj))
        for t in range(1,len(obj)):
            x_1, y_1 = obj[t-1]
            x_2, y_2 = obj[t]
            distances[t] = distances[t-1] + np.sqrt((x_2 -x_1)**2 + (y_2 - y_1)**2)
        plt.plot(distances)
    plt.title("Root Growth over Time")
    plt.show()
    """

    #output distance data to .csv
    dist_outputs = np.zeros((len(points[0]), len(points)))
    for plant, obj in enumerate(list(points)):
        #dist_outputs[plant][0] = plant
        for t in range(1,len(obj)):
            x_1, y_1 = obj[t-1]
            x_2, y_2 = obj[t]
            dist_outputs[t][plant] = dist_outputs[t - 1][plant] + np.sqrt((x_2 -x_1)**2 + (y_2 - y_1)**2)



    #output distance data to .csv
    #print(points.shape)
    dist_outputs = np.zeros((len(points),len(points[0])-1))
    for plant_num, plant in enumerate(points):
        #dist_outputs[plant][0] = plant
        for t in range(len(plant)-1):
            #print(plant)
            x_1, y_1 = plant[t]
            x_2, y_2 = plant[t+1]
            dist_outputs[plant_num][t] = np.sqrt((x_2 -x_1)**2 + (y_2 - y_1)**2)




    #print("skel sizes shape,", skel_sizes.shape)
    #delta skele = distance
    #growth_skele = [[] for i in skel_sizes]
    #TODO exclude outliers
    growth_skele = np.zeros_like(skel_sizes)
    iffy_plants = []
    for p, plant in enumerate(skel_sizes):
        for i in range(len(plant)-1):
            #growth_skele[p].append(plant[i+1]-plant[i])
            gr = plant[i+1]-plant[i]
            if np.abs(gr) < MAX_DELTA: #TODO scaled
                growth_skele[p][i] = gr
            else:
                iffy_plants.append(p)
                break


    growth_skele = growth_skele[:,0:-1]
    iffy_plants.sort(reverse=True)
    for bad in iffy_plants:
        print("deleting plant ", bad, " because of unreliable growth rates ")
    #TODO same for root tip tracing

    #print(timing)

    if skeletonize:
        growth = growth_skele
    else:
        growth = dist_outputs

    #TODO remove bad plants
    bad_plants = np.where(lost_points < 0.75)[0]#TODO adjustable threshold
    bad_plants_r = reversed(bad_plants)
    for bad in bad_plants_r:
        if bad not in iffy_plants:
            print("deleting plant ", bad, " with quality score ", lost_points[bad])

    #TODO add labels before deleting bad plants for consistency
    #for i in range(len(growth)):
    #    growth[]
    to_delete = list(bad_plants)+iffy_plants
    to_delete = list(dict.fromkeys(to_delete))
    to_delete.sort(reverse=True)
    for bad in to_delete:
        growth = np.delete(growth, bad, 0)


    """
    #average, smoothed data
    avg_growth = np.mean(growth, axis = 0) #TODO test on dist_outputs
    smoothed_growth = np.zeros_like(avg_growth)
    for i in range(len(avg_growth)):
        count = 0
        sum = 0.0
        offset = -7
        while i+offset < len(avg_growth) and offset <= 3:
            if i+offset >=0:
                sum += avg_growth[i+offset]
                count+=1
            offset+=1
        smoothed_growth[i] = sum/count
    """
    #smoothed, averaged data
    smoothed_growth = np.zeros_like(growth)
    smooth_window = 3 #actual window size is 2* smoothed_window + 1
    for p,plant in enumerate(growth):
        for i in range(len(plant)):
            count = 0
            sum = 0.0
            offset = -1*smooth_window
            while i+offset < len(plant) and offset <= smooth_window:
                if i+offset >=0:
                    sum += plant[i+offset]
                    count+=1
                offset+=1
            smoothed_growth[p][i] = sum/count
    standard_dev = np.std(smoothed_growth, 0)
    average_smoothed = np.mean(smoothed_growth, 0)
    smoothed_output = pd.DataFrame({"Average Growth": average_smoothed, "Standard Deviation": standard_dev})

    #check for existing file
    #TODO better output file names
    try:
        os.remove("output"+slash+output_name+slash+output_name+".csv") #TODO ask before deleting old file?
        os.remove("output"+slash+output_name+slash+output_name+"_skel_growth.csv") #TODO ask before deleting old file?
        os.remove("output"+slash+output_name+slash+output_name+"_skel_smoothed.csv") #TODO ask before deleting old file?
    except OSError:
        pass
    pd.DataFrame(dist_outputs).to_csv("output"+slash+output_name+slash+output_name+".csv")
    pd.DataFrame(growth_skele).to_csv("output"+slash+output_name+slash+output_name+"_skel_growth.csv")
    smoothed_output.to_csv("output"+slash+output_name+slash+output_name+"_skel_smoothed.csv")

    start_points = points[:, 0]
    normalized_points = (points.transpose([1, 0, 2]) - start_points).transpose([1, 0, 2])

    # points = np.array(root_tips)
    # for obj in points:
    #     start = obj[0].copy()
    #     for p in obj:
    #         p[0] = p[0] - start[0]
    #         p[1] = p[1] - start[1]

    #for obj in list(normalized_points):
    #    plt.plot(*zip(*obj))
    #plt.title("Normalized Root Growth")
    #plt.show()

def str2bool(v): #TODO from stack overflow
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("folder", metavar="FOLDER",
                        help="The folder of images to analyse")
    parser.add_argument("--name", "-n", metavar="NAME",
                        help="Name of dataset to be used in naming output files." +
                        " Default: same name as folder")
    #parser.add_argument("--image_output", "-i", metavar="DIR",
    #                    help="A directory to output the images for each " +
    #                         "frame before and after applying morphological" +
    #                         "operators")
    parser.add_argument("--param_file", "-p", metavar="PARAMS",
                        help="pkl file of input parameters for mask " +
                            "creation and object detection. (include .pkl)") #TODO set default?
    parser.add_argument("--skeletonize", "-s", metavar="SKELETONIZE", type=str2bool,
                        help="#TODO", default=True)
    args = parser.parse_args()

    main(args.folder, args.name, args.param_file, skeletonize=args.skeletonize)
