# %%
# EMPM673 - Perception for Autonomous Robots - Project #2 - Date: March 8, 2023
# Author: Arshad Shaik, UID: 118438832
# 
# Problem 1: Python program to estimate the camera pose 
# through homography, hough transform, canny edge detection
# 
# Import necessary packages
import cv2
import numpy as np
from matplotlib import pyplot as plt
from numpy.linalg import inv, norm
from scipy.spatial.transform import Rotation

##### Function 1: Find Homography
def Homography(xw, yw, xp, yp): # xw, yw -> World axes points, xp, yp -> Camera Projective Points
    x1 = xw[0]
    x2 = xw[1]
    x3 = xw[2]
    x4 = xw[3]
    
    y1 = yw[0]
    y2 = yw[1]
    y3 = yw[2]
    y4 = yw[3]
    
    xp1 = xp[0]
    xp2 = xp[1]
    xp3 = xp[2]
    xp4 = xp[3]
    
    yp1 = yp[0]
    yp2 = yp[1]
    yp3 = yp[2]
    yp4 = yp[3]
    
    A = np.array([
    [-x1,  -y1,  -1,   0,    0,    0,   x1*xp1,   y1*xp1,   xp1],
    [0,     0,    0,  -x1,  -y1,  -1,   x1*yp1,   y1*yp1,   yp1],
    [-x2,  -y2,  -1,   0,    0,    0,   x2*xp2,   y2*xp2,   xp2],
    [0,     0,    0,  -x2,  -y2,  -1,   x2*yp2,   y2*yp2,   yp2],
    [-x3,  -y3,  -1,   0,    0,    0,   x3*xp3,   y3*xp3,   xp3],
    [0,     0,    0,  -x3,  -y3,  -1,   x3*yp3,   y3*yp3,   yp3],
    [-x4,  -y4,  -1,   0,    0,    0,   x4*xp4,   y4*xp4,   xp4],
    [0,     0,    0,  -x4,  -y4,  -1,   x4*yp4,   y4*yp4,   yp4]
    ])
    
    [U,S,VH] = np.linalg.svd(A)
    
    H = VH[-1, :]
    H = np.reshape(H, (3,3))
    
    return H

# Function 2: Finding Hough Lines
def customHoughLines(edges, rho_acc, theta_acc, threshold, width, height):
    y1, x1 = np.nonzero(edges) #get the x and y coordinates of the edges
    img_diag_size = np.ceil(np.sqrt(height**2 + width**2)) #calculate the diagonal of the image
    #create a numpy array of d values from -diagonal to +diagonal
    d = np.arange(-img_diag_size, img_diag_size+1, rho_acc) 
    #create a numpy array of theta values from 0 to 180 degrees with step size of 1
    theta = np.deg2rad(np.arange(0, 180, theta_acc))

    #declare numpy array to store the hough space
    H_Space = np.zeros((len(d), len(theta)), dtype=np.uint8) 

    #loop through all the edges
    for i in range(len(x1)): 
        x = x1[i] 
        y = y1[i]

        for j in range(len(theta)): #loop through all the theta values
            d1 = int((x * np.cos(theta[j])) + (y * np.sin(theta[j])) + img_diag_size) #d = xcos(theta) + ysin(theta) + diagonal of image
            H_Space[d1, j] += 1 #increment the value of the corresponding d and theta in hough space for vote

    maximas =  np.argpartition(H_Space.flatten(), -2)[-threshold:] #get the 4 highest values or maximas in hough space
    indices = np.vstack(np.unravel_index(maximas, H_Space.shape)).T #get the indices (d,theta) of those maxima values
    d1_list = []
    thetas_list = []
    #loop through all the indices   
    for i in range(len(indices)): 
        d1, thetas = d[indices[i][0]], theta[indices[i][1]] #get the d and theta values from the indices
        d1_list = np.append(d1_list, d1)
        thetas_list = np.append(thetas_list, thetas)
    
    return np.c_[d1_list, thetas_list]


# Creating a OpenCV window
cv2.namedWindow('Original Video', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Original Video', 640, 480)

# Import video
given_video = cv2.VideoCapture("project2.avi")
width  = given_video.get(3)  
height = given_video.get(4) 
print("Width of Video: ", width, "\nHeight of Video: ", height)

# cv2.resizeWindow('Original Video', width, height)

num_of_lines = np.array([])
num_of_slopes = np.array([])
frame_num = np.array([1])
fn = 0

# Open a file to save a frame processing time
f = open('HomographY_check.txt','a')

#Create a list to store values outside the loop
T_x = [] #list to store the x component of translation vector
T_y = [] #list to store the y component of translation vector
T_z = [] #list to store the z component of translation vector
angle_x = [] #list to store the x component of rotation matrix in degrees
angle_y = [] #list to store the y component of rotation matrix in degrees
angle_z = [] #list to store the z component of rotation matrix in degrees

while(given_video.isOpened()):
    ret, img = given_video.read()
    
    if ret == True:    

        #cv2.imshow("Original Video", img)   

        # Convert to BGR image to HSV image
        #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Convert to BGR image to HSV image
        hsvIm = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # Define the threshold values for HSV mask (Ref: colorpicker.py)
        minHSV = np.array([80, 32, 234]) 
        maxHSV = np.array([179, 255, 255]) 

        # Create a mask
        maskHSV = cv2.inRange(hsvIm, minHSV, maxHSV)

        # Apply erosion and dilation to remove noise
        kernel = np.ones((5,5), np.uint8)

        procIm1 = cv2.dilate(maskHSV, kernel, iterations=1)
        procIm2 = cv2.erode(procIm1, kernel, iterations=1)

        # Renaming a processed image to a gray image
        gray = procIm2

        # Apply edge detection method on the image
        edges = cv2.Canny(gray, 300, 400)

        # This returns an array of r and theta values
        lines = customHoughLines(edges, 1, 1, 5, width, height)

        print("\n No. of lines: ",len(lines))
        num_of_lines = np.append(num_of_lines, len(lines))
        
        print("\n Lines (rho, theta): \n", lines)

        # The below for loop runs till r and theta values
        # are in the range of the 2d array
        itr = 1
        m = np.array([])
        c = np.array([])
        for i in range(len(lines)):
            
            r, theta = np.array(lines[i], dtype=np.float64)
            # Stores the value of cos(theta) in a
            a = np.cos(theta)

            # Stores the value of sin(theta) in b
            b = np.sin(theta)

            # x0 stores the value rcos(theta)
            x0 = a*r

            # y0 stores the value rsin(theta)
            y0 = b*r

            # x1 stores the rounded off value of (rcos(theta)-1000sin(theta))
            x1 = int(x0 + 1000*(-b))

            # y1 stores the rounded off value of (rsin(theta)+1000cos(theta))
            y1 = int(y0 + 1000*(a))

            # x2 stores the rounded off value of (rcos(theta)+1000sin(theta))
            x2 = int(x0 - 1000*(-b))

            # y2 stores the rounded off value of (rsin(theta)-1000cos(theta))
            y2 = int(y0 - 1000*(a))

            # slope of the line
            slope = (y2 - y1) / (x2 - x1)
            intercept = y1 - (slope*x1)

            m = np.append(m,slope)
            c = np.append(c, intercept)

            # cv2.line draws a line in img from the point(x1,y1) to (x2,y2).
            # (0,0,255) denotes the colour of the line to be
            # drawn. In this case, it is red.    
            if itr == 1:
                clr = (255,0,0)
            elif itr == 2:
                clr = (0,255,0)
            elif itr == 3:
                clr = (0,0,255)
            elif itr == 4:
                clr = (0,255,255)
            else:
                clr = (255,255,255)

            cv2.line(img, (x1, y1), (x2, y2), clr, 2)

            # print((x1, y1), (x2, y2))
            itr += 1 

        print("\n Slopes of the lines:\n ",m)

        print("\n y-intercepts of the lines:\n ",c)

        # group positive slopes and perpendicular slopes
        pos_m = np.array([])
        neg_m = np.array([])
        pos_c = np.array([])
        neg_c = np.array([])

        for i in range(len(m)):
            if m[i] > 0:
                pos_m = np.append(pos_m, m[i])
                pos_c = np.append(pos_c, c[i])
            else:
                neg_m = np.append(neg_m, m[i])
                neg_c = np.append(neg_c, c[i])

        del_inx_pos = np.array([]).astype(int)
        del_inx_neg = np.array([]).astype(int)

        for i in range(len(pos_c)-1):
            for j in range(len(pos_c) - (i+1)):
                if abs(pos_c[i] - pos_c[i+j+1]) < 100:
                    del_inx_pos = np.append(del_inx_pos, i+j)
                    
        
        for i in range(len(neg_c)-1):
            for j in range(len(neg_c) - (i+1)):
                if abs(neg_c[i] - neg_c[i+j+1]) < 100:
                    del_inx_neg = np.append(del_inx_neg, i+j)

        pos_m = np.delete(pos_m, del_inx_pos)
        neg_m = np.delete(neg_m, del_inx_neg)

        pos_c = np.delete(pos_c, del_inx_pos)
        neg_c = np.delete(neg_c, del_inx_neg)

        m = np.hstack((pos_m, neg_m))
        c = np.hstack((pos_c, neg_c))

        

        if len(m) < 4:
            m = m_prev
            c = c_prev

        num_of_slopes = np.append(num_of_slopes, len(m))

        xp = np.array([])
        yp = np.array([])

        for i in range(len(m) - 1):
            # if the product of the lines is -ve, then compute the intersection point
            for j in range(len(m) - (i+1)):
                if (m[i] * m[i+j+1]) < 0:
                    x = (c[i+j+1] - c[i]) / (m[i] - m[i+j+1])
                    y = (m[i] * x) + c[i]
                    xp = np.append(xp, x)
                    yp = np.append(yp, y)
                    cv2.circle(img,(int(x),int(y)),20,(0,0,255),-1)         

        print("\n Intersection points (x,y) of the lines (Camera Frame):\n ", xp, "\n", yp)

        # Sorting the points along the x-axis
        inds = xp.argsort()

        yp = yp[inds]

        xp = np.sort(xp)

        print("\n Sorted Intersection points (x,y) of the lines:\n ", xp,"\n", yp)

        # Designate the points in the same order as the world axes points
        ints_pt = 1
        for i in range(len(xp)):
            # Using cv2.putText()
            cv2.putText(
              img = img,
              text = "Point" + str(ints_pt),
              org = (int(xp[i]),int(yp[i])),
              fontFace = cv2.FONT_HERSHEY_DUPLEX,
              fontScale = 2.0,
              color = (125, 246, 55),
              thickness = 3
            )
            ints_pt += 1

        # Show the original video with points marked and lines shown
        cv2.imshow("Original Video", img)

        # World axes co-ordinates - center of the paper is origin
        xw = np.array([-13.95, -13.95, 13.95, 13.95])
        yw = np.array([-10.8, 10.8, -10.8, 10.8,])

        print("\n World axes co-ordinates (x,y): \n", xw, "\n", yw)

        # Compute the homography only when you have atleast 4 points
        if (len(xp) >= 4):
            H = Homography(xw, yw, xp, yp)

            # making last element as '1'
            H = H / H[2,2]

            print("\n Homography Matrix: \n",H)

            # reconstructing the camera projected point from Homography matric
            sample_point = np.array([xw[0], yw[0], 1])
            print("\n Check for Homography - sample_point\n", sample_point)

            sample_point = np.reshape(sample_point,(3,1))

            check_pt = np.dot(H,sample_point)

            # Converting to Homogenous co-ordinates
            check_pt = check_pt / check_pt[2]

            print("\n Reconstructed Point: ", [check_pt[0,0], check_pt[1,0]],"\n", "\n Actual Point: ",[xp[0],yp[0]])

            if int(check_pt[0,0]) == int(xp[0]) and int(check_pt[1,0]) == int(yp[0]):
                print("\n Homography Test Passed!")  
                outstring = "Homography Test Passed!" + '\n'
                f.write(outstring)
            else:
                print("\n Homography Test Failed!")
                outstring = "Homography Test Failed!" + '\n'
                f.write(outstring)

            K = np.array([
                [1382.58398, 0,          945.743164],
                [0,          1383.57251, 527.04834],
                [0,          0,          1]
            ])

            RT = np.dot(inv(K), H)

            print("\n Matrix[r1 r2 t]: \n", RT)

            lamda1 = norm(RT[:, 0]) #norm of the first column of the RT matrix
            lamda2 = norm(RT[:, 1]) #norm of the second column of the RT matrix
            lamda = (lamda1 + lamda2)/2 #average of the two norms
            print("\nAvg Lambda: ", lamda)

            r1 = np.array(RT[:, 0]/lamda) #first column of the RT matrix divided by the average of the two norms
            r1 = r1.reshape(3, 1)
           
            r2 = np.array(RT[:, 1]/lamda) #second column of the RT matrix divided by the average of the two norms
            r2 = r2.reshape(3, 1)

            r3 = np.cross(r1.T, r2.T) #cross product of r1 and r2
            r3 = r3.reshape(3, 1)

            R = np.hstack([r1, r2, r3]).reshape(3, 3) #rotation matrix
            print("\nRotation Matrix: \n", R)

            T = np.array(RT[:, 2]/lamda) #translation matrix
            print("\nTranslation Matrix: \n", T.reshape(3, 1))

            ##Plot the translation and rotation values over no of frames:
            T_x = np.append(T_x, T[0]) #append the x translation value to the array for plotting
            T_y = np.append(T_y, T[1]) #append the y translation value to the array for plotting
            T_z = np.append(T_z, T[2]) #append the z translation value to the array for plotting

            #Convert the rotation matrix to Euler angles:         
            r = Rotation.from_matrix(R) #convert the rotation matrix to a rotation object
            angles_op = r.as_euler('xyz', degrees=True) #convert the rotation object to euler angles
            print("\nRoll, Pitch & Yaw Angles from Rotation Matrix\n", angles_op)
            angle_x = np.append(angle_x, angles_op[0]) #append the roll angle to the array for plotting
            angle_y = np.append(angle_y, angles_op[1]) #append the pitch angle to the array for plotting
            angle_z = np.append(angle_z, angles_op[2]) #append the yaw angle to the array for plotting

            print("\n---------------------------------\n")

        else:
            print("\n Not enough points to compute Homography" + str(len(xp)))  
        
        m_prev = m
        c_prev = c
        # Exit while each video frame is processed, by preessing 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
        
    # when no image frame is read, break from video loop    
    else:
        break

    fn += 1
    frame_num = np.append(frame_num, fn)
given_video.release()
cv2.destroyAllWindows()

# Close the opened file
f.close()

# Plots
fig1 = plt.figure()
plot1 = fig1.add_subplot(211)
plot1.set_title("No. of Lines per Frame")
plot1.set_xlabel("Frames")
plot1.set_ylabel("No. of Raw Lines detected")
plot1.plot(num_of_lines, color = 'magenta', label = 'lines/frame')
# plot1.plot(num_of_slopes, color = 'cyan', label = 'lines/frame')
plot2 = fig1.add_subplot(212)
plot2.set_ylabel("No. of Processed Lines")
plot2.plot(num_of_slopes, color = 'cyan', label = 'lines/frame')
# plot1.invert_yaxis()

#Creating 1 output plot window which will have all the plots:
fig = plt.figure('Camera Pose Estmimation from the given video')

#Setting the space between subplots to make it viewable
plt.subplots_adjust(left=0.1,
                    bottom=0.1,
                    right=0.9,
                    top=0.9,
                    wspace=0.4,
                    hspace=0.4)
    
plot1 = fig.add_subplot(231)
plot1.set_title("Roll Angle")
plot1.set_xlabel("Frame Number")
plot1.set_ylabel("Roll Angle")
plot1.plot(angle_x, color = 'magenta', label = "Roll Angle")
plot1.legend(loc='best', fontsize = 8)

plot2 = fig.add_subplot(232)
plot2.set_title("Pitch Angle")
plot2.set_xlabel("Frame Number")
plot2.set_ylabel("Pitch Angle")
plot2.plot(angle_y, color = 'magenta', label = "Pitch Angle")
plot2.legend(loc='best', fontsize = 8)

plot3 = fig.add_subplot(233)
plot3.set_title("Yaw Angle")
plot3.set_xlabel("Frame Number")
plot3.set_ylabel("Yaw Angle")
plot3.plot(angle_z, color = 'magenta', label = "Yaw Angle")
plot3.legend(loc='best', fontsize = 8)

plot4 = fig.add_subplot(234)
plot4.set_title("X - Translation")
plot4.set_xlabel("Frame Number")
plot4.set_ylabel("Translation in X")
plot4.plot(T_x, color = 'magenta', label = "Translation in x")
plot4.legend(loc='best', fontsize = 8)

plot5 = fig.add_subplot(235)
plot5.set_title("Y - Translation")
plot5.set_xlabel("Frame Number")
plot5.set_ylabel("Translation in Y")
plot5.plot(T_y, color = 'magenta', label = "Translation in y")
plot5.legend(loc='best', fontsize = 8)

plot6 = fig.add_subplot(236)
plot6.set_title("Z - Translation")
plot6.set_xlabel("Frame Number")
plot6.set_ylabel("Translation in Z")
plot6.plot(T_z, color = 'magenta', label = "Translation in z")
plot6.legend(loc='best', fontsize = 8)


plt.show()