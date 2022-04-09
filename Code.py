import numpy as np
import cv2
import random
from scipy.spatial import distance
from scipy.optimize import least_squares
from math import cos, sin


# Parameters for the disparity map computation
window_size = 4
min_disp = 2
num_disp = 128
stereo = cv2.StereoSGBM_create(minDisparity = min_disp,
    numDisparities = num_disp,
    blockSize = 5,
    P1 = 8*3*window_size**2,
    P2 = 32*3*window_size**2,
    disp12MaxDiff = -1,
    uniquenessRatio = 10,
    speckleWindowSize = 100,
    speckleRange = 32
)

fast = cv2.FastFeatureDetector_create()

CamMatrix = np.asarray([[707.0912, 0.0, 601.8873, 0],
             [ 0.0, 707.0912, 183.1104, 0],
             [ 0.0, 0.0, 1.0, 0]])

GroundTruth = np.asarray([[0.9999978, 0.0005272628, -0.002066935, -0.04690294],
                          [-0.0005296506, 0.9999992, -0.001154865, -0.02839928],
                          [0.002066324, 0.001155958, 0.9999971, 0.8586941],
                          [0, 0, 0, 1]])

fMetric = 0.004 
fPixels = CamMatrix[0,0]
b = 0.54 

def compute_blur(img):
    return cv2.bilateralFilter(img, 5, 20, 20)

def compute_disparity(im1, im2):
    disparity = stereo.compute(im1, im2).astype(np.float32)
    disparity = np.divide(disparity, 16.0)
    return disparity

def keypoint_extraction(img, disparity):
    #size of descriptor
    m = 13
    m = int(m/2) 
    keypoints = fast.detect(img)
    
    # Store keypoints by response and keep only the 1200 firsts one to reduce computation time
    keypoints = sorted(keypoints, key=lambda x: -x.response)
    keypoints = keypoints[0:1200]
    
    # Remove kpts with unknow disparity (=1) or too close from edge
    list_of_bad_kp = []
    for i, kpt in enumerate(keypoints):
        kpt_x = int(kpt.pt[0])
        kpt_y = int(kpt.pt[1])
        if kpt_x<7 or kpt_x>len(img[0])-7:
            list_of_bad_kp.append(i)
        elif kpt_y<7 or kpt_y>len(img)-7:
            list_of_bad_kp.append(i)
        elif (disparity[kpt_y][kpt_x] == 1):
            list_of_bad_kp.append(i)
    keypoints = np.delete(keypoints,list_of_bad_kp)

    #Descriptor creation
    descriptors = []
    for i, kpt in enumerate(keypoints):
        kpt_x = int(kpt.pt[0])
        kpt_y = int(kpt.pt[1])
        
        descriptor = img[kpt_y-m:kpt_y+m+1, kpt_x-m:kpt_x+m+1]
        descriptor = descriptor.ravel()
        descriptor = np.delete(descriptor, int((m*m-1)/2))
        descriptors.append(descriptor)
    return keypoints, descriptors

def compute_S_matrix(des1, des2):
    S = np.zeros((len(des1), len(des2)))
    for i,d1 in enumerate(des1):
        for j, d2 in enumerate(des2):
            SAD = np.sum(np.abs(np.subtract(d1,d2)))
            S[i][j] = SAD
    return S

def match_features(S):
    #index of minimum value for each fa
    fa_matches = np.zeros(len(S))
    #index of minimum value for each fb
    fb_matches = np.zeros(len(S[0]))
    matches = []
    # We create a list of matching features starting from fa and fb
    for i in range(len(S)):
        fa_matches[i] = np.argmin(S[i])
    for i in range(len(S[0])):
        fb_matches[i] = np.argmin(S[:,i])
        
    # We check constistency between those two lists to obtain the matches
    for i,fb_bar in enumerate(fa_matches):
        fb_bar = int(fb_bar)
        if fb_matches[fb_bar] == i:
            matches.append([i,fb_bar])
    return matches

# Give 3D coordinates from 2D coordinates
def get_coordinate(kp, disparity_map):
    depth_map = (b*fPixels)/disparity_map 
    matrix = CamMatrix
    u = int(kp.pt[0])
    v = int(kp.pt[1])
    depth = depth_map[v][u]
    
    X = depth*(u-matrix[0][2])/(matrix[0][0])
    Y = depth*(v-matrix[1][2])/(matrix[1][1])
    Z = depth
    return [X,Y,Z]

# Create the consistency matrix of the matches by comparing real world distance between features
def compute_consistency(matches, Da, Db):
    # Maximum distance difference allowed
    treshold = 1
    W = np.zeros((len(matches),len(matches)))
    for i, match1 in enumerate(matches):
        kpa1 = match1[0]
        kpb1 = match1[1]
        Wa1 = get_coordinate(kpa1, Da)
        Wb1 = get_coordinate(kpb1, Db)
        for j, match2 in enumerate(matches):
            kpa2 = match2[0]
            kpb2 = match2[1]
            Wa2 = get_coordinate(kpa2, Da)
            Wb2 = get_coordinate(kpb2, Db)
            consistency = abs(distance.euclidean(Wa1, Wa2) - distance.euclidean(Wb1, Wb2))
            if consistency < treshold:
                W[i][j] = 1
    return W.astype(int)

def compute_Q(W):
    matchesConsistencyScore = []
    # List of indexes of matches that we are going to keep
    Q = []
    for i in range(len(W)):
        matchesConsistencyScore.append(sum(W[i]))
    # Clique initialisation, we take the match with the best score
    Q.append(np.argmax(matchesConsistencyScore))

    # We start by adding all the matches as candidates
    candidates = np.arange(len(W))
    
    while True:
        # Find set of matches compatibles with the matches in the clique
        for q in Q:
            compatibles = []
            for i in range(len(W)):
                if W[q][i]==1 and not i in Q and not i in compatibles:
                    compatibles.append(i)
            #Intersect this list of compatibles with the list of candidates to keep only matches compatibles with all matches in the clique
            candidates = np.intersect1d(candidates,compatibles)

        # If there is no more candidtes, exit the while loop
        if len(candidates) == 0:
            break
        
        ## Choose the candidates with the best score to add it in the clique
        #score of candidates
        candidatesScore = [matchesConsistencyScore[i] for i in candidates]
        # find the index of the best candidate
        bestCandidateIndex = np.argmax(candidatesScore)
        # add this candidate to Q
        Q.append(candidates[bestCandidateIndex])
    
    return Q

# Create a transformation matrix with pitch, roll, yaw, Tx, Ty, Tz
def createTransfomrationMatrix(params):
    motionMatrix = np.eye(4)
    #https://mathworld.wolfram.com/EulerAngles.html
    theta = params[0]
    psi = params[1]
    phi = params[2]

    D = [[cos(phi), sin(phi), 0],
         [-sin(phi), cos(phi), 0],
         [0, 0, 1]]

    C = [[cos(theta), 0, sin(theta)],
         [0, 1, 0],
         [-sin(theta), 0, cos(theta)]]

    B = [ [1, 0, 0],
        [0, cos(psi), sin(psi)],
         [0, -sin(psi), cos(psi)]]

    motionMatrix[:3,:3] = np.matmul(D,np.matmul(C,B))
    
    motionMatrix[0,3] = params[3]
    motionMatrix[1,3] = params[4]
    motionMatrix[2,3] = params[5]
    
    return motionMatrix

# Compute the equation describe in A5 and return the residuals (the error)
def estimateMotion(params, imPtsa, imPtsb, worldsPtsa, worldsPtsb, CamMatrix):
    motionMatrix = createTransfomrationMatrix(params)

    Pdelt = np.matmul(CamMatrix,motionMatrix)
    Pdeltinv = np.matmul(CamMatrix, np.linalg.inv(motionMatrix))
    error1 = np.zeros((len(imPtsa),3))
    error2 = np.zeros((len(imPtsa),3))
    for i in range(len(imPtsa)):
        Pred1 = np.matmul(Pdelt, np.asarray(worldsPtsb[i]).reshape(-1,1))
        # Normalisation to stay in homogeneous space
        Pred1 /= Pred1[-1]
        Pred2 = np.matmul(Pdeltinv, np.asarray(worldsPtsa[i]).reshape(-1,1))
        Pred2 /= Pred2[-1]
        err1 = np.asarray(imPtsa[i]).reshape(-1,1) - Pred1
        err2 = np.asarray(imPtsb[i]).reshape(-1,1) - Pred2
        #Creation of the reisuldals : list of error for x and y coordinate
        error1[i,:] = np.squeeze(err1)
        error2[i,:] = np.squeeze(err2)
        residual = np.vstack((error1,error2))
    return residual.flatten()

#Download images, they are already rectified
Ja_L = cv2.imread('KITTI/00/image_0/000000.png', 0)
Ja_R = cv2.imread('KITTI/00/image_1/000000.png', 0)

Jb_L = cv2.imread('KITTI/00/image_0/000001.png', 0)
Jb_R = cv2.imread('KITTI/00/image_1/000001.png', 0)

# We skiped pre-filtering to obtain better results
'''
# Pre-filtering
Ja_L = compute_blur(Ja_L)
Ja_R = compute_blur(Ja_R)
Jb_L = compute_blur(Jb_L)
Jb_R = compute_blur(Jb_R)
'''

# Disparity images
Da = compute_disparity(Ja_L, Ja_R)
Db = compute_disparity(Jb_L, Jb_R)

##plt.figure("StereoBM disparity map")
##plt.imshow(Da, 'jet', vmin=0, vmax=70)
##plt.show()

# Detect keypoints of each image and get descriptors
Kpa, Dsa = keypoint_extraction(Ja_L, Da)
Kpb, Dsb = keypoint_extraction(Jb_L, Db)

# Get the score matrix S
S = compute_S_matrix(Dsa, Dsb)

#Match the features according to the score matrix
matchesIndexes = match_features(S)

# Construct the matches matrix full of features instead of feature's indexes
matches = []
for match in matchesIndexes:
    matches.append([Kpa[match[0]], Kpb[match[1]]])

# Compute consistency matrix :
W = compute_consistency(matches, Da, Db)
np.savetxt('W.csv', W, delimiter=',')

# Compute Q, set of matches in the inlier
Q = compute_Q(W)
# Q is the set of indexes of the matches that we are going to keep, let's exctract those matches
inlierMatches = [matches[i] for i in Q]

# Create parameters to run the optimizer
imPtsa = []
imPtsb = []
worldsPtsa = []
worldsPtsb = []

for i in range(len(inlierMatches)):
    imPtsa.append([inlierMatches[i][0].pt[0], inlierMatches[i][0].pt[1]])
    imPtsa[i].append(1)
    imPtsb.append([inlierMatches[i][1].pt[0], inlierMatches[i][1].pt[1]])
    imPtsb[i].append(1)
    worldsPtsa.append(get_coordinate(inlierMatches[i][0], Da))
    worldsPtsa[i].append(1)
    worldsPtsb.append(get_coordinate(inlierMatches[i][1], Db))
    worldsPtsb[i].append(1)

x0 = np.array([0,0,0,0,0,0])
optRes = least_squares(estimateMotion, x0, method='lm',
                       args=(imPtsa, imPtsb, worldsPtsa, worldsPtsb, CamMatrix))

# Print the found parameters and the transformation matrix coresponding
print(optRes.x)
print(createTransfomrationMatrix(optRes.x))

#Draw and save the keypoints that have been retained
output = Ja_L
kpaa=np.asarray(inlierMatches)[:,0]
output = cv2.drawKeypoints(Ja_L, kpaa, output)
output2 = Jb_L
kpbb=np.asarray(inlierMatches)[:,1]
output2 = cv2.drawKeypoints(Jb_L, kpbb, output2)
cv2.imwrite('keypointsA.png', output)
cv2.imwrite('keypointsB.png', output2)

#Draw and save the matches that have been retained
image = cv2.vconcat([output, output2])
for match in inlierMatches:
    Point_a = match[0].pt
    Point_b = match[1].pt
    cv2.line(image, (int(Point_a[0]),int(Point_a[1])),
             (int(Point_b[0]),int(Point_b[1] + 370)),
             (random.randrange(255), random.randrange(255), random.randrange(255)))
cv2.imwrite('matches.png', image)

#Save disparity images
cv2.imwrite('Da.png',Da)
cv2.imwrite('Db.png',Db)

#Save images used
cv2.imwrite('La.png',Ja_L)
cv2.imwrite('Lb.png',Jb_L)

cv2.imwrite('Ra.png',Ja_R)
cv2.imwrite('Rb.png',Jb_R)
