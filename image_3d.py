import numpy as np
import matplotlib.pyplot as plt
import cv2
import open3d as o3d

ply_header = '''ply
format ascii 1.0
element vertex %(vert_num)d
property float x
property float y
property float z
property uchar red
property uchar green
property uchar blue
end_header
'''

def find_disparity(image_left,image_right):
    window_size = 5
    min_disp = -39
    num_disp = 144
    stereo = cv2.StereoSGBM_create(minDisparity = -39, numDisparities = 144, disp12MaxDiff = 1,
        							blockSize = 5, P1 = 8*3*window_size**2, P2 = 32*3*window_size**2,
        							uniquenessRatio = 10, speckleWindowSize = 100, speckleRange = 32, preFilterCap = 63 )

    scale_factor = 16
    disparity = ( stereo.compute(image_left, image_right).astype(np.float32) )/scale_factor
    disparity = (disparity-min_disp)/num_disp
    return disparity

gt_file = open('./resources/poses.txt','r')

K =[[ 7.070912e+02 , 0.000000e+00 , 6.018873e+02 ],
	[ 0.000000e+00 , 7.070912e+02 , 1.831104e+02 ],
	[ 0.000000e+00 , 0.000000e+00 , 1.000000e+00 ]]
K = np.array(K)

# baseline:
base = 0.53790448812
#b = 0.53790448812;

focus = K[0][0];

total_imgs = 1

path_left = './resources/img2/'
path_right = './resources/img3/'

proj_mat = []

for lines in gt_file:
	line1 = lines.split(' ')
	line1 = np.array(line1)

	P = np.zeros((4,4) , dtype='float')
	P[0,:] = line1[0:4]
	P[1,:] = line1[4:8]
	P[2,:] = line1[8:12]
	P[3,:] = np.array( [0,0,0,1] )

	proj_mat.append(P)
#print(len(proj_mat))

points_3d = []
all_colors = []

for index in range(0,total_imgs):
	img_num = 60 + index
	print("Processing image no. ...",img_num)

	cur_left_img = path_left + '00000004' + str(img_num) + '.png'
	left_img = cv2.imread(cur_left_img)
	col_left_img = left_img

	cur_right_img = path_right + '00000004' + str(img_num) + '.png'
	right_img = cv2.imread(cur_right_img)

	disparity = find_disparity(left_img , right_img)
	min_disp = disparity.min()
	disparity_map = []

	image_shape = left_img.shape
	height = image_shape[0]
	width = image_shape[1]

	Q=[ [ 1, 0, 0, -width/2 ],
		[ 0,-1, 0, height/2 ],
		[ 0, 0, 0,    focus ],
		[ 0, 0, 1/base,   0 ] ]
	Q = np.array(Q)
	Q = np.float32(Q)

	for i in range(height):
		for j in range(width):
			disparity_map.append( [j,i,disparity[i,j],1] )


	point_cloud = []
	for dis in disparity_map:
		point = Q.dot(dis)
		point_cloud.append(point)
	point_cloud = np.array(point_cloud)

	colors = cv2.cvtColor(col_left_img, cv2.COLOR_BGR2RGB)
	mask = ( disparity >= min_disp )
	colors = ( colors[mask] )/ 255

	[ N,M ] = point_cloud.shape

	for i in range(N):

		points_tr = np.transpose(point_cloud[i])
		p1 = np.matmul( proj_mat[index] , points_tr )

		if(p1[3] > 0):
			p1 = p1/p1[3]
			points_3d.append(p1[0:3])
			all_colors.append(colors[i])

points_3d = np.array(points_3d)
all_colors = np.array(all_colors)

mask =( (-1500 <= points_3d[:,0]) & (-1500 <= points_3d[:,1]) & (-1500 <= points_3d[:,2]) &
		(points_3d[:,0] < 1500) & (points_3d[:,1] < 1500) & (points_3d[:,2] < 1500) )

points_3d = points_3d[mask]
all_colors = all_colors[mask]

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points_3d)
pcd.colors = o3d.utility.Vector3dVector(all_colors)
o3d.visualization.draw_geometries([pcd])


verts = np.array(points_3d)
colors = np.array(all_colors*255)
verts = verts.reshape(-1, 3)
colors = colors.reshape(-1, 3)
verts = np.hstack([verts, colors])
with open('output_file.ply', 'wb') as f:
	f.write((ply_header % dict(vert_num=len(verts))).encode('utf-8'))
	np.savetxt(f, verts, fmt='%f %f %f %d %d %d ')
