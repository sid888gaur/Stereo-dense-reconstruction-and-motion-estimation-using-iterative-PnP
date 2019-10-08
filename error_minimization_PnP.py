import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d

def find_Jacob_mat(W_coords,proj):

	J=[]
	W_coords_tr = np.transpose(W_coords)
	# print(proj)
	# print(W_coords_tr.shape)
	imgcord = np.transpose(np.dot(proj,W_coords_tr))

	Shape1 = imgcord.shape
	# print(Shape1)
	x = Shape1[0]
	# print(x)
	y = Shape1[1]
	# print(y)

	for i in range(x):

		temp1 = []
		temp2 = []
		z1 = imgcord[i][2]

		temp1.append( -W_coords[i][0]/z1 )
		temp1.append( -W_coords[i][1]/z1 )
		temp1.append( -W_coords[i][2]/z1 )
		temp1.append( -1/z1 )
		temp1.append( 0 )
		temp1.append( 0 )
		temp1.append( 0 )
		temp1.append( 0 )
		temp1.append( W_coords[i][0]*imgcord[i][0]/(z1**2) )
		temp1.append( W_coords[i][1]*imgcord[i][0]/(z1**2) )
		temp1.append( W_coords[i][2]*imgcord[i][0]/(z1**2) )
		temp1.append( imgcord[i][0]/(z1**2) )

		J.append(temp1)

		temp2.append( 0 )
		temp2.append( 0 )
		temp2.append( 0 )
		temp2.append( 0 )
		temp2.append( -W_coords[i][0]/z1 )
		temp2.append( -W_coords[i][1]/z1 )
		temp2.append( -W_coords[i][2]/z1 )
		temp2.append( -1/z1 )
		temp2.append( W_coords[i][0]*imgcord[i][1]/(z1**2) )
		temp2.append( W_coords[i][1]*imgcord[i][1]/(z1**2) )
		temp2.append( W_coords[i][2]*imgcord[i][1]/(z1**2) )
		temp2.append( imgcord[i][1]/(z1**2) )

		J.append(temp2)


	J = np.array(J)
	# print(J[0:12,:])
	return J


def find_error(img_orig,proj,W_h_coords):

	error = 0
	W_h_coords_tr = np.transpose(W_h_coords)
	temp = np.transpose( np.dot(proj , W_h_coords_tr) )

	Shape1 = W_h_coords.shape
	x = Shape1[0]
	y = Shape1[1]

	for i in range(x):
		temp[i]=temp[i]/temp[i][2]

	for i in range(x):

		a1 = temp[i][0] - img_orig[i][0]
		b1 = temp[i][1] - img_orig[i][1]
		error = error + a1**2 + b1**2

	return error


def find_pinv(X):

	X_tr = np.transpose(X)
	X_prod = np.dot(X_tr , X)
	# print(X_prod[0:12,:])
	X_prod_inv = np.linalg.pinv(X_prod)
	X_pinv = np.dot(X_prod_inv , X_tr)
	return X_pinv



P_init = [  [-9.098548e-01, 5.445376e-02,-4.113381e-01,-1.872835e+02],
			[ 4.117828e-02, 9.983072e-01, 4.107410e-02, 1.870218e+0 ],
			[ 4.128785e-01, 2.043327e-02,-9.105569e-01, 5.417085e+01] ]
P_init = np.array(P_init)
z23 = P_init[2][3]

P_new =[[ 12,  0,  0 ],
		[  0, 12,  0 ],
		[  0,  0, 12 ]]
P_new = np.array(P_new)
tr_new = [ 4, 5, 6 ]
tr_new = np.array(tr_new)
P_new = np.vstack((P_new,tr_new))
#print(P_new)
P_new = np.array(P_new,dtype='float')
P_new = np.transpose(P_new)
oldP = P_new
# print(oldP)
# Reading the point cloud from q1
pcd = o3d.io.read_point_cloud("./output_file.ply")
points = np.asarray(pcd.points)
Shape = points.shape
x = Shape[0]
y = Shape[1]

x_ones = [1]*x
x_ones = np.ones((x,1))

colors = np.asarray(pcd.colors)

# R=[ [1, 3, 5],
# 	[2, 2, 8],
# 	[3, 6, 7] ]
# R = np.array(R)

# t = [1,-2,1]
# t = np.transpose( t )
# t = t.reshape(3,1)

W_h_coords = np.hstack( (points,x_ones) )
W_h_coords_tr = np.transpose(W_h_coords)

img_orig = np.transpose( np.dot(P_init , W_h_coords_tr) )

for index in range(x):
	img_orig[index] = img_orig[index]/img_orig[index][2]

# min_error = 0.001


# print(find_error(img_orig,P_new,W_h_coords))
# o3d.visualization.draw_geometries([pcd]) # Visualize the point cloud
err_vec=[]

for index in range(25):
	print(index)
	# print(oldP)
	Jacob = find_Jacob_mat(W_h_coords , oldP)
	# print(Jacob)
	# print(Jacob.shape)
	Jacob_tr = np.transpose(Jacob)
	J_psinv = find_pinv(Jacob)
	# print(J_psinv)
	# print(J_psinv.shape)

	imgcord = np.transpose( np.dot(oldP , W_h_coords_tr) )
	Shape1 = imgcord.shape
	# print(imgcord)
	# print(Shape1)
	x = Shape1[0]
	y = Shape1[1]
	ret = []
	for i in range(x):

		z1 = imgcord[i][2]

		d1 = img_orig[i][0]-(imgcord[i][0]/z1)
		ret.append( d1 )

		d2 = img_orig[i][1]-(imgcord[i][1]/z1)
		ret.append( d2 )

	# print(ret)
	ret = np.array(ret)
	# print(dife.shape)
	ret = ret.reshape(2*x,1)
	# print(ret[0:12,:])

	oldP = oldP.reshape(12,1)

	eta = 0.6
	# print(J_psinv)
	cost = 	np.dot(J_psinv , ret)
	# print(cost)
	P_new = oldP - ( 0.6*cost )
	# print(P_new)
	oldP = P_new.reshape(3,4)
	# print(oldP)
	newerr = find_error(img_orig,oldP,W_h_coords)
	print(newerr)
	err_vec.append( newerr )

	temp_er_vec = np.dot( Jacob_tr , ret )
	norm_error = np.linalg.norm(temp_er_vec)
	# print(norm_error)
	# if norm_error < min_error:
	# 	break

tp23 = oldP[2][3]
oldP = ( oldP * z23) / tp23
print(oldP)

err_vec = np.array(err_vec)
plt.plot(err_vec)

plt.show()
