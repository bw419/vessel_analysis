from global_imports import *


''' mathematical functions implemented with numpy arrays '''
def gaussian(x, u, s):
	return np.exp(-np.power(x - u, 2.) / (2 * np.power(s, 2.)))

def sigmoid_pulse(x, centre, width, steepness):
	return 1/(1+np.exp(-steepness*(x-centre+width*0.5))) + 1/(1+np.exp(steepness*(x-centre-width*0.5)))-1



''' sum multiplied images (by pixel) and normalise value '''
def get_correlation(img1, img2):
	return np.sum((img1*img2).flatten())/np.sqrt((np.sum((img2*img2).flatten())*np.sum((img1*img1).flatten())))


''' convert similarity transform matrix to parameters '''
def params_from_transform_matrix(M):

	dx, dy = M[:,2]
	da = math.atan2(M[1,0], M[0,0])
	s = np.sqrt(np.square(M[1,0]) + np.square(M[0,0]))

	return dx, dy, da, s


''' convert similarity transform parameters to matrix '''
def transform_matrix_from_params(dx, dy, da, s):
	co = np.cos(da)*s
	si = np.sin(da)*s

	return np.float32([[co, -si, dx],
					   [si,  co, dy]])


''' construct an affine transformation matrix which
	rotates the input in order to transform a pair
	of points to being horizontally aligned with
	the centre of the image at dest_centre. '''
def get_rot_matrix(pt_pair, dest_centre):
	dp1 = pt_pair[1] - pt_pair[0]
	mx,my = np.mean(pt_pair, axis=0)
	angle = math.atan2(dp1[1], dp1[0]) * 180 / math.pi

	M = cv2.getRotationMatrix2D((mx,my), angle, 1)

	dx,dy = int(dest_centre[1]-mx), int(dest_centre[0]-my)

	M[0,2] += dx
	M[1,2] += dy

	return M