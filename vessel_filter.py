from global_imports import *
from img_utils import *


''' calculate and return Hessian matrix of the image 
	for each pixel at a particular scale sigma. 
	force_separable_diff: True uses the first method,
	to get Hessian values, False uses the second. The
	difference is not significant except for smallest scales,
	where (at sigma <= 1) the first method is used '''
def get_hessian(img, sigma, force_separable_diff=False):


	# gaussian blur, then take derivatives in each dimension
	# independently and compose together.
	if sigma <= 1 or force_separable_diff:
		img2 = img.astype(float)
		
		ksize = round(3*sigma)
		img2 = cv2.GaussianBlur(img2, (ksize, ksize), sigma)

		dL_dx, dL_dy = [x*sigma**2 for x in np.gradient(img2)]

		d2L_dxx = np.gradient(dL_dx, axis=0) 
		d2L_dyy = np.gradient(dL_dy, axis=1)
		d2L_dxy = np.gradient(dL_dx, axis=1)

		# show_img(d2L_dxx)
		# show_img(d2L_dxy)
		# show_img(d2L_dyy)

	# construct gaussian derivative kernels and convolve.
	else:
		grid_lim = round(3*sigma)
		X, Y = np.mgrid[-grid_lim:grid_lim+1:1, -grid_lim:grid_lim+1:1]

		exponential_part = np.exp(-(np.square(X) + np.square(Y))/(2*sigma**2));

		# already incorporates an additional sigma squared
		# multiplication to correct for scale
		d2G_dxx = 1/(2*np.pi*sigma**2) * (np.square(X)/sigma**2 - 1) * exponential_part
		d2G_dxy = 1/(2*np.pi*sigma**4) * (X * Y) * exponential_part
		d2G_dyy = d2G_dxx.T

		d2L_dxx = cv2.filter2D(img, cv2.CV_64F, d2G_dxx)
		d2L_dxy = cv2.filter2D(img, cv2.CV_64F, d2G_dxy)
		d2L_dyy = cv2.filter2D(img, cv2.CV_64F, d2G_dyy)


	hessian = [[d2L_dxx, d2L_dxy],
			   [d2L_dxy, d2L_dyy]]

	rows = []
	for row in hessian:
		rows.append(np.stack(row, axis=-1))
	hessian = np.stack(rows, axis=-2)

	return hessian



''' compare Hessian eigenvalues at each pixel to give a 'vesselness' score.
	beta:   lower -> more sensitive to stretched features
	gamma:  lower -> more sensitive to strong features against background
	c:      only used for well-posedness of diffusion equation in VED
		    where it is set to a small value. Can be set to 0/None otherwise.'''
def get_vesselness(hessian, beta=1, gamma=15, c=None):

	#
	# obtain eigenvalues and order by decreasing size
	#-------------------------------------------------

	# this calculation is much faster than numpy functions
	# for the case of 2x2 matrices.
	dets = hessian[:,:,0,0]*hessian[:,:,1,1]-hessian[:,:,1,0]*hessian[:,:,0,1]
	traces = hessian[:,:,0,0] + hessian[:,:,1,1]

	sqrt_terms = np.sqrt(np.square(traces) - 4 * dets)
	v1 = 0.5*(traces + sqrt_terms)
	v2 = 0.5*(traces - sqrt_terms)
	v = np.dstack((v1, v2))


	# sort by largest eigval:
	index = list(np.ix_(*[np.arange(i) for i in v.shape]))
	index[2] = np.abs(v).argsort(-1)
	rearrangement_tuple = tuple(index)
	v = v[rearrangement_tuple]

	eig1 = v[:,:,0]
	eig2 = v[:,:,1]
	eig2[eig2 == 0] = 1e-10


	#
	# get vesselness score components and multiply together
	#------------------------------------------------------

	# deviation from blob-ness
	B = np.divide(np.abs(eig1), np.abs(eig2))

	# strength vs. background
	S_squared = np.square(eig1) + np.square(eig2)
	S = np.sqrt(S_squared)

	if gamma is None:
		gamma = S.max()*0.5

	# Vesselness factors
	stretchedness = np.exp(np.negative(np.square(B) / (2 * np.square(beta))))
	feature_strength = 1 - np.exp(np.negative(S_squared / (2 * np.square(gamma))))
	
	if c is None:
		smoothing_factor = 1
	else:
		# This is different from the paper as in the 2D case the lowest common
		# denominator requires is just the second eigenvalue squared
		smoothing_factor = np.exp(-2 * np.square(c) / np.square(eig2))

	vesselness = stretchedness * feature_strength * smoothing_factor

	vesselness[eig2 <= 0] = 0
	vesselness[np.isnan(vesselness)] = 0

	return vesselness





''' apply a Frangi vesselness filter to the frame, taking the maximum response
	for each pixel over the range of scales given by sigmas. 
	beta:   lower -> more sensitive to stretched features
	gamma:  lower -> more sensitive to strong features against background
	c:      only used for well-posedness of diffusion equation in VED
		    where it is set to a small value. Can be set to 0/None otherwise.
	scale_dependence:
		    response at each length scale is adjusted by raising it to minus
		    this power, before the maximum is computer, in case a stronger 
		    output is desired from smaller/larger scales. Derivative kernels
		    have already been normalised so when set to 0, response is uniform over scale.
	return_eigens:
			return eigenvectors and eigenvalues of Hessian at maximum response.
			This is required in the unused VED functions below. However, it can
			also be useful in future functions, as a vector field is returned 
			giving strength and direction of vessel features.'''
def apply_frangi(frame, sigmas=[1,2,3,5], beta=1, gamma=None, c=None, scale_dependence=0, return_eigens=False):


	if return_eigens:
		v = np.ones(frame.shape + (2,))
		w = np.zeros(frame.shape + (2,2))
		w[:,:,0,0] = 1
		w[:,:,1,1] = 1


	vesselness_max = np.zeros(frame.shape)

	for i, scale in enumerate(sigmas):

		H = get_hessian(frame, scale, False)
		vesselness = get_vesselness(H, beta, gamma, c)

		vesselness *= np.power(scale, -scale_dependence)
		vesselness_gt_max = vesselness > vesselness_max
		vesselness_max[vesselness_gt_max] = vesselness[vesselness_gt_max]


		if return_eigens:
			bool_mask = vesselness_gt_max

			v[bool_mask], w[bool_mask] = np.linalg.eigh(H[bool_mask])

			# sort by largest eigval:
			index = list(np.ix_(*[np.arange(i) for i in v.shape]))
			index[2] = np.abs(v).argsort(-1)
			rearrangement_tuple = tuple(index)
			w = w[rearrangement_tuple]



	if return_eigens:
		return rescale_to_ubyte(vesselness_max), w
	else: 
		return rescale_to_ubyte(vesselness_max)




if __name__ == "__main__":

	frame_path = "frames2/average.bmp"
	frame = cv2.cvtColor(cv2.imread(frame_path), cv2.COLOR_BGR2GRAY)
	filtered = apply_frangi(frame, [1,2,3,5], 1, None)
	show_img(filtered, "plasma")






# # ------THESE FUNCTIONS SHOULD NOT BE USED: -------


''' initial attempt at filtering images.
	not particularly theoretically justified but does an OK job.
	better methods are below. '''
def filter_raw_img(grey):

	overall_out = np.zeros_like(grey)
	# overall_out = (overall_out)

	print("  denoising...")
	denoised = cv2.fastNlMeansDenoising(grey,3,3,19)
	# show_img(denoised)

	s.stop()
	print("  applying laplacian pyramid...")

	base = gamma_brighten(denoised, 0.3)

	h, w = grey.shape

	N_UPSCALES = 5

	imgs = []

	for layer in range(N_UPSCALES):
		for ksize in range(5, 15, 4):

			lap = np.abs(cv2.Laplacian(base, cv2.CV_64F, ksize=ksize))
			# threshed = cv2.adaptiveThreshold(base,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,3*ksize,2)
			# imgs.append(np.abs(lap))

			# if ksize == 13:
				# show_img(lap)
				# show_img(threshed)

			upscaled = cv2.resize(lap, (w, h))
			gt_max = upscaled>overall_out
			overall_out[gt_max] = upscaled[gt_max]

		base = cv2.resize(base, (w//2**layer, h//2**layer), interpolation=cv2.INTER_LINEAR)

	overall_out[overall_out < 0.1*overall_out.max()] = 0
	show_img(overall_out)
	# overall_out = imgs.max(axis=0)
	return cv2.bitwise_not(overall_out)



# VED FUNCTIONS
#--------------------------------------------------------------------------
# Idea is to blur anisotropically/apply anisotropic diffusion
# directed along edge boundaries to remove noise while preserving
# vessel edges, repeating vessel detection and blurring steps iteratively.

# from the paper:
# Vessel enhancing diffusion: A scale space representation of vessel structures

def test():

	print("applying VED algorithm...")

	frame_path = "frames2/average.bmp"
	frame = cv2.cvtColor(cv2.imread(frame_path), cv2.COLOR_BGR2GRAY)

	f1 = frame.copy()

	frames1 = [frame]
	frames2 = []
	frames3 = []

	for i in range(1):

		filtered, w = apply_frangi(frame, [1,2,3,5], 1, 10, 1e-3, 0.05, True)
		frames2.append(filtered)

		ret, frame = cv2.threshold(filtered,5,255,cv2.THRESH_BINARY)

		frames3.append(frame)
		frame2 = anisotropic_blur(frame, filtered, w, 4, 0.002, 1.5, 5, 0)

		frames1.append(frame.astype(np.uint8))

		print("completed iteration", i)



	for f in frames1:
		show_img(f, "plasma")

	import frame_io as fio

	# fio.save_frames_to_folder(frames1, "frames1/VED_process2", 0)
	# fio.save_frames_to_folder(frames2, "frames1/VED_process2", 100)
	# fio.save_frames_to_folder(frames3, "frames1/VED_process2", 200)

	fio.write_video_from_frames(frames1, "frames2/vid4.avi")
	fio.write_video_from_frames(frames3, "frames2/vid5.avi")





''' this function does seem to blur as intended in tests but it doesn't give good results.
	cos_scale: damps transverse blurring further - preserves smaller veins slightly better
	- this didn't end up actually helping much, just set to 0 '''
def anisotropic_blur(frame, vesselness, w, omega=2, epsilon=0.01, sensitivity=1, kernel_rad=3, cos_scale=0):


	sensitivity_scaled_vesselness = np.power(vesselness, 1/sensitivity)

	scaling_matrix = np.zeros(frame.shape + (2,2))
	scaling_matrix[:,:,0,0] = 1/(1. + (omega - 1.) * sensitivity_scaled_vesselness)
	scaling_matrix[:,:,1,1] = 1/(1. + (epsilon - 1.) * sensitivity_scaled_vesselness)


	D_tensor = np.transpose(w, axes=(0,1,3,2)) @ scaling_matrix @ w
	

	dets = D_tensor[:,:,0,0]*D_tensor[:,:,1,1]-D_tensor[:,:,1,0]*D_tensor[:,:,0,1]

	k_r = kernel_rad
	X, Y = np.mgrid[-k_r:k_r+1:1, -k_r:k_r+1:1]
	xys = np.dstack((X, Y))

	r2 = np.einsum('ijkl,mnl,mnk->ijmn', D_tensor, xys, xys)
	z = np.exp(-0.5 * r2)


	if cos_scale != 0:
		k = cos_scale*np.sqrt(scaling_matrix[:,:,1,1])
		w[:,:,1,0] *= k
		w[:,:,1,1] *= k

		cos_cmpt = np.square(np.cos(np.einsum('ijk,lmk->ijlm', w[:,:,1,:], xys)))

		z *= cos_cmpt;


	# fig = plt.figure()
	# ax = fig.gca(projection='3d')

	# print(w[0,0])
	# print(D_tensor)
	# # Plot the surface.
	# surf = ax.plot_wireframe(X, Y, z[0,0])
	# plt.show()

	# scale_factors = (np.sqrt(dets) / (2*np.pi))
	scale_factors = 1.00/z.sum((2,3))


	h, w = frame.shape
	new_frame = np.zeros((h,w))


	for i in range(k_r, h-k_r):
		for j in range(k_r, w-k_r):
			new_frame[i,j] = np.sum(z[i,j] * frame[i-k_r:i+k_r+1,j-k_r:j+k_r+1])

	new_frame *= scale_factors


	m = new_frame == 0

	new_frame[m] = frame[m]

	# show_img(new_frame)

	return new_frame



''' this function doesn't work.
	D tensor construction is fine but application of diffusion equation
	leads to instabilities - sharp black and white 1-pixel-wide bands '''
def VED_step(frame, vesselness, e_vecs):

	raise Exception("Not implemented.")

	omega = 5
	epsilon = 0.1
	sensitivity = 1

	sensitivity_scaled_vesselness = np.power(vesselness, 1/sensitivity)

	e_vecs2 = e_vecs.copy()
	e_vecs2[:,:,0,0] *= 1. + (omega - 1.) * sensitivity_scaled_vesselness
	e_vecs2[:,:,0,1] *= 1. + (omega - 1.) * sensitivity_scaled_vesselness
	e_vecs2[:,:,1,0] *= 1. + (epsilon - 1.) * sensitivity_scaled_vesselness
	e_vecs2[:,:,1,1] *= 1. + (epsilon - 1.) * sensitivity_scaled_vesselness

	# H,W = v.shape[:2]
	# X = np.arange(0, W, 1)
	# Y = np.arange(0, H, 1)
	# plt.quiver(X, Y, e_vecs2[:,:,0,0], e_vecs2[:,:,0,1], sensitivity_scaled_vesselness)#, scale=60)
	# show_img(frame)
	# plt.show()

	Qt = e_vecs2.copy()
	D_tensor = Qt.copy()

	scaling_matrix = np.zeros_like(e_vecs)
	scaling_matrix[:,:,0,0] = 1. + (omega - 1.) * sensitivity_scaled_vesselness
	scaling_matrix[:,:,1,1] = 1. + (epsilon - 1.) * sensitivity_scaled_vesselness

	D_tensor = Qt @ scaling_matrix @ np.transpose(Qt, axes=(0,1,3,2))

	img = frame.astype(float)

	# Apply diffusion. Currently just starting with the first frame
	for i in range(5):
		grad = np.stack(np.gradient(img), axis=-1)
		# print(grad)

		directed_grad_x = np.sum(D_tensor[:,:,0] * grad, axis=-1)
		directed_grad_y = np.sum(D_tensor[:,:,1] * grad, axis=-1)
		# print(directed_grad_x)
		print(directed_grad_y)

		plt.quiver(X, Y, e_vecs2[:,:,0,0], e_vecs2[:,:,0,1], sensitivity_scaled_vesselness, scale=60)
		plt.quiver(X, Y, e_vecs2[:,:,1,0], e_vecs2[:,:,1,1], sensitivity_scaled_vesselness, scale=60)
		show_img(grad[:,:,0])
		plt.show()
		show_img(directed_grad_x)
	# p

		div_x = np.gradient(directed_grad_x, axis=0)
		div_y = np.gradient(directed_grad_y, axis=1)
		div = div_x + div_y


		# H,W = v.shape[:2]
		# X = np.arange(0, W, 1)
		# Y = np.arange(0, H, 1)
		# plt.quiver(X, Y, directed_grad_x[:,:], directed_grad_y[:,:], div)
		# show_img(grey)
		# plt.show()
		# show_img(directed_grad_x)
		# show_img(directed_grad_y)
		img = img + div
	# print(div)
		show_img(img)