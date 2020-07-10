from global_imports import *
from img_utils import *
from maths_utils import *
from scipy.optimize import curve_fit
import warnings



''' currently this just creates x and y second derivatives at 
	spatial scale sigma (which are added together to give a Laplacian) '''
def get_filter_bank(sigma):
	grid_lim = round(3*sigma)
	X, Y = np.mgrid[-grid_lim:grid_lim+1:1, -grid_lim:grid_lim+1:1]

	exponential_part = np.exp(-(np.square(X) + np.square(Y))/(2*sigma**2));
	# already incorporates an additional sigma squared for scale correction
	d2G_dxx = 1/(2*np.pi*sigma**2) * (np.square(X)/sigma**2 - 1) * exponential_part
	d2G_dyy = d2G_dxx.T

	return [d2G_dxx + d2G_dyy]


''' build a filter bank from the get_filter_bank function
	and then apply each to the vessel, stacking outputs along
	the y-axis if there is more than one filter. '''
def filter_vessel_internal(frames):

	filter_bank = get_filter_bank(1)

	h,w = frames[0].shape
	outputs = np.zeros((len(frames), h*len(filter_bank), w))
	for i, f in enumerate(frames):
		for j, ker in enumerate(filter_bank):
			outputs[i,j*h:(j+1)*h,:] = cv2.filter2D(f, cv2.CV_64F, ker)

	return outputs


''' search before and after an index to find when the 
	array value first drops below cutoff_val. '''
def get_peak_start_end_idx(y, peak_idx, cutoff_val):

		# split at maximum value.
		before, after = np.split(y, [peak_idx])
		after = after[1:]

		# find first value left of peak less than a third of max.
		# take start index to be one before that (accounting for end)
		b = np.argwhere(np.abs(before) < cutoff_val)
		if len(b) == 0:
			b = [[0]]
		bi = b[-1][0]

		# find first value right of peak less than a third of max.
		# take start index to be one after that (accounting for end)
		a = np.argwhere(np.abs(after) < cutoff_val)
		if len(a) == 0:
			ai = len(y)
		else:
			ai = a[0][0] + len(before) + 1

		return bi, ai


''' fit a sigmoid pulse function to a filtered vessel
	to obtain its centre, width, and boundary indices. '''
def get_vessel_boundaries(filtered_img, show_graph=False):
	h,w = filtered_img.shape

	vsl_params = {"centre":[], "width":[], "start":[], "end":[], "steepness":[]}
	
	a = filtered_img.sum(axis=1)
	a = a/a.max()

	# this is done so that other vessels which might happen to be
	# in the image, but away from the vessel being analysed,
	# do not interfere with the curve fitting:
	i1, i2 = get_peak_start_end_idx(a, len(a)//2, .1)
	i1 = max(i1-2, 0)
	i2 = min(i2+2, h-1)
	a[:i1] = 0
	a[i2:] = 0

	warnings.filterwarnings('ignore')
	try:
		params_opt, cov = curve_fit(sigmoid_pulse, np.arange(len(a)), a, p0=(len(a)*0.5, len(a)*0.1, 3))
	except:
		print("failed to fit")
		return np.nan, np.nan, 0, h
	warnings.resetwarnings()


	# not steep enough to give good result
	if params_opt[2] < 0:
		# print("skipping - not steep enough fit to give good result")
		return np.nan, np.nan, 0, h


	vsl_centre = params_opt[0]
	vsl_width = params_opt[1]
	vsl_steepness = params_opt[2]
	vsl_start = params_opt[0] - params_opt[1]/2 - 1/vsl_steepness
	vsl_end   = params_opt[0] + params_opt[1]/2 + 1/vsl_steepness
	vsl_start_idx = max(0, int(np.round(vsl_start-.5)))
	vsl_end_idx   = min(h, int(np.round(vsl_end+.5)) + 1)


	# print(vsl_centre, vsl_width, vsl_start_idx, vsl_end_idx)
	if show_graph:

		x_axis = np.linspace(0, len(a), 200)
		y = sigmoid_pulse(x_axis, vsl_centre, vsl_width, vsl_steepness)

		# plt.hist(vsl_params["centre"])
		# plt.hist(vsl_params["start"])
		# plt.hist(vsl_params["end"])
		plt.plot(np.arange(len(a)), a, "ko-")
		plt.plot(x_axis, y, "b")

		plt.plot([vsl_start, vsl_start], [0, 1], "g")
		plt.plot([vsl_end,     vsl_end], [0, 1], "g")

		plt.plot([vsl_start_idx, vsl_start_idx], [0, 1], "r")
		plt.plot([vsl_end_idx-1, vsl_end_idx-1], [0, 1], "r")
		plt.show()

	return vsl_centre, vsl_width, vsl_start_idx, vsl_end_idx


''' calculate correlation of each frame relative to the next
	as it is shifted by different numbers up pixels sideways
	up to a maximum of slide_dist in each direction. 
	This is done separately for each pixel along the cross-
	section of the vessel. '''
def get_shift_correlation(frames, slide_dist, plot_3d=False):

	h, w = frames[0].shape
	window_len = w - slide_dist
	out_w = 2*slide_dist+1

	if window_len < 1:
		raise ValueError("Insufficient vessel length for this window slide distance ({} px window)".format(window_len))

	z = np.zeros((h, out_w))

	# min samples (frames * length in pixels) = 500
	# mean becomes stable at around 2000.

	# print(w, h, slide_dist, window_len)

	N_COMPARISON_FRAMES = len(frames)-1

	for i in range(N_COMPARISON_FRAMES):
		img1, img2 = frames[i], frames[i+1]
		window = img1[:, slide_dist:w-slide_dist]

		for j in range(0, 2*slide_dist+1):
			dx = j - slide_dist
			img3 = np.zeros((h,w))
			img3[:, slide_dist+dx:w-slide_dist+dx] = window

			norm1 = np.sqrt(np.sum(img2*img2, axis=1))
			norm2 = np.sqrt(np.sum(img3*img3, axis=1))
			z[:,j] = z[:,j] + np.sum(img2 * img3, axis=1) / (norm1 * norm2)


		# below: use different windows for sliding left
		# and right, reducing the necessary vessel length
		#------------------------------------------------
		# something seems wrong with normalisation, causing
		# a jump in value where different window used
		#------------------------------------------------
		# img1, img2 = frames[i], frames[i+1]

		# window = img1[:, :w-slide_dist]

		# for j in range(0, slide_dist+1):
		# 	dx = j
		# 	img3 = np.zeros((h,w))
		# 	img3[:, dx:w-slide_dist+dx] = window

		# 	norm1 = np.sqrt(np.sum(img2*img2, axis=1))
		# 	norm2 = np.sqrt(np.sum(img3*img3, axis=1))

		# 	z[:,dx+slide_dist] = z[:,dx+slide_dist] + np.sum(img2 * img3, axis=1) / (norm1 * norm2)


		# window = img1[:, slide_dist:]

		# for j in range(1, slide_dist+1):
		# 	dx = -j
		# 	img3 = np.zeros((h,w))
		# 	img3[:, slide_dist+dx:w+dx] = window

		# 	norm1 = np.sqrt(np.sum(img2*img2, axis=1))
		# 	norm2 = np.sqrt(np.sum(img3*img3, axis=1))

		# 	z[:,dx+slide_dist] = z[:,dx+slide_dist] + np.sum(img2 * img3, axis=1) / (norm1 * norm2)
		#------------------------------------------------------

	if plot_3d:
		from mpl_toolkits.mplot3d import axes3d
		import matplotlib.pyplot as plt

		fig = plt.figure()
		ax = fig.add_subplot(111, projection='3d')
		X, Y = np.mgrid[0:z.shape[0]:1, -slide_dist:slide_dist+1:1]

		ax.plot_surface(X, Y, z/N_COMPARISON_FRAMES, cmap="plasma")
		
		plt.show()

	return z / N_COMPARISON_FRAMES



''' from correlation vs. shift data at different positions 
	along the cross-section of the vessel, find the position
	of the peak value to obtain a velocity profile. '''
def extract_flow_profile(correlations, plot_peaks=False, plot_v_profile=False):

	z = correlations
	h, w = z.shape

	std_devs = np.std(z, axis=1)
	rel_std_devs = std_devs / std_devs.max()

	rough_peaks = z.argmax(axis=1)
	max_vals = z.max(axis=1)

	peaks = np.zeros(h)
	sigmas = np.zeros(h)
	confidences = np.zeros(h)

	# Localise peaks more precisely and measure significance
	# by fitting gaussians to points nearby peaks
	# and finding means and standard deviations.
	#------------------------------------------------------------
	# Only local values to peaks taken as there is a lot of 
	# noise which messes with the curve fitting.
	#------------------------------------------------------------
	# A better approach would be a Bayesian one using a gaussian
	# process which contains noise and a gaussian peak.
	for row in range(0, h):

		peak_idx = rough_peaks[row]

		# bi, ai = get_peak_start_end_idx(z[row], 0.5*max_vals[row])

		# this is more stable for now.
		if peak_idx >= w - 2 or peak_idx <= 1:
			peaks[row] = max_vals[row]
			sigmas[row] = 0
			confidences[row] = 0
			continue

		bi = max(0, peak_idx-3)
		ai = min(w, peak_idx+4)

		y = z[row, bi:ai]
		x = np.arange(bi, ai) - (w-1)//2

		y_sum = np.sum(y)
		y_others_sum = np.sum(z[row]) - y_sum 
		y_mean = y_sum/y.size
		y_others_mean = y_sum/(z[row].size - y.size)

		confidences[row] = y_mean - y_others_mean

		# # not enough points to fit gaussian (2 parameters!)
		# if len(x) < 3:
		# 	print("WARNING - not enough points to fit Gaussian")
		# 	peaks[row] = peak_idx - (w-1)//2
		# 	sigmas[row] = max_vals[row]
		# 	continue


		# scale to have peak equal to rough peak value
		fn = lambda x, u, s: max_vals[row] * gaussian(x, u, s)
		try:
			params_opt, cov = curve_fit(fn, x, y, p0=(peak_idx - (w-1)//2, 2))
			peaks[row] = params_opt[0]
			sigmas[row] = params_opt[1]

		except:
			print("WARNING - unable to fit gaussian")
			peaks[row] = max_vals[row]
			sigmas[row] = 0#max_vals[row]
			confidences[row] = 0


		if plot_peaks:
			x2 = np.arange(0,w) - (w-1)//2 - peaks[row]
			y2 = z[row,:]
			x3 = np.arange(bi,ai) - (w-1)//2 - peaks[row]
			y3 = z[row,bi:ai]
			plt.plot(x2,y2,"k")
			plt.plot(x3,y3)
			try:
				y4 = fn(x4, *params_opt)
				x4 = np.arange(0,w,0.1) - (w-1)//2 
				x4 = x4 - peak_idx
				plt.plot(x4,y4)
			except:
				pass

	if plot_peaks:
		plt.title("Correlation vs. shift distance, x-axes adjusted to align peaks")
		plt.xlabel("shift in pixels")
		plt.ylabel("correlation")
		plt.show()


	x = np.arange(0, h) - (h-1)/2

	# print(np.max(peaks)/(np.sum(np.abs(x)*peaks)/np.sum(np.abs(x))))

	valid_readings = 2*sigmas < np.abs(peaks).max()
	valid_readings = np.bitwise_and(valid_readings, confidences > 0.1)

	UM_PER_PIXEL = 4.2735
	FRAMERATE = 30

	if plot_v_profile:
		plt.title("Measured flow velocity vs. radius")
		plt.xlabel("Radial distance (um)")
		plt.ylabel("Flow velocity (mm/s)")

		# plt.plot(x*UM_PER_PIXEL, peaks*UM_PER_PIXEL*FRAMERATE*0.001)
		# plt.errorbar(x*UM_PER_PIXEL, peaks*UM_PER_PIXEL*FRAMERATE*0.001, yerr=0.001*UM_PER_PIXEL*FRAMERATE*sigmas*3, uplims=True, lolims=True)
		plt.plot(x, peaks)
		plt.errorbar(x[valid_readings], peaks[valid_readings], yerr=sigmas[valid_readings]*1.5, uplims=True, lolims=True)
		plt.show()


	return x[valid_readings], peaks[valid_readings], sigmas[valid_readings]


''' return a function transforming a frame so that a vessel
	(between two points) is horizontal and crop to that vessel'''
def get_vessel_centering_fn(pts):

	w = int(np.hypot(*(pts[1] - pts[0]))+.5)
	h = 30 # seems a sensible upper bound for maximum possible vessel width
	M = get_rot_matrix(pts, (h//2,w//2))

	return lambda f: cv2.warpAffine(f, M, (w, h))


''' get the difference of each frame from the mean frame '''
def get_frame_deltas(frames):
	mean = np.mean(frames, axis=0)
	return np.array(frames) - mean


''' given all necessary data, attempt to extract relevant information
	including velocity profile and vessel thickness. '''
def data_from_vessel_pts(pts, filtered, deltas, max_slide, show_graphs=True, space_time_image=False):

	fn = get_vessel_centering_fn(pts)

	filt = fn(filtered)
	frames = np.array([fn(f) for f in deltas]).astype(float)

	if show_graphs:
		show_img(filt)



	vsl_centre, vsl_width, vsl_start_idx, vsl_end_idx = get_vessel_boundaries(filt, show_graphs)
	
	# print("Vessel dims: {:.2f}, {:.2f}, {}, {}".format(vsl_centre, vsl_width, vsl_start_idx, vsl_end_idx))


	vsl_data = {
		"x"    : None,
		"v_x"  : None,
		"v_sd" : None,
		"v_mean" : None,
		"v_peak" : None,
		"Q" : None, # estimate this properly.
		"width" : vsl_width
	}


	frames = frames[:, vsl_start_idx:vsl_end_idx, :]

	filtereds = filter_vessel_internal(frames)

	if vsl_width < 3 and space_time_image:
		print(vsl_width)
		# space-time image may be a useful technique
		# but I haven't written code to analyse the gradient
		img = np.mean(filtereds, axis=1)
		show_img(img)


	try:
		z = get_shift_correlation(filtereds, max_slide, show_graphs)
	except ValueError as e:
		print(e)
		return vsl_data

	peak_xs, peaks, std_devs = extract_flow_profile(z, show_graphs, show_graphs)

	if len(peaks) == 0:
		return vsl_data

	vsl_data = {
		"x"    : peak_xs,
		"v_x"  : peaks,
		"v_sd" : std_devs,
		"v_mean" : np.mean(peaks),
		"v_peak" : np.max(peaks),
		"Q" : None, # estimate this properly.
		"width" : vsl_width
	}

	return vsl_data


