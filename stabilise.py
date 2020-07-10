from global_imports import *
from img_utils import *
from maths_utils import *



''' find upright rectangle containing only preserved pixels. '''
def find_max_preserved_rect(border_mask):

	if not (border_mask>0).any() or not (border_mask<1).any():
		h, w = border_mask.shape
		return 0, w, 0, h

	border_mask = rescale_to_ubyte(border_mask)


	cnt, h = cv2.findContours(border_mask,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
	cnt = np.array([x[0] for x in cv2.approxPolyDP(cnt[0],5,True)])
	x,y,w,h = cv2.boundingRect(cnt)

	x1 = np.max(cnt[cnt[:,0]<x+w/2,0])
	y1 = np.max(cnt[cnt[:,1]<y+h/2,1])
	x2 = np.min(cnt[cnt[:,0]>x+w/2,0])-1
	y2 = np.min(cnt[cnt[:,1]>y+h/2,1])-1

	return x1, x2, y1, y2


''' register two frames and find transform to stabilise them. '''
def get_transform(frame, frame0, p0, downscale, mark_keypoints=False):

	# calculate optical flow
	p1, st, err = cv2.calcOpticalFlowPyrLK(frame0, frame, p0, None)

	# Select good points
	good_p0 = p0[st==1]
	good_p1 = p1[st==1]

	if len(good_p0) < 0.3 * len(p0):
		return 0, 0, 0, 1, None

	transform_mat, inliers = cv2.estimateAffinePartial2D(good_p0, good_p1)

	if all(transform_mat.flatten() == 0):
		return 0, 0, 0, 1, None

	dx, dy, da, s = params_from_transform_matrix(transform_mat)

	if downscale is not None:
		dx *= downscale
		dy *= downscale

	if mark_keypoints:
		return dx, dy, da, s, p1*downscale
	else:
		return dx, dy, da, s, None


''' generator which yields successive stabilised frames.
	for detailed description, see README.md. '''
def yield_stabilised(frames, split_score=0.3, crop=False, downscale=8, mark_keypoints=False, registration_filter=None):

	if registration_filter is None:
		registration_filter = lambda x: x

	first_loop = True
	for frame in frames:
		frame_fullsize = frame.copy()

		# initial setup on first loop
		if first_loop:
			first_loop = False
			h_fullsize, w_fullsize = frame.shape
			if crop:
				ones = np.ones_like(frame)

		# runs when there is a reference frame
		else:
			#
			# register keypoints and transform later frames:
			#-------------------------------------------------
			frame = cv2.resize(frame, (w_fullsize//downscale, h_fullsize//downscale))

			dx, dy, da, s, keypoints = get_transform(registration_filter(frame), frame0, p0, downscale, mark_keypoints=mark_keypoints)
			M = transform_matrix_from_params(dx, dy, da, s)
			M_inv = cv2.invertAffineTransform(M)


			if mark_keypoints:
				if keypoints is not None:
					for kpt in keypoints:
						cv2_cross(frame_fullsize, kpt[0], 1, 255, 1)


			adjusted = cv2.warpAffine(frame_fullsize, M_inv, (w_fullsize, h_fullsize))

			#
			# score transformed frames based on their correlation with the original
			#----------------------------------------------------------------------
			adjusted_ds = cv2.resize(adjusted, (w_fullsize//downscale, h_fullsize//downscale))
			frame_laplacian = cv2.Laplacian(adjusted_ds, cv2.CV_64F)
			score = get_correlation(frame0_laplacian, frame_laplacian)


			#
			# if score sufficiently high, return adjusted
			# - otherwise, start again with new reference frame
			#------------------------------------------------------
			if score > split_score:
				if crop:
					border_mask = cv2.warpAffine(ones, M_inv, (w_fullsize, h_fullsize), flags=cv2.INTER_NEAREST)
					yield adjusted, score, border_mask.copy()
				else:
					yield adjusted, score

				continue

		#
		# if here, either there frame0 is None (first loop)
		# or score was low enough to trigger a new frame0
		#------------------------------------------------

		# reference frame processing
		frame0 = cv2.resize(frame, (w_fullsize//downscale, h_fullsize//downscale))
		frame0 = registration_filter(frame0)
		frame0_laplacian = cv2.Laplacian(frame0, cv2.CV_64F)
		h,w = frame0.shape

		# heuristic scaling of feature detection parameters based on image dimensions
		N_PTS, Q, MIN_SEP = int(np.sqrt(h*w)), 0.01, int(0.3*np.power(h*w, 0.3))

		p0 = cv2.goodFeaturesToTrack(frame0, N_PTS, Q, MIN_SEP)


		if mark_keypoints:
			for kpt in p0*downscale:
				cv2_cross(frame_fullsize, kpt[0], 1, 255, 1)

		if crop:
			border_mask = ones.copy()
			yield frame_fullsize, None, border_mask.copy()
		else:
			yield frame_fullsize, None


''' 
	apply stabilisation to frame array. Parameters:
	`crop`					- if `False`, pad with zeros up to full resolution, otherwise crop to the minimum upright rectangle which is common to all of the frames for each frame sequence.
	`ignore_score`  		- if not `None`, do not return frames with a correlation score below its value (but do not split into multiple sequences). Mutually exclusive with `split_score`.
	`split_score`	 		- if not `None`, start a new frame sequence each time the correlation score falls below its value. Mutually exclusive with `ignore_score`.
	`downscale` 	 		- if not `None`, how much to downsample by during registration. Higher resolutions are slower computationally and seem to occasionally be affected by the motion of blood cells, causing minor rotations.
	`mark_keypoints` 		- if `True`, draw on the output frames the keypoints used for matching.
	`correspondence_filter` - if not `None`, supply a function to be used just before registration - e.g. a function which masks a portion of the frame that contains a needle which would disrupt registration.
'''
def simple_stabilisation(frames, ignore_score=None, split_score=None, crop=False, downscale=8, mark_keypoints=False, registration_filter=None):
	
	if ignore_score is not None and split_score is not None:
		raise Exception("Either set ignore_score or split_score")

	if ignore_score is None and split_score is None:
		ignore_score = -2

	if ignore_score is not None:
		stabilised_gen = yield_stabilised(frames, -2, crop, downscale, mark_keypoints, registration_filter)
		if crop:
			stabilised_frames, scores, border_masks = zip(*stabilised_gen)
			stabilised_frames = np.array(stabilised_frames)

			scores = np.array(scores)
			border_masks = np.array(border_masks)
			scores[scores == None] = 1

			len1 = len(stabilised_frames)
			stabilised_frames = stabilised_frames[scores > ignore_score]
			border_masks = border_masks[scores > ignore_score]
			len2 = len(stabilised_frames)

			# multiply masks to get overall mask
			x1, x2, y1, y2 = find_max_preserved_rect(np.prod(border_masks, axis=0))

			if len1 > len2:
				print("ignored", len1-len2, "out of", len1, "frames")

			return stabilised_frames[:,y1:y2,x1:x2]
		else:
			stabilised_frames, scores = zip(*stabilised_gen)
			stabilised_frames = np.array(stabilised_frames)

			scores = np.array(scores)
			scores[scores == None] = 1

			len1 = len(stabilised_frames)
			stabilised_frames = stabilised_frames[scores > ignore_score]
			len2 = len(stabilised_frames)

			if len1 > len2:
				print("ignored", len1-len2, "out of", len1, "frames")

			return stabilised_frames


	if split_score is not None:
		stabilised_gen = yield_stabilised(frames, split_score, crop, downscale, mark_keypoints, registration_filter)
		
		if crop:
			stabilised_frames, scores, border_masks = zip(*stabilised_gen)
			splits = get_indices_of_value(scores, None)

			frame_sequences = np.split(stabilised_frames, splits)[1:]
			x = np.split(border_masks, splits, axis=0)[1:]

			# multiply together masks for each sequence to get its overall mask
			seq_masks = [np.prod(y, axis=0) for y in x]

			good_seqs = []
			for i, seq in enumerate(frame_sequences):
				if len(seq) > 3:
					mask = seq_masks[i]
					x1, x2, y1, y2 = find_max_preserved_rect(mask)
					good_seqs.append(seq[:,y1:y2,x1:x2])

			return np.array(good_seqs)

		else:
			stabilised_frames, scores = zip(*stabilised_gen)
			splits = get_indices_of_value(scores, None)
			frame_sequences = np.split(stabilised_frames, splits)[1:]
			return np.array([seq for seq in frame_sequences if len(seq) > 3])
	





if __name__ == "__main__":

	import frame_io as fio

	S = Stopwatch()

	# optional function to use as a filter in order to help registration stage
	# by removing moving single blood cells that may disrupt the transformation estimation?
	# - set registration_filter=f
	f = lambda x: cv2.dilate(x, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3)), iterations=3)


	frames = fio.get_frame_range_from_folder("frames4/unstabilised", "pgm", 1180, 1409)

	# to ignore parts of the frame during stabilisation, use this as registration_filter:
	def mask(frame):
		h, w = frame.shape
		frame[:int(0.7*h),int(0.15*w):] = 0
		# show_img(frame)
		return frame

	stab = simple_stabilisation(frames, ignore_score=0.1, crop=False, registration_filter=mask)
	fio.write_video_from_frames(stab, "frames4/stabilised1.avi")



	FOLDER_PATH = "frames1"
	frames = fio.get_frame_range_from_folder(FOLDER_PATH + "/frames", "tif", 0, 50)
	print("stabilising...")

	RUN_STABILISATION_TESTS = False

	if RUN_STABILISATION_TESTS:

		if not os.path.exists(FOLDER_PATH + "/stabilisation_tests"):
			os.makedirs(FOLDER_PATH + "/stabilisation_tests")


		stab = simple_stabilisation(frames, crop=False)
		fio.write_video_from_frames(stab, FOLDER_PATH + "/stabilisation_tests/test1.avi")

		stab = simple_stabilisation(frames, crop=False, mark_keypoints=True)
		fio.write_video_from_frames(stab, FOLDER_PATH + "/stabilisation_tests/test2.avi")

		stab = simple_stabilisation(frames, crop=True)
		fio.write_video_from_frames(stab, FOLDER_PATH + "/stabilisation_tests/test3.avi")

		stab = simple_stabilisation(frames, ignore_score=0.3, crop=False)
		fio.write_video_from_frames(stab, FOLDER_PATH + "/stabilisation_tests/test4.avi")

		stab = simple_stabilisation(frames, ignore_score=0.1, crop=True)
		fio.write_video_from_frames(stab, FOLDER_PATH + "/stabilisation_tests/test5.avi")


		async_frames = fio.sim_async_input(frames, 30, print_timing=True)
		stab = simple_stabilisation(async_frames, ignore_score=0.1, crop=False)
		fio.write_video_from_frames(stab, FOLDER_PATH + "/stabilisation_tests/test6.avi")

		async_frames = fio.sim_async_input(frames, 30, print_timing=True)
		stab = simple_stabilisation(async_frames, ignore_score=0.1, crop=True)
		fio.write_video_from_frames(stab, FOLDER_PATH + "/stabilisation_tests/test7.avi")


		stab = simple_stabilisation(frames, split_score=0.1, crop=False)
		for i, seq in enumerate(stab):
			fio.write_video_from_frames(seq, FOLDER_PATH + "/stabilisation_tests/test{}.avi".format(100+i))
		
		stab = simple_stabilisation(frames, split_score=0.5, crop=True)
		for i, seq in enumerate(stab):
			fio.write_video_from_frames(seq, FOLDER_PATH + "/stabilisation_tests/test{}.avi".format(200+i))










# ------------ THIS FUNCTION SHOULDN'T BE USED ------------
''' This was an attempt to detect relative motion of conjunctival
	and scleral vessels and is based on an older version of the
	simple_stabilisation function. 
	Kept because it is a kind of proof-of-concept. '''
def two_layer_stabilisation(frames, similarity_threshold=0):

	frame0 = frames[0]

	p0 = cv2.goodFeaturesToTrack(frame0, 1000, 0.01, 15)

	frame0 = cv2.GaussianBlur(frame0, (5, 5), 2)
	# kernel = np.ones((3,3),np.uint8)
	# frame0 = cv2.erode(frame0, kernel, iterations=3)
	transforms = np.zeros((len(frames), 5))
	transforms[:,3] = 1

	X = []

	for i, frame in enumerate(frames):
		if i == 0:
			continue

		frame = cv2.GaussianBlur(frame, (5, 5), 2)
		# kernel = np.ones((3,3),np.uint8)
		# frame = cv2.erode(frame, kernel, iterations=3)

		# show_img(frame)

		# frame = resize_img(frame, 1./2.)

		# calculate optical flow
		p1, st, err = cv2.calcOpticalFlowPyrLK(frame0, frame, p0, None)

		# Select good points
		good_p0 = p0[st==1]
		good_p1 = p1[st==1]

		deltas = good_p1 - good_p0

		deltas_mean = np.mean(deltas, 0)
		displacements_from_mean = deltas - deltas_mean
		deltas_std = np.std(displacements_from_mean)
		dists_from_mean = np.hypot(displacements_from_mean[:,0],
									displacements_from_mean[:,1])

		# print(deltas_mean)
		# print(deltas_std)

		filter_array = dists_from_mean<5*deltas_std
		good_p0 = good_p0[filter_array]
		good_p1 = good_p1[filter_array]
		deltas = deltas[filter_array]
		z = np.float32(deltas)

		criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
		flags = cv2.KMEANS_RANDOM_CENTERS

		# Apply KMeans
		compactness,labels,centers = cv2.kmeans(z,2,None,criteria,10,flags)
		l = labels.ravel()
		if len(l[l==0]) >= len(l) / 2:
			l = 1 - l
			centers = centers[::-1]

		dists_from_centres = np.zeros(deltas.shape[0])
		d1 = centers[0] - deltas[l==0]

		dists_from_centres[l==0] = np.hypot(d1[:,0], d1[:,1])
		dists_from_centres[l==0] = dists_from_centres[l==0]/dists_from_centres[l==0].max()

		d2 = centers[1] - deltas[l==1]
		dists_from_centres[l==1] = np.hypot(d2[:,0], d2[:,1])
		dists_from_centres[l==1] = dists_from_centres[l==1]/dists_from_centres[l==1].max()

		# A = z[l==0]
		# B = z[l==1]

		# plt.scatter(A[:,0], A[:,1], s=0.1)
		# plt.scatter(B[:,0], B[:,1], s=0.1)
		# plt.show()

		X.append([good_p0, l, z, dists_from_centres])

		transform_mat, inliers = cv2.estimateAffinePartial2D(good_p0, good_p1)

		if len(transform_mat[transform_mat!=0]) == 0 or len(good_p0) < len(p0) * 0.9:
			transforms[i] = np.array([0, 0, 0, 1, 0])
			continue

		dx, dy, da, s = params_from_transform_matrix(transform_mat)

		transforms[i] = np.array([dx, dy, da, s, len(good_p0)])


	max_matched = transforms[:,4].max()
	out_frames = np.zeros(((len(frames),) + frame0.shape))
	height, width = frame0.shape[:2]
	res = (width, height)

	for i, frame in enumerate(frames):

		if transforms[i,4] < max_matched * similarity_threshold and i > 0:
			print("skipping frame {} - {} out of {} matched".format(i, transforms[i,4], max_matched))
			continue

		M = transform_matrix_from_params(*transforms[i,:4])
		M_inv = cv2.invertAffineTransform(M)

		adjusted = cv2.warpAffine(frame, M_inv, res)

		if i > 0:
			p, l, z, d = X[i-1]
			for j in range(len(z)):
				if l[j]:
					adjusted = cv2.circle(adjusted, tuple(p[j]), 1, 255, 2)
				else:
					adjusted = cv2.circle(adjusted, tuple(p[j]), 1, 0, 2)

		out_frames[i] = adjusted

	# remove skipped frames (will be all zeros)
	out_frames = out_frames[np.any(out_frames, axis=(1,2))]
	out_frames = np.array([rescale_to_ubyte(x) for x in out_frames])

	return out_frames
