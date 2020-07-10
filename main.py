from global_imports import *
from img_utils import *
from maths_utils import *
import vessel_filter
import stabilise
import frame_io as fio
import skeleton as skel
import flow_analysis 


# THIS PROGRAMS DEMONSTRATES SOME OF THE FUNCTIONS
# AND ATTEMPTS TO DO A FLOW ANALYSIS ON ALL APPLICABLE
# VESSELS IN THE FRAMES.


s = Stopwatch()
s.start()

frames = fio.get_frames_from_folder("frames1/unstabilised")
frames = stabilise.simple_stabilisation(frames, crop=True, ignore_score=0.5)

mean = np.mean(frames, axis=0)
mean_c = cv2.cvtColor(mean.astype(np.uint8), cv2.COLOR_GRAY2BGR)

filtered = vessel_filter.apply_frangi(mean, [1,2,3,5])

skeleton = skel.skeletonisation(filtered)
G = skel.skel_to_graph(skeleton)
G = skel.simplify_graph(G, skeleton.shape)

# G = skel.filter_graph_edges(G, lambda x: x["score"] > 220)
# img = skel.graph_to_img(G, mean_c, dilate=True, colourcode=True)
# show_img(img)

G = skel.order_graph_cnt_pixels(G)
G = skel.make_graph_line_chains(G)


show_img(filtered)

img = mean_c.copy()
deltas = flow_analysis.get_frame_deltas(frames)

min_cnt_length = 0.
cnts = []
ends = []
j = 0
for u,v,w in list(G.edges):
	j+=1

	try:
		edge_pts = G[u][v][w]["point chain"]
	except:
		print("something went wrong... no point chain attribute or edge doesn't exist")
		print(G[u][v][w])
		continue


	segment_data = {
		"x"    : [],
		"v_x"  : [],
		"v_sd" : [],
		"v_mean" : [],
		"v_peak" : [],
		"Q" : [],
		"width" : []
	}

	# show_list = [(244, 166), (461, 441), (194, 692)]
	# show_graphs = (u,v) in show_list or (v,u) in show_list
	show_graphs = False
	show_space_time_images = False
	some_valid_data = False

	for i in range(len(edge_pts)-1):

		pts = [edge_pts[i], edge_pts[i+1]]

		dist = np.hypot(*(pts[1] - pts[0]))

		slide_dist = 15
		if dist > 2*slide_dist+1:
			vsl_data = flow_analysis.data_from_vessel_pts(pts, filtered, deltas, slide_dist, show_graphs, show_space_time_images)
		else:
			continue

		for key in vsl_data.keys():
			segment_data[key].append(vsl_data[key])

		if vsl_data["v_mean"] is not None:
			some_valid_data = True

			mean_pt = tuple(np.mean(pts, axis=0).astype(int))
			v_txt = "{:.2f}".format(np.abs(vsl_data["v_mean"]))

			cv2_text(img, v_txt, mean_pt, (255, 0, 0), 0.35, 1)

			if vsl_data["v_mean"] < 0:
				cv2.arrowedLine(img, tuple(pts[1]), tuple(pts[0]), (0, 255, 0), 1, tipLength=4/dist)
			else:
				cv2.arrowedLine(img, tuple(pts[0]), tuple(pts[1]), (0, 255, 0), 1, tipLength=4/dist)


	for pt in edge_pts:
		img[pt[1], pt[0]] = (0, 0, 255)



	# draw node end ids in graph
	if some_valid_data:
		print(j)
		# s.stop()
		# cv2_text(img, str(u), tuple(edge_pts[0]), (0, 0, 255), 0.35, 1)
		# cv2_text(img, str(v), tuple(edge_pts[-1]), (0, 0, 255), 0.35, 1)


	G[u][v][w]["segment_data"] = segment_data 

show_img(img)










# After investigation cv2.approxPolyDP is identical but
# faster and issues with it were due to the structure of
# the contour data (non-uniqueness, non-orderedness)
# (kept just in case it still turns out to have issues)
#-----------------------------------------------------------------
# this is not a very good way to get straight line approximations.
# only implemented because it was quick and easy
# in order to get a functional tool.
# def perp_dist(start, end, pt):

# 	# print("perpendicular dist")
# 	# print(start, end, pt)
# 	d1 = end - start
# 	# print(d1)
# 	d1_unit = d1 / np.hypot(*d1)
# 	d2 = pt - start

# 	cross_prod = np.cross(d1_unit, d2)

# 	return np.abs(cross_prod)

# approximate contours with the Ramer-Douglas-Peucker algorithm.
# def approx_line_RDP(contour, max_d_allowed=1):
# 	if len(contour) <= 3:
# 		return contour

# 	start = contour[0]
# 	end = contour[-1]
# 	# print(contour)
# 	dists = [perp_dist(start[0], end[0], pt[0]) for pt in contour[1:-1]]
# 	# print(dists)
# 	d_max = np.max(dists)
# 	i = np.argmax(dists) + 1

# 	if d_max > max_d_allowed:
# 		# print("not allowed", contour[:i], contour[i:])
# 		cnt1 = approx_line_RDP(contour[:i],  max_d_allowed)
# 		cnt2 = approx_line_RDP(contour[i:],  max_d_allowed)
# 		results = np.concatenate((cnt1, cnt2))
# 	else:
# 		results = np.array([start, end])

# 	return results
#-----------------------------------------------------------------
