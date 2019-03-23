# -*- coding: utf-8 -*-

#
# * Copyright (c) 2009-2016. Authors: see NOTICE file.
# *
# * Licensed under the Apache License, Version 2.0 (the "License");
# * you may not use this file except in compliance with the License.
# * You may obtain a copy of the License at
# *
# *      http://www.apache.org/licenses/LICENSE-2.0
# *
# * Unless required by applicable law or agreed to in writing, software
# * distributed under the License is distributed on an "AS IS" BASIS,
# * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# * See the License for the specific language governing permissions and
# * limitations under the License.
# */


__author__ = "Vandaele Rémy <remy.vandaele@ulg.ac.be>"
__contributors__ = ["Marée Raphaël <raphael.maree@ulg.ac.be>"]
__copyright__ = "Copyright 2010-2016 University of Liège, Belgium, http://www.cytomine.be/"

from neubiaswg5.helpers.data_upload import imwrite
from cytomine.models import *
import numpy as np
from ldmtools import *
from scipy.stats import multivariate_normal
from sumproduct import Variable,Factor,FactorGraph
from ldmtools import *
import numpy as np
import scipy.ndimage as snd
from sklearn.externals import joblib
import sys, os
from neubiaswg5 import CLASS_LNDDET
from neubiaswg5.helpers import NeubiasJob, prepare_data, get_discipline, upload_data, upload_metrics
from cytomine.models import Job, AttachedFile, Property
import joblib

def filter_perso(img, filter_size):
	if np.sum(img > 0) <= 1:
		return img
	(h, w) = img.shape
	y, x = np.where(img > 0.)
	nimg = np.zeros((h, w))
	ymin = np.clip(y - filter_size, a_min=0, a_max=h-1).astype('int')
	ymax = np.clip(y + filter_size+1, a_min=0, a_max=h-1).astype('int')
	xmin = np.clip(x - filter_size, a_min=0, a_max=w-1).astype('int')
	xmax = np.clip(x + filter_size+1, a_min=0, a_max=w-1).astype('int')
	for i in range(x.size):
		val = img[y[i], x[i]]
		if val == np.max(img[ymin[i]:ymax[i], xmin[i]:xmax[i]]) or filter_size==0:
			nimg[y[i], x[i]] = val
	return nimg


def agregation_phase_2(repository, image_number, ip, probability_maps, reg, delta, feature_offsets, filter_size, beta, n_iterations):
	img = makesize(snd.zoom(readimage(repository, image_number, image_type='tif'), delta), 1)
	(h, w, nldms) = probability_maps.shape
	nldms -= 1
	mh = h - 1
	mw = w - 1
	for iteration in range(n_iterations):
		y, x = np.where(probability_maps[:, :, ip] >= beta * np.max(probability_maps[:, :, ip]))
		if y.size>10000:
			(y, x) = np.unravel_index(np.argsort(probability_maps[:, :, ip].ravel()), (h, w))
			y = y[:10000]
			x = x[:10000]
		dataset = dataset_from_coordinates(img, x + 1, y + 1, feature_offsets)
		offsets = reg.predict(dataset)
		n_x = (x - offsets[:, 0]).clip(min=0, max=mw).astype('int')
		n_y = (y - offsets[:, 1]).clip(min=0, max=mh).astype('int')
		y = y.astype('int')
		x = x.astype('int')
		new_pmap = np.zeros((h, w))
		for i in range(n_x.size):
			new_pmap[n_y[i], n_x[i]] += probability_maps[y[i], x[i], ip]
		probability_maps[:, :, ip] = new_pmap
		probability_maps[0, :, ip] = 0
		probability_maps[:, 0, ip] = 0
		probability_maps[mh, :, ip] = 0
		probability_maps[:, mw, ip] = 0

	return filter_perso(probability_maps[:, :, ip], filter_size)


def build_bmat_phase_3(xc, yc, T, x_candidates, y_candidates, edges, sde):
	B_mat = {}  # np.zeros((ncandidates,ncandidates,T*nldms))

	c = 0
	(nims, nldms) = xc.shape
	c1 = np.zeros((nims, 2))
	c2 = np.zeros((nims, 2))

	std_matrix = np.eye(2) * (sde ** 2)

	for ip in range(nldms):
		c1[:, 0] = xc[:, ip]
		c1[:, 1] = yc[:, ip]
		for ipl in edges[ip, :]:
			rel = np.zeros((len(x_candidates[ip]), len(x_candidates[ipl])))

			c2[:, 0] = xc[:, ipl]
			c2[:, 1] = yc[:, ipl]

			diff = c1 - c2

			gaussians = [multivariate_normal(diff[i, :], std_matrix) for i in range(nims)]

			for cand1 in range(len(x_candidates[ip])):
				pos1 = np.array([x_candidates[ip][cand1], y_candidates[ip][cand1]])
				for cand2 in range(len(x_candidates[ipl])):
					pos2 = np.array([x_candidates[ipl][cand2], y_candidates[ipl][cand2]])
					diff = pos1 - pos2
					rel[cand1, cand2] = np.max([gaussians[i].pdf(diff) for i in range(nims)])
			B_mat[(ip, ipl)] = rel / multivariate_normal([0, 0], std_matrix).pdf([0, 0])

	for (ip, ipl) in B_mat.keys():
		rel = B_mat[(ip, ipl)]
		for i in range(len(x_candidates[ip])):
			rel[i, :] = rel[i, :] / np.sum(rel[i, :])
		B_mat[(ip, ipl)] = rel
	return B_mat

def compute_final_solution_phase_3(xc, yc, probability_map_phase_2, ncandidates, sde, delta, T, edges):
	(height, width, nldms) = probability_map_phase_2.shape
	# nldms-=1
	x_candidates = []  # np.zeros((nldms,ncandidates))
	y_candidates = []  # np.zeros((nldms,ncandidates))

	for i in range(nldms):
		val = -np.sort(-probability_map_phase_2[:, :, i].flatten())[ncandidates]
		if (val > 0):
			(y, x) = np.where(probability_map_phase_2[:, :, i] >= val)
		else:
			(y, x) = np.where(probability_map_phase_2[:, :, i] > val)

		if (y.size > ncandidates):
			vals = -probability_map_phase_2[y, x, i]
			order = np.argsort(vals)[0:ncandidates]
			y = y[order]
			x = x[order]

		x_candidates.append(x.tolist())
		y_candidates.append(y.tolist())

	b_mat = build_bmat_phase_3(xc, yc, T, x_candidates, y_candidates, edges, sde)

	g = FactorGraph(silent=True)
	nodes = [Variable('x%d' % i, len(x_candidates[i])) for i in range(nldms)]
	for ip in range(nldms):
		for ipl in edges[ip, :].astype(int):
			g.add(Factor('f2_%d_%d' % (ip, ipl), b_mat[(ip, ipl)]))
			g.append('f2_%d_%d' % (ip, ipl), nodes[ip])
			g.append('f2_%d_%d' % (ip, ipl), nodes[ipl])

	for i in range(nldms):
		v = probability_map_phase_2[np.array(y_candidates[i]).astype('int'), np.array(x_candidates[i]).astype('int'), i]
		g.add(Factor('f1_%d' % i, v / np.sum(v)))
		g.append('f1_%d' % i, nodes[i])

	g.compute_marginals()

	x_final = np.zeros(nldms)
	y_final = np.zeros(nldms)

	for i in range(nldms):
		amin = np.argmax(g.nodes['x%d' % i].marginal())
		x_final[i] = x_candidates[i][amin]
		y_final[i] = y_candidates[i][amin]

	return x_final / delta, (y_final / delta)

def find_by_attribute(att_fil, attr, val):
	return next(iter([i for i in att_fil if hasattr(i, attr) and getattr(i, attr) == val]), None)

def main():
	with NeubiasJob.from_cli(sys.argv) as conn:
		problem_cls = get_discipline(conn, default=CLASS_LNDDET)
		conn.job.update(status=Job.RUNNING, progress=0, statusComment="Initialization of the prediction phase")
		in_images, gt_images, in_path, gt_path, out_path, tmp_path = prepare_data(problem_cls, conn, is_2d=True, **conn.flags)
		train_job = Job().fetch(conn.parameters.model_to_use)
		properties = PropertyCollection(train_job).fetch()
		str_terms = ""
		for prop in properties:
			if prop.fetch(key='id_terms') != None:
				str_terms = prop.fetch(key='id_terms').value
		term_list = [int(x) for x in str_terms.split(' ')]
		attached_files = AttachedFileCollection(train_job).fetch()
		model_file = find_by_attribute(attached_files, "filename", "model_phase1.joblib")
		model_filepath = os.path.join(in_path, "model_phase1.joblib")
		model_file.download(model_filepath, override=True)
		clf = joblib.load(model_filepath)
		pr_ims = [int(p) for p in conn.parameters.cytomine_predict_images.split(',')]
		offset_file = find_by_attribute(attached_files, "filename", "offsets_phase1.joblib")
		offset_filepath = os.path.join(in_path, "offsets_phase1.joblib")
		offset_file.download(offset_filepath, override=True)
		feature_offsets_1 = joblib.load(offset_filepath)
		train_parameters = {}
		for hashmap in train_job.jobParameters:
			train_parameters[hashmap['name']] = hashmap['value']
		train_parameters['model_delta'] = float(train_parameters['model_delta'])
		train_parameters['model_sde'] = float(train_parameters['model_sde'])
		train_parameters['model_T'] = int(train_parameters['model_T'])
		for j in conn.monitor(pr_ims, start=10, end=33, period=0.05,prefix="Phase 1 for images..."):
			probability_map = probability_map_phase_1(in_path, j, clf, feature_offsets_1, float(train_parameters['model_delta']))
			filesave = os.path.join(out_path, 'pmap_%d.npy'%j)
			np.savez_compressed(filesave,probability_map)

		clf = None

		coords_file = find_by_attribute(attached_files, "filename", "coords.joblib")
		coords_filepath = os.path.join(in_path, "coords.joblib")
		coords_file.download(coords_filepath, override=True)
		(Xc, Yc) = joblib.load(coords_filepath)

		for j in conn.monitor(pr_ims, start=33, end=66, period=0.05,prefix="Phase 2 for images..."):
			filesave = os.path.join(out_path, 'pmap_%d.npy.npz' % j)
			probability_map = np.load(filesave)['arr_0']
			for id_term in term_list:
				reg_file = find_by_attribute(attached_files, "filename", "reg_%d_phase2.joblib"%id_term)
				reg_filepath = os.path.join(in_path, "reg_%d_phase2.joblib"%id_term)
				reg_file.download(reg_filepath, override=True)
				reg = joblib.load(reg_filepath)

				off_file = find_by_attribute(attached_files, "filename", 'offsets_%d_phase2.joblib' % id_term)
				off_filepath = os.path.join(in_path, 'offsets_%d_phase2.joblib' % id_term)
				off_file.download(off_filepath, override=True)
				feature_offsets_2 = joblib.load(off_filepath)

				probability_map_phase_2 = agregation_phase_2(in_path, j, id_term, probability_map, reg, train_parameters['model_delta'], feature_offsets_2, conn.parameters.model_filter_size, conn.parameters.model_beta, conn.parameters.model_n_iterations)
				filesave = os.path.join(out_path, 'pmap2_%d_%d.npy' % (j, id_term))
				np.savez_compressed(filesave, probability_map_phase_2)

		edge_file = find_by_attribute(attached_files, "filename", "model_edges.joblib")
		edge_filepath = os.path.join(in_path, "model_edges.joblib")
		edge_file.download(edge_filepath, override=True)
		edges = joblib.load(edge_filepath)
		for j in conn.monitor(pr_ims, start=66, end=90, period=0.05,prefix="Phase 3 for images..."):
			filesave = os.path.join(out_path, 'pmap2_%d_%d.npy.npz' % (j, term_list[0]))
			probability_map = np.load(filesave)['arr_0']
			(hpmap,wpmap) = probability_map.shape
			probability_volume = np.zeros((hpmap,wpmap,len(term_list)))
			probability_volume[:,:,0] = probability_map
			for i in range(1,len(term_list)):
				filesave = os.path.join(out_path, 'pmap2_%d_%d.npy.npz' % (j, term_list[0]))
				probability_volume[:,:,i] = np.load(filesave)['arr_0']
			x_final, y_final = compute_final_solution_phase_3(Xc, Yc, probability_volume, conn.parameters.model_n_candidates, train_parameters['model_sde'], train_parameters['model_delta'], train_parameters['model_T'], edges)
			lbl_img = np.zeros((hpmap, wpmap), 'uint8')
			for i in range(x_final.size):
				x = int(x_final[i])
				y = int(y_final[i])
				lbl_img[y, x] = term_list[i]
			imwrite(path=os.path.join(out_path, '%d.tif' % j), image=lbl_img.astype(np.uint8), is_2d=True)
		upload_data(problem_cls, conn, in_images, out_path, **conn.flags, is_2d=True, monitor_params={"start": 90, "end": 95, "period": 0.1})
		conn.job.update(progress=90, statusComment="Computing and uploading metrics (if necessary)...")
		upload_metrics(problem_cls, conn, in_images, gt_path, out_path, tmp_path, **conn.flags)
		conn.job.update(status=Job.TERMINATED, progress=100, statusComment="Finished.")

if __name__ == "__main__":
	main()
