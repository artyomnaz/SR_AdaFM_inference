import os
import shutil
import time

import options.options as option
import utils.util as util
from collections import OrderedDict
from data import create_dataset, create_dataloader
from models import create_model

class Test:
	def __init__(self):
		self.opt = { 'name': 'basicmodel', 
				'model': 'sr', 
				'crop_size': 4, 
				'gpu_ids': [0],
				'interpolate_stride': 0.01, 
				'datasets': {
					'test': {
						'name': 'personal_images', 
						'dataroot_LR': '../data', 
						'data_type': 'img'
					}
				}, 
				'path': {
					'root': '../', 
					'pretrain_model_G': '../AdaFM_23_09_clear_dataset.pth', 
					'results_root': '../results', 
				}, 
				'network_G': {
					'which_model_G': 'adaptive_resnet', 
					'norm_type': 'basic', 
					'nf': 64, 
					'nb': 16, 
					'in_nc': 3, 
					'out_nc': 3, 
					'adafm_ksize': 1
				},
				'is_train': False,
				'train': False
			 }
				
	def inference(self, path_to_image):
		# copy image
		filename = os.path.basename(path_to_image)
		if not os.path.exists('../data'):
			os.mkdir('../data')
		if not os.path.exists('../results'):
			os.mkdir('../results')
		shutil.copyfile(path_to_image, f'../data/{filename}')
		
		# create the model
		model = create_model(self.opt)
		
		# create generator
		self.test_loaders = []
		for phase, dataset_opt in sorted(self.opt['datasets'].items()):
			test_set = create_dataset(dataset_opt)
			test_loader = create_dataloader(test_set, dataset_opt)
			self.test_loaders.append(test_loader)
			
		for test_loader in self.test_loaders:
			test_set_name = test_loader.dataset.opt['name']
			test_start_time = time.time()
			dataset_dir = self.opt['path']['results_root']
			util.mkdir(dataset_dir)
			test_results = OrderedDict()
		
		# test the model
		for data in test_loader:
			model.feed_data(data, need_HR=False)
			img_path = data['LR_path'][0]
			img_name = os.path.splitext(os.path.basename(img_path))[0]

			model.test()  # test
			visuals = model.get_current_visuals(need_HR=False)
			sr_img = util.tensor2img(visuals['SR'])  # uint8

			# save images
			save_img_path = os.path.join(dataset_dir, img_name + '.png')
			util.save_img(sr_img, save_img_path)
		shutil.rmtree('../data', ignore_errors=True)
