import os
from glob import glob
import numpy as np
from numpy.random import uniform as unif
from lenstronomy.LensModel.lens_model import LensModel
from lenstronomy.LightModel.light_model import LightModel
from astropy.convolution import convolve, Gaussian2DKernel
from lensing_utils import theta_E_from_M, sp_ray_tracing, lens_source
import h5py
import time

# total number of slurm workers detected
# defaults to 1 if not running under SLURM
N_WORKERS = int(os.getenv('SLURM_ARRAY_TASK_COUNT', 1))

# this worker's array index. Assumes slurm array job is zero-indexed
# defaults to zero if not running under SLURM
THIS_WORKER = int(os.getenv('SLURM_ARRAY_TASK_ID', 0)) ## it starts from 1!!


class GalaxyLenser():
	"""
	Class for the lensing of a single Sersic source galaxy by an SIE + SHEAR of parameters randomly sampled parameters
	"""
	def __init__(self, zl, zs, beta1, beta2, theta1, theta2, npix, shear_bool=True, lens_light_bool=False, noise="gaussian", psf="gaussian", normalize=True, mass_function="beta", mass_function_kw={"a": 6.5, "b": 2.}):

		self.zl = zl
		self.zs = zs
		self.beta1, self.beta2 = beta1, beta2
		self.theta1, self.theta2 = theta1, theta2
		self.npix = npix
		self.add_shear = shear_bool
		self.noise = noise
		self.psf = psf
		self.normalize = normalize
		self.mass_function = mass_function
		self.mass_function_kw = mass_function_kw
		self.add_lens_light = lens_light_bool

		self.sample_mass()
		self.lens_paramsampler()
		self.source_paramsampler()
		if self.add_lens_light:
			self.lens_light_paramsampler()

		self.format_params()

	def sample_mass(self, log_mlow=10.7, log_mhigh=12.):

		if self.mass_function == "beta":
			sample = np.random.beta(**self.mass_function_kw)
		else:
			raise ValueError(f"mass function {self.mass_function} not implemented")

		self.log_lens_mass = GalaxyLenser._min_max_scale(sample, log_mlow, log_mhigh)
		self.lens_mass = 10 ** self.log_lens_mass


	def lens_paramsampler(self):
		if not hasattr(self, "lens_mass"):
			self.sample_mass()

		# SIE (+ SHEAR) lens
		SIE_kwargs = {"theta_E": theta_E_from_M(self.lens_mass, self.zl, self.zs),
					  "e1": unif(low=SIE_pb["e1"][0], high=SIE_pb["e1"][1]),
					  "e2": unif(low=SIE_pb["e2"][0], high=SIE_pb["e2"][1]),
					  "center_x": unif(low=SIE_pb["center_x"][0], high=SIE_pb["center_x"][1]),
					  "center_y": unif(low=SIE_pb["center_y"][0], high=SIE_pb["center_y"][1])}

		self.lens_kwargs = [SIE_kwargs]
		self.lens_redshift_list = [self.zl]
		self.lens_model_list = ["SIE"]

		if self.add_shear:
			SHEAR_kwargs = {"gamma1": unif(low=SHEAR_pb["gamma1"][0], high=SHEAR_pb["gamma1"][1]),
							"gamma2": unif(low=SHEAR_pb["gamma2"][0], high=SHEAR_pb["gamma2"][1]),
							"ra_0": unif(low=SHEAR_pb["ra_0"][0], high=SHEAR_pb["ra_0"][1]),
							"dec_0": unif(low=SHEAR_pb["dec_0"][0], high=SHEAR_pb["dec_0"][1])}

			self.lens_kwargs.append(SHEAR_kwargs)
			self.lens_redshift_list.append(self.zl)
			self.lens_model_list.append("SHEAR")


	def source_paramsampler(self):
		# Sersic Ellise source
		SERSIC_E_kwargs = {"amp": unif(low=SERSIC_E_pb["amp"][0], high=SERSIC_E_pb["amp"][1]),
					  	 "R_sersic": unif(low=SERSIC_E_pb["R_sersic"][0], high=SERSIC_E_pb["R_sersic"][1]),
					  	 "n_sersic": unif(low=SERSIC_E_pb["n_sersic"][0], high=SERSIC_E_pb["n_sersic"][1]),
						 "e1": unif(low=SERSIC_E_pb["e1"][0], high=SERSIC_E_pb["e1"][1]),
						 "e2": unif(low=SERSIC_E_pb["e2"][0], high=SERSIC_E_pb["e2"][1]),
						 "center_x": unif(low=SERSIC_E_pb["center_x"][0], high=SERSIC_E_pb["center_y"][1]),
						 "center_y": unif(low=SERSIC_E_pb["center_y"][0], high=SERSIC_E_pb["center_y"][1])}

		self.source_kwargs = [SERSIC_E_kwargs]
		self.source_redshift = self.zs
		self.source_model_list = ["SERSIC_ELLIPSE"]


	def lens_light_paramsampler(self):
		if not hasattr(self, "lens_kwargs"):
			self.lens_paramsampler()

		# Sersic Ellise lens light
		SERSIC_E_kwargs = {"amp": unif(low=SERSIC_E_lens_pb["amp"][0], high=SERSIC_E_lens_pb["amp"][1]),
					  	 "R_sersic": unif(low=SERSIC_E_lens_pb["R_sersic"][0], high=SERSIC_E_lens_pb["R_sersic"][1]),
					  	 "n_sersic": unif(low=SERSIC_E_lens_pb["n_sersic"][0], high=SERSIC_E_lens_pb["n_sersic"][1]),
						 "e1": self.lens_kwargs[0]["e1"] + unif(-d_e_lim, d_e_lim),
						 "e2": self.lens_kwargs[0]["e2"] + unif(-d_e_lim, d_e_lim),
						 "center_x": self.lens_kwargs[0]["center_x"] + unif(-d_center_lim, d_center_lim),
						 "center_y": self.lens_kwargs[0]["center_y"] + unif(-d_center_lim, d_center_lim)}

		self.lens_light_kwargs = [SERSIC_E_kwargs]
		self.lens_light_redshift = self.zl
		self.lens_light_model_list = ["SERSIC_ELLIPSE"]


	def produce_lens(self):
		# Source light
		self.source_light_model = LightModel(self.source_model_list)
		self.source_light = self.source_light_model.surface_brightness(self.beta1, self.beta2, self.source_kwargs)

		# Raytracing and lensing
		self.lens_model = LensModel(lens_model_list=self.lens_model_list,
									z_source=self.zs,
									lens_redshift_list=self.lens_redshift_list)

		self.beta1_def, self.beta2_def = self.lens_model.ray_shooting(self.theta1, self.theta2, self.lens_kwargs)
		self.lensed_src = lens_source(self.theta1, self.theta2, self.beta1_def, self.beta2_def, self.source_light, self.npix)


		if self.add_lens_light:
			# Lens light
			self.lens_light_model = LightModel(self.lens_light_model_list)
			self.lens_light = self.lens_light_model.surface_brightness(self.theta1, self.theta2, self.lens_light_kwargs)

			self.lensed_image = self.lensed_src + self.lens_light
		else:
			self.lensed_image = self.lensed_src

		if self.normalize:
			self.lensed_image = self.lensed_image / np.max(np.abs(self.lensed_image))


	def corrupt_image(self, image):

		# PSF first
		if self.psf is not None:
			if self.psf == "gaussian":
				kernel = Gaussian2DKernel(x_stddev=2)
			else:
				raise ValueError(f"psf of type '{self.psf}' not implemented")
			image = convolve(image, kernel)

		# Noise
		if self.noise is not None:
			if self.noise == "poisson":
				mask = np.random.poisson(image)
			elif self.noise == "gaussian":
				mask = np.random.normal(loc=0., scale=0.02*np.max(np.abs(self.lensed_image)), size=(self.npix, self.npix))
			else:
				raise ValueError(f"noise of type '{self.noise}' not implemented")
			image += mask

		return image


	def format_params(self):
		self.lens_param_values = list(self.lens_kwargs[0].values())
		self.lens_param_keys = list(self.lens_kwargs[0].keys())

		if self.add_shear:
			# don't save SHEAR's 'ra_0' and 'dec_0' params, since always 0
			self.lens_param_values += list(self.lens_kwargs[1].values())[:-2]
			self.lens_param_keys += list(self.lens_kwargs[1].keys())[:-2]

		self.source_param_values = list(self.source_kwargs[0].values())
		self.source_param_keys = list(self.source_kwargs[0].keys())

		self.param_values = self.lens_param_values + self.source_param_values
		self.param_keys = self.lens_param_keys + self.source_param_keys

		if self.add_lens_light:
			self.lens_light_param_values = list(self.lens_light_kwargs[0].values())
			self.lens_light_param_keys = list(self.lens_light_kwargs[0].keys())

			self.param_values = self.param_values + self.lens_light_param_values
			self.param_keys = self.param_keys + self.lens_light_param_keys

		for i, v in enumerate(self.param_keys):
			if i in range(7):
				v_desc = "__lens_SIE_SHEAR"
			elif i in range(7, 14):
				v_desc = "__src_SERSIC_E"
			elif i in range(14, 21):
				v_desc = "__lenslight_SERSIC_E"

			self.param_keys[i] = v + v_desc

	@staticmethod
	def _min_max_scale(x, a, b):
		return a + x*(b-a)


def produce_dataset(output_path, set_size, wmode="w-", rpf=50, gen_params={}):
	"""
	Produce a dataset of lensed galaxy image realizations
	:param output_path: path to dataset directory (str)
	:param set_size: number of examples in dataset to be produced (int)
	:param mode: specifier for type of dataset to produce (str)
	:param wmode: h5py write mode (str)
	:param rpf: number of realizations to save per file (int)
	:param gen_params: kwargs for generator function
	"""
	dataset_name = os.path.basename(output_path)
	if dataset_name == "":
		raise ValueError("Invalid output_path")

	if not os.path.isdir(output_path):
		os.makedirs(output_path, exist_ok=True)

	def basegrp_header(grp):
		header_dic = {"set_size": set_size,
					  "zl": gen_params["zl"],
					  "zs": gen_params["zs"],
					  "npix": gen_params["npix"],
					  "lens_fov": gen_params["lens_fov"],
					  "src_fov": gen_params["src_fov"],
					  "lens": gen_params["lens"],
					  "shear": gen_params["shear_bool"],
					  "source": gen_params["source"],
					  "params": gen_params["param_keys"],
					  "noise": gen_params["noise"],
					  "rpf": rpf}

		if gen_params["lens_light_bool"]:
			header_dic.update({"lens_light": gen_params["lens_light"]})

		grp.attrs["dataset_descriptor"] = str(header_dic)


	# SLURM array job separation
	real_ids = np.arange(set_size)
	rpw = set_size // N_WORKERS			# realizations per worker
	work_ids = real_ids[THIS_WORKER * rpw : (THIS_WORKER+1) * rpw]
	if THIS_WORKER + 1 == N_WORKERS:
		work_ids = real_ids[THIS_WORKER * rpw:]

	prod_start = time.time()
	ind_file = (rpw // rpf + 1) * THIS_WORKER
	output_file = h5py.File(os.path.join(output_path, f"{dataset_name}_{ind_file:04d}.h5"), mode=wmode)

	# GL bidon for param values
	GL_null = GalaxyLenser(**keyword_parse_GL(gen_params))
	GL_null.format_params()
	gen_params.update({"param_keys": GL_null.param_keys})
	basegrp = output_file.create_group("base")
	basegrp_header(basegrp)

	lens_dset = basegrp.create_dataset(f"lenses", (rpf, gen_params["npix"], gen_params["npix"]), dtype=float, compression="gzip")
	param_dset = basegrp.create_dataset(f"params", (rpf, len(gen_params["param_keys"])), dtype=float, compression="gzip")
	if gen_params["lens_light_bool"]:
		deblend_dset = basegrp.create_dataset(f"lenses_deblended", (rpf, gen_params["npix"], gen_params["npix"]), dtype=float, compression="gzip")

	for i, r in enumerate(work_ids):
		if i % rpf == 0 and i!= 0:
			output_file.close()
			ind_file += 1
			output_file = h5py.File(os.path.join(output_path, f"{dataset_name}_{ind_file:04d}.h5"), mode=wmode)
			basegrp = output_file.create_group("base")
			basegrp_header(basegrp)

			lens_dset = basegrp.create_dataset(f"lenses_{ind_file:04d}", (rpf, gen_params["npix"], gen_params["npix"]),
											   dtype=float, compression="gzip")
			param_dset = basegrp.create_dataset(f"params_{ind_file:04d}", (rpf, len(gen_params["param_keys"])), dtype=float,
											compression="gzip")
			if gen_params["lens_light_bool"]:
				deblend_dset = basegrp.create_dataset(f"lenses_deblended_{ind_file:04d}",
													  (rpf, gen_params["npix"], gen_params["npix"]), dtype=float,
													  compression="gzip")

		# generation
		GL = GalaxyLenser(**keyword_parse_GL(gen_params))
		GL.produce_lens()
		corrupt_lensed_image = GL.corrupt_image(GL.lensed_image)

		lens_dset[i%rpf, ...] = corrupt_lensed_image
		param_dset[i%rpf, ...] = np.array(GL.param_values)

		if gen_params["lens_light_bool"]:
			corrupt_lens_deblended = GL.corrupt_image(GL.lensed_src)
			deblend_dset[i%rpf, ...] = corrupt_lens_deblended

	prod_end = time.time()
	timer = prod_end - prod_start
	print(f"dataset produced in {int(timer // 60)} minutes {(timer % 60):.02f} seconds")


def keyword_parse_GL(keywords_dic):

	args_GL = {}

	required_keys = ["zs", "zl", "beta1", "beta2", "theta1", "theta2", "npix", "shear_bool", "lens_light_bool", "noise", "psf", "normalize"]

	for key in required_keys:
		if key not in keywords_dic:
			continue
		else:
			args_GL[key] = keywords_dic[key]

	return args_GL


if __name__ == "__main__":
	import argparse

	parser = argparse.ArgumentParser(
		description="Produce a dataset of strong lens images with or without lens light")

	parser.add_argument("--path_in", type=str, default=os.path.join(os.getenv("ASTROMATIC_PATH")),
						help="principal repo path")
	parser.add_argument("--path_out", type=str,	default=os.path.join(os.getenv("ASTROMATIC_PATH"), "Problems", "P2_lens_inference", "datasets"),
						help="output path")
	parser.add_argument("--rpf", type=int, default=50, help="number of realizations per file")
	parser.add_argument("--ow", action="store_const", const="w", default="w-",
						help="toggle to overwrite existing dataset in same path")
	parser.add_argument("--dataset_name", type=str, required=True, help="name of dataset directory")
	parser.add_argument("--data_type", type=str, default="lens", help="type of data, in ['lens', 'lens_light']")
	parser.add_argument("--zl", type=float, default="0.5", help="redshift of lens")
	parser.add_argument("--zs", type=float, default="1.5", help="redshift of source")
	parser.add_argument("--npix", type=int, default=128, help="number of pixels at which to create data")
	parser.add_argument("--pixel_scale", type=float, default=0.05, help="scale of a single pixel in arcsecs")
	parser.add_argument("--shear", type=bool, default=True, help="toggle for inclusion of shear")
	parser.add_argument("--noise", type=str, default="poisson", help="type of noise to corrupt images, in ['poisson']")
	parser.add_argument("--psf", type=str, default="gaussian", help="type of psf to corrupt images, in ['gaussian']")
	parser.add_argument("--normalize", action="store_true", help="normalize lensed images")
	parser.add_argument("--size", type=int, required=True, help="size of dataset to produce")
	parser.add_argument("--seed", type=int, default=None, help="random seed")

	args = parser.parse_args()
	np.random.seed(args.seed)
	gen_params = {
		"npix": args.npix,
		"zl": args.zl,
		"zs": args.zs,
		"pixel_scale": args.pixel_scale,
		"lens_fov": args.npix * args.pixel_scale,
		"src_fov": args.npix * args.pixel_scale / 2,
		"lens": "SIE",
		"shear_bool": args.shear,
		"lens_light_bool": True if "light" in args.data_type else False,
		"source": "SERSIC_ELLIPSE",
		"noise": args.noise,
		"psf": args.psf,
		"normalize": args.normalize
	}

	if "light" in args.data_type:
		gen_params.update({"lens_light": "SERSIC_ELLIPSE"})

	# source plane coordinates
	src_grid_side = np.linspace(-gen_params["src_fov"] / 2, gen_params["src_fov"] / 2, gen_params["npix"])
	beta1, beta2 = np.meshgrid(src_grid_side, src_grid_side)

	# lens plane coordinates
	lens_grid_side = np.linspace(-gen_params["lens_fov"] / 2, gen_params["lens_fov"] / 2, gen_params["npix"])
	theta1, theta2 = np.meshgrid(lens_grid_side, lens_grid_side)

	gen_params.update({"beta1": beta1, "beta2": beta2, "theta1": theta1, "theta2": theta2})

	# Lens model parambounds
	SIE_pb = {"e1": (-0.3, 0.3),  # prior bounds on theta_E defined from log mass bounds in GalaxyLenser.sample_mass
			  "e2": (-0.3, 0.3),
			  "center_x": (-0.15, 0.15),
			  "center_y": (-0.15, 0.15)}

	gamma_fac = 0.05
	SHEAR_pb = {"gamma1": (-gamma_fac, gamma_fac),
				"gamma2": (-gamma_fac, gamma_fac),
				"ra_0": (0, 0),
				"dec_0": (0, 0)}

	# Source model parambounds
	SERSIC_E_pb = {"amp": (10, 15),
				   "R_sersic": (0.05, 0.4),
				   "n_sersic": (1, 4),
				   "e1": (-0.4, 0.4),
				   "e2": (-0.4, 0.4),
				   "center_x": (-0.1, 0.1),
				   "center_y": (-0.1, 0.1)}

	# Lens light model priorbounds
	SERSIC_E_lens_pb = {"amp": (50, 60),
				   "R_sersic": (0.7, 1.2),
				   "n_sersic": (1, 2),
				   "e1": (-0.4, 0.4),
				   "e2": (-0.4, 0.4),
				   "center_x": (-0.1, 0.1),
				   "center_y": (-0.1, 0.1)}

	# Modifiers for lens light
	d_e_lim = 0.01
	d_center_lim = 0.01


	wmode = args.ow
	dataset_dir = os.path.join(args.path_out, args.dataset_name)
	produce_dataset(dataset_dir, args.size, wmode, args.rpf, gen_params)
