'''
Module to read the matrices and power spectra provided by Beutler et al. (arxiv:2106.06324)
'''
import numpy as np
import os 

def read_matrix(filename):
	return np.loadtxt(filename)

def dict_to_vec(pk_input, use_ell=[0,1,2,3,4]):
	'''
	Take a dictionary and return the model vector
	'''
	k_output = np.concatenate([pk_input['k'] for ell in use_ell], axis=0)
	pk_output = np.concatenate([pk_input['pk%d' % ell] for ell in use_ell], axis=0)
	return k_output, pk_output

def read_power(filename, combine_bins=10):
	'''
	Read power spectrum files provided as part of 
	Beutler et al. (arxiv:X)

	Input:
	filename: location of power spectrum file
	combine_bins: What k-bins sahould the output have?
	combine_bins=1  -> \Delta k_o = 0.001h/Mpc
	combine_bins=10 -> \Delta k_o = 0.01h/Mpc
	'''
	if not os.path.isfile(filename):
		print("WARNING: file %s not found" % filename)
		return {}
	else:
		output = {}
		output['header'] = []
		with open(filename, "r") as f:

			pks = [[] for x in range(0,5)]
			sigs = [[] for x in range(0,5)]
			k_ps = []
			k_eff = []
			modes = []
			within_header = 0
			for i, line in enumerate(f):
				if within_header < 2:
					output['header'].append(line)
					if line[:5] == 'kx_ny':
						dummy = list(map(str, line.split()))
						output['kx_ny'] = float(dummy[2])
						output['ky_ny'] = float(dummy[5])
						output['kz_ny'] = float(dummy[8])
					if line[:2] == 'Lx':
						dummy = list(map(str, line.split()))
						output['Lx'] = float(dummy[2])
						output['Ly'] = float(dummy[5])
						output['Lz'] = float(dummy[8])
					if line[:8] == 'SN(data)':
						dummy = list(map(str, line.split()))
						output['SN(data)'] = float(dummy[2])
					if line[:7] == 'SN(ran)':
						dummy = list(map(str, line.split()))
						output['SN(ran)'] = float(dummy[2])
					if line[:12] == 'SN(data+ran)':
						dummy = list(map(str, line.split()))
						output['SN(data+ran)'] = float(dummy[2])
					if line[:14] == '### header ###':
						within_header += 1
				else:
					dummy = list(map(float, line.split()))
					k_ps.append(dummy[0])
					k_eff.append(dummy[1])
					for ell in range(0,5):
						pks[ell].append(dummy[2+ell*2])
						sigs[ell].append(dummy[3+ell*2])
					modes.append(dummy[12])

		# Right now we can only average if the number of bins is divisible by combine_bins
		if len(k_ps)%combine_bins != 0:
			print("ERROR: Number of bins not divisible by combine_bins", len(k_ps), combine_bins)
			print("ABORT")
			sys.exit()
		# Average bins
		modes = np.array(modes).reshape(-1, combine_bins)
		output['k'] = np.ma.average(np.array(k_eff).reshape(-1, combine_bins), axis=1, weights=modes)
		output['k_center'] = np.ma.average(np.array(k_ps).reshape(-1, combine_bins), axis=1, weights=modes)
		for ell in range(0,5):
			output['pk%d' % ell] = np.ma.average(np.array(pks[ell]).reshape(-1, combine_bins), axis=1, weights=modes)
			output['sig%d' % ell] = np.ma.average(np.array(sigs[ell]).reshape(-1, combine_bins), axis=1, weights=modes)
		output['Nmodes'] = np.ma.sum(modes, axis=1).astype(int)
		# Store the bin size
		output['delta_k'] = output['k_center'][-1] - output['k_center'][-2]
		return output
