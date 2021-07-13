import numpy as np

def get_end_point_LOS_M(d, Nkth=400, kmin=0., kmax=0.4):
	'''
	Returns the transformation matrix M assuming the 
	End-point LOS definition
	Input:
	d = Comoving distance to the effective redshift
	Nkth = The number of bins in kth
	kmin = The lower k-range limit
	kmax = The upper k-range limit

	-> Note that large k-bins can lead to significant error
	-> The matrix is always 3*Nkth x 5*Nkth, mapping from 
	(P_0, P_2, P_4) to (P_0, P_1, P_2, P_3, P_4)
	'''
	M = np.zeros((Nkth*5, Nkth*3))

	dkp_th = (kmax-kmin)/Nkth
	kp_th = np.array([kmin + i*dkp_th + dkp_th/2. for i in range(0,Nkth)])

	# Populate matrix M
	# We start with the three unity matrices
	M[:Nkth, :Nkth] = np.identity(Nkth)
	M[2*Nkth:3*Nkth, Nkth:2*Nkth] = np.identity(Nkth)
	M[4*Nkth:5*Nkth, 2*Nkth:3*Nkth] = np.identity(Nkth)

	# Now we add the K matrices, which however have off-diagonal elements
	# We start with the diagonal elements
	M[Nkth:2*Nkth, Nkth:2*Nkth] = (3./(5.*d))*( 3.*np.identity(Nkth) )
	M[3*Nkth:4*Nkth, Nkth:2*Nkth] = (3./(5.*d))*( 2.*np.identity(Nkth) )
	M[3*Nkth:4*Nkth, 2*Nkth:3*Nkth] = (10./(9.*d))*( 5.*np.identity(Nkth) )

	# Now we add the (forward) derivative
	for ik, k in enumerate(kp_th):
		# K1 derivative (see eq. 4.3)
		M = _populate_derivative(d, M.copy(), Nkth+ik, Nkth+ik, ik, 3./5., kp_th)
		# K2 derivative (see eq. 4.4)
		M = _populate_derivative(d, M.copy(), 3*Nkth+ik, Nkth+ik, ik, -3./5., kp_th)
		# K3 derivative (see eq. 4.5)
		M = _populate_derivative(d, M.copy(), 3*Nkth+ik, 2*Nkth+ik, ik, 10./9., kp_th)
	return M


def _populate_derivative(d, M, index1, index2, ik, pre_factor, kp_th):
	'''
	Populate the derivative part of the M matrix
	'''
	delta_k = kp_th[1] - kp_th[0]
	norm = 0
	# If we are at the edge we do a one sided derivative
	# otherwise two sided
	if ik > 0 and ik < len(kp_th)-1:
		norm = 2.*delta_k
	else:
		norm = delta_k

	if ik > 0:
		M[index1, index2-1] = -pre_factor*kp_th[ik]/(d*norm)
	else:
		M[index1, index2] += -pre_factor*kp_th[ik]/(d*norm)

	if ik < len(kp_th)-1:
		M[index1, index2+1] = pre_factor*kp_th[ik]/(d*norm)
	else:
		M[index1, index2] += pre_factor*kp_th[ik]/(d*norm)
	return M
