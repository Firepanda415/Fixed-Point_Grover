# -*- coding: utf-8 -*-
"""
Created on Wed May  5 13:30:05 2021

@author: Muqing Zheng
"""




import mpmath as mpm
import numpy as np
from collections import Counter

import matplotlib.pyplot as plt
from matplotlib import cm
plt.rcParams["figure.dpi"] = 150

from qiskit import QuantumCircuit
from qiskit import Aer,execute, QuantumRegister,ClassicalRegister
from qiskit.circuit.library import UGate
from qiskit.visualization import plot_histogram
from qiskit.exceptions import QiskitError

from qiskit import IBMQ
from qiskit import transpile
from qiskit.tools.monitor import job_monitor

from qiskit.ignis.mitigation.measurement import (complete_meas_cal,CompleteMeasFitter,MeasurementFilter)

from qiskit.ignis.verification.tomography import state_tomography_circuits, StateTomographyFitter
from qiskit.quantum_info import DensityMatrix, Statevector
from qiskit.quantum_info import state_fidelity
import qiskit.ignis.mitigation.measurement as mc

import scipy.stats as ss

import measfilter as mf # package for measurement error filter


import warnings
warnings.filterwarnings('ignore')

import logging
logger = logging.getLogger()
logger.setLevel(logging.CRITICAL)


##############################################################################

# Load IBMQ Account and choose a real backend
IBMQ.load_account()
provider = IBMQ.get_provider('ibm-q')
name = 'ibmq_quito'
backend = provider.get_backend(name)


##############################################################################

# Device parameter
# Athens
# n = 2
# interested_qubits = [2,3] # Qubit Order
# itr = 5
# rep = 75 # Repeat "rep" times for each itr in itrs

# shots = 1024

# DataFileAddress = 'BellAthens/'
# NoiseFileAddress = 'BellAthens/'


# Santiago
# n = 2
# interested_qubits = [1,2] # Qubit Order

# itr = 15
# param_itr = 5 # The itr for inferrenced parameters

# rep = 75 # Repeat "rep" times for each itr in itrs
# resample_jobs = 20 # How many jobs submit for each sources

# shots = 1024

# DataFileAddress = 'BellSantiago/'
# NoiseFileAddress = 'BellSantiago/'
# Quito
n = 2
interested_qubits = [1,3] # Qubit Order

itr = 10
param_itr = 5 # The itr for inferrenced parameters

rep = 75# Repeat "rep" times for each itr in itrs
resample_jobs = 20 # How many jobs submit for each sources

shots = 1024

DataFileAddress = 'BellQuito/'
NoiseFileAddress = 'BellQuito/'


print("Device:", name)
print("Inferenced Data: {:d} itrs, Denoised circuit: {:d} itrs".format(param_itr, itr))
print("Resample posteriors {:d} times, measure {:d} times".format(rep*resample_jobs, rep*resample_jobs*shots))
print("Error Filtering Data file:", NoiseFileAddress, "Data File:", DataFileAddress)


##############################################################################


def one_itr(qc):
    qc.h(0)
    qc.cx(0,1)
    qc.barrier()
    qc.cx(0,1)
    qc.h(0)
    qc.barrier()

# Bell0 state
def pure_circuit(name=None):
    r = QuantumRegister(n)
    qc = QuantumCircuit(r, name=name)
    for it in range(itr):
        one_itr(qc)
    return r, qc  

def noise_gates_circuit(noise_params):
    angles = noise_params.reshape((3, n)) #NOT A NUMPY ARRAY FROM SCIPY.OPTIMIZE
    
    r = QuantumRegister(n)
    qc = QuantumCircuit(r)
    for it in range(itr):
        one_itr(qc)
        for q in range(n):
            qc.u(angles[0,n-q-1], angles[1,n-q-1], angles[2,n-q-1], n-q-1) # Add noise/denoise here
        qc.barrier()
    return r, qc


def denoise_gates_circuit(angles):
#     _, num_post = post_params.shape # find how many posteriors
#     selected_index = np.random.randint(0, high=num_post)
#     angles = post_params[:,selected_index].reshape((3,n))
    angles = angles.reshape((3,n))
    
    r = QuantumRegister(n)
    qc = QuantumCircuit(r)
    for it in range(itr):
        one_itr(qc)
        # Add denoise gates for each iteration, for all qubits
        for q in range(n):
            qc.append(UGate(angles[0,n-q-1], angles[1,n-q-1], angles[2,n-q-1]).inverse(), [n-q-1], [])
        qc.barrier()
    return r, qc



##############################################################################





def circs_gen(post_lambdas, selected_indices, prefix=""):
    circs = []
    for k in range(len(selected_indices)):
        if post_lambdas is not None:
#             angles = post_kde.resample(size=1).reshape(3,2)
#             r,qc = denoise_gates_circuit(angles)
            selected_index = selected_indices[k]
            r,qc = denoise_gates_circuit(post_lambdas[:,selected_index])
        else:
            r,qc = pure_circuit()
        qc.measure_all()
        trans_qc = transpile(qc, backend, initial_layout = interested_qubits, optimization_level=0)
        circs.append(trans_qc.copy(prefix+"Rep{:d}".format(k)))
        if selected_indices[k] == selected_indices[0]:
            print(prefix + " Original Depth = {:d}, Transpiled Depth = {:d}".format(qc.depth(), trans_qc.depth()))
    return circs

def mem_read(job_id_list, prefix=""):
    itr_mems = np.array([], dtype=str)
    for job_id_str in job_id_list:
        job_retrieved = backend.retrieve_job(job_id_str)
        resultObj = job_retrieved.result()
        for k in range(rep):
            new_mem = resultObj.get_memory(prefix+"Rep{:d}".format(k))
            itr_mems = np.append(itr_mems, new_mem)
    np.save(DataFileAddress+'DenoiseBellItr{:d}use{:d}_{:d}jobs_'.format(itr, param_itr, resample_jobs) + prefix + '.npy',  itr_mems)
    return itr_mems






##############################################################################



def job_execution(post_lambdas, prefix=""):
    lambda_indices = np.random.randint(0, high=post_lambdas.shape[1], size=resample_jobs*rep)
    circs_submitted = []
    job_ids = []
    job_lists = []
    for k in range(resample_jobs):
        # Generate circuit for this submission
        selected_circs = circs_gen(post_lambdas, lambda_indices[k*rep:(k+1)*rep], prefix=prefix)
        circs_submitted.append(selected_circs)

        job = execute(selected_circs, backend=backend, shots=shots, memory=True, optimization_level=0)
        job_monitor(job)
        job_lists.append(job)
        job_ids.append(job.job_id())
        print(job.job_id())

    np.save(DataFileAddress+"jobids_{:d}use{:d}_{:d}jobs_".format(itr, param_itr, resample_jobs)+prefix+".npy", np.array(job_ids))
    return lambda_indices, circs_submitted, job_lists, job_ids



##############################################################################




