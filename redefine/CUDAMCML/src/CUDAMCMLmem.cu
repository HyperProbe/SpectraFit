/*	This file is part of CUDAMCML.

    CUDAMCML is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    CUDAMCML is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with CUDAMCML.  If not, see <http://www.gnu.org/licenses/>.*/

int CopyDeviceToHostMem(MemStruct* HostMem, MemStruct* DeviceMem, SimulationStruct* sim)
{ //Copy data from Device to Host memory

	int rz_size = sim->det.nr*sim->det.nz;
	int ra_size = sim->det.nr*sim->det.na;

	// Copy A_rz, Rd_ra and Tt_ra
	checkCudaErrors( cudaMemcpy(HostMem->A_rz,DeviceMem->A_rz,rz_size*sizeof(unsigned long long),cudaMemcpyDeviceToHost) );
	checkCudaErrors( cudaMemcpy(HostMem->Rd_ra,DeviceMem->Rd_ra,ra_size*sizeof(unsigned long long),cudaMemcpyDeviceToHost) );
	checkCudaErrors( cudaMemcpy(HostMem->Tt_ra,DeviceMem->Tt_ra,ra_size*sizeof(unsigned long long),cudaMemcpyDeviceToHost) );

	//Also copy the state of the RNG's
	checkCudaErrors( cudaMemcpy(HostMem->x,DeviceMem->x,NUM_THREADS*sizeof(unsigned long long),cudaMemcpyDeviceToHost) );

	return 0;
}
int InitDCMem(SimulationStruct* sim)
{
	// Copy det-data to constant device memory
	checkCudaErrors( cudaMemcpyToSymbol(det_dc,&(sim->det),sizeof(DetStruct)) );
	
	// Copy num_photons_dc to constant device memory
	checkCudaErrors( cudaMemcpyToSymbol(n_layers_dc,&(sim->n_layers),sizeof(unsigned int)));

	// Copy start_weight_dc to constant device memory
	checkCudaErrors( cudaMemcpyToSymbol(start_weight_dc,&(sim->start_weight),sizeof(unsigned int)));

	// Copy layer data to constant device memory
	checkCudaErrors( cudaMemcpyToSymbol(layers_dc,sim->layers,(sim->n_layers+2)*sizeof(LayerStruct)) );

	// Copy num_photons_dc to constant device memory
	checkCudaErrors( cudaMemcpyToSymbol(num_photons_dc,&(sim->number_of_photons),sizeof(unsigned int)));

	return 0;
	
}

int InitMemStructs(MemStruct* HostMem, MemStruct* DeviceMem, SimulationStruct* sim)
{
	int rz_size,ra_size;

	rz_size = sim->det.nr*sim->det.nz;
	ra_size = sim->det.nr*sim->det.na;

	
	// Allocate p on the device!!
	checkCudaErrors( cudaMalloc((void**)&DeviceMem->p,NUM_THREADS*sizeof(PhotonStruct)) );
	
	// Allocate A_rz on host and device
	HostMem->A_rz = (unsigned long long*) malloc(rz_size*sizeof(unsigned long long));
	if(HostMem->A_rz==NULL){printf("Error allocating HostMem->A_rz"); exit (1);}
	checkCudaErrors( cudaMalloc((void**)&DeviceMem->A_rz,rz_size*sizeof(unsigned long long)) );
	checkCudaErrors( cudaMemset(DeviceMem->A_rz,0,rz_size*sizeof(unsigned long long)) );

	// Allocate Rd_ra on host and device
	HostMem->Rd_ra = (unsigned long long*) malloc(ra_size*sizeof(unsigned long long));
	if(HostMem->Rd_ra==NULL){printf("Error allocating HostMem->Rd_ra"); exit (1);}
	checkCudaErrors( cudaMalloc((void**)&DeviceMem->Rd_ra,ra_size*sizeof(unsigned long long)) );
	checkCudaErrors( cudaMemset(DeviceMem->Rd_ra,0,ra_size*sizeof(unsigned long long)) );

	// Allocate Tt_ra on host and device
	HostMem->Tt_ra = (unsigned long long*) malloc(ra_size*sizeof(unsigned long long));
	if(HostMem->Tt_ra==NULL){printf("Error allocating HostMem->Tt_ra"); exit (1);}
	checkCudaErrors( cudaMalloc((void**)&DeviceMem->Tt_ra,ra_size*sizeof(unsigned long long)) );
	checkCudaErrors( cudaMemset(DeviceMem->Tt_ra,0,ra_size*sizeof(unsigned long long)) );


	// Allocate x and a on the device (For MWC RNG)
    checkCudaErrors(cudaMalloc((void**)&DeviceMem->x,NUM_THREADS*sizeof(unsigned long long)));
    checkCudaErrors(cudaMemcpy(DeviceMem->x,HostMem->x,NUM_THREADS*sizeof(unsigned long long),cudaMemcpyHostToDevice));
	
    checkCudaErrors(cudaMalloc((void**)&DeviceMem->a,NUM_THREADS*sizeof(unsigned int)));
    checkCudaErrors(cudaMemcpy(DeviceMem->a,HostMem->a,NUM_THREADS*sizeof(unsigned int),cudaMemcpyHostToDevice));


	// Allocate thread_active on the device and host
	HostMem->thread_active = (unsigned int*) malloc(NUM_THREADS*sizeof(unsigned int));
	if(HostMem->thread_active==NULL){printf("Error allocating HostMem->thread_active"); exit (1);}
	for(int i=0;i<NUM_THREADS;i++)HostMem->thread_active[i]=1u;

	checkCudaErrors( cudaMalloc((void**)&DeviceMem->thread_active,NUM_THREADS*sizeof(unsigned int)) );
	checkCudaErrors( cudaMemcpy(DeviceMem->thread_active,HostMem->thread_active,NUM_THREADS*sizeof(unsigned int),cudaMemcpyHostToDevice));


	//Allocate num_launched_photons on the device and host
	HostMem->num_terminated_photons = (unsigned int*) malloc(sizeof(unsigned int));
	if(HostMem->num_terminated_photons==NULL){printf("Error allocating HostMem->num_terminated_photons"); exit (1);}
	*HostMem->num_terminated_photons=0;

	checkCudaErrors( cudaMalloc((void**)&DeviceMem->num_terminated_photons,sizeof(unsigned int)) );
	checkCudaErrors( cudaMemcpy(DeviceMem->num_terminated_photons,HostMem->num_terminated_photons,sizeof(unsigned int),cudaMemcpyHostToDevice));

	return 1;
}

void FreeMemStructs(MemStruct* HostMem, MemStruct* DeviceMem)
{
	free(HostMem->A_rz);
	free(HostMem->Rd_ra);
	free(HostMem->Tt_ra);
	free(HostMem->thread_active);
	free(HostMem->num_terminated_photons);
	
	cudaFree(DeviceMem->A_rz);
	cudaFree(DeviceMem->Rd_ra);
	cudaFree(DeviceMem->Tt_ra);
    cudaFree(DeviceMem->x);
    cudaFree(DeviceMem->a);
	cudaFree(DeviceMem->thread_active);
	cudaFree(DeviceMem->num_terminated_photons);

}

void FreeSimulationStruct(SimulationStruct* sim, int n_simulations)
{
	for(int i=0;i<n_simulations;i++)free(sim[i].layers);
	free(sim);
}

