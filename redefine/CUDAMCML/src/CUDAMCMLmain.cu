/////////////////////////////////////////////////////////////
//
//		CUDA-based Monte Carlo simulation of photon migration in layered media (CUDAMCML).
//	
//			Some documentation is avialable for CUDAMCML and should have been distrbuted along 
//			with this source code. If that is not the case: Documentation, source code and executables
//			for CUDAMCML are available for download on our webpage:
//			http://www.atomic.physics.lu.se/Biophotonics
//			or, directly
//			http://www.atomic.physics.lu.se/fileadmin/atomfysik/Biophotonics/Software/CUDAMCML.zip
//
//			We encourage the use, and modifcation of this code, and hope it will help 
//			users/programmers to utilize the power of GPGPU for their simulation needs. While we
//			don't have a scientifc publication describing this code, we would very much appreciate
//			if you cite our original GPGPU Monte Carlo letter (on which CUDAMCML is based) if you 
//			use this code or derivations thereof for your own scientifc work:
//			E. Alerstam, T. Svensson and S. Andersson-Engels, "Parallel computing with graphics processing
//			units for high-speed Monte Carlo simulations of photon migration", Journal of Biomedical Optics
//			Letters, 13(6) 060504 (2008).
//
//			To compile and run this code, please visit www.nvidia.com and download the necessary 
//			CUDA Toolkit and SKD. We also highly recommend the Visual Studio wizard 
//			(available at:http://forums.nvidia.com/index.php?showtopic=69183) 
//			if you use Visual Studio 2005 
//			(The express edition is available for free at: http://www.microsoft.com/express/2005/). 
//
//			This code is distributed under the terms of the GNU General Public Licence (see
//			below). 
//
//
///////////////////////////////////////////////////////////////

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

#include <float.h> //for FLT_MAX 
#include <stdio.h>
#include <helper_cuda.h>
#include "CUDAMCML.h"

__device__ __constant__ unsigned int num_photons_dc[1];	
__device__ __constant__ unsigned int n_layers_dc[1];		
__device__ __constant__ unsigned int start_weight_dc[1];	
__device__ __constant__ LayerStruct layers_dc[MAX_LAYERS];	
__device__ __constant__ DetStruct det_dc[1];		

#include "CUDAMCMLmem.cu"
#include "CUDAMCMLio.cu"
#include "CUDAMCMLrng.cu"
#include "CUDAMCMLtransport.cu"


// wrapper for device code
void DoOneSimulation(SimulationStruct* simulation, unsigned long long* x,unsigned int* a)
{
	MemStruct DeviceMem;
	MemStruct HostMem;	
	unsigned int threads_active_total=1;
	unsigned int i,ii;

    cudaError_t cudastat;
    clock_t time1,time2;


	// Start the clock
    time1=clock();


	// x and a are already initialised in memory
	HostMem.x=x;
	HostMem.a=a;

	InitMemStructs(&HostMem,&DeviceMem, simulation);
	
	InitDCMem(simulation);




    dim3 dimBlock(NUM_THREADS_PER_BLOCK);
    dim3 dimGrid(NUM_BLOCKS);
	
	LaunchPhoton_Global<<<dimGrid,dimBlock>>>(DeviceMem);
	checkCudaErrors( cudaDeviceSynchronize() ); // Wait for all threads to finish
	cudastat=cudaGetLastError(); // Check if there was an error
	if(cudastat)printf("Error code=%i, %s.\n",cudastat,cudaGetErrorString(cudastat));

	printf("ignoreAdetection = %d\n\n",simulation->ignoreAdetection);

	i=0;
	while(threads_active_total>0)
	{
		i++;
		//run the kernel
		if(simulation->ignoreAdetection == 1){
			MCd<1><<<dimGrid,dimBlock>>>(DeviceMem);
		}
		else{
			MCd<0><<<dimGrid,dimBlock>>>(DeviceMem);
		}	
		checkCudaErrors( cudaDeviceSynchronize() ); // Wait for all threads to finish
		cudastat=cudaGetLastError(); // Check if there was an error
		if(cudastat)printf("Error code=%i, %s.\n",cudastat,cudaGetErrorString(cudastat));

		// Copy thread_active from device to host
		checkCudaErrors( cudaMemcpy(HostMem.thread_active,DeviceMem.thread_active,NUM_THREADS*sizeof(unsigned int),cudaMemcpyDeviceToHost) );
		threads_active_total = 0;
		for(ii=0;ii<NUM_THREADS;ii++) threads_active_total+=HostMem.thread_active[ii];

		checkCudaErrors( cudaMemcpy(HostMem.num_terminated_photons,DeviceMem.num_terminated_photons,sizeof(unsigned int),cudaMemcpyDeviceToHost) );

		printf("Run %u, Number of photons terminated %u, Threads active %u\n",i,*HostMem.num_terminated_photons,threads_active_total);
	}
	printf("Simulation done!\n");
	
	CopyDeviceToHostMem(&HostMem, &DeviceMem, simulation);

    time2=clock();

	printf("Simulation time: %.2f sec\n",(double)(time2-time1)/CLOCKS_PER_SEC);

	Write_Simulation_Results(&HostMem, simulation, time2-time1);


	FreeMemStructs(&HostMem,&DeviceMem);
}



int main(int argc,char* argv[])
{
	int i;
	SimulationStruct* simulations;
	int n_simulations;
	unsigned long long seed = (unsigned long long) time(NULL);// Default, use time(NULL) as seed
	int ignoreAdetection = 0;
	char* filename;

	if(argc<2){printf("Not enough input arguments!\n");return 1;}
	else{filename=argv[1];}

	if(interpret_arg(argc,argv,&seed,&ignoreAdetection)) return 1;


	n_simulations = read_simulation_data(filename, &simulations, ignoreAdetection);


	if(n_simulations == 0)
	{
		printf("Something wrong with read_simulation_data!\n");
		return 1;
	}
	else
	{
		printf("Read %d simulations\n",n_simulations);
	}

	// Allocate memory for RNG's
	unsigned long long x[NUM_THREADS];
	unsigned int a[NUM_THREADS];

	//Init RNG's
	if(init_RNG(x, a, NUM_THREADS, "safeprimes_base32.txt", seed)) return 1;

	
	//perform all the simulations
	for(i=0;i<n_simulations;i++)
	{
		// Run a simulation
		DoOneSimulation(&simulations[i],x,a);
	}

	FreeSimulationStruct(simulations, n_simulations);

	return 0; 
}
