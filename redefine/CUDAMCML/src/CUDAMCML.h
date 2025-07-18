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

// DEFINES 
#define NUM_BLOCKS 56 //Keep numblocks a multiple of the #MP's of the GPU (8800GT=14MP)
//The register usage varies with platform. 64-bit Linux and 32.bit Windows XP have been tested.

#ifdef __linux__ //uses 25 registers per thread (64-bit)
	#define NUM_THREADS_PER_BLOCK 320 //Keep above 192 to eliminate global memory access overhead However, keep low to allow enough registers per thread
	#define NUM_THREADS 17920
#endif

#ifdef _WIN32 //uses 26 registers per thread
	#define NUM_THREADS_PER_BLOCK 288 //Keep above 192 to eliminate global memory access overhead However, keep low to allow enough registers per thread
	#define NUM_THREADS 16128
#endif




#define NUMSTEPS_GPU 1000
#define PI 3.141592654f
#define RPI 0.318309886f
#define MAX_LAYERS 100
#define STR_LEN 200

//#define WEIGHT 0.0001f
#define WEIGHTI 429497u //0xFFFFFFFFu*WEIGHT
#define CHANCE 0.1f


// TYPEDEFS
typedef struct __align__(16)
{
	float z_min;		// Layer z_min [cm]
	float z_max;		// Layer z_max [cm]
	float mutr;			// Reciprocal mu_total [cm]
	float mua;			// Absorption coefficient [1/cm]
	float g;			// Anisotropy factor [-]
	float n;			// Refractive index [-]
}LayerStruct;

typedef struct __align__(16) 
{
	float x;		// Global x coordinate [cm]
	float y;		// Global y coordinate [cm]
	float z;		// Global z coordinate [cm]
	float dx;		// (Global, normalized) x-direction
	float dy;		// (Global, normalized) y-direction
	float dz;		// (Global, normalized) z-direction
	unsigned int weight;			// Photon weight
	int layer;				// Current layer
}PhotonStruct;

typedef struct __align__(16)
{
	float dr;		// Detection grid resolution, r-direction [cm]
	float dz;		// Detection grid resolution, z-direction [cm]
	
	int na;			// Number of grid elements in angular-direction [-]
	int nr;			// Number of grid elements in r-direction
	int nz;			// Number of grid elements in z-direction

}DetStruct;


typedef struct 
{
	unsigned long number_of_photons;
	int ignoreAdetection;
	unsigned int n_layers;
	unsigned int start_weight;
	char outp_filename[STR_LEN];
	char inp_filename[STR_LEN];
	long begin,end;
	char AorB;
	DetStruct det;
	LayerStruct* layers;
}SimulationStruct;


typedef struct
{
	PhotonStruct* p;					// Pointer to structure array containing all the photon data
	unsigned long long* x;				// Pointer to the array containing all the WMC x's
	unsigned int* a;					// Pointer to the array containing all the WMC a's
	unsigned int* thread_active;		// Pointer to the array containing the thread active status
	unsigned int* num_terminated_photons;	//Pointer to a scalar keeping track of the number of terminated photons

	unsigned long long* Rd_ra;
	unsigned long long* A_rz;			// Pointer to the 2D detection matrix!
	unsigned long long* Tt_ra;
}MemStruct;

