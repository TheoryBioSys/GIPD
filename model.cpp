/*****************************************************************************/
/***************** Copyright (C) 2021-2022, Emanuele Crosato *****************/
/*****************************************************************************/
// 
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
// 
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU General Public License for more details.
// 
// You should have received a copy of the GNU General Public License
// along with this program. If not, see <http://www.gnu.org/licenses/>
// 
/*****************************************************************************/


#include <iostream>
#include <vector>
#include <math.h>
#include <random>
#include <chrono>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <algorithm>

#include "utilities.cpp"

using namespace std;


 
class TriStateGrowth
{
	// model parameters
	double const b_rate;		// birth rate
	double const m_rate;		// mutation rate
	double const d_rate;		// death rate
	double const startPop;		// actual initial population size
	double const startPopName;	// just for naming the results (might differ from startPop)
	int const m ;   			// average number of rounds
	double const T;   			// temptation to defect
	double const R;   			// mutual reward for cooperation
	double const P;   			// punishment for mutual defection
	double const S;				// Sucker's payoff
	double const c;  			// TFT complexity cost*/
	double const MP[3][3];		// payoff matrix
	vector<vector<double> > nu;	// stoichiometric matrix
	bool const noGrowth;		// if 1 then no growth
	
	// run parameters
	int const runId;				// run id
	int const numIt;				// number of repetitions
	int const numTrajToRec;			// number of stoc. trajectories to record
	double const timeout;			// simulation stops when timeout is reached
	double const maxPop;			// simulation stops when maxPop is reached
	double const methodThreshold;	// population threshold for switching alghorithm
	int const initMode;				// -1)specific 0)random 1)fix-point 2)TFT 3)MTC 4)centre
	
	// possible initial states
	vector<vector<double> > startdemog;		// possible starting pop. for the random init.
	int const gridSize;						// possible values for an individual NN
	double mixfix[3];						// fixed point (in concentrations)
	double orig_NN[3];						// starting population

	// state of the system
	int currIt;				// current iteration (no extintions)
	int numExtinctions;		// counter for the extintions
	double t;				// current time
	double NN[3], totNN;	// current population and its total
	bool isInMix;			// current side of the separatrix
	bool isLangevin;		// current algorithm
	double timeInMix;		// current amount of time spent in Mix side of sepa
	double timeInAllD;		// current amount of time spent in AllD side of sepa
	bool allGood;			// zero if something goes wrong

	// separatrix
	int const sepaSize;								// size of the separatrices
	vector<vector<double> > separatrix;			// separatrix in N space
	vector<vector<double> > separatrixHiRes;	// high resolution around n2==0
	vector<vector<double> > separatrixBoth;		// merge of the two separatrices (for distance)
	double const sepaZoom;						// fraction of hiRes (e.g. 100 to zoom into 0-0.01)

	// stochastic limit-cycle
	double const refVec[3];				// reference vector for stochastic limit-cycle
	double const normVec[3];			// vector orthogonal to the simplex
	double const st_lc_sec_size;		// size of sectors of limit-cycle (radians)
	vector<double> st_lc_max_rad;		// keep track of max radius per sector
	vector<vector<double> > st_lc;		// stohastic limit-cycle coorddinates (one per sector)

	// random numbers generators
	std::mt19937_64 generator;				// generator of random numbers
	uniform_int_distribution<int> uid;		// uniform integer distribution
	uniform_real_distribution<double> urd;	// uniform real distribution
	normal_distribution<double> nd;			// normal distribution

	// output
	ofstream outFile;			// for storing final populations (one for each run)
	ofstream slcFile;			// for storing the stochastic limit-cycle distribution
	ofstream sepaOutFile;		// for storing the separatrix
	ofstream sepaHrOutFile;		// for storing the separatrix HiRes
	ofstream detTrajOutFile;	// for storing the deterministic trajectory
	ofstream trajOutFile;		// for storing (a subset of) the trajectories
	ofstream jumpsFile;			// for storing the separatrix jumps
	ofstream currStateFile;		// for storing the current state (for checking progress)




	//////////////////////////////////////////////////////////////
	////////////////////// output functions //////////////////////
	//////////////////////////////////////////////////////////////	

	///// separatrices /////

	void writeSeparatrixFile() {
		stringstream path;
		stringstream pathHr;
		path << fixed << setprecision(6) << "../results/separatrices/b_" << b_rate << "_m_" << log10(m_rate) << ".dat";
		pathHr << fixed << setprecision(6) << "../results/separatrices/b_" << b_rate << "_m_" << log10(m_rate) << "_HR.dat";
		sepaOutFile.open(path.str());
		sepaHrOutFile.open(pathHr.str());
		sepaOutFile << setprecision(10);
		sepaHrOutFile << setprecision(10);
		for (int i = 0; i < separatrix.size(); i++) {
			sepaOutFile << separatrix[i][0] << '\t' << separatrix[i][1] << '\t' << separatrix[i][2] << endl;
		}
		for (int i = 0; i < separatrixHiRes.size(); i++) {
			sepaHrOutFile << separatrixHiRes[i][0] << '\t' << separatrixHiRes[i][1] << '\t' << separatrixHiRes[i][2] << endl;
		}
		sepaOutFile.close();
		sepaHrOutFile.close();
	}

	///// deterministic trajectory /////

	void openDeterministicTrajectoryFile() {
		stringstream path;
		if (initMode == -1)
			path << fixed << setprecision(6) << "../results/deterministic-trajectories/" << b_rate <<
				"_m_" << log10(m_rate) << "_" << orig_NN[0] << "_" << orig_NN[1] << "_" << orig_NN[2] << ".dat";
		else
			path << fixed << setprecision(6) << "../results/deterministic-trajectories/b_" << b_rate <<
				"_m_" << log10(m_rate) << ".dat";
		detTrajOutFile.open(path.str());
	}

	void addToDeterministicTrajectoryFile() {
		detTrajOutFile << setprecision(10) << NN[0] << '\t' << NN[1] << '\t' << NN[2] << '\n';
	}

	void closeDeterministicTrajectoryFile() {
		detTrajOutFile.close();
	}

	///// stochastic trajectory file /////

	void openTrajectoryFile() {
		stringstream path;
		if (initMode == -1)
			path << fixed << setprecision(6) << "../results/trajectories/" << b_rate <<
				"_m_" << log10(m_rate) << "_pop_" << startPopName << "_" << orig_NN[0] <<
				"_" << orig_NN[1] << "_" << orig_NN[2] << "_run_" << runId << "_num_" << currIt << ".dat";
		else
			path << fixed << setprecision(6) << "../results/trajectories/b_" << b_rate <<
				"_m_" << log10(m_rate) << "_pop_" << startPopName << "_run_" << runId << "_num_" << currIt << ".dat";
		trajOutFile.open(path.str());
	}

	void addToTrajectoryFile() {
		trajOutFile << setprecision(10) << t << '\t' << NN[0] << '\t' << NN[1] << '\t' << NN[2] << '\n';
	}

	void closeTrajectoryFile() {
		trajOutFile.close();
	}

	///// jumps /////

	void openJumpsFile() {
		/*stringstream path;
		if (initMode == -1)
			path << fixed << setprecision(6) << "../results/jumps/b_" << b_rate <<
				"_m_" << log10(m_rate) << "_pop_" << startPopName << "_" << orig_NN[0] <<
				"_" << orig_NN[1] << "_" << orig_NN[2] << ".dat";
		else
			path << fixed << setprecision(6) << "../results/jumps/b_" << b_rate <<
				"_m_" << log10(m_rate) << "_pop_" << startPopName << ".dat";
		jumpsFile.open(path.str());*/
	}

	void addToJumpsFile() {
		/*jumpsFile << setprecision(10);
		jumpsFile << currIt << '\t' << t << '\t' << NN[0] << '\t' << NN[1] << '\t' << NN[2] << '\t'
			<< isInMix << endl;*/
	}

	void closeJumpsFile() {
		//jumpsFile.close();
	}

	///// final populations /////

	void openOutputFile() {
		stringstream path;
		if (initMode == -1)
			path << fixed << setprecision(6) << "../results/final-populations/b_" << b_rate << "_m_" << log10(m_rate)
				<< "_pop_" << startPopName << "_" << orig_NN[0] << "_" << orig_NN[1] << "_" << orig_NN[2]
				<< "_run_" << runId << ".dat";
		else
			path << fixed << setprecision(6) << "../results/final-populations/b_" << b_rate << "_m_" << log10(m_rate)
				<< "_pop_" << startPopName << "_run_" << runId << ".dat";
		outFile.open(path.str());
	}

	void addToOutputFile() {
		outFile << setprecision(10);
		for (int d = 0; d < 3; d++) {
			outFile << orig_NN[d] << '\t';
		}
		for (int d = 0; d < 3; d++) {
			outFile << NN[d] << '\t';
		}
		outFile << isInMix << '\t';
		outFile << timeInMix << '\t';
		outFile << timeInAllD << endl;
	}
	
	void closeOutputFile() {
		outFile.close();
	}

	///// stochastic limit-cycle /////

	void writeStocLimitCycleFile() {
	/*	stringstream path;
		if (initMode == -1)
			path << fixed << setprecision(6) << "../results/stochastic-limit-cycles/b_" << b_rate << "_m_" << log10(m_rate)
				<< "_pop_" << startPopName << "_" << orig_NN[0] << "_" << orig_NN[1] << "_" << orig_NN[2]
				<< "_run_" << runId << "_num_" << currIt << ".dat";
		else
			path << fixed << setprecision(6) << "../results/stochastic-limit-cycles/b_" << b_rate << "_m_" << log10(m_rate)
				<< "_pop_" << startPopName << "_run_" << runId << "_num_" << currIt << ".dat";
		slcFile.open(path.str());
		for (int i = 0; i < st_lc.size(); i++) {
			for (int d = 0; d < st_lc[0].size(); d++) {
				slcFile << st_lc[i][d];
				if (d < 2)
					slcFile << '\t';
				else
					slcFile << endl;
			}

		}
		slcFile.close();*/
	}
	
	///// current state /////

	void writeCurrStateFile() {
		stringstream path;
		if (initMode == -1)
			path << fixed << setprecision(6) << "../results/current-state/b_" << b_rate << "_m_" << log10(m_rate)
				<< "_pop_" << startPopName << "_" << orig_NN[0] << "_" << orig_NN[1] << "_" << orig_NN[2]
				<< "_run_" << runId << ".dat";
		else
			path << fixed << setprecision(6) << "../results/current-state/b_" << b_rate << "_m_" << log10(m_rate)
				<< "_pop_" << startPopName << "_run_" << runId << ".dat";
		currStateFile.open(path.str());
		currStateFile << setprecision(10) << currIt << '\t' << t << '\t' << NN[0] << '\t' << NN[1] << '\t'
			<< NN[2] << '\t' << timeInMix << '\t' << timeInAllD << '\t' << numExtinctions << '\n';
		currStateFile.close();
	}




	//////////////////////////////////////////////////////////////
	/////////////////////// other functions //////////////////////
	//////////////////////////////////////////////////////////////

	void reset()
	{
		// reset random number generator
		unsigned seed = chrono::system_clock::now().time_since_epoch().count();
		generator = mt19937_64(seed);

		// times
		t = 0.0;
		timeInMix = 0.0;
		timeInAllD = 0.0;

		// generate new random original point if needed
		if (initMode == 0) {
			int randIdx = uid(generator);
			for (int d = 0; d < 3; d++) {
				orig_NN[d] = startdemog[randIdx][d];
			}
		}
		
		// set NN to original
		for (int d = 0; d < 3; d++) {
			NN[d] = orig_NN[d];
		}
		checkNNandUpdateSum();

		// check what side of the separatrix
		if (NN[1] / totNN < 1.0 / sepaZoom) {
			isInMix = isPointBelowSeparatrix(NN[1] / totNN, NN[2] / totNN, separatrixHiRes, true);
		} else {
			isInMix = isPointBelowSeparatrix(NN[1] / totNN, NN[2] / totNN, separatrix, true);
		}

		// reset stochastic limit-cycle
		for (int i = 0; i < st_lc.size(); i++) {
			st_lc_max_rad[i] = 0.0;
			for (int d = 0; d < st_lc[0].size(); d++) {
				st_lc[i][d] = 0.0;
			}
		}

		// reset other things
		isLangevin = totNN >= methodThreshold;
	}


	void checkNNandUpdateSum()
	{
		// check zeros in NN
		for (int d = 0; d < 3; d++) {
			if (NN[d] < 0.0) {
				cout << "BAD: one of the three popualations became negative!\n";
				allGood = false;
				return;
			}
		}

		// update totNN
		totNN = 0.0;
		for (int d = 0; d < 3; d++) {
			totNN = totNN + NN[d];
		}
	}


	void calcFitness(double & phi, double *f)
	{
     	phi = 0.0;
     	for (int d = 0; d < 3; d++) {
     		
     		// fitness
     		if (totNN == 1.0 && NN[d] == 1.0) {
     			f[d] = 1.0;
     		}
     		else if (NN[d] < 1.0) {
     			f[d] = 0.0;
     		}
     		else {
     			double sumTerm = 0.0;
	     		for (int d2 = 0; d2 < 3; d2++) {
	     			sumTerm = sumTerm + MP[d][d2] * NN[d2];
	     		}
     			f[d] = (sumTerm - MP[d][d]) / (totNN - 1.0);
     		}

     		// phi
     		phi = phi + (f[d] * NN[d]) / totNN;
     	}
	}


	void calcFitnessDeterministicLimit(double & phi, double *f)
	{
		phi = 0.0;
		for (int d = 0; d < 3; d++) {
     		
     		// fitness
     		f[d] = 0.0;
	     	for (int d2 = 0; d2 < 3; d2++) {
	     		f[d] = f[d] + MP[d][d2] * NN[d2];
	     	}

     		// phi
     		phi = phi + f[d] * NN[d];
     	}
	}


	void calcTransitionRates(double phi, double *f, double *h)
	{
     	for (int d = 0; d < 3; d++) {
     		h[d] = b_rate * (f[d] / phi) * NN[d];				// birth
     		h[d+3] = d_rate * NN[d];							// death
     		h[d+6] = b_rate * (f[d] / phi) * m_rate * NN[d];	// mutation 1
     		h[d+9] = b_rate * (f[d] / phi) * m_rate * NN[d];	// mutation 2
     	}
	}



	////////////////////////////////////////////////////////
	/////////////////////// Gillespie //////////////////////
	////////////////////////////////////////////////////////

	void killRandomOne()
	{
		double kill = urd(generator);
		double totPop = NN[0] + NN[1] + NN[2];
	    if (kill < NN[0] / totPop) {
	    	NN[0] = NN[0] - 1.0;
	    } else if (kill < (NN[0] + NN[1]) / totPop) {
	    	NN[1] = NN[1] - 1.0;
	    } else {
	    	NN[2] = NN[2] - 1.0;
	    }
	}

	void gillespieStep(double *h)
	{
     	// total transition rate
     	double totH = 0;
     	for (int i = 0; i < 12; i++) {
     		totH = totH + h[i];
     	}

     	// time to next event
     	double r1 = urd(generator);
     	while (r1 == 0.0) {
     		r1 = urd(generator);
     	}
     	double t_next = -log(r1) / totH;

     	// update time
     	t = t + t_next;

     	// determine next reaction
     	int i = 0;
     	int u = 0;
     	double amu = 0.0;
     	double r2 = urd(generator);
     	while (amu < r2 * totH) {
     		u = u + 1;
     		amu = amu + h[i];
     		i = i + 1;
     	}

     	// reaction channels (no growth)
     	if (noGrowth == true)
     	{
     		switch (u)
	     	{
	     		case 1: { // birth 1
	     			NN[0] = NN[0] + 1.0;
	     			killRandomOne();
	     			break;
	     		}
	     		case 2: { // birth 2
	     			NN[1] = NN[1] + 1.0;
	     			killRandomOne();
	     			break;
	     		}
	     		case 3: { // birth 3
	     			NN[2] = NN[2] + 1.0;
	     			killRandomOne();
	     			break;
	     		}
				case 4: { // death 1
					cout << "death should not happen" << endl;
					allGood = false;
					return;
				}
				case 5: { // death 2
					cout << "death should not happen" << endl;
					allGood = false;
					return;
				}
				case 6: { // death 3
					cout << "death should not happen" << endl;
					allGood = false;
					return;
				}
				case 7: { // mutation 1->2
					NN[0] = NN[0] - 1.0;
					NN[1] = NN[1] + 1.0;
					break;
				}
				case 8: { // mutation 2->3
					NN[1] = NN[1] - 1.0;
					NN[2] = NN[2] + 1.0;
					break;
				}
				case 9: { // mutation 3->1
					NN[2] = NN[2] - 1.0;
					NN[0] = NN[0] + 1.0;
					break;
				}
				case 10: { // mutation 1->3
					NN[0] = NN[0] - 1.0;
					NN[2] = NN[2] + 1.0;
					break;
				}
				case 11: { // mutation 2->1
					NN[1] = NN[1] - 1.0;
					NN[0] = NN[0] + 1.0;
					break;
				}
				case 12: { // mutation 3->2
					NN[2] = NN[2] - 1.0;
					NN[1] = NN[1] + 1.0;
					break;
				}
			}
     	}

     	// reaction channels (with growth)
     	else
     	{
     		switch (u)
     		{
	     		case 1: { // birth 1
	     			NN[0] = NN[0] + 1.0;
	     			break;
	     		}
	     		case 2: { // birth 2
	     			NN[1] = NN[1] + 1.0;
	     			break;
	     		}
	     		case 3: { // birth 3
	     			NN[2] = NN[2] + 1.0;
	     			break;
	     		}
				case 4: { // death 1
					NN[0] = NN[0] - 1.0;
					break;
				}
				case 5: { // death 2
					NN[1] = NN[1] - 1.0;
					break;
				}
				case 6: { // death 3
					NN[2] = NN[2] - 1.0;
					break;
				}
				case 7: { // mutation 1->2
					NN[0] = NN[0] - 1.0;
					NN[1] = NN[1] + 1.0;
					break;
				}
				case 8: { // mutation 2->3
					NN[1] = NN[1] - 1.0;
					NN[2] = NN[2] + 1.0;
					break;
				}
				case 9: { // mutation 3->1
					NN[2] = NN[2] - 1.0;
					NN[0] = NN[0] + 1.0;
					break;
				}
				case 10: { // mutation 1->3
					NN[0] = NN[0] - 1.0;
					NN[2] = NN[2] + 1.0;
					break;
				}
				case 11: { // mutation 2->1
					NN[1] = NN[1] - 1.0;
					NN[0] = NN[0] + 1.0;
					break;
				}
				case 12: { // mutation 3->2
					NN[2] = NN[2] - 1.0;
					NN[1] = NN[1] + 1.0;
					break;
				}
     		}
		}
	}



	////////////////////////////////////////////////////////
	/////////////////////// Langevin ///////////////////////
	////////////////////////////////////////////////////////

	void langevinStep(double *h)
	{
		// update time
		double t_sample = 0.01; // 0.01
     	t = t + t_sample;

     	// generate noise
     	double noise[12];
     	for (int i = 0; i < 12; i++) {
     		noise[i] = nd(generator);
     	}

     	// update stoichiometric matrix (if no growth)
     	if (noGrowth == true) {
	     	double n1 = NN[0] / totNN;
	     	double n2 = NN[1] / totNN;
	     	double n3 = NN[2] / totNN;
	     	nu = {	{1.0-n1, -n1, -n1, 0.0, 0.0, 0.0,-1.0, 0.0, 1.0,-1.0, 1.0, 0.0},
					{-n2, 1.0-n2, -n2, 0.0, 0.0, 0.0, 1.0,-1.0, 0.0, 0.0,-1.0, 1.0},
					{-n3, -n3, 1.0-n3, 0.0, 0.0, 0.0, 0.0, 1.0,-1.0, 1.0, 0.0,-1.0}};
     	}

     	// update NN
     	double term1, term2;
     	for (int d = 0; d < 3; d++) {
     		term1 = 0;
     		term2 = 0;
     		for (int i = 0; i < 12; i++) {
     			term1 = term1 + t_sample * nu[d][i] * h[i];
     			term2 = term2 + nu[d][i] * sqrt(t_sample * h[i]) * noise[i];
     		}
     		NN[d] = NN[d] + term1 + term2;
     		if (NN[d] < 0) {
     			NN[d] = 0;
     		}
     	}
	}


	
	/////////////////////////////////////////////////////////////
	/////////////////////// deterministic ///////////////////////
	/////////////////////////////////////////////////////////////

	void deterministicRun(bool store)
	{
		// convert populations into concentrations
		for (int d = 0; d < 3; d++) {
			NN[d] = NN[d] / totNN;
		}
		checkNNandUpdateSum();

		// parameters
		double t_end = 2000.0;
		double dt = 0.01;
		double storeInterval = 1.0;
		double timeLastStore = 0.0;

		// initialise
		t = 0.0;

		// variables
		double f[3], phi;	// fitness
		double dNN[3];		// increments
		int o[2];			// indices of other two populations, given one

		// open trajectory file if needed
		if (store) {
			cout << "\tOpening deterministic trajectory file...\n";
			openDeterministicTrajectoryFile();
			if (NN[1] < 1.0 / sepaZoom) {
				isInMix = isPointBelowSeparatrix(NN[1], NN[2], separatrixHiRes, true);
			} else {
				isInMix = isPointBelowSeparatrix(NN[1], NN[2], separatrix, true);	
			}
			addToDeterministicTrajectoryFile();
		}

		// run deterministically
		while (t < t_end)
		{
			// update time
			t = t + dt;
			timeLastStore = timeLastStore + dt;

			// fitness
			calcFitnessDeterministicLimit(phi, f);			

			// NN increment (from Mathematica)
			for (int d = 0; d < 3; d++) {
				o[0] = (d+1) % 3;
				o[1] = (d+2) % 3;
				dNN[d] = ((b_rate * m_rate) / phi) * (f[o[0]]*NN[o[0]] + f[o[1]]*NN[o[1]] -2.0*f[d]*NN[d])
					+ b_rate * ((f[d] / phi) - 1.0) * NN[d];
			}

            // update NN
            for (int d = 0; d < 3; d++) {
				NN[d] = NN[d] + dNN[d] * dt;
			}
			checkNNandUpdateSum();

            // store if needed
            if (store) {
            	if (NN[1] < 1.0 / sepaZoom) {
            		isInMix = isPointBelowSeparatrix(NN[1], NN[2], separatrixHiRes, isInMix);
            	} else {
            		isInMix = isPointBelowSeparatrix(NN[1], NN[2], separatrix, isInMix);
            	}
            	if (timeLastStore >= storeInterval) {
					addToDeterministicTrajectoryFile();
					timeLastStore = 0.0;
				}
			}
		}

		if (store) {
			closeDeterministicTrajectoryFile();
		}
	}



public:


	TriStateGrowth(
		double br, double mr, double dr,
		int nit, double maxTime,
		int mm, double TT, double RR, double PP, double SS, double cc,
		double N0, double N0name,
		double methThr, double maxp,
		int imode,
		int nttr,
		int rid,
		bool noGr,
		double nn1, double nn2, double nn3) :

		b_rate(br), m_rate(mr), d_rate(dr),
		numIt(nit), timeout(maxTime),
		m(mm), T(TT), R(RR), P(PP), S(SS), c(cc),
		totNN(round(N0)), startPop(round(N0)), startPopName(N0name),
		methodThreshold(max(methThr, 1.0/m_rate)),
		maxPop(maxp),
		initMode(imode),
		numTrajToRec(nttr),
		runId(rid),
		noGrowth(noGr), 		
		MP{{R*m, S*m, R*m}, {T*m, P*m, T+P*(m-1)}, {R*m-c, S+P*(m-1)-c, R*m-c}},
		gridSize(150),
		sepaZoom(100.0), sepaSize(500),
		refVec{sqrt(0.5), -sqrt(0.5), 0},
		normVec{sqrt(1.0/3.0), sqrt(1.0/3.0), sqrt(1.0/3.0)},
		st_lc_sec_size((2.0*M_PI) / 1000.0)
	{
		// stoichiometric matrix
		nu = {	{1.0, 0.0, 0.0,-1.0, 0.0, 0.0,-1.0, 0.0, 1.0,-1.0, 1.0, 0.0},
				{0.0, 1.0, 0.0, 0.0,-1.0, 0.0, 1.0,-1.0, 0.0, 0.0,-1.0, 1.0},
				{0.0, 0.0, 1.0, 0.0, 0.0,-1.0, 0.0, 1.0,-1.0, 1.0, 0.0,-1.0}};

		// starting populations
		switch(initMode) {
			
			// start from specific point
		   	case -1 : {

		   		// get original population from input
		   		orig_NN[0] = nn1;
				orig_NN[1] = nn2;
				orig_NN[2] = nn3;
		    	break;
		    }
			
			// random initial distributions
			case 0 : {

				// genearte all possible starting points
				generateAllPoints(gridSize, startPop, startdemog);
				break;
			}
			
			// start from mixed fixed point
		   	case 1 : {

		   		// search for fixed point in file
			   	ifstream points("fixed_points.csv");
				string line;
				double bRat, mRatExp, r1, r2, r3;
				bool found = 0;
			    while (getline(points, line) && !found)
				{
					// read line
					stringstream lineStream(line);
					string cell;

					// get birth rate
					getline(lineStream, cell, ',');
					string2double(cell, bRat);
					if (bRat >= b_rate-0.001 && bRat < b_rate+0.001)
					{
						// get mutation rate
						getline(lineStream, cell, ',');
						string2double(cell, mRatExp);
						if (pow(10,mRatExp) > 0.9999*m_rate && pow(10,mRatExp) < 1.0001*m_rate)
						{
							// read r1, r2 and r3
							getline(lineStream, cell, ',');
							string2double(cell, r1);
							getline(lineStream, cell, ',');
							string2double(cell, r2);
							getline(lineStream, cell, ',');
							string2double(cell, r3);
							found = 1;
						}
					}
				}

				// check if found
				if (found)
				{
					// set mixed strategy fixed point
					mixfix[0] = r1;
					mixfix[1] = r2;
					mixfix[2] = r3;

					// use it to set original population
					orig_NN[0] = round(mixfix[0] * startPop);
					orig_NN[1] = round(mixfix[1] * startPop);
					orig_NN[2] = round(mixfix[2] * startPop);
					double fixTot = orig_NN[0] + orig_NN[1] + orig_NN[2];
					int pos = 0;
					while (fixTot != startPop)
					{
						if (fixTot > startPop) {
							orig_NN[pos] = orig_NN[pos] - 1;
							fixTot = fixTot - 1;
						}
						else {
							orig_NN[pos] = orig_NN[pos] + 1;
							fixTot = fixTot + 1;
						}
						pos = pos + 1;
					}
				}
				else {
					cout << "BAD: Fixed point not found in file.\n";
					allGood = false;
				}
		    	break;
		    }

		    // start from TFT corner
		   	case 2 : {
		    	orig_NN[0] = 0;
				orig_NN[1] = 0;
				orig_NN[2] = startPop;
		    	break;
		    }

		    // start from MTC edge
		   	case 3 : {
				orig_NN[0] = floor(startPop / 5.0);
				orig_NN[1] = 0;
				orig_NN[2] = startPop - orig_NN[0];
		    	break;
		    }

		    // start from centre
		   	case 4 : {
		    	double third = round(startPop / 3.0);
				orig_NN[0] = third;
				orig_NN[1] = third;
				orig_NN[2] = startPop - 2.0*third;
		    	break;
		    }

		    // start from AllD corner
		   	case 5 : {
				orig_NN[0] = 0;
				orig_NN[1] = startPop;
				orig_NN[2] = 0;
		    	break;
		    }

		    // start from p1
		   	case 6 : {
				orig_NN[0] = floor(startPop / 8.0);
				orig_NN[1] = floor(startPop / 8.0);
				orig_NN[2] = startPop - orig_NN[0] - orig_NN[1];
		    	break;
		    }

		    // start from p2
		   	case 7 : {
				orig_NN[0] = floor(startPop * 0.65);
				orig_NN[1] = floor(startPop * 0.05);
				orig_NN[2] = startPop - orig_NN[0] - orig_NN[1];
		    	break;
		    }

		    // no default case
		    default : {
		    	cout << "BAD: wrong init mode.\n";
		    	allGood = false;
		    	return;
		    }
		}

		// random numbers
		unsigned seed = chrono::system_clock::now().time_since_epoch().count();
		//generator = default_random_engine(seed);
		generator = mt19937_64(seed);
		uid = uniform_int_distribution<int>(0, nChoosek(gridSize+2,3-1)-1);
		urd =  uniform_real_distribution<double>(0.0, 1.0);
		nd = normal_distribution<double>(0.0, 1.0);

		// initialize stochastic limit-cycle
		int numSecs = round((2.0*M_PI) / st_lc_sec_size);
		st_lc_max_rad = vector<double>(numSecs, 0.0);
		st_lc = vector<vector<double> >(numSecs, vector<double>(3, 0.0));

		// errors indicator
		allGood = true;
	}


	void run()
	{
		// open output files
		openOutputFile();
		openJumpsFile();

		// timer
		chrono::high_resolution_clock::time_point tstart, tend;
		chrono::duration<double> span;

		// variables
		double f[3], phi;			// fitness
		double h[12];				// transition rates
		double eta, xi;				// for checking side of separatrix
		double normNN[3];			// for the distance
		double trajInt = 1.0;		// record trajectory at this interval (0.1)
		double stateInt = 1000.0;	// write out state at this interval
		double tLastTrajRec;		// time elapsed from last output
		double tLastStateOut;		// time elapsed from last output
		double prevT;				// need this for comparison with t
		double deltaT;				// t - pervT
		bool isInMix2;				// needed for comparison with isInMix

		// cycle over all iterations
		numExtinctions = 0;
		for (currIt = 0; currIt < numIt; currIt++)
		{
			// reset
			reset();
			writeCurrStateFile();
			cout << "\nRunning: it = " << currIt << ", ";
			cout << "b = " << b_rate << ", m = " << m_rate << ", ";
			cout << "NN = " << NN[0] << " " << NN[1] << " " << NN[2] << ", ";
			cout << "Langevin = " << isLangevin << " ..." << endl;

			// time
			tstart = chrono::high_resolution_clock::now();
			tLastStateOut = 0.0;
			tLastTrajRec = 0.0;
			prevT = 0.0;

			// open trajectory file and write first element
			if (currIt < numTrajToRec) {
				openTrajectoryFile();
				addToTrajectoryFile();
			}

			// Hybrid Gillespie-Ito stochastic simulation algorithm
			//bool jumped = false; // !!!!!
			while (totNN >= 1.0 && totNN < maxPop && t < timeout) // && !jumped) // !!!!!!
			{
				// calculate transition rates
				calcFitness(phi, f);
				calcTransitionRates(phi, f, h);

				// step with either Gillespie or Langevin
				if (totNN < methodThreshold) {
					if (isLangevin == true) {
						cout << "\tswitch to Gillespie" << "\n";
						for (int d = 0; d < 3; d++) {
							NN[d] = floor(NN[d]); // convert to integer
						}
						checkNNandUpdateSum();
						isLangevin = false;
					}
					gillespieStep(h);
				}
				else {
					if (isLangevin == false) {
						cout << "\tswitch to Langevin" << "\n";
						isLangevin = true;
					}
					langevinStep(h);
				}
				checkNNandUpdateSum();

				// check all good after step
				if (checkAllGood() == false)
					return;			

				// check if population died out
				if (totNN >= 1.0) {

					/*// update stochastic-limit cycle
					double unit_NN[3];
					double VR[3];
					for (int d = 0; d < 3; d++) {
						unit_NN[d] = NN[d] / totNN;
						VR[d] = unit_NN[d] - mixfix[d];
					}
					double alpha = angleBetweenVectors(refVec, normVec, VR);
					int sec_id = floor(alpha / st_lc_sec_size);
					if (sec_id >= st_lc.size()) {
						cout << "ERROR: sector id out of bound!\n";
						allGood = false;
						return;
					}
					else {
						double r = sqrt(VR[0]*VR[0] + VR[1]*VR[1] + VR[2]*VR[2]);
						if (r > st_lc_max_rad[sec_id]) {
							st_lc_max_rad[sec_id] = r;
							for (int d = 0; d < st_lc[0].size(); d++) {
								st_lc[sec_id][d] = unit_NN[d];
							}
						}
					}*/

					// check if it jumped the separatrix
					if (NN[1] / totNN < 1.0 / sepaZoom) {
						isInMix2 = isPointBelowSeparatrix(NN[1] / totNN, NN[2] / totNN, separatrixHiRes, isInMix);
					} else {
						isInMix2 = isPointBelowSeparatrix(NN[1] / totNN, NN[2] / totNN, separatrix, isInMix);
					}
					if (isInMix2 != isInMix) {
						isInMix = !isInMix;
						addToJumpsFile();
						//jumped = true;
					}

					// update time in Mix and AllD and time from output
					deltaT = t - prevT;
					prevT = t;
					if (isInMix) {
						timeInMix = timeInMix + deltaT;
					} else {
						timeInAllD = timeInAllD + deltaT;
					}
					tLastStateOut = tLastStateOut + deltaT;
					if (tLastStateOut >= stateInt) {
						writeCurrStateFile();
						writeStocLimitCycleFile();
						tLastStateOut = 0.0;
					}

					// write out trajectory if needed
					if (currIt < numTrajToRec) {
						tLastTrajRec = tLastTrajRec + deltaT;
						if (tLastTrajRec >= trajInt) {
							addToTrajectoryFile();
							tLastTrajRec = 0.0;
						}
					}
				}
			}

			// write out state last time
			writeCurrStateFile();
			writeStocLimitCycleFile();

			// close trajectory file
			if (currIt < numTrajToRec) {
				cout << "\tClosing trajectory file...\n";
				closeTrajectoryFile();
			}

			// check if population died out
			if (totNN >= 1.0)
			{
				if (totNN >= maxPop)
					cout << "Simulation ended properly: population limit\n";
				if (t >= timeout)
					cout << "Simulation ended properly: time limit\n";
				
				// print final result
				tend = chrono::high_resolution_clock::now();
				span = chrono::duration_cast<chrono::duration<double> >(tend - tstart);
				cout << "\tDone in " << span.count() << "sec.\n";
				cout << "\tResult: ";
				for (int d = 0; d < 3; d++) {
					cout << NN[d] << " ";
				}
				cout << "\n";

				// write output
				cout << "\tWriting output file...\n";
				addToOutputFile();
			}
			else
			{
				cout << "Population died out: need to repeat\n";
				numExtinctions++;
				currIt--;
			}
		}
		
		// close output file
		closeOutputFile();
		closeJumpsFile();
	}


	void computeSeparatrix()
	{
		cout << "Computing separatrix..." << endl;
		cout << "\tRunning deterministic over NN space" << endl;

		// parameters and variables
		bool lastOutcome, currOutcome; // true if ends in the mix side
		separatrix = vector<vector<double> >(sepaSize+1, vector<double>(3, 0.0));
		separatrixHiRes = vector<vector<double> >(sepaSize+1, vector<double>(3, 0.0));
		int crosses;

		// for each n2
		for (int n2 = 0; n2 <= sepaSize; n2++) {
			
			// store n2 coordinate
			separatrix[n2][1] = (double)n2 / sepaSize;

			// initialise search
			crosses = 0;
			lastOutcome = false;

			// find the jumping n3
			for (int n3 = 0; n3 <= sepaSize - n2; n3++) {

				// get corrensponding n1
				int n1 = sepaSize - n2 - n3;

				// deterministic run
				NN[0] = n1; NN[1] = n2; NN[2] = n3;
				checkNNandUpdateSum();
				deterministicRun(false); // NN gets converted into concentrations (totNN becomes 1)
				currOutcome = NN[1] / totNN <= 0.95;

				// check if it crosses the sepa
				if (currOutcome != lastOutcome) {
					crosses++;
					separatrix[n2][0] = double(n1) / sepaSize;
					separatrix[n2][2] = double(n3) / sepaSize;
				}
				lastOutcome = currOutcome;
			}

			// if no crosses, assign n2 edge (remove later)
			if (crosses == 0) {
				separatrix[n2][0] = 0.0;
				separatrix[n2][2] = (sepaSize - n2) / sepaSize;
			}

			// if more than one cross, raise attention
			if (crosses > 1) {
				cout << "\tBAD: separatrix is not well-shaped at n2=" << n2 << "!\n";
				allGood = false;
				return;
			}
		}

		// skip elements with n3==0 and fix last element
		for (int i = 1; i < separatrix.size(); i++) {
			if (separatrix[i][0] == 0.0 && separatrix[i-1][0] == 0.0) {
				separatrix.erase(separatrix.begin() + i, separatrix.end());
				break;
			}
		}

		// separatrix HiRes
		for (int n2 = 0; n2 <= sepaSize; n2++) {
			double n2zoom = (double)n2 / sepaZoom;
			separatrixHiRes[n2][1] = n2zoom / sepaSize;
			crosses = 0;
			lastOutcome = false;
			for (int n3 = 0; n3 <= sepaSize - n2zoom; n3++) {
				double n1 = sepaSize - n2zoom - n3;
				NN[0] = n1; NN[1] = n2zoom; NN[2] = n3;
				checkNNandUpdateSum();
				deterministicRun(false); // NN gets converted into concentrations (totNN becomes 1)
				currOutcome = NN[1] / totNN <= 0.95;
				if (currOutcome != lastOutcome) {
					crosses++;
					separatrixHiRes[n2][0] = double(n1) / sepaSize;
					separatrixHiRes[n2][2] = double(n3) / sepaSize;
				}
				lastOutcome = currOutcome;
			}
			if (crosses > 1) {
				cout << "\tBAD: separatrixHiRes is not well-shaped at n2=" << n2zoom << "!\n";
			}
		}

		// merge two separatrices
		separatrixBoth = vector<vector<double> >(separatrix.size()+separatrixHiRes.size(), vector<double>(3, 0.0));
		int j = 0;
		for (int i = 0; i < separatrix.size(); i++) {
			for (int d = 0; d < 3; d++) {
				separatrixBoth[j][d] = separatrix[i][d];
			}
			j++;
		}
		for (int i = 0; i < separatrixHiRes.size(); i++) {
			for (int d = 0; d < 3; d++) {
				separatrixBoth[j][d] = separatrixHiRes[i][d];
			}
			j++;
		}

		// write separatrix
		cout << "\tWriting separatrix line on file..." << endl;
		writeSeparatrixFile();
	}


	void loadSeparatrix()
	{
		// open file
		stringstream path;
		stringstream pathHr;
		path << fixed << setprecision(6) << "../results/separatrices/b_" << b_rate << "_m_" << log10(m_rate) << ".dat";
		pathHr << fixed << setprecision(6) << "../results/separatrices/b_" << b_rate << "_m_" << log10(m_rate) << "_HR.dat";
		ifstream f(path.str());
		ifstream fHr(pathHr.str());
		
		if (f.good() && fHr.good()) {
					
			// count lines
			int numLines = 0;
			int numLinesHr = 0;
		    string line;
		    while (getline(f, line))
		        numLines++;
		    while (getline(fHr, line))
		        numLinesHr++;

		    // back to beginning of files
		    f.clear(); fHr.clear();
			f.seekg(0); fHr.seekg(0);

			// load separatrix
			separatrix = vector<vector<double> >(numLines, vector<double>(3, 0.0));
			for (int i = 0; i < separatrix.size(); i++) {
				for (int d = 0; d < 3; d++) {
					f >> separatrix[i][d];
				}
			}

			// load separatrix high res
			separatrixHiRes = vector<vector<double> >(numLinesHr, vector<double>(3, 0.0));
			for (int i = 0; i < separatrixHiRes.size(); i++) {
				for (int d = 0; d < 3; d++) {
					fHr >> separatrixHiRes[i][d];
				}
			}

			cout << "Separatrix found and loaded." << endl;
		}
		else {
			cout << "Error: separatrix not found." << endl;
			allGood = false;
			return;
		}
	}


	void storeDeterministicRun()
	{
		cout << "\nDeterministic run from fix point for storing..." << endl;
		reset();
		deterministicRun(true); // NN gets converted into concentrations (totNN becomes 1)
		cout << "\tDone..." << endl;
	}


	bool checkAllGood() {
		return allGood;
	}
};