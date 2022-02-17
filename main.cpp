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


#include <float.h> 

#include "model.cpp"
#include "utilities.cpp"

using namespace std;


int main(int argc, char * argv[])
{
	// variables for arguments
	double b_rate; 		// argv[1]
	double m_rate; 		// argv[2] is the exponent
	double N0;			// argv[3]
	bool withSepa; 		// argv[4]
	bool recDetTraj;	// argv[5]
	int numTrajToRec; 	// argv[6]
	int runId;			// argv[7]
	bool noGrowth;		// argv[8]
	double startNN1;	// argv[9]
	double startNN2;	// argv[10]
	double startNN3;	// argv[11]
	double N0name;

	// check arguments
	if ( strtod(argv[1], NULL) >= 0.0 &&
		 strtod(argv[3], NULL) > 0 &&
		 (strtod(argv[4], NULL) == 0 || strtod(argv[4], NULL) == 1) &&
		 (strtod(argv[5], NULL) == 0 || strtod(argv[5], NULL) == 1) &&
		 strtod(argv[6], NULL) >= 0 &&
		 strtod(argv[7], NULL) >= 0)
	{
		b_rate = strtod(argv[1], NULL);
		m_rate = pow(10.0, strtod(argv[2], NULL));
		//N0 = pow(10.0, strtod(argv[3], NULL)); N0name = log10(N0);
		N0 = strtod(argv[3], NULL); N0name = N0;
		withSepa = strtod(argv[4], NULL);
		recDetTraj = strtod(argv[5], NULL);
		numTrajToRec = strtod(argv[6], NULL);
		runId = strtod(argv[7], NULL);
		noGrowth = strtod(argv[8], NULL);
	}
	else {
		cout << "wrong execution type\n";
		return 1;
	}

	// simulation parameters
	int numIt = 1000;			// number of repetitions (10000 full run, 1000 cond-prob)
	double maxTime = DBL_MAX;	// time limit per repetition (should be DBL_MAX, 20000 for fix size)
	double maxPop = pow(10.0, 10.0);	// population limit
										// full run: pow(10.0, 10.0)
										// cond-prob-1: 500.0
										// extinction pow(2.0, 15.0)
										// example trajectories pow(10, 6.0)
										// fixed size DBL_MAX (shouldn't change)
	double methThr = 10000; 	// threshold for switch method (should be 10000)
	int initMode = 1;			// starting point
									// -1: specific
									// 0: random
									// 1: fix point
									// 2: TFT corner
									// 3: MTC edge
									// 4: centre
									// 5: AllD corner
									// 6: p1
									// 7: p2

	// specific start-point (if necessary)
	if (initMode == -1) {
		startNN1 = strtod(argv[9], NULL);
		startNN2 = strtod(argv[10], NULL);
		startNN3 = strtod(argv[11], NULL);
	} else {
		startNN1 = 0;
		startNN2 = 0;
		startNN3 = 0;
	}

	// other fixed parameters
	double d_rate = 1.0; 	// death rate (may change if noGroth == 1)
	if (noGrowth == true)
		d_rate = 0;
	int m = 10;   			// average number of rounds
	double T = 5.0;    		// temptation to defect
	double R = 3.0;    		// mutual reward for cooperation
	double P = 1.0;    		// punishment for mutual defection
	double S = 0.1;			// looser's payoff
	double c = 0.8;  		// TFT complexity cost

	// create model
	TriStateGrowth TSG(
		b_rate, m_rate, d_rate,
		numIt, maxTime,
		m, T, R, P, S, c,
		N0, N0name,
		methThr, maxPop,
		initMode,
		numTrajToRec,
		runId,
		noGrowth,
		startNN1, startNN2, startNN3);
	if (TSG.checkAllGood() == false) {
		return 0;
	}

	// compute separatrix
	if (withSepa == 1) {
		TSG.computeSeparatrix();
	} else {
		TSG.loadSeparatrix();
	}
	if (TSG.checkAllGood() == false) {
		return 0;
	}

	// record deterministic trajectory
	if (recDetTraj == 1) {
		TSG.storeDeterministicRun();
	}
	if (TSG.checkAllGood() == false) {
		return 0;
	}

	// run
	TSG.run();
	if (TSG.checkAllGood() == false) {
		return 0;
	}

	// done
	cout << "\nAll done mate!\n";
	return 1;
}