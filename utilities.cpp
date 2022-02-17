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


#ifndef UTILITIES
#define UTILITIES

using namespace std;


void string2double(string cell, double & val)
{
	stringstream convertor(cell);
	convertor >> val;
}


unsigned nChoosek(unsigned n, unsigned k)
{
	if (k > n) return 0;
	if (k * 2 > n) k = n-k;
	if (k == 0) return 1;

	int result = n;
	for(int i = 2; i <= k; ++i) {
		result *= (n-i+1);
		result /= i;
	}
	return result;
}


void generateAllPoints(int gridSize, double startPop, vector<vector<double> > & startdemog)
{
	startdemog = vector<vector<double> >(nChoosek(gridSize+2,3-1), vector<double>(3, 0.0));
	int i = 0;
	for (int n1 = 0; n1 <= gridSize; n1++) {
		for (int n2 = 0; n2 <= gridSize-n1; n2++) {
			startdemog[i][0] = n1;
			startdemog[i][1] = n2;
			startdemog[i][2] = gridSize-n1-n2;
			i = i + 1;
		}
	}
	for (int i = 0; i < startdemog.size(); i++) {
		for (int d = 0; d < 3; d++) {
			startdemog[i][d] = round((startdemog[i][d] / gridSize) * startPop);
		}
	}
}


bool isPointBelowSeparatrix(double n2test, double n3test, vector<vector<double> > & separatrix, bool prev)
{
	if (n2test >= separatrix[separatrix.size()-1][1]) {
		return false;
	}
	double n2step = separatrix[1][1] - separatrix[0][1];
	int idx = floor(n2test / n2step);
	double n3delta = separatrix[idx+1][2] - separatrix[idx][2];
	double n3val = separatrix[idx][2] + (n3delta / n2step) * (n2test - separatrix[idx][1]);
	double dist = n3test - n3val;
	double tolerance = 0.0 * n2step;
	if (abs(dist) >= tolerance) {
		return dist > 0.0;
	}
	else {
		return prev;
	}
}


double minimumDistance(double *p, vector<vector<double> > & l, bool inside)
{
	double minDist2 = numeric_limits<double>::max();
	double d2;
	for (int i = 0; i < l.size(); i++)
	{
		d2 = (p[0]-l[i][0])*(p[0]-l[i][0]) + (p[1]-l[i][1])*(p[1]-l[i][1]) + (p[2]-l[i][2])*(p[2]-l[i][2]);
		if (d2 < minDist2) {
			minDist2 = d2;
		}
	}
	int sign;
	if (inside)
		sign = 1;
	else
		sign = -1;
	return sign * sqrt(minDist2);
}


void crossProduct(double const *v, double const *u, double *cp)
{
	cp[2] = v[0]*u[1] - v[1]*u[0];
	cp[0] = v[1]*u[2] - v[2]*u[1];
	cp[1] = v[2]*u[0] - v[0]*u[2];
}


double dotProduct(double const *v, double const *u)
{
	double dp = 0.0;
	for (int d = 0; d < 3; d++) {
		dp = dp + v[d]*u[d];
	}
	return dp;
}


void rotateVector(double const *v, double const *k, double t, double *vr)
{
	double c[3];
	crossProduct(k, v, c);
	double cos_t = cos(t);
	double sin_t = sin(t);
	for (int d = 0; d < 3; d++) {
		vr[d] = v[d] * cos_t + c[d] * sin_t;
	}
}

double angleBetweenVectors(double const *v, double const *k, double const *vr)
{
	double norm_v = sqrt(v[0]*v[0] + v[1]*v[1] + v[2]*v[2]);
	double norm_vr = sqrt(vr[0]*vr[0] + vr[1]*vr[1] + vr[2]*vr[2]);
	double t = acos(dotProduct(v, vr) / (norm_v * norm_vr));
	double ov[3];
	rotateVector(v, k, M_PI / 2.0, ov);
	if (dotProduct(ov, vr) < 0) {
		t = 2.0*M_PI - t;
	}
	if (t == 2*M_PI) {
		t = 0;
	}
	return t;
}


#endif