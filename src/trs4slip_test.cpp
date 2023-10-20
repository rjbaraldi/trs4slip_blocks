#include <algorithm>
#include <cmath>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <fstream>
#include <vector>


#include "trs4slip.h"


using namespace std;


double const ABS_TOL = 1e-14;
double const REL_TOL = 1e-8;


int main(int argc, char* argv[])
{ 
    vector<double> const lo_alpha({ 1e-5,1e-4,1e-3 });
    vector<int> const lo_delta({ 512,256,128,64,32 });
    vector<int32_t> const lo_bangs({ 0, 1, 2 });
    int const NUM_TESTS = 54;
    int const N_MAX = 16384;
    int const M = lo_bangs.size();
    int const DELTA_MAX = *max_element(lo_delta.begin(), lo_delta.end());
    int const BUFFER_SIZE = N_MAX*M*(DELTA_MAX + 1) + 2;

    vector<int32_t> x;
    x.reserve(N_MAX);
    vector<double> c;
    c.reserve(N_MAX);
    vector<double> s;
    s.reserve(N_MAX);
    vector<int32_t> x_out(N_MAX, 0.);
    vector<double> costs_buffer(BUFFER_SIZE);
    vector<int32_t> layer_buffer(BUFFER_SIZE);
    vector<int32_t> value_buffer(BUFFER_SIZE);
    vector<int32_t> prev_buffer(BUFFER_SIZE);
    vector<int32_t> remcap_buffer(BUFFER_SIZE);

    stringstream fnc, fnx, fns;
    string line;
    string const current_path = filesystem::current_path().string();

    int num_errors = 0;
    for (int i = 0; i < NUM_TESTS; i++)
    {
        c.clear();        
        fnc.str(string());
        fnc << current_path << "/inputs/c_" << setfill('0') 
            << setw(4) << i << ".csv"; 
        ifstream ifc(fnc.str().c_str());
        while (getline(ifc, line))
        {
            c.push_back(stod(line));
        }        

        x.clear();
        fnx.str(string());
        fnx << current_path << "/inputs/x_" << setfill('0') 
            << setw(4) << i << ".csv";
        ifstream ifx(fnx.str().c_str());
        while (getline(ifx, line))
        {
            x.push_back(stod(line));
        }

        for (int j = 0; j < lo_alpha.size(); j++)
        {
            vector<double> c_(c.size(), 0.);
            for (int l = 0; l < c.size(); l++)
            {
                c_[l] = c[l] / lo_alpha[j];
            }
            for (int k = 0; k < lo_delta.size(); k++)
            {
                s.clear();
                fns.str(string());
                fns << current_path << "/solutions/sol_" 
                    << setfill('0') << setw(4) << i 
                    << "_" << j << "_" << k << ".csv";

                ifstream ifs(fns.str().c_str());
                while (getline(ifs, line))
                {
                    s.push_back(stod(line));
                }

                trs4slip_astar(
                    &x_out[0],
                    &c_[0],
                    &x[0],
                    &lo_bangs[0],
                    lo_delta[k],
                    x.size(),
                    M,
                    &costs_buffer[0],
                    &layer_buffer[0],
                    &value_buffer[0],
                    &prev_buffer[0],
                    &remcap_buffer[0]
                );

                double l2err = 0;
                for (int l = 0; l < x.size(); l++)
                {
                    l2err += (x_out[l] - s[l])*(x_out[l] - s[l]);
                }
                l2err = sqrt(l2err);

                if(l2err > 0.)
                {
                    double obj = c[0] * (x_out[0] - x[0]);
                    double obj_ref = c[0] * (s[0] - x[0]);
                    for (int l = 1; l < x.size(); l++)
                    {
                        obj += c[l] * (x_out[l] - x[l]) 
                            + lo_alpha[j] * abs(x_out[l] - x_out[l - 1]);
                        obj_ref += c[l] * (s[l] - x[l])
                            + lo_alpha[j] * abs(s[l] - s[l - 1]);
                    }

                    double diff = abs(obj - obj_ref);
                    if(diff / abs(obj_ref) > REL_TOL or diff > ABS_TOL)
                    {
                        num_errors++;
                        cout << setw(2) << i << " " << j << " " << k 
                            << "  " << l2err << "  " << obj << " " << obj_ref 
                            << "\n";
                    }
                }
            }
        }
    }
    return num_errors;
}
