#include "../../POLAR/NeuralNetwork.h"
#include "../../flowstar/flowstar-toolbox/Discrete.h"
#include <chrono>
#include <fstream>
#include <string>
// #include "../flowstar-toolbox/Constraint.h"

using namespace std;
using namespace flowstar;

double my_relu(double v)
{
	return max(v, 0.0);
}

vector<double> vector_minus(const vector<double> &lhs, const vector<double> &rhs)
{
	assert(lhs.size() == rhs.size());
	vector<double> rst;
	for (int i = 0; i < lhs.size(); ++i)
	{
		rst.push_back(lhs[i] - rhs[i]);
	}
	return rst;
}

double vector_max(const vector<double> &input)
{
	assert(input.size() > 0);
	double m = input[0];
	for (int i = 1; i < input.size(); ++i)
	{
		m = max(m, input[i]);
	}
	return m;
}

double vector_min(const vector<double> &input)
{
	assert(input.size() > 0);
	double m = input[0];
	for (int i = 1; i < input.size(); ++i)
	{
		m = min(m, input[i]);
	}
	return m;
}

double reach_loss(const vector<double> &final_lbs, const vector<double> &final_ubs,
				  const vector<double> &reach_set_lbs, const vector<double> &reach_set_ubs)
{
	double lb_loss = my_relu(vector_max(vector_minus(final_lbs, reach_set_lbs)));
	double ub_loss = my_relu(vector_max(vector_minus(reach_set_ubs, final_ubs)));
	return max(lb_loss, ub_loss);
}

double calculate_safe_loss(const vector<vector<double>> &reached_set, const vector<vector<double>> &unsafe_set)
{
	double loss = 0.0;
	bool initialized = false;
	for (int i = 0; i < reached_set.size(); ++i)
	{
		if (!initialized)
		{
			initialized = true;
			loss = my_relu(unsafe_set[i][1] - reached_set[i][0]);
			loss = min(loss, my_relu(reached_set[i][1] - unsafe_set[i][0]));
		}
		else
		{
			loss = min(loss, my_relu(unsafe_set[i][1] - reached_set[i][0]));
			loss = min(loss, my_relu(reached_set[i][1] - unsafe_set[i][0]));
		}
	}
	return loss;
}

int main(int argc, char *argv[])
{
	bool plot = false;
	bool print_safe_sets = false;
	if (argc >= 3)
	{
		if (string(argv[2]) == "--plot")
		{
			plot = true;
		}
		else if (string(argv[2]) == "--safe_sets")
		{
			print_safe_sets = true;
		}
	}
	string benchmark_name = "acc";
	// Declaration of the state variables.
	unsigned int numVars = 3;

	intervalNumPrecision = 200;

	Variables vars;

	int x1_id = vars.declareVar("x1");
	int x2_id = vars.declareVar("x2");
	int x3_id = vars.declareVar("x3");
	int x4_id = vars.declareVar("x4");
	int u_id = vars.declareVar("u");

	int domainDim = numVars + 1;

	vector<string> dynamics_str{
		"x1+0.02*x2",
		"x2+0.02*(((u + 0.05 * x4 * x4 * sin(x3)) / 1.1)  -  0.05 * ((9.8 * sin(x3) - cos(x3) *  ((u + 0.05 * x4 * x4 * sin(x3)) / 1.1)) / (0.5 * (4.0/3.0 - 0.1 * cos(x3) * cos(x3) / 1.1))) * cos(x3) / 1.1)",
		"x3+0.02*x4",
		"x4+0.02*((9.8 * sin(x3) - cos(x3) *  ((u + 0.05 * x4 * x4 * sin(x3)) / 1.1)) / (0.5 * (4.0/3.0 - 0.1 * cos(x3) * cos(x3) / 1.1)))",
		"0"};
	DDE<Real> dynamics(dynamics_str, vars);

	// vector<string> dynamics_str{
	// "x2",
	// "(((u + 0.05 * x4 * x4 * sin(x3)) / 1.1)  -  0.05 * ((9.8 * sin(x3) - cos(x3) *  ((u + 0.05 * x4 * x4 * sin(x3)) / 1.1)) / (0.5 * (4.0/3.0 - 0.1 * cos(x3) * cos(x3) / 1.1))) * cos(x3) / 1.1)",
	// "x4",
	// "((9.8 * sin(x3) - cos(x3) *  ((u + 0.05 * x4 * x4 * sin(x3)) / 1.1)) / (0.5 * (4.0/3.0 - 0.1 * cos(x3) * cos(x3) / 1.1)))",
	// "0"
	// };
	// ODE<Real> dynamics(dynamics_str, vars);

	// Specify the parameters for reachability computation.
	Computational_Setting setting(vars);
	// Computational_Setting setting;

	unsigned int order = 4;

	// stepsize and order for reachability analysis
	setting.setFixedStepsize(0.02, order); // the stepsize will be ignored

	// cutoff threshold
	setting.setCutoffThreshold(1e-8);

	// print out the steps
	setting.printOff();

	/*	// DDE does not require a remainder estimation
		Interval I(-0.01, 0.01);
		vector<Interval> remainder_estimation(numVars, I);
		setting.setRemainderEstimation(remainder_estimation);
	*/
	// setting.printOn();

	// setting.prepare();

	/*
	 * Initial set can be a box which is represented by a vector of intervals.
	 * The i-th component denotes the initial set of the i-th state variable.
	 */
	int steps = 200;
	Interval init_x1(-0.05, 0.05), init_x2(-0.05, 0.05), init_x3(-0.05, 0.05), init_x4(-0.05, 0.05);
	Interval init_u(0);

	std::vector<Interval> X0;
	X0.push_back(init_x1);
	X0.push_back(init_x2);
	X0.push_back(init_x3);
	X0.push_back(init_x4);
	X0.push_back(init_u);

	// translate the initial set to a flowpipe
	Flowpipe initial_set(X0);

	Symbolic_Remainder symbolic_remainder(initial_set, 100);

	// no unsafe set
	vector<Constraint> safeSet;
	// vector<Constraint> unsafeSet;

	// result of the reachability computation
	Result_of_Reachability result;

	// the order in use
	// unsigned int order = 5;
	Interval cutoff_threshold(-1e-12, 1e-12);
	unsigned int bernstein_order = 4;
	unsigned int partition_num = 4000;
	// unsigned int partition_num = 200;

	unsigned int if_symbo = 1;
	;

	double err_max = 0;
	time_t start_timer;
	time_t end_timer;
	double seconds;
	double nn_total_time = 0.0, flowstar_total_time = 0.0;

	auto begin = std::chrono::high_resolution_clock::now();
	string controller_base = string(argv[1]); //+net_name;
	// cout << "controller base is " << controller_base;
	int interval = 100;
	NeuralNetwork *nn = nullptr;
	double safe_loss = 0.0;
	vector<vector<double>> unsafe_set = {{1.0, 2.0}, {1.0, 2.0}};
	for (int iter = 0; iter < steps; ++iter)
	{
		// cout << "Step " << iter << " starts.      " << endl;
		// vector<Interval> box;
		// initial_set.intEval(box, order, setting.tm_setting.cutoff_threshold);
		// define the neural network controller
		if (iter % interval == 0)
		{
			nn = new NeuralNetwork(controller_base + "_" + to_string(iter));
		}

		TaylorModelVec<Real> tmv_input;

		tmv_input.tms.push_back(initial_set.tmvPre.tms[0]);
		tmv_input.tms.push_back(initial_set.tmvPre.tms[1]);
		tmv_input.tms.push_back(initial_set.tmvPre.tms[2]);
		tmv_input.tms.push_back(initial_set.tmvPre.tms[3]);

		// taylor propagation
		PolarSetting polar_setting(order, bernstein_order, partition_num, "Mix", "Concrete");
		polar_setting.set_num_threads(-1);
		TaylorModelVec<Real> tmv_output;

		if (if_symbo == 0)
		{
			// not using symbolic remainder
			nn->get_output_tmv(tmv_output, tmv_input, initial_set.domain, polar_setting, setting);
		}
		else
		{
			// using symbolic remainder
			nn->get_output_tmv_symbolic(tmv_output, tmv_input, initial_set.domain, polar_setting, setting);
		}

		initial_set.tmvPre.tms[u_id] = tmv_output.tms[0];

		// for (int i = 0; i < noise_var_ids.size(); ++i) {
		// initial_set.tmvPre.tms[noise_var_ids[i]].remainder = initial_set.tmvPre.tms[noise_var_ids[i]].remainder + Interval(-3 * stds[i], 3 * stds[i]);
		// }
		// Always using symbolic remainder
		// cout << "before reach is called" << endl;
		dynamics.reach(result, setting, initial_set, 1, safeSet, symbolic_remainder);
		// dynamics.reach(result, initial_set, 0.02, setting, safeSet, symbolic_remainder);

		if (result.status == COMPLETED_SAFE || result.status == COMPLETED_UNSAFE || result.status == COMPLETED_UNKNOWN)
		{
			initial_set = result.fp_end_of_time;
			// for (int i = 0; i < 4; ++i) {
			// initial_set.tmvPre.tms[i].remainder = initial_set.tmvPre.tms[i].remainder + Interval(-3.0 * stds[i], 3.0 * stds[i]);
			// }
			vector<Interval> inter_box;
			result.fp_end_of_time.intEval(inter_box, order, setting.tm_setting.cutoff_threshold);
			vector<vector<double>> reached_set = {{inter_box[0].inf(), inter_box[0].sup()}, {inter_box[1].inf(), inter_box[1].sup()}};
			safe_loss += calculate_safe_loss(reached_set, unsafe_set);
			if (plot || print_safe_sets)
			{
				cout << inter_box[0].inf() << " " << inter_box[0].sup() << " " << inter_box[1].inf() << " " << inter_box[1].sup() << " "
					 << inter_box[2].inf() << " " << inter_box[2].sup() << " " << inter_box[3].inf() << " " << inter_box[3].sup() << "\n";
			}

			if (inter_box[0].inf() >= -0.05 && inter_box[0].sup() <= 0.05 && inter_box[1].inf() >= -0.05 && inter_box[1].sup() <= 0.05 &&
				inter_box[2].inf() >= -0.05 && inter_box[2].sup() <= 0.05 && inter_box[3].inf() >= -0.05 && inter_box[3].sup() <= 0.05)
			{
				cout << "returned to initial region at step " << iter << "\n";
			}
			// cout << "Flowpipe taylor remainder: " << initial_set.tmv.tms[0].remainder << "     " << initial_set.tmv.tms[1].remainder << endl;
		}
		else
		{
			printf("Terminated due to too large overestimation.\n");
			return 1;
		}
	}

	vector<Interval> inter_box;
	result.fp_end_of_time.intEval(inter_box, order, setting.tm_setting.cutoff_threshold);
	vector<double> final_lbs(4, -0.05);
	vector<double> final_ubs(4, 0.05);
	vector<double> reach_set_lbs;
	vector<double> reach_set_ubs;
	for (int i = 0; i < 4; ++i)
	{
		reach_set_lbs.push_back(inter_box[i].inf());
		reach_set_ubs.push_back(inter_box[i].sup());
		// if (print)
		// {
		// 	cout << inter_box[i].inf() << " ";
		// 	cout << inter_box[i].sup() << " ";
		// }
	}
	// plot the flowpipes in the x-y plane
	if (plot)
	{
		result.transformToTaylorModels(setting);

		Plot_Setting plot_setting(vars);
		plot_setting.setOutputDims("x1", "x2");

		int mkres = mkdir("./outputs", S_IRWXU | S_IRGRP | S_IXGRP | S_IROTH | S_IXOTH);
		if (mkres < 0 && errno != EEXIST)
		{
			printf("Can not create the directory for images.\n");
			exit(1);
		}

		std::string running_time = "Running Time: " + to_string(seconds) + " seconds";

		// you need to create a subdir named outputs
		// the file name is example.m and it is put in the subdir outputs
		plot_setting.plot_2D_interval_GNUPLOT("./outputs/", benchmark_name + "_" + to_string(if_symbo), result.tmv_flowpipes, setting);
		// plot_setting.plot_2D_octagon_GNUPLOT("./outputs/", benchmark_name + "_" + to_string(if_symbo), result);
	}

	cout << reach_loss(final_lbs, final_ubs, reach_set_lbs, reach_set_ubs) << endl;
	return 0;
}
