#include "../../POLAR/NeuralNetwork.h"
//#include "../flowstar-toolbox/Constraint.h"

using namespace std;
using namespace flowstar;

int main(int argc, char *argv[])
{
	// Declaration of the state variables.
	unsigned int numVars = 9;

	Variables vars;

	int x1_id = vars.declareVar("x1");
	int x2_id = vars.declareVar("x2");
	int x3_id = vars.declareVar("x3");
	int x4_id = vars.declareVar("x4");
	int x5_id = vars.declareVar("x5");
	int x6_id = vars.declareVar("x6");
	int u1_id = vars.declareVar("u1");
	int u2_id = vars.declareVar("u2");
	int u3_id = vars.declareVar("u3");

	int domainDim = numVars + 1;

	// Define the continuous dynamics.
	// Expression<Real> deriv_x1("x4-0.25", vars);
	Expression<Real> deriv_x1("-0.25+x4", vars);	// Segmentation fault when the expression is "x4-0.25"
	Expression<Real> deriv_x2("x5+0.25", vars);
	Expression<Real> deriv_x3("x6", vars);
	Expression<Real> deriv_x4("9.81*sin(u1)/cos(u1)", vars);
	Expression<Real> deriv_x5("-9.81*sin(u2)/cos(u2)", vars);
	Expression<Real> deriv_x6("-9.81+u3", vars);	// Segmentation fault when the expression is "u3-9.81"
	Expression<Real> deriv_u1("0", vars);
	Expression<Real> deriv_u2("0", vars);
	Expression<Real> deriv_u3("0", vars);

	vector<Expression<Real> > ode_rhs(numVars);
	ode_rhs[x1_id] = deriv_x1;
	ode_rhs[x2_id] = deriv_x2;
	ode_rhs[x3_id] = deriv_x3;
	ode_rhs[x4_id] = deriv_x4;
	ode_rhs[x5_id] = deriv_x5;
	ode_rhs[x6_id] = deriv_x6;
	ode_rhs[u1_id] = deriv_u1;
	ode_rhs[u2_id] = deriv_u2;
	ode_rhs[u3_id] = deriv_u3;

	Deterministic_Continuous_Dynamics dynamics(ode_rhs);

	// Specify the parameters for reachability computation.
	Computational_Setting setting;

	unsigned int order = stoi(argv[4]);

	// stepsize and order for reachability analysis
	setting.setFixedStepsize(0.0001, order);

	// time horizon for a single control step
	setting.setTime(0.01);

	// cutoff threshold
	setting.setCutoffThreshold(1e-8);

	// print out the steps
	setting.printOff();

	// remainder estimation
	Interval I(-0.01, 0.01);
	vector<Interval> remainder_estimation(numVars, I);
	setting.setRemainderEstimation(remainder_estimation);

	//setting.printOn();

	setting.prepare();

	/*
	 * Initial set can be a box which is represented by a vector of intervals.
	 * The i-th component denotes the initial set of the i-th state variable.
	 */
	double w = stod(argv[1]);
	int steps = stoi(argv[2]);
	Interval init_x1(-0.05, -0.025), init_x2(-0.025, 0);
	Interval init_x3(0), init_x4(0);
	Interval init_x5(0), init_x6(0);
	Interval init_u1(0), init_u2(0), init_u3(0);

	std::vector<Interval> X0;
	X0.push_back(init_x1);
	X0.push_back(init_x2);
	X0.push_back(init_x3);
	X0.push_back(init_x4);
	X0.push_back(init_x5);
	X0.push_back(init_x6);
	X0.push_back(init_u1);
	X0.push_back(init_u2);
	X0.push_back(init_u3);

	// translate the initial set to a flowpipe
	Flowpipe initial_set(X0);

	Symbolic_Remainder symbolic_remainder(initial_set, 500);

	// no unsafe set
	vector<Constraint> unsafeSet;

	// result of the reachability computation
	Result_of_Reachability result;

	// define the neural network controller
	string nn_name = "tanh20x20_remodel";
	NeuralNetwork nn(nn_name);

	// the order in use
	// unsigned int order = 5;
	Interval cutoff_threshold(-1e-12, 1e-12);
	unsigned int bernstein_order = stoi(argv[3]);
	unsigned int partition_num = 4000;

	unsigned int if_symbo = stoi(argv[5]);

	double err_max = 0;
	time_t start_timer;
	time_t end_timer;
	double seconds;
	time(&start_timer);

	vector<string> state_vars;
	state_vars.push_back("x1");
	state_vars.push_back("x2");
	state_vars.push_back("x3");
	state_vars.push_back("x4");
	state_vars.push_back("x5");
	state_vars.push_back("x6");

	if (if_symbo == 0)
	{
		cout << "High order abstraction starts." << endl;
	}
	else
	{
		cout << "High order abstraction with symbolic remainder starts." << endl;
	}
	// return 0;

	// perform 35 control steps
	for (int iter = 0; iter < steps; ++iter)
	{
		cout << "Step " << iter << " starts.      " << endl;
		//vector<Interval> box;
		//initial_set.intEval(box, order, setting.tm_setting.cutoff_threshold);
		TaylorModelVec<Real> tmv_input;

		// tmv_input.tms.push_back(initial_set.tmvPre.tms[0]);
		// tmv_input.tms.push_back(initial_set.tmvPre.tms[1]);

		TaylorModelVec<Real> tmv_temp;
		initial_set.compose(tmv_temp, order, cutoff_threshold);
		tmv_input.tms.push_back(tmv_temp.tms[0]);
		tmv_input.tms.push_back(tmv_temp.tms[1]);
		tmv_input.tms.push_back(tmv_temp.tms[2]);
		tmv_input.tms.push_back(tmv_temp.tms[3]);
		tmv_input.tms.push_back(tmv_temp.tms[4]);
		tmv_input.tms.push_back(tmv_temp.tms[5]);

		// taylor propagation
        PolarSetting polar_setting(order, bernstein_order, partition_num, "Berns", "Concrete");
		TaylorModelVec<Real> tmv_output;

		if(if_symbo == 0){
			// not using symbolic remainder
			nn.get_output_tmv(tmv_output, tmv_input, initial_set.domain, polar_setting, setting);
		}
		else{
			// using symbolic remainder
			nn.get_output_tmv_symbolic(tmv_output, tmv_input, initial_set.domain, polar_setting, setting);
		}

		
		Matrix<Interval> rm1(1, 3);
		tmv_output.Remainder(rm1);
		cout << "Neural network taylor remainder: " << rm1 << endl;

		initial_set.tmvPre.tms[u1_id] = tmv_output.tms[0];
		initial_set.tmvPre.tms[u2_id] = tmv_output.tms[1];
		initial_set.tmvPre.tms[u3_id] = tmv_output.tms[2];

		if(if_symbo == 0){
			dynamics.reach(result, setting, initial_set, unsafeSet);
		}
		else{
			dynamics.reach_sr(result, setting, initial_set, unsafeSet, symbolic_remainder);
		}

		if (result.status == COMPLETED_SAFE || result.status == COMPLETED_UNSAFE || result.status == COMPLETED_UNKNOWN)
		{
			initial_set = result.fp_end_of_time;
			cout << "Flowpipe taylor remainder: " << initial_set.tmv.tms[0].remainder << "     " << initial_set.tmv.tms[1].remainder << endl;
		}
		else
		{
			printf("Terminated due to too large overestimation.\n");
			// return 1;
			break;
		}
	}


	vector<Constraint> targetSet;
	Constraint c1("x1 - 0.32", vars);		// x1 <= 0.32
	Constraint c2("-x1 - 0.32", vars);		// x1 >= -0.32
	Constraint c3("x2 - 0.32", vars);		// x2 <= 0.32
	Constraint c4("-x2 - 0.32", vars);		// x2 >= -0.32
	Constraint c5("x3 - 0.32", vars);		// x3 <= 0.32
	Constraint c6("-x3 - 0.32", vars);		// x3 >= -0.32

	targetSet.push_back(c1);
	targetSet.push_back(c2);
	targetSet.push_back(c3);
	targetSet.push_back(c4);
	targetSet.push_back(c5);
	targetSet.push_back(c6);

	bool b = result.fp_end_of_time.isInTarget(targetSet, setting);
	string reach_result;

	if(b)
	{
		reach_result = "Verification result: Yes(35)";
	}
	else
	{
		reach_result = "Verification result: No(35)";
	}

/*
	vector<Interval> end_box;
	string reach_result;
	reach_result = "Verification result: Unknown(35)";
	result.fp_end_of_time.intEval(end_box, order, setting.tm_setting.cutoff_threshold);

	if (end_box[0].inf() >= 0.0 && end_box[0].sup() <= 0.2 && end_box[1].inf() >= 0.05 && end_box[1].sup() <= 0.3)
	{
		reach_result = "Verification result: Yes(35)";
	}

	if (end_box[0].inf() >= 0.2 || end_box[0].sup() <= 0.0 || end_box[1].inf() >= 0.3 || end_box[1].sup() <= 0.05)
	{
		reach_result = "Verification result: No(35)";
	}
*/


	time(&end_timer);
	seconds = difftime(start_timer, end_timer);

	// plot the flowpipes in the x-y plane
	result.transformToTaylorModels(setting);

	Plot_Setting plot_setting(vars);
	plot_setting.setOutputDims("x1", "x2");

	int mkres = mkdir("./outputs", S_IRWXU | S_IRGRP | S_IXGRP | S_IROTH | S_IXOTH);
	if (mkres < 0 && errno != EEXIST)
	{
		printf("Can not create the directory for images.\n");
		exit(1);
	}

	std::string running_time = "Running Time: " + to_string(-seconds) + " seconds";

	ofstream result_output("./outputs/polar_quadrotor_verisig_" + to_string(if_symbo) + ".txt");
	if (result_output.is_open())
	{
		result_output << reach_result << endl;
		result_output << running_time << endl;
	}
	// you need to create a subdir named outputs
	// the file name is example.m and it is put in the subdir outputs
	plot_setting.plot_2D_octagon_GNUPLOT("./outputs/", "polar_quadrotor_verisig_" + to_string(if_symbo), result);

	return 0;
}