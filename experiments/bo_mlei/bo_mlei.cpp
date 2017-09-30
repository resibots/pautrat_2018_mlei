#include <limbo/limbo.hpp>
#include "meanMap.hpp"
#include "statTransferts.hpp"
#include "bo_optimizer.hpp"
#include "gp_mlei.hpp"
#include "exhaustiveSearchIntersection.hpp"
#include "ei_acq.hpp"
#include "hexapod_dart/hexapod_dart_simu.hpp"
#include "hexapod_dart/binary_map.hpp"

#ifdef ROBOT
#include <ros/ros.h>
#include <hexapod_driver/hexapod.hpp>
#include <thread>
#include <mutex>
//#include "controller_manager_msgs/SwitchController.h"
#endif

#include <iostream>
#include <cstdlib>
#include <algorithm>
#include <vector>
#include <ctime>

using namespace limbo;


Eigen::VectorXd make_v1(double x)
{
    Eigen::VectorXd v1(1);
    v1 << x;
    return v1;
}

template <typename T>
inline T gaussian_rand(T m = 0.0, T v = 1.0)
{
    std::random_device rd;
    std::mt19937 gen(rd());

    std::normal_distribution<T> gaussian(m, v);

    return gaussian(gen);
}

struct Params {
    struct opt_rprop : public defaults::opt_rprop {
        BO_PARAM(int, iterations, 300);
    };
    struct bayes_opt_bobase {
        BO_PARAM(int, stats_enabled, true);
        BO_PARAM(bool, bounded, true);
    };
    struct bayes_opt_boptimizer : public defaults::bayes_opt_boptimizer {
        BO_PARAM(double, noise, 0.00001);
        BO_PARAM(int, hp_period, 1);
    };
    struct stop_maxiterations {
        BO_DYN_PARAM(int, iterations);
    };
    struct acqui_ucb : public defaults::acqui_ucb {
        BO_DYN_PARAM(float, alpha);
    };
    struct acqui_ei : public defaults::acqui_ei {
        BO_PARAM(double, jitter, 0.0);
    };
    struct kernel_squared_exp_ard : public defaults::kernel_squared_exp_ard {
    };

    struct data {
        typedef int nb_priors_t;
        static int nb_priors;
        typedef int brk_leg_t;
        static int brk_leg;
    };
    struct archiveparams {

        struct elem_archive {
            std::vector<double> descriptor;
            float fit;
            std::vector<double> controller;
        };

        struct classcomp {
            bool operator()(const std::vector<double>& lhs, const std::vector<double>& rhs) const
            {
                assert(lhs.size() == 6 && rhs.size() == 6);
                int i = 0;
                while (i < 5 && std::round(lhs[i] * 4) == std::round(rhs[i] * 4)) //lhs[i]==rhs[i])
                    i++;
                return std::round(lhs[i] * 4) < std::round(rhs[i] * 4); //lhs[i]<rhs[i];
            }
        };
        typedef std::map<std::vector<double>, elem_archive, classcomp> archive_t;
        static std::map<std::vector<double>, elem_archive, classcomp> archive;
        typedef std::vector<std::map<std::vector<double>, elem_archive, classcomp> > archives_t;
        static std::vector<std::map<std::vector<double>, elem_archive, classcomp> > archives;
        typedef std::vector<std::vector<double> > descriptors_list_t;
        static std::vector<std::vector<double> > descriptors_list;
    };
};

Params::archiveparams::archive_t load_archive(std::string archive_name);
Params::archiveparams::archive_t create_random_map(int size);

namespace global {

    int nb_controllers = 54;
    struct timeval timev_selection; // Initial absolute time (static)
    std::string res_dir;
#ifdef ROBOT
    std::shared_ptr<hexapod_ros::Hexapod> hexa;
    std::mutex mutex;
#else
    std::vector<int> brokenLegs;
#endif
    std::shared_ptr<hexapod_dart::Hexapod> global_robot;
};
///---------------------------

#ifdef ROBOT
void init_ros_node(int argc, char** argv)
{
    ros::init(argc, argv, "hexapod_driver");
    ros::NodeHandle n;
    global::hexa = std::make_shared<hexapod_ros::Hexapod>(n);
}
#else
void init_simu(std::string robot_file, std::vector<int> broken_legs = std::vector<int>())
{
    std::vector<hexapod_dart::HexapodDamage> damages(broken_legs.size());
    for (size_t i = 0; i < broken_legs.size(); ++i)
        damages.push_back(hexapod_dart::HexapodDamage("leg_removal", std::to_string(broken_legs[i]))); // the kind of damage can be modified here (e.g. "leg_shortening")
    global::global_robot = std::make_shared<hexapod_dart::Hexapod>(robot_file, damages);
}
#endif

// stop the robot if he moves to far to the left to the right or too far in the x direction
#ifdef ROBOT
void safety_check()
{
    bool ok = true;
    // attempt to get the frequency for the control loop; otherwise, set the frequency to 50 Hz.
    int frequency;
    if (!ros::param::get("/dynamixel_control_hw/loop_frequency", frequency))
        frequency = 50;

    ros::Rate rate(frequency);
    while (ok) {
        rate.sleep();
        auto t = global::hexa->transform();
        auto p = t.getOrigin();
        if(global::mutex.try_lock())
        {
            global::mutex.unlock();
            break;
        }
        if(fabs(p[1]) > 0.4 || p[0] > 3.0)
            ok = false;
    }
    if(!ok) {
        global::hexa->stop_trajectories();
        ROS_INFO("Robot stepped out of the corridor.");
    }
    ROS_DEBUG("Corridor-related thread terminated.");
}
#endif


template <typename Params, int obs_size = 1>
struct fit_eval_map {

    BOOST_STATIC_CONSTEXPR int dim_in = 6;

    BOOST_STATIC_CONSTEXPR int dim_out = obs_size;

    fit_eval_map()
    {
        timerclear(&global::timev_selection);
        gettimeofday(&global::timev_selection, NULL);
    }

    Eigen::VectorXd operator()(Eigen::VectorXd x) const
    {
        std::vector<double> key(x.size(), 0);
        for (int i = 0; i < x.size(); i++)
            key[i] = x[i];
        if (Params::archiveparams::archive.count(key) == 0) {
            throw limbo::EvaluationError();   
        }

#ifdef ROBOT
        std::vector<double> ctrl = Params::archiveparams::archive.at(key).controller;
        double duration = 10.0;

        std::cout << "Robot is about to move. Type any key when ready" << std::endl;
        getchar();

        global::hexa->zero();

        global::mutex.lock();
        std::thread safeguard(safety_check);
        global::hexa->move(ctrl, duration);
        global::mutex.unlock();
        safeguard.join();
        
        auto t = global::hexa->transform();
        auto p = t.getOrigin();
        ROS_INFO_STREAM("Pos: " << p[0] << " " << p[1] << " " << p[2]);

        float obs = p[0];

        return make_v1(obs);
#else
        std::vector<double> ctrl = Params::archiveparams::archive.at(key).controller;
        hexapod_dart::HexapodDARTSimu<> simu(ctrl, global::global_robot->clone());
        simu.run(10);

     // std::cout << "covered distance: " << simu.covered_distance() << std::endl;

        float value = simu.covered_distance();

        return make_v1(value);
#endif
    }
};


// Load an archive file
std::map<std::vector<double>, Params::archiveparams::elem_archive, Params::archiveparams::classcomp> load_archive(std::string archive_name)
{

    std::map<std::vector<double>, Params::archiveparams::elem_archive, Params::archiveparams::classcomp> archive;

    std::ifstream monFlux(archive_name.c_str());
    if (monFlux) {
        while (!monFlux.eof()) {
            Params::archiveparams::elem_archive elem;
            std::vector<double> candidate(6);
            for (int i = 0; i < global::nb_controllers+7; i++) {
                if (monFlux.eof())
                    break;
                double data;
                monFlux >> data;
                if (i <= 5) {
                    candidate[i] = data;
                    elem.descriptor.push_back(data);
                }
                if (i == 6) {

                    elem.fit = data;
                }
                if (i >= 7)
                    elem.controller.push_back(data);
            }
            if ((int) elem.controller.size() == global::nb_controllers) {
                archive[candidate] = elem;
            }
        }
    }
    else {
        std::cerr << "ERROR: Could not load the archive." << std::endl;
        return archive;
    }
    std::cout << archive.size() << " elements loaded" << std::endl;
    return archive;
}


// Load a binary file
std::map<std::vector<double>, Params::archiveparams::elem_archive, Params::archiveparams::classcomp> load_binary(std::string binary_name)
{

    std::map<std::vector<double>, Params::archiveparams::elem_archive, Params::archiveparams::classcomp> archive;

    try {
        binary_map::BinaryMap binary_map = binary_map::load(binary_name);
        std::vector<binary_map::Elem> elems = binary_map.elems;
        for (auto& line : elems) {
            Params::archiveparams::elem_archive elem;
            std::vector<double> candidate(6);
            for(int i=0; i < 6; i++) {
                candidate[i] = float(round(line.pos[i] * 100)) / 100.;
                elem.descriptor.push_back(candidate[i]);
            }
            elem.fit = line.fit;
            for(double c : line.phen)
                elem.controller.push_back(c);
            if ((int) elem.controller.size() == global::nb_controllers) {
                    archive[candidate] = elem;
                }
        }
    }
    catch(int e) {
        std::cerr << "ERROR: Could not load the binary." << std::endl;
        return archive;
    }
    std::cout << archive.size() << " elements loaded" << std::endl;
    return archive;
}


// Replay a controller given by the user
void lecture(const std::vector<double>& controller)
{
    if ((int) controller.size() != global::nb_controllers) {
        std::cerr << "Controller needs " << global::nb_controllers << " parameters!" << std::endl;
        return;
    }
#ifdef ROBOT
    double duration = 10.0;

    std::cout << "Robot is about to move. Type any key when ready" << std::endl;
    getchar();

    global::hexa->zero();

    global::mutex.lock();
    std::thread safeguard(safety_check);
    global::hexa->move(controller, duration);
    global::mutex.unlock();
    safeguard.join();

    auto t = global::hexa->transform();
    auto p = t.getOrigin();
    ROS_INFO_STREAM("Pos: " << p[0] << " " << p[1] << " " << p[2]);

    std::cout << "covered distance: " << p[0] << std::endl;

    global::hexa.reset();
#else
    hexapod_dart::HexapodDARTSimu<> simu(controller, global::global_robot->clone());
    simu.run(10);
    std::cout << "covered distance: " << simu.covered_distance() << std::endl;
#endif

    return;
}


// fill descriptors_list with the descriptors that are present in all the prior archives
void fill_descriptors_list() {
    Params::archiveparams::descriptors_list.clear();
    for(Params::archiveparams::archive_t::iterator it = Params::archiveparams::archives[0].begin(); it != Params::archiveparams::archives[0].end(); it++) {
        int i = 1;
        while(i < Params::data::nb_priors) {
            if(Params::archiveparams::archives[i].count(it->first) == 0) // check that the descriptor it->first is in all the priors
                break;
            i++;
        }
        if(i == Params::data::nb_priors)
            Params::archiveparams::descriptors_list.push_back(it->first);
    }
}


// choose randomly 5 different numbers between 0 and exp_nb
std::vector<int> select_priors(int exp_nb) {
    std::vector<int> priors;
    int next = rand() % exp_nb;
    priors.push_back(next);
    for(int i=0; i < 4; i++) {
        next = rand() % exp_nb;
        while(std::find(priors.begin(), priors.end(), next) != priors.end())
            next = rand() % exp_nb;
        priors.push_back(next);
    }
    return priors;
}


Params::archiveparams::archive_t Params::archiveparams::archive;
Params::archiveparams::archives_t Params::archiveparams::archives;
Params::archiveparams::descriptors_list_t Params::archiveparams::descriptors_list;
Params::data::nb_priors_t Params::data::nb_priors;
Params::data::brk_leg_t Params::data::brk_leg;
BO_DECLARE_DYN_PARAM(int, Params::stop_maxiterations, iterations);
BO_DECLARE_DYN_PARAM(float, Params::acqui_ucb, alpha);

int main(int argc, char** argv)
{
    std::vector<std::string> cmd_args;
    for (int i = 0; i < argc; i++)
        cmd_args.push_back(std::string(argv[i]));

    std::vector<std::string>::iterator legs_it = std::find(cmd_args.begin(), cmd_args.end(), "-l"); // id of the broken leg
    std::vector<std::string>::iterator ctrl_it = std::find(cmd_args.begin(), cmd_args.end(), "-c"); // controller with 54 paramteres
    std::vector<std::string>::iterator n_it = std::find(cmd_args.begin(), cmd_args.end(), "-n"); // number of iterations in Bayesian optimization
    std::vector<std::string>::iterator prior_it = std::find(cmd_args.begin(), cmd_args.end(), "-s"); // method of prior selection
    // 0: selects a prior at random
    // 1: always selects the same prior (its value should be set by hand in the code in boptimizer.hpp)
    // 2: selects the prior using MLEI
    std::vector<std::string>::iterator type_it = std::find(cmd_args.begin(), cmd_args.end(), "-t"); // number of different types of priors
    std::vector<std::string>::iterator nb_priors_it = std::find(cmd_args.begin(), cmd_args.end(), "-p"); // number of priors for each type of prior
    std::vector<std::string>::iterator help = std::find(cmd_args.begin(), cmd_args.end(), "-h"); // help command

    if (help != cmd_args.end()) {
        std::cout << "Please provide the directory of your priors as first argument (unless you are just replaying a controller)." << std::endl;
        std::cout << "Available options:" << std::endl;
        std::cout << "-l : id of the leg you want to remove (no broken leg if -l is not specified)" << std::endl;
        std::cout << "-s : method used of prior selection (0: random, 1: constant prior, 2: MLEI, default: MLEI)" << std::endl;
        std::cout << "-t : number of types of different priors that you use (default: 4)" << std::endl;
        std::cout << "-p : number of priors used for each type of prior (default: 15)" << std::endl;
        std::cout << "-n : number of iterations for Bayesian optimization (default: 10)" << std::endl; 
        std::cout << "-c : controller of the robot (only for replaying one gait, there is no learning here)" << std::endl;
        return 0;
    }

#ifdef ROBOT
    init_ros_node(argc, argv);

#else
    std::vector<int> brokenleg;
    if (legs_it != cmd_args.end()) {
        std::vector<std::string>::iterator end_it = (legs_it < ctrl_it) ? ctrl_it : cmd_args.end();
        end_it = (end_it < n_it || n_it < legs_it) ? end_it : n_it;

        for (std::vector<std::string>::iterator ii = legs_it + 1; ii != end_it; ii++) {
            brokenleg.push_back(atoi((*ii).c_str()));
        }
        if (brokenleg.size() >= 6) {
            std::cerr << "The robot should at least have one leg!" << std::endl;
            if (global::global_robot)
                global::global_robot.reset();
            return -1;
        }
    }
    global::brokenLegs = brokenleg;

    // initialization of the simulation and the simulated robot
    const char* env_p = std::getenv("RESIBOTS_DIR");

    if (env_p) //if the environment variable exists
        init_simu(std::string(env_p) + "/share/hexapod_models/URDF/pexod.urdf", global::brokenLegs);
    else {
        std::cout << "You have to set the RESIBOTS_DIR environment variable at the root of your installation." << std::endl;
        return 0;
    }
#endif

    if (ctrl_it != cmd_args.end()) {
        std::vector<std::string>::iterator end_it = ctrl_it + global::nb_controllers + 1;

        std::vector<double> ctrl;
        for (std::vector<std::string>::iterator ii = ctrl_it + 1; ii != end_it; ii++) {
            ctrl.push_back(float(atof((*ii).c_str())));
        }
        if ((int) ctrl.size() != global::nb_controllers) {
            std::cerr << "You have to provide " << global::nb_controllers << " controller parameters!" << std::endl;
            if (global::global_robot)
                global::global_robot.reset();
            return -1;
        }
        lecture(ctrl);

        if (global::global_robot)
            global::global_robot.reset();
        return 1;
    }

    if (n_it != cmd_args.end()) {
        Params::stop_maxiterations::set_iterations(atoi((n_it + 1)->c_str()));
    }
    else
        Params::stop_maxiterations::set_iterations(10);

    Params::acqui_ucb::set_alpha(0.2);
    srand(time(NULL));

    typedef kernel::SquaredExpARD<Params> Kernel_t;
    typedef opt::Rprop<Params> ParametersOpt_t;
    typedef opt::ExhaustiveSearchIntersection<Params> InnerOpt_t;
    typedef boost::fusion::vector<stop::MaxIterations<Params>> Stop_t;
    typedef mean_functions::MeanArchive_Map<Params> Mean_t;
    typedef boost::fusion::vector<stat::Samples<Params>, stat::BestObservations<Params>, stat::AggregatedObservations<Params>, stat::StatTransferts<Params>> Stat_t;

    typedef init::NoInit<Params> Init_t;
    typedef model::GP_MLEI<Params, Kernel_t, Mean_t, model::gp::KernelLFOpt<Params, ParametersOpt_t>> GP_t;
    typedef acqui::EIAcq<Params, GP_t> Acqui_t;

    int nb_prior_type = 4;
    if (type_it != cmd_args.end())
        nb_prior_type = atoi((type_it + 1)->c_str());

    int nb_prior_per_type = 15;
    if (nb_priors_it != cmd_args.end())
        nb_prior_per_type = atoi((nb_priors_it + 1)->c_str());

    // load the maps
    Params::archiveparams::archives.clear();
    for(int prior_type=0; prior_type < nb_prior_type; prior_type++) { 
        std::vector<int> indexes = select_priors(nb_prior_per_type); // select several priors among the nb_prior_per_type priors available
        for(int e : indexes) {
            std::ostringstream archive_oss;
            archive_oss << argv[1] << "/prior" << prior_type << "_" << e << ".bin";
            Params::archiveparams::archives.push_back(load_binary(archive_oss.str()));
        }    
    }
    Params::data::nb_priors = nb_prior_type * nb_prior_per_type;

    fill_descriptors_list();

    if (legs_it != cmd_args.end())
        Params::data::brk_leg = atoi((legs_it + 1)->c_str());
    else
        Params::data::brk_leg = -1;

    int prior_mode = 2;
    if (prior_it != cmd_args.end())
        prior_mode = atoi((prior_it + 1)->c_str());

    bayes_opt::BO_Optimizer<Params, modelfun<GP_t>, initfun<Init_t>, acquifun<Acqui_t>, acquiopt<InnerOpt_t>, statsfun<Stat_t>, stopcrit<Stop_t>> opt;
    global::res_dir = opt.res_dir();
    Eigen::VectorXd result(1);

    opt.optimize(fit_eval_map<Params>(), prior_mode);
    auto val = opt.best_observation();
    result = opt.best_sample().transpose();

    std::cout << "Best performance obtained with descriptor " << val << ": " << result.transpose() << std::endl;

#ifdef ROBOT
    global::hexa.reset();
#else
    if (global::global_robot)
        global::global_robot.reset();
#endif
    return 0;
}
