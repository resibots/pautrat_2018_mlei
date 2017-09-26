#include <iostream>
#include <mutex>
#include <boost/random.hpp>

#include <sferes/phen/parameters.hpp>
#include <sferes/gen/evo_float.hpp>
#include <sferes/gen/sampled.hpp>
#include <sferes/stat/pareto_front.hpp>
#include <sferes/modif/dummy.hpp>
#include <sferes/run.hpp>
#include <modules/map_elites/map_elites.hpp>
#include <modules/map_elites/fit_map.hpp>
#include <hexapod_dart/hexapod_dart_simu.hpp>
#include "stat_map.hpp"
#include "stat_map_binary.hpp"

#ifdef GRAPHIC
#define NO_PARALLEL
#endif

#define NO_MPI

#ifndef NO_PARALLEL
#include <sferes/eval/parallel.hpp>
#ifndef NO_MPI
#include <sferes/eval/mpi.hpp>
#endif
#else
#include <sferes/eval/eval.hpp>
#endif

using namespace sferes;
using namespace sferes::gen::evo_float;

struct Params {
    struct surrogate {
        SFERES_CONST int nb_transf_max = 10;
        SFERES_CONST float tau_div = 0.05f;
    };

    struct ea {
        SFERES_CONST size_t behav_dim = 6;
        SFERES_ARRAY(size_t, behav_shape, 5, 5, 5, 5, 5, 5);
        SFERES_CONST float epsilon = 0.01;
    };

    struct sampled {
        SFERES_ARRAY(float, values, 0.00, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35,
            0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85,
            0.90, 0.95, 1);
        SFERES_CONST float mutation_rate = 0.05f;
        SFERES_CONST float cross_rate = 0.00f;
        SFERES_CONST bool ordered = false;
    };
    struct evo_float {
        SFERES_CONST float cross_rate = 0.0f;
        SFERES_CONST float mutation_rate = 1.0f / 54.0f;
        SFERES_CONST float eta_m = 10.0f;
        SFERES_CONST float eta_c = 10.0f;
        SFERES_CONST mutation_t mutation_type = polynomial;
        SFERES_CONST cross_over_t cross_over_type = sbx;
    };
    struct pop {
        SFERES_CONST unsigned size = 200;
        SFERES_CONST unsigned init_size = 200;
#ifndef TURNING
        SFERES_CONST unsigned nb_gen = 100001;
#else
        SFERES_CONST unsigned nb_gen = 50001;
#endif
        SFERES_CONST int dump_period = 50;
        SFERES_CONST int initial_aleat = 1;
    };
    struct parameters {
        SFERES_CONST float min = 0.0f;
        SFERES_CONST float max = 1.0f;
    };
};

namespace global {
    std::shared_ptr<hexapod_dart::Hexapod> global_robot;
    std::vector<hexapod_dart::HexapodDamage> damages;
};

void init_simu(std::string robot_file, std::vector<hexapod_dart::HexapodDamage> damages = std::vector<hexapod_dart::HexapodDamage>())
{
    global::global_robot = std::make_shared<hexapod_dart::Hexapod>(robot_file, damages);
}

FIT_MAP(FitAdapt)
{
public:
    template <typename Indiv>
    void eval(Indiv & indiv)
    {

        this->_objs.resize(2);
        std::fill(this->_objs.begin(), this->_objs.end(), 0);
        this->_dead = false;
        _eval(indiv);
    }

    template <class Archive>
    void serialize(Archive & ar, const unsigned int version)
    {
        dbg::trace trace("fit", DBG_HERE);

        ar& boost::serialization::make_nvp("_value", this->_value);
        ar& boost::serialization::make_nvp("_objs", this->_objs);
    }

    bool dead() { return _dead; }
    std::vector<double> ctrl() { return _ctrl; }

protected:
    bool _dead;
    std::vector<double> _ctrl;

    template <typename Indiv>
    void _eval(Indiv & indiv)
    {
        // copy of controler's parameters
        _ctrl.clear();
        for (size_t i = 0; i < 54; i++)
            _ctrl.push_back(indiv.data(i));
        // launching the simulation
        auto robot = global::global_robot->clone();
        using safe_t = boost::fusion::vector<hexapod_dart::safety_measures::TurnOver>;
        hexapod_dart::HexapodDARTSimu<hexapod_dart::safety<safe_t>> simu(_ctrl, robot);
        simu.run(10);

#ifndef TURNING
        this->_value = simu.covered_distance();
#else
        this->_value = simu.arrival_angle();
#endif

        std::vector<float> desc;

        if (this->_value < -1000) {
            this->_dead = true;
            // mort subite
            desc.resize(6);
            desc[0] = 0;
            desc[1] = 0;
            desc[2] = 0;
            desc[3] = 0;
            desc[4] = 0;
            desc[5] = 0;
            this->_value = -1000;
        }
        else {
            desc.resize(6);
            std::vector<double> v;
            simu.get_descriptor<hexapod_dart::descriptors::BodyOrientation>(v);
            desc[0] = v[0];
            desc[1] = v[1];
            desc[2] = v[2];
            desc[3] = v[3];
            desc[4] = v[4];
            desc[5] = v[5];
        }

#ifdef BACKWD
        this->_value = -this->_value;
#endif

        this->set_desc(desc);
    }
};

int main(int argc, char** argv)
{
#ifndef NO_PARALLEL
#ifndef NO_MPI
    typedef eval::Mpi<Params> eval_t;
#else
    typedef eval::Parallel<Params> eval_t;
#endif
#else
    typedef eval::Eval<Params> eval_t;
#endif

    typedef gen::Sampled<54, Params> gen_t;
    typedef FitAdapt<Params> fit_t;
    typedef phen::Parameters<gen_t, fit_t, Params> phen_t;

    typedef boost::fusion::vector<sferes::stat::MapBinary<phen_t, Params>> stat_t;
    typedef modif::Dummy<> modifier_t;
    typedef ea::MapElites<phen_t, eval_t, stat_t, modifier_t, Params> ea_t;

    ea_t ea;

    // Choose the damage that you want in the prior
    /*hexapod_dart::HexapodDamage dmg;
    dmg.type = "leg_removal";
    dmg.data = "0";
    global::damages.push_back(dmg);*/

    std::cout << "init SIMU" << std::endl;
    // initialization of the simulation and the simulated robot
    const char* env_p = std::getenv("RESIBOTS_DIR");

    if (env_p) //if the environment variable exists
        init_simu(std::string(env_p) + "/share/hexapod_models/URDF/pexod.urdf", global::damages);
    else {
        cout << "You have to set the RESIBOTS_DIR environment variable at the root of your installation." << endl;
        return 0;
    }   

    std::cout << "debut run" << std::endl;

    run_ea(argc, argv, ea);
    std::cout << "fin run" << std::endl;

    std::cout << "fin" << std::endl;
    return 0;
}
