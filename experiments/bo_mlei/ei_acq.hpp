//| Copyright Inria May 2015
//| This project has received funding from the European Research Council (ERC) under
//| the European Union's Horizon 2020 research and innovation programme (grant
//| agreement No 637972) - see http://www.resibots.eu
//|
//| Contributor(s):
//|   - Jean-Baptiste Mouret (jean-baptiste.mouret@inria.fr)
//|   - Antoine Cully (antoinecully@gmail.com)
//|   - Kontantinos Chatzilygeroudis (konstantinos.chatzilygeroudis@inria.fr)
//|   - Federico Allocati (fede.allocati@gmail.com)
//|   - Vaios Papaspyros (b.papaspyros@gmail.com)
//|   - Roberto Rama (bertoski@gmail.com)
//|
//| This software is a computer library whose purpose is to optimize continuous,
//| black-box functions. It mainly implements Gaussian processes and Bayesian
//| optimization.
//| Main repository: http://github.com/resibots/limbo
//| Documentation: http://www.resibots.eu/limbo
//|
//| This software is governed by the CeCILL-C license under French law and
//| abiding by the rules of distribution of free software.  You can  use,
//| modify and/ or redistribute the software under the terms of the CeCILL-C
//| license as circulated by CEA, CNRS and INRIA at the following URL
//| "http://www.cecill.info".
//|
//| As a counterpart to the access to the source code and  rights to copy,
//| modify and redistribute granted by the license, users are provided only
//| with a limited warranty  and the software's author,  the holder of the
//| economic rights,  and the successive licensors  have only  limited
//| liability.
//|
//| In this respect, the user's attention is drawn to the risks associated
//| with loading,  using,  modifying and/or developing or reproducing the
//| software by the user in light of its specific status of free software,
//| that may mean  that it is complicated to manipulate,  and  that  also
//| therefore means  that it is reserved for developers  and  experienced
//| professionals having in-depth computer knowledge. Users are therefore
//| encouraged to load and test the software's suitability as regards their
//| requirements in conditions enabling the security of their systems and/or
//| data to be ensured and,  more generally, to use and operate it in the
//| same conditions as regards security.
//|
//| The fact that you are presently reading this means that you have had
//| knowledge of the CeCILL-C license and that you accept its terms.
//|
#ifndef LIMBO_ACQUI_EI_ACQ_HPP
#define LIMBO_ACQUI_EI_ACQ_HPP

#include <cmath>
#include <vector>
#include <Eigen/Core>

#include <limbo/tools/macros.hpp>
#include <limbo/opt/optimizer.hpp>

namespace limbo {
    /*namespace defaults {
        struct acqui_ei {
            /// @ingroup acqui_defaults
            BO_PARAM(double, jitter, 0.0);
        };
    }*/
    namespace acqui {
        /** @ingroup acqui
        \rst
        Classic EI (Expected Improvement). See :cite:`brochu2010tutorial`, p. 14

          .. math::
            EI(x) = (\mu(x) - f(x^+) - \xi)\Phi(Z) + \sigma(x)\phi(Z),\\\text{with } Z = \frac{\mu(x)-f(x^+) - \xi}{\sigma(x)}.

        Parameters:
          - ``double jitter`` - :math:`\xi`
        \endrst
        */
        template <typename Params, typename Model>
        class EIAcq {
        public:
            EIAcq(const Model& model, int iteration = 0) : _model(model), _nb_samples(-1) {}

            size_t dim_in() const { return _model.dim_in(); }

            size_t dim_out() const { return _model.dim_out(); }

            template <typename AggregatorFunction>
            opt::eval_t operator()(const Eigen::VectorXd& v, const AggregatorFunction& afun, bool gradient)
            {
                assert(!gradient);

                Eigen::VectorXd mu;
                double sigma_sq;
                std::tie(mu, sigma_sq) = _model.query(v);
                double sigma = std::sqrt(sigma_sq);

                // First find the best so far (predicted) observation -- if needed
                _f_max = 0.0;
                if(_model.nb_samples() > 0) { // if we have no observations yet, we consider their max to be 0
                    std::vector<double> rewards;
                    for (auto s : _model.samples())
                        rewards.push_back(afun(_model.mu(s))); // sigma(s)~0, so mu(s) should be equal to the real observation
                    _f_max = *std::max_element(rewards.begin(), rewards.end());
                }
                _nb_samples = _model.nb_samples();

                // if sigma is too low, we consider that the value of the model on v is exactly mu(v)
                if(sigma < 1e-10) {
                    if(afun(mu) < _f_max - Params::acqui_ei::jitter())
                        return opt::no_grad(0.0);
                    else
                        return opt::no_grad((afun(mu) - _f_max - Params::acqui_ei::jitter()));
                }

                // Calculate Z and \Phi(Z) and \phi(Z)
                double X = afun(mu) - _f_max - Params::acqui_ei::jitter();
                double Z = X / sigma;
                double phi = std::exp(-0.5 * std::pow(Z, 2.0)) / std::sqrt(2.0 * M_PI);
                double Phi = 0.5 * std::erfc(-Z / std::sqrt(2)); //0.5 * (1.0 + std::erf(Z / std::sqrt(2)));

                return opt::no_grad(X * Phi + sigma * phi);
            }

        protected:
            const Model& _model;
            int _nb_samples;
            double _f_max;
        };
    }
}

#endif
