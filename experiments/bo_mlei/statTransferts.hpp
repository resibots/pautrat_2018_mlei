#ifndef STAT_TRANSFERTS_HPP_
#define STAT_TRANSFERTS_HPP_

namespace limbo {
    namespace stat {

        template <typename Params>
        struct StatTransferts : public StatBase<Params> {
            std::ofstream _ofs;
            StatTransferts()
            {
            }

            template <typename BO, typename AggregatorFunction>
            void operator()(const BO& bo, const AggregatorFunction& afun)//, bool blacklisted)
            {
                if (!bo.stats_enabled())// || blacklisted)
                    return;

                this->_create_log_file(bo, "transferts.dat");

                if (bo.total_iterations() == 0)
                    (*this->_log_file) << "#iteration : sample : archive_performance : aggregated_observation \\n controller" << std::endl;

                (*this->_log_file) << bo.total_iterations() << " : ";
                std::vector<double> sample;
                for (int i = 0; i < bo.samples()[0].size(); i++) {
                    (*this->_log_file) << bo.samples()[bo.samples().size() - 1][i] << " ";
                    sample.push_back(bo.samples()[bo.samples().size() - 1][i]);
                }

                (*this->_log_file) << " : " << Params::archiveparams::archive[sample].fit << " : ";

                (*this->_log_file) << afun(bo.observations()[bo.observations().size() - 1]) << std::endl;

                for (size_t i = 0; i < Params::archiveparams::archive[sample].controller.size(); i++)
                    (*this->_log_file) << Params::archiveparams::archive[sample].controller[i] << " ";
                (*this->_log_file) << std::endl;
            }
        };
    }
}
#endif
