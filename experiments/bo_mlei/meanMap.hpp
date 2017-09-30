#ifndef MEAN_ARCHIVE_HPP_
#define MEAN_ARCHIVE_HPP_

namespace limbo {
    namespace mean_functions {
        template <typename Params>
        struct MeanArchive_Map {
            int map_id;
            MeanArchive_Map(size_t dim_out = 1, int id = 0)
            {
                map_id = id;
            }
            template <typename GP>
            Eigen::VectorXd operator()(const Eigen::VectorXd& v, const GP&) const
            {
                std::vector<double> key(v.size(), 0);
                for (int i = 0; i < v.size(); i++)
                    key[i] = v[i];
                Eigen::VectorXd r(1);
                //std::cout << "GP " << map_id << " key: " << key[0] << key[1] << key[2] << key[3] << key[4] << key[5] << std::endl;
                r(0) = Params::archiveparams::archives[map_id].at(key).fit;
                return r;
            }
        };
    }
}

#endif
