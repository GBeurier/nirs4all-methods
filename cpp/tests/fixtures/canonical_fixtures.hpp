// SPDX-License-Identifier: CECILL-2.1
// Generated mechanically from parity/fixtures/*canonical*.json.
#pragma once

#include <cstddef>
#include <cstdint>

#include "phase1_fixtures.hpp"

namespace pls4all::test::fixtures {

struct CanonicalFixture {
    const char* id;
    std::int32_t n_components;
    MatrixRef X;
    MatrixRef Y;
    MatrixRef coefficients;
    MatrixRef intercept;
    MatrixRef x_mean;
    MatrixRef x_scale;
    MatrixRef y_mean;
    MatrixRef y_scale;
    MatrixRef weights_w;
    MatrixRef loadings_p;
    MatrixRef y_loadings_q;
    MatrixRef rotations_r;
    MatrixRef scores_t;
    MatrixRef predict_train;
};

inline const double synthetic_canonical_tiny_pls1_v1_x[] = {
    -1.1575496471201177, 0.28975580232775139, 0.78085406922509848, 0.54397364470857956,
    -0.96138264124543649, 1.0710086656015809, 0.70145566011045068, 0.70497345509881937,
    0.74506260266182944, 1.1043472383723143, 2.242972395734681, -0.61149312272307454,
    0.047211183090740669, 1.7542346824437531, -1.3379798662995022, 0.32557446888715391,
    -0.68911771598769356, -0.019821809991921974, 0.47475324599756469, -1.9311014161570534,
    -0.99247827806663802, -1.4054710825938028, -0.23109550849022736, -0.68884708495962477,
    1.5151057798585295, -0.60317154655094207, 1.7136844875562847, -0.40624919079488403,
    0.27140952079766212, 0.039840200086869802, 0.011518318252040626, -1.127178008332383,
    0.33471297537835931, 0.38389154856474461, 0.23783551398709574, 0.62141167436006728,
};

inline const double synthetic_canonical_tiny_pls1_v1_y[] = {
    -0.20253470175040716, -0.30336843027862981, 0.30568862874200531, -0.76633412374308429,
    -0.85080268801212666, -0.31684559955135949, 1.0975242920357919, -0.24462036877819271,
    0.31433682957961956,
};

inline const double synthetic_canonical_tiny_pls1_v1_coefficients[] = {
    0.45860342172018825, -0.15675410522208585, 0.37066495298599872, 0.12290531021749562,
};

inline const double synthetic_canonical_tiny_pls1_v1_intercept[] = {
    -0.10743957352848704,
};

inline const double synthetic_canonical_tiny_pls1_v1_x_mean[] = {
    -0.098558468959196094, 0.29051263314003861, 0.51044425734149845, -0.2854372866569333,
};

inline const double synthetic_canonical_tiny_pls1_v1_x_scale[] = {
    0.91345763358563647, 0.95242319677031895, 1.0517911542572422, 0.90587882698207822,
};

inline const double synthetic_canonical_tiny_pls1_v1_y_mean[] = {
    -0.10743957352848704,
};

inline const double synthetic_canonical_tiny_pls1_v1_y_scale[] = {
    0.60180349429464719,
};

inline const double synthetic_canonical_tiny_pls1_v1_weights_w[] = {
    0.69609897637732099, -0.24808138772519903, 0.64782295623057784, 0.18500610133576567,
};

inline const double synthetic_canonical_tiny_pls1_v1_loadings_p[] = {
    0.71993501990980191, -0.15148597083147577, 0.72825486231383607, -0.05679325168061912,
};

inline const double synthetic_canonical_tiny_pls1_v1_y_loadings_q[] = {
    1.0000000000000002,
};

inline const double synthetic_canonical_tiny_pls1_v1_rotations_r[] = {
    0.69609897637732088, -0.24808138772519897, 0.64782295623057773, 0.18500610133576564,
};

inline const double synthetic_canonical_tiny_pls1_v1_scores_t[] = {
    -0.47086448682951798, -0.54089433835442169, 1.4314125294882929, -1.283879933849275,
    -0.7272748746130937, -0.77857022542324228, 2.1789025507374618, -0.1319803657718443,
    0.3231491446156402,
};

inline const double synthetic_canonical_tiny_pls1_v1_predict_train[] = {
    -0.39080746704174685, -0.43295167639436921, 0.75398948849470737, -0.88008300397376105,
    -0.54511613438334827, -0.57598585574196548, 1.2038315952328376, -0.18686581882826855,
    0.087032710879531489,
};

inline const double synthetic_canonical_small_pls2_v1_x[] = {
    -0.79015249996301462, -2.0346254818318728, 0.60330174692476468, 0.74429452987991185,
    -0.30968679986627135, 0.36732137294554024, 1.7103942914776449, 1.0607978400658138,
    0.70763902080607544, 0.68774938850544498, -0.86356745375235988, 0.96401673167356172,
    -1.6484628156748702, -0.33209181276866229, -0.4372938409341125, -1.7285187058460105,
    -0.11033888700145329, 1.6434550766344018, -0.34012733762511482, -1.2076033402593367,
    -0.091053718506435913, 0.85392431650098788, 0.19321811565038277, -0.0026657164631725457,
    -1.4092222584076113, 0.50101574245542613, 0.48573108678369781, 1.3412662057172517,
    1.489053273620953, 0.68858298010104879, -1.6424168811861135, -1.4353935022152693,
    -1.0979784550410603, 0.29492495778261718, -0.78724610205555678, 1.3095376610763625,
    0.24385526074207703, 0.36476739570056421, 0.75050810471515816, 0.34024790142790512,
    -1.4105670085192841, 2.4519291799137406, 1.4629889056618062, 1.5231286713207461,
    -0.74748263300550599, -1.8474977757527729, -1.5066347122204127, -1.0534632139707452,
    0.88213946616834249, 0.90031941080800182, 1.458392914492266, 2.1424499559696035,
    0.52090760864139496, 0.35456182486107002, -0.36300233224128459, -0.33646269482473018,
    -1.1577837086639562, 0.51080012678065279, 0.31817103995393936, -0.40519828505149047,
    -0.81058551805147638, 0.45695142835272667, 1.3061600648002516, 0.08149811962435799,
    0.28376904399171499, -0.5481664388036902, -0.24852584993024826, -0.316502228872869,
    -0.28507864375164177, -0.59724477894249017, 0.39638337815465297, 1.299729294584818,
    -0.94717551737762373, 1.3992067337629219, -0.40300244276432512, -0.3670887832022261,
    -2.1564658910317949, -2.4214757028147136, 0.73550304189520133, -1.0910834111861687,
    0.8053083193545435, 0.050625874930255298, 0.25047314441891266, 1.3333047650292276,
};

inline const double synthetic_canonical_small_pls2_v1_y[] = {
    0.24731295349642007, -0.85851925557840758, 0.64505960545944152, -0.020346030994014927,
    0.39124385191316363, 1.0513216744654519, -1.117065606636551, 1.1982004076153687,
    -0.20792553633368682, 0.28636607689616028, -0.80074839699938016, 0.16417983078082887,
    0.03838195260437241, 0.55057302897560079, -0.1898788827516972, -0.99947229666107451,
    -0.31584914569215505, 0.12738617247783995, -0.93257971983746435, 0.90598025524675385,
    1.487438882317716, -0.62537627676763441, 0.31865907192456411, -0.35707561915533709,
    0.65848085120132915, 0.0055258342222494258, 0.83245820746284738, 0.067491273967309973,
    -0.65283353715582115, -0.52016727539474372, 0.024804295526147067, 0.46357391930813829,
    0.10043358090191506, -0.28019545572770732, 0.49404450334867056, 0.010567537533471685,
    -1.2393975580905361, -0.23415157009212714, -0.54317344432148529, 0.50724270354617929,
    0.083126327613403245, 0.89230021309326524,
};

inline const double synthetic_canonical_small_pls2_v1_coefficients[] = {
    0.136130603453535, 0.030221402261722589, 0.19533876763186672, -0.16800583737745386,
    0.20395362596288308, 0.030586390306270968, 0.23000441451112313, 0.10543705728074504,
    0.39127178712823196, 0.31531127219077121, -0.23725035645030651, 0.10646859992508603,
    0.23220013331655667, -0.20317916908651146, 0.0463525539536329, -0.024987891255001393,
    0.24805274788645953, 0.24971347056203894,
};

inline const double synthetic_canonical_small_pls2_v1_intercept[] = {
    -0.24173948839121889, 0.11063037818928727, 0.24949463903827335,
};

inline const double synthetic_canonical_small_pls2_v1_x_mean[] = {
    -0.22605643175087753, -0.039112517820907709, 0.13552627251497995, 0.17910223809824921,
    -0.26375223214394156, 0.42724782355113994,
};

inline const double synthetic_canonical_small_pls2_v1_x_scale[] = {
    1.0960624676630049, 1.107926583710011, 0.85197566643088651, 1.0763725440871212,
    0.97417566200993533, 1.2539580583387901,
};

inline const double synthetic_canonical_small_pls2_v1_y_mean[] = {
    -0.24173948839121889, 0.11063037818928727, 0.24949463903827335,
};

inline const double synthetic_canonical_small_pls2_v1_y_scale[] = {
    0.62911316205544354, 0.62436658204725948, 0.63107698588848893,
};

inline const double synthetic_canonical_small_pls2_v1_weights_w[] = {
    0.40578628323220262, 0.1226792464968687, -0.17721085241946641, 0.42137012631432957,
    0.58995370116637014, 0.26067514947179488, 0.5159285894497958, -0.431670800440629,
    0.30973186529700114, -0.34729809067439688, 0.30622262949273732, 0.6576395987450333,
};

inline const double synthetic_canonical_small_pls2_v1_loadings_p[] = {
    0.25499641931870759, 0.50305832967525432, -0.11599620338825496, 0.47143224215006824,
    0.59950153741421264, 0.22138325396180966, 0.56161051922779603, -0.47396471006384583,
    0.4124322334872757, -0.48558867461235006, 0.34222711109383308, 0.46938758280403436,
};

inline const double synthetic_canonical_small_pls2_v1_y_loadings_q[] = {
    0.71575792713273789, -0.40488610344752352, -0.13215698942116791, 0.81078927033372672,
    0.69826157292989732, 0.42501521417107807,
};

inline const double synthetic_canonical_small_pls2_v1_rotations_r[] = {
    0.40578628323220262, 0.13157615254967039, -0.17721085241946638, 0.41748476012364771,
    0.58995370116637014, 0.27360994513066234, 0.5159285894497958, -0.42035901298017747,
    0.30973186529700114, -0.34050718762770726, 0.30622262949273737, 0.66435356143528967,
};

inline const double synthetic_canonical_small_pls2_v1_scores_t[] = {
    0.67592083196970087, -0.90585326238725383, 1.1213296607775325, 1.1260518970490629,
    -1.4449793930997539, 0.8706077317128621, 0.35153041879752572, -1.1778028140010528,
    0.89623521135873185, -0.75410705723369131, -1.050645071625177, -0.48711668632689631,
    0.74227456699369565, 1.8167032063169386, -1.9641540465487075, 0.94925711513968469,
    2.3488431924104707, -0.16766516209314852, -0.75797628478985091, -1.317804441531532,
    0.40395537227505296, -0.17733731787148754, -0.20437061720102434, 0.29239624001374792,
    -2.4291841836557451, -0.35192733121806702, 1.3112203423375486, 0.28459788243083339,
};

inline const double synthetic_canonical_small_pls2_v1_predict_train[] = {
    0.29336094750397007, -0.40371267730830745, 0.30437822122291236, -0.023639903328836614,
    0.58814546802380363, 1.0456428113394751, -1.1141636929048409, 0.67058951008789836,
    -0.15373329626973681, 0.21656106778252515, -0.51461474133391882, 0.088491796952140767,
    0.3539142430141467, -0.34507334807134538, 0.44216221309302073, -0.59075936754847402,
    -0.049269212335545584, -0.34413239769993048, -0.37024779035327932, 0.96905119729511191,
    1.0638542467953482, -1.3679777726125828, 0.75324355493211503, -0.36141601540446294,
    0.8586350637175173, -0.16806011577592578, 1.2395576841054583, -0.24738063852406939,
    -0.49393743823242292, -0.43797074777601169, -0.014670104252029692, -0.012475181866193119,
    0.37993538409369354, -0.40824502167493493, 0.2755135669481773, 0.23786307101559789,
    -1.2459408328300099, 0.13291699621763256, -0.91533491705955061, 0.27620096453383364,
    0.14650771606894211, 0.90362689212787273,
};

inline const CanonicalFixture kCanonicalFixtures[] = {
    {
        "synthetic_canonical_tiny_pls1_v1",
        1,
        MatrixRef{9, 4, synthetic_canonical_tiny_pls1_v1_x, sizeof(synthetic_canonical_tiny_pls1_v1_x) / sizeof(double), false},
        MatrixRef{9, 1, synthetic_canonical_tiny_pls1_v1_y, sizeof(synthetic_canonical_tiny_pls1_v1_y) / sizeof(double), false},
        MatrixRef{4, 1, synthetic_canonical_tiny_pls1_v1_coefficients, sizeof(synthetic_canonical_tiny_pls1_v1_coefficients) / sizeof(double), false},
        MatrixRef{1, 1, synthetic_canonical_tiny_pls1_v1_intercept, sizeof(synthetic_canonical_tiny_pls1_v1_intercept) / sizeof(double), false},
        MatrixRef{1, 4, synthetic_canonical_tiny_pls1_v1_x_mean, sizeof(synthetic_canonical_tiny_pls1_v1_x_mean) / sizeof(double), false},
        MatrixRef{1, 4, synthetic_canonical_tiny_pls1_v1_x_scale, sizeof(synthetic_canonical_tiny_pls1_v1_x_scale) / sizeof(double), false},
        MatrixRef{1, 1, synthetic_canonical_tiny_pls1_v1_y_mean, sizeof(synthetic_canonical_tiny_pls1_v1_y_mean) / sizeof(double), false},
        MatrixRef{1, 1, synthetic_canonical_tiny_pls1_v1_y_scale, sizeof(synthetic_canonical_tiny_pls1_v1_y_scale) / sizeof(double), false},
        MatrixRef{4, 1, synthetic_canonical_tiny_pls1_v1_weights_w, sizeof(synthetic_canonical_tiny_pls1_v1_weights_w) / sizeof(double), true},
        MatrixRef{4, 1, synthetic_canonical_tiny_pls1_v1_loadings_p, sizeof(synthetic_canonical_tiny_pls1_v1_loadings_p) / sizeof(double), true},
        MatrixRef{1, 1, synthetic_canonical_tiny_pls1_v1_y_loadings_q, sizeof(synthetic_canonical_tiny_pls1_v1_y_loadings_q) / sizeof(double), true},
        MatrixRef{4, 1, synthetic_canonical_tiny_pls1_v1_rotations_r, sizeof(synthetic_canonical_tiny_pls1_v1_rotations_r) / sizeof(double), true},
        MatrixRef{9, 1, synthetic_canonical_tiny_pls1_v1_scores_t, sizeof(synthetic_canonical_tiny_pls1_v1_scores_t) / sizeof(double), true},
        MatrixRef{9, 1, synthetic_canonical_tiny_pls1_v1_predict_train, sizeof(synthetic_canonical_tiny_pls1_v1_predict_train) / sizeof(double), false}
    },
    {
        "synthetic_canonical_small_pls2_v1",
        2,
        MatrixRef{14, 6, synthetic_canonical_small_pls2_v1_x, sizeof(synthetic_canonical_small_pls2_v1_x) / sizeof(double), false},
        MatrixRef{14, 3, synthetic_canonical_small_pls2_v1_y, sizeof(synthetic_canonical_small_pls2_v1_y) / sizeof(double), false},
        MatrixRef{6, 3, synthetic_canonical_small_pls2_v1_coefficients, sizeof(synthetic_canonical_small_pls2_v1_coefficients) / sizeof(double), false},
        MatrixRef{1, 3, synthetic_canonical_small_pls2_v1_intercept, sizeof(synthetic_canonical_small_pls2_v1_intercept) / sizeof(double), false},
        MatrixRef{1, 6, synthetic_canonical_small_pls2_v1_x_mean, sizeof(synthetic_canonical_small_pls2_v1_x_mean) / sizeof(double), false},
        MatrixRef{1, 6, synthetic_canonical_small_pls2_v1_x_scale, sizeof(synthetic_canonical_small_pls2_v1_x_scale) / sizeof(double), false},
        MatrixRef{1, 3, synthetic_canonical_small_pls2_v1_y_mean, sizeof(synthetic_canonical_small_pls2_v1_y_mean) / sizeof(double), false},
        MatrixRef{1, 3, synthetic_canonical_small_pls2_v1_y_scale, sizeof(synthetic_canonical_small_pls2_v1_y_scale) / sizeof(double), false},
        MatrixRef{6, 2, synthetic_canonical_small_pls2_v1_weights_w, sizeof(synthetic_canonical_small_pls2_v1_weights_w) / sizeof(double), true},
        MatrixRef{6, 2, synthetic_canonical_small_pls2_v1_loadings_p, sizeof(synthetic_canonical_small_pls2_v1_loadings_p) / sizeof(double), true},
        MatrixRef{3, 2, synthetic_canonical_small_pls2_v1_y_loadings_q, sizeof(synthetic_canonical_small_pls2_v1_y_loadings_q) / sizeof(double), true},
        MatrixRef{6, 2, synthetic_canonical_small_pls2_v1_rotations_r, sizeof(synthetic_canonical_small_pls2_v1_rotations_r) / sizeof(double), true},
        MatrixRef{14, 2, synthetic_canonical_small_pls2_v1_scores_t, sizeof(synthetic_canonical_small_pls2_v1_scores_t) / sizeof(double), true},
        MatrixRef{14, 3, synthetic_canonical_small_pls2_v1_predict_train, sizeof(synthetic_canonical_small_pls2_v1_predict_train) / sizeof(double), false}
    }
};

}  // namespace pls4all::test::fixtures
