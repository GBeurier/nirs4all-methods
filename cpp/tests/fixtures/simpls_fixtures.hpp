// SPDX-License-Identifier: CECILL-2.1
// Generated mechanically from parity/fixtures/*simpls*.json.
#pragma once

#include <cstddef>
#include <cstdint>

#include "phase1_fixtures.hpp"

namespace n4m::test::fixtures {

struct SimplsFixture {
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

inline const double synthetic_simpls_tiny_pls1_v1_x[] = {
    -0.35966845567139044, 1.2036751014877107, 1.3968681196180939, 0.31723592758455404,
    0.41413419203577684, -0.48954980350522342, -0.91412070483088947, -0.90036062903098812,
    -0.99843126798853443, 0.92920482409395733, -0.056291916680931781, 0.1282790949733999,
    -0.63997043922044561, -1.0878172085644096, -1.2019541755133116, -0.84156918447725637,
    0.59915671471951848, 0.01844008196322814, -0.45677163924420827, -0.23927840186540572,
    -1.4273353500870181, 1.2306824377006564, -1.2164154219415413, 0.04238178066477212,
    2.1369751648960755, -2.5511902614867554, -1.4069595793191529, -0.72370598399872466,
    0.1165516397529902, -1.7148763387690875, -0.23527948055323214, -0.028427676959011872,
};

inline const double synthetic_simpls_tiny_pls1_v1_y[] = {
    -0.020547373552374775, 0.065017967442902835, -0.94715470099958232, -0.52474371039941392,
    0.26911683948076065, -1.8193031916954523, 1.6248639466005865, 0.33964696114015758,
};

inline const double synthetic_simpls_tiny_pls1_v1_coefficients[] = {
    0.65444603334812546, -0.29524133172184341, 0.39353606839712846, -0.021333135041413907,
};

inline const double synthetic_simpls_tiny_pls1_v1_intercept[] = {
    -0.12663790774780195,
};

inline const double synthetic_simpls_tiny_pls1_v1_x_mean[] = {
    -0.019823475195378387, -0.30767889588499042, -0.51136559980814666, -0.28068063413858257,
};

inline const double synthetic_simpls_tiny_pls1_v1_x_scale[] = {
    1.1135961076077572, 1.4112923220175753, 0.91542028709289192, 0.47649584326014111,
};

inline const double synthetic_simpls_tiny_pls1_v1_y_mean[] = {
    -0.12663790774780195,
};

inline const double synthetic_simpls_tiny_pls1_v1_y_scale[] = {
    1.0132438094858149,
};

inline const double synthetic_simpls_tiny_pls1_v1_weights_w[] = {
    0.19041797392201154, 0.21889412521528029, -0.15171626182599129, -0.047368679246263215,
    -0.0060651601178723837, 0.29007519577153751, -0.079784144243515376, 0.13695371543540052,
};

inline const double synthetic_simpls_tiny_pls1_v1_loadings_p[] = {
    2.4274939249273833, 0.81290442705064525, -2.4598024779854666, 0.17947485658084483,
    -1.3065302047608109, 2.186888082188537, -1.9633622057567082, 1.4325970951297795,
};

inline const double synthetic_simpls_tiny_pls1_v1_y_loadings_q[] = {
    2.3127102717444492, 1.2740459421153016,
};

inline const double synthetic_simpls_tiny_pls1_v1_rotations_r[] = {
    0.1904179739220116, 0.21889412521528032, -0.15171626182599132, -0.047368679246263201,
    -0.0060651601178723898, 0.29007519577153756, -0.07978414424351539, 0.13695371543540058,
};

inline const double synthetic_simpls_tiny_pls1_v1_scores_t[] = {
    -0.33334226513236054, 0.65899792218657571, 0.20018274780516659, -0.21432587081066767,
    -0.37179391877655166, 0.027869797738602053, 0.076315235469606668, -0.47575569471027673,
    0.06348933653323538, 0.13992333936764556, -0.45547413725163249, -0.45886095887321371,
    0.690093940994296, 0.088126266245321716, 0.13052906035824, 0.23402519885601281,
};

inline const double synthetic_simpls_tiny_pls1_v1_predict_train[] = {
    -0.057058913289952634, 0.065780831635870979, -0.96189957765526057, -0.56196755585961178,
    0.20276887412985781, -1.786320789106512, 1.6042502763424742, 0.48134359182071806,
};

inline const double synthetic_simpls_small_pls2_v1_x[] = {
    0.35877340800391416, 1.5106773081434572, -1.7863312373111822, 1.6866135086657521,
    -0.047317212156137566, -0.79997858437515323, -0.80295671839036198, -1.0828165165825909,
    -0.22364535840745078, 0.83388417955215743, 0.58406373164297032, 0.63828602425618286,
    -1.6948083917690129, -1.5709621176914044, 1.5538031742896521, 0.96888536545666004,
    2.1832140014726402, 1.2098158348986146, -1.0243913350545566, 1.2852724548122736,
    0.62839426427033118, 0.21482492767891934, -0.81987332257188328, 0.0027802218072239555,
    -0.14699870178042784, 0.89253511161482213, 0.12449694399871385, 1.5847459143396285,
    -0.7744857532181485, 0.47110255941042922, 0.49884237745245574, -0.86433853981418096,
    -1.1812660464710163, 1.0135795003256129, 0.51744047138143934, -0.16250518501238939,
    0.3943673986045515, -0.67534402878431565, 0.2383482020485698, -0.60512669466153035,
    0.18878846665371848, 0.98608946288340649, -0.33963024501217354, 0.84977380994192264,
    -1.0341993407801506, -0.28849076965017367, -1.3647425964001083, -0.10058608537896181,
    0.078654083325231425, -0.19726266289721334, -0.26767571670742146, 1.3401946901039739,
    1.4237928368424078, -1.2156310049465249, 1.1741144932558947, -1.7230696128709919,
    -0.73447925083704557, -0.1734455959217569, 0.11106060267130161, -1.0474166760042456,
};

inline const double synthetic_simpls_small_pls2_v1_y[] = {
    -0.66537984070031331, -0.30743079148811076, -0.12427166705750206, -0.17147398564492849,
    0.14391346346930278, 1.0287055617480449, 0.29424091764037985, 1.7988327035803953,
    0.030289218802810363, -0.14519050325630539, 0.80179355987434664, 0.59054215833191526,
    0.54753313367746992, -0.89546455672962733, -0.52946816199454949, -0.14618500640053755,
    -0.56520724571983927, -0.16770110419632167, 0.24710937935268426, -0.69719719392748825,
    -0.018671694653079947, 1.5893850267035421, -0.8969309557883689, -0.45345419085703464,
};

inline const double synthetic_simpls_small_pls2_v1_coefficients[] = {
    0.46089969686624982, -0.020815487723529372, -0.3275579750834241, 0.47020085740279921,
    0.082515535728788933, 0.18116182687401619, -0.071389686281369436, -0.33327395909494495,
    0.27996842635919333, 0.20923913274090111,
};

inline const double synthetic_simpls_small_pls2_v1_intercept[] = {
    -0.061254157758054928, 0.16861400982196195,
};

inline const double synthetic_simpls_small_pls2_v1_x_mean[] = {
    0.07321357137322855, 0.30213613764167907, -0.30297890697685348, -0.069025470044372553,
    0.22977467036851174,
};

inline const double synthetic_simpls_small_pls2_v1_x_scale[] = {
    0.77521069173179902, 1.0929532658038983, 1.1605552240596499, 0.96564590448524501,
    0.88378581888076679,
};

inline const double synthetic_simpls_small_pls2_v1_y_mean[] = {
    -0.061254157758054928, 0.16861400982196195,
};

inline const double synthetic_simpls_small_pls2_v1_y_scale[] = {
    0.51607902708760456, 0.87986585967270636,
};

inline const double synthetic_simpls_small_pls2_v1_weights_w[] = {
    0.076955020886279688, -0.21363740125339079, 0.063018745481530289, 0.32753669303502886,
    0.081034021359544281, -0.0083419742446084431, -0.10632667140612963, -0.032859377826538839,
    0.10849926077735488, -0.10324128796533162,
};

inline const double synthetic_simpls_small_pls2_v1_loadings_p[] = {
    1.9474034781375487, -1.0167917915893538, 1.9031382113485094, 2.1377029602933826,
    1.9305966206871665, 0.56420621032434404, -2.6467351152360457, 0.074087669296491951,
    2.6944109786560357, -0.8692273787440028,
};

inline const double synthetic_simpls_small_pls2_v1_y_loadings_q[] = {
    2.0316381026571415, -2.5088311667164085, 3.0715587846516055, 1.1922606704006182,
};

inline const double synthetic_simpls_small_pls2_v1_rotations_r[] = {
    0.076955020886279729, -0.21363740125339079, 0.063018745481530219, 0.32753669303502886,
    0.081034021359544281, -0.0083419742446084518, -0.10632667140612963, -0.032859377826538826,
    0.1084992607773549, -0.10324128796533164,
};

inline const double synthetic_simpls_small_pls2_v1_scores_t[] = {
    -0.23287183646291448, 0.26676947081033348, -0.11366182458285425, -0.15023838701330716,
    0.30083532650791911, -0.13360205034201994, 0.53777759399725777, 0.21522292120562192,
    -0.040174594958991762, -0.13388008994770215, 0.31020368312430829, -0.29678308344486082,
    -0.17022016000000595, -0.53099759759738929, -0.18042422937762742, 0.18234859500751133,
    -0.20799190539718604, 0.28976851895673722, -0.18657211071915566, -0.35644499973978105,
    0.38876807068397234, 0.32132044135775301, -0.4056680128147222, 0.32651626074710338,
};

inline const double synthetic_simpls_small_pls2_v1_predict_train[] = {
    -0.65081812066518663, -0.18088699930156499, 0.014094956372336861, -0.2961683518852915,
    0.42714935727715742, 0.84148695607803703, 0.22393703627281664, 1.8477652533692821,
    0.069965284914568826, -0.080404390328158326, 0.64825142891128196, 0.69562378823218396,
    0.4477842918614518, -0.84844825944278535, -0.48652318506939446, -0.12770419520805587,
    -0.65451000183282293, -0.089520820949998803, 0.20463662886695266, -0.70952956522336674,
    -0.0696676456521301, 1.5563575434105645, -0.90934992435369111, -0.58520284088730323,
};

inline const SimplsFixture kSimplsFixtures[] = {
    {
        "synthetic_simpls_tiny_pls1_v1",
        2,
        MatrixRef{8, 4, synthetic_simpls_tiny_pls1_v1_x, sizeof(synthetic_simpls_tiny_pls1_v1_x) / sizeof(double), false},
        MatrixRef{8, 1, synthetic_simpls_tiny_pls1_v1_y, sizeof(synthetic_simpls_tiny_pls1_v1_y) / sizeof(double), false},
        MatrixRef{4, 1, synthetic_simpls_tiny_pls1_v1_coefficients, sizeof(synthetic_simpls_tiny_pls1_v1_coefficients) / sizeof(double), false},
        MatrixRef{1, 1, synthetic_simpls_tiny_pls1_v1_intercept, sizeof(synthetic_simpls_tiny_pls1_v1_intercept) / sizeof(double), false},
        MatrixRef{1, 4, synthetic_simpls_tiny_pls1_v1_x_mean, sizeof(synthetic_simpls_tiny_pls1_v1_x_mean) / sizeof(double), false},
        MatrixRef{1, 4, synthetic_simpls_tiny_pls1_v1_x_scale, sizeof(synthetic_simpls_tiny_pls1_v1_x_scale) / sizeof(double), false},
        MatrixRef{1, 1, synthetic_simpls_tiny_pls1_v1_y_mean, sizeof(synthetic_simpls_tiny_pls1_v1_y_mean) / sizeof(double), false},
        MatrixRef{1, 1, synthetic_simpls_tiny_pls1_v1_y_scale, sizeof(synthetic_simpls_tiny_pls1_v1_y_scale) / sizeof(double), false},
        MatrixRef{4, 2, synthetic_simpls_tiny_pls1_v1_weights_w, sizeof(synthetic_simpls_tiny_pls1_v1_weights_w) / sizeof(double), true},
        MatrixRef{4, 2, synthetic_simpls_tiny_pls1_v1_loadings_p, sizeof(synthetic_simpls_tiny_pls1_v1_loadings_p) / sizeof(double), true},
        MatrixRef{1, 2, synthetic_simpls_tiny_pls1_v1_y_loadings_q, sizeof(synthetic_simpls_tiny_pls1_v1_y_loadings_q) / sizeof(double), true},
        MatrixRef{4, 2, synthetic_simpls_tiny_pls1_v1_rotations_r, sizeof(synthetic_simpls_tiny_pls1_v1_rotations_r) / sizeof(double), true},
        MatrixRef{8, 2, synthetic_simpls_tiny_pls1_v1_scores_t, sizeof(synthetic_simpls_tiny_pls1_v1_scores_t) / sizeof(double), true},
        MatrixRef{8, 1, synthetic_simpls_tiny_pls1_v1_predict_train, sizeof(synthetic_simpls_tiny_pls1_v1_predict_train) / sizeof(double), false}
    },
    {
        "synthetic_simpls_small_pls2_v1",
        2,
        MatrixRef{12, 5, synthetic_simpls_small_pls2_v1_x, sizeof(synthetic_simpls_small_pls2_v1_x) / sizeof(double), false},
        MatrixRef{12, 2, synthetic_simpls_small_pls2_v1_y, sizeof(synthetic_simpls_small_pls2_v1_y) / sizeof(double), false},
        MatrixRef{5, 2, synthetic_simpls_small_pls2_v1_coefficients, sizeof(synthetic_simpls_small_pls2_v1_coefficients) / sizeof(double), false},
        MatrixRef{1, 2, synthetic_simpls_small_pls2_v1_intercept, sizeof(synthetic_simpls_small_pls2_v1_intercept) / sizeof(double), false},
        MatrixRef{1, 5, synthetic_simpls_small_pls2_v1_x_mean, sizeof(synthetic_simpls_small_pls2_v1_x_mean) / sizeof(double), false},
        MatrixRef{1, 5, synthetic_simpls_small_pls2_v1_x_scale, sizeof(synthetic_simpls_small_pls2_v1_x_scale) / sizeof(double), false},
        MatrixRef{1, 2, synthetic_simpls_small_pls2_v1_y_mean, sizeof(synthetic_simpls_small_pls2_v1_y_mean) / sizeof(double), false},
        MatrixRef{1, 2, synthetic_simpls_small_pls2_v1_y_scale, sizeof(synthetic_simpls_small_pls2_v1_y_scale) / sizeof(double), false},
        MatrixRef{5, 2, synthetic_simpls_small_pls2_v1_weights_w, sizeof(synthetic_simpls_small_pls2_v1_weights_w) / sizeof(double), true},
        MatrixRef{5, 2, synthetic_simpls_small_pls2_v1_loadings_p, sizeof(synthetic_simpls_small_pls2_v1_loadings_p) / sizeof(double), true},
        MatrixRef{2, 2, synthetic_simpls_small_pls2_v1_y_loadings_q, sizeof(synthetic_simpls_small_pls2_v1_y_loadings_q) / sizeof(double), true},
        MatrixRef{5, 2, synthetic_simpls_small_pls2_v1_rotations_r, sizeof(synthetic_simpls_small_pls2_v1_rotations_r) / sizeof(double), true},
        MatrixRef{12, 2, synthetic_simpls_small_pls2_v1_scores_t, sizeof(synthetic_simpls_small_pls2_v1_scores_t) / sizeof(double), true},
        MatrixRef{12, 2, synthetic_simpls_small_pls2_v1_predict_train, sizeof(synthetic_simpls_small_pls2_v1_predict_train) / sizeof(double), false}
    }
};

}  // namespace n4m::test::fixtures
