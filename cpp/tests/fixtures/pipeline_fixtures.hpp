// SPDX-License-Identifier: CeCILL-2.1
// Generated mechanically from parity/fixtures/*pipeline*.json.
#pragma once

#include <cstddef>
#include <cstdint>

#include "pls4all/p4a.h"
#include "phase1_fixtures.hpp"

namespace pls4all::test::fixtures {

struct PipelineFixture {
    const char* id;
    const p4a_operator_kind_t* operators;
    std::size_t n_operators;
    const double* params;
    const std::int32_t* n_params;
    MatrixRef X;
    MatrixRef Y;
    MatrixRef transform_train;
};

inline const double synthetic_pipeline_identity_v1_x[] = {
    1, 2, 3, 4,
    5, 6, 7, 8,
    9, 2, 4, 8,
};

inline const double synthetic_pipeline_identity_v1_y[] = {
    0, 0, 0, 0,
};

inline const double synthetic_pipeline_identity_v1_transform_train[] = {
    1, 2, 3, 4,
    5, 6, 7, 8,
    9, 2, 4, 8,
};

inline const p4a_operator_kind_t synthetic_pipeline_identity_v1_operators[] = {
    P4A_OP_IDENTITY,
};

inline const double synthetic_pipeline_identity_v1_params[] = {
    0.0,
};

inline const std::int32_t synthetic_pipeline_identity_v1_n_params[] = {
    0,
};

inline const double synthetic_pipeline_center_v1_x[] = {
    1, 2, 3, 4,
    5, 6, 7, 8,
    9, 2, 4, 8,
};

inline const double synthetic_pipeline_center_v1_y[] = {
    0, 0, 0, 0,
};

inline const double synthetic_pipeline_center_v1_transform_train[] = {
    -2.5, -2.75, -3.5, 0.5,
    0.25, -0.5, 3.5, 3.25,
    2.5, -1.5, -0.75, 1.5,
};

inline const p4a_operator_kind_t synthetic_pipeline_center_v1_operators[] = {
    P4A_OP_CENTER,
};

inline const double synthetic_pipeline_center_v1_params[] = {
    0.0,
};

inline const std::int32_t synthetic_pipeline_center_v1_n_params[] = {
    0,
};

inline const double synthetic_pipeline_autoscale_v1_x[] = {
    1, 2, 3, 4,
    5, 6, 7, 8,
    9, 2, 4, 8,
};

inline const double synthetic_pipeline_autoscale_v1_y[] = {
    0, 0, 0, 0,
};

inline const double synthetic_pipeline_autoscale_v1_transform_train[] = {
    -0.94491118252306805, -1.1000000000000001, -1.3228756555322951, 0.1889822365046136,
    0.10000000000000001, -0.1889822365046136, 1.3228756555322951, 1.3,
    0.94491118252306805, -0.56694670951384085, -0.29999999999999999, 0.56694670951384085,
};

inline const p4a_operator_kind_t synthetic_pipeline_autoscale_v1_operators[] = {
    P4A_OP_AUTOSCALE,
};

inline const double synthetic_pipeline_autoscale_v1_params[] = {
    0.0,
};

inline const std::int32_t synthetic_pipeline_autoscale_v1_n_params[] = {
    0,
};

inline const double synthetic_pipeline_pareto_v1_x[] = {
    1, 2, 3, 4,
    5, 6, 7, 8,
    9, 2, 4, 8,
};

inline const double synthetic_pipeline_pareto_v1_y[] = {
    0, 0, 0, 0,
};

inline const double synthetic_pipeline_pareto_v1_transform_train[] = {
    -1.5369703823781609, -1.7392527130926085, -2.1517585353294253, 0.30739407647563216,
    0.15811388300841897, -0.30739407647563216, 2.1517585353294253, 2.0554804791094465,
    1.5369703823781609, -0.92218222942689643, -0.47434164902525688, 0.92218222942689643,
};

inline const p4a_operator_kind_t synthetic_pipeline_pareto_v1_operators[] = {
    P4A_OP_PARETO_SCALE,
};

inline const double synthetic_pipeline_pareto_v1_params[] = {
    0.0,
};

inline const std::int32_t synthetic_pipeline_pareto_v1_n_params[] = {
    0,
};

inline const double synthetic_pipeline_snv_v1_x[] = {
    1, 2, 3, 2,
    2, 2, 3, 5,
    7,
};

inline const double synthetic_pipeline_snv_v1_y[] = {
    0, 0, 0,
};

inline const double synthetic_pipeline_snv_v1_transform_train[] = {
    -1, 0, 1, 0,
    0, 0, -1, 0,
    1,
};

inline const p4a_operator_kind_t synthetic_pipeline_snv_v1_operators[] = {
    P4A_OP_SNV,
};

inline const double synthetic_pipeline_snv_v1_params[] = {
    0.0,
};

inline const std::int32_t synthetic_pipeline_snv_v1_n_params[] = {
    0,
};

inline const double synthetic_pipeline_msc_v1_x[] = {
    1.3500000000000001, 2.4500000000000002, 4.6500000000000004, 7.9500000000000011,
    12.350000000000001, 0.46999999999999997, 1.29, 2.9299999999999997,
    5.3899999999999997, 8.6699999999999999, 2.25, 3.6000000000000001,
    6.3000000000000007, 10.350000000000001, 15.750000000000002, 1.0700000000000001,
    1.6899999999999999, 2.9300000000000002, 4.79, 7.2700000000000005,
};

inline const double synthetic_pipeline_msc_v1_y[] = {
    0, 0, 0, 0,
};

inline const double synthetic_pipeline_msc_v1_transform_train[] = {
    1.2849999999999995, 2.2574999999999998, 4.2024999999999997, 7.1200000000000019,
    11.010000000000002, 1.2850000000000006, 2.2575000000000007, 4.2025000000000006,
    7.120000000000001, 11.010000000000002, 1.2849999999999995, 2.2574999999999998,
    4.2025000000000006, 7.1200000000000019, 11.010000000000003, 1.285000000000001,
    2.2575000000000007, 4.2025000000000015, 7.120000000000001, 11.010000000000002,
};

inline const p4a_operator_kind_t synthetic_pipeline_msc_v1_operators[] = {
    P4A_OP_MSC,
};

inline const double synthetic_pipeline_msc_v1_params[] = {
    0.0,
};

inline const std::int32_t synthetic_pipeline_msc_v1_n_params[] = {
    0,
};

inline const double synthetic_pipeline_emsc_v1_x[] = {
    1.1800000000000002, 2.6620408163265301, 4.4010204081632649, 6.0269387755102048,
    8.4497959183673501, 9.0595918367346968, 12.21632653061225, 14.350000000000001,
    0.53499999999999992, 1.542938775510204, 2.7094693877551013, 3.8085918367346938,
    5.6183061224489785, 6.0186122448979589, 8.4465102040816333, 10.028999999999996,
    2.1425000000000001, 3.9275510204081634, 5.9902040816326529, 7.9154591836734705,
    10.898316326530614, 11.598775510204083, 15.514336734693879, 18.097499999999997,
    0.97000000000000008, 1.830938775510204, 2.7860408163265307, 3.6713061224489802,
    5.1787346938775523, 5.4683265306122468, 7.4780816326530637, 8.7260000000000026,
};

inline const double synthetic_pipeline_emsc_v1_y[] = {
    0, 0, 0, 0,
};

inline const double synthetic_pipeline_emsc_v1_transform_train[] = {
    1.2155523279016054, 2.4688727264769503, 3.9712808187552042, 5.3806460299620085,
    7.5384246385964619, 8.0312090766556512, 10.890592063102297, 12.815475889978387,
    1.2022168885781221, 2.5026743679427939, 3.9718999315348604, 5.3421149510149046,
    7.5351414301354387, 8.0390736517302273, 10.926279516695665, 12.792652833796559,
    1.2090438768856053, 2.4853698453497794, 3.9715829809188969, 5.3618406772595062,
    7.5368222465848032, 8.0350474355122952, 10.908009565290561, 12.804336943627119,
    1.1951760744460946, 2.5205208797183296, 3.9722268092553903, 5.3217714018754787,
    7.5334079693944727, 8.0432259717239116, 10.945121695157217, 12.780602769857673,
};

inline const p4a_operator_kind_t synthetic_pipeline_emsc_v1_operators[] = {
    P4A_OP_EMSC,
};

inline const double synthetic_pipeline_emsc_v1_params[] = {
    2,
};

inline const std::int32_t synthetic_pipeline_emsc_v1_n_params[] = {
    1,
};

inline const double synthetic_pipeline_detrend_poly_v1_x[] = {
    1.52, 1.0680000000000001, 1.002, 1.0720000000000001,
    1.478, 2.1000000000000001, -1.3100000000000001, -0.86199999999999999,
    -0.56800000000000006, -0.2279999999999999, -0.082000000000000003, 0.10999999999999992,
    1.1000000000000001, 0.66400000000000003, 0.28600000000000003, 0.12599999999999995,
    0.034000000000000009, 0.12000000000000004,
};

inline const double synthetic_pipeline_detrend_poly_v1_y[] = {
    0, 0, 0,
};

inline const double synthetic_pipeline_detrend_poly_v1_transform_train[] = {
    0.014642857142856736, -0.038928571428571646, 0.034285714285714253, -0.01571428571428557,
    0.011071428571428621, -0.0053571428571426161, -0.010000000000000231, 0.024857142857142689,
    -0.027428571428571802, 0.033142857142856946, -0.033428571428571606, 0.012857142857142775,
    -0.0082142857142857295, 0.022499999999999964, -0.022857142857142798, 0.015714285714285722,
    -0.011785714285714274, 0.0046428571428571014,
};

inline const p4a_operator_kind_t synthetic_pipeline_detrend_poly_v1_operators[] = {
    P4A_OP_DETREND_POLY,
};

inline const double synthetic_pipeline_detrend_poly_v1_params[] = {
    2,
};

inline const std::int32_t synthetic_pipeline_detrend_poly_v1_n_params[] = {
    1,
};

inline const double synthetic_pipeline_savgol_smooth_v1_x[] = {
    1.55, 0.89749999999999974, 0.56999999999999995, 0.3075,
    0.31, 0.45750000000000002, 0.73999999999999999, 1.2674999999999998,
    1.8799999999999999, -0.92929742682568173, -0.96749498660405442, -0.85147098480789651,
    -0.43942553860420303, -0.02, 0.47942553860420301, 0.86147098480789652,
    0.96749498660405442, 0.91929742682568172, 1.9000000000000001, 1.6637500000000001,
    1.385, 1.2537500000000001, 1.1300000000000001, 0.97375000000000012,
    0.93500000000000005, 0.88375000000000004, 0.92000000000000015,
};

inline const double synthetic_pipeline_savgol_smooth_v1_y[] = {
    0, 0, 0,
};

inline const double synthetic_pipeline_savgol_smooth_v1_transform_train[] = {
    1.4102857142857137, 1.0035714285714279, 0.53057142857142803, 0.33492857142857113,
    0.30057142857142838, 0.4472142857142854, 0.76314285714285657, 1.3135714285714277,
    1.7677142857142847, -0.94906457092264773, -0.96315619473091973, -0.81457602182160771,
    -0.47039078942761853, 0.0031428571428571347, 0.47610507514190425, 0.83743316467875051,
    0.96058476615949118, 0.94077885663693339, 1.8631428571428561, 1.6640714285714275,
    1.4132857142857136, 1.2451785714285708, 1.113714285714285, 0.99774999999999969,
    0.91528571428571381, 0.90292857142857097, 0.90628571428571392,
};

inline const p4a_operator_kind_t synthetic_pipeline_savgol_smooth_v1_operators[] = {
    P4A_OP_SAVGOL_SMOOTH,
};

inline const double synthetic_pipeline_savgol_smooth_v1_params[] = {
    5, 2,
};

inline const std::int32_t synthetic_pipeline_savgol_smooth_v1_n_params[] = {
    2,
};

inline const double synthetic_pipeline_savgol_derivative_v1_x[] = {
    -1.8999999999999999, -1.090625, -0.47499999999999998, 0.0031250000000000219,
    0.40000000000000002, 0.77187500000000009, 1.1749999999999998, 1.6656249999999999,
    2.2999999999999998, -1.0011436155469338, -0.44953350618957416, 0.21532236239526867,
    0.76096311950521789, 1, 0.86096311950521798, 0.41532236239526865,
    -0.14953350618957412, -0.60114361554693363, -0.30000000000000004, -0.54999999999999993,
    -0.69999999999999996, -0.74999999999999989, -0.69999999999999996, -0.54999999999999993,
    -0.29999999999999993, 0.050000000000000155, 0.50000000000000011,
};

inline const double synthetic_pipeline_savgol_derivative_v1_y[] = {
    0, 0, 0,
};

inline const double synthetic_pipeline_savgol_derivative_v1_transform_train[] = {
    0.36593749999999953, 0.52312499999999973, 0.56937499999999996, 0.46000000000000008,
    0.40687500000000021, 0.41000000000000031, 0.46937500000000043, 0.41812500000000058,
    0.28843750000000073, 0.29845420652417626, 0.47406794480465053, 0.52127838567886609,
    0.34056708889943188, 0.050000000000000336, -0.24056708889943132, -0.42127838567886583,
    -0.39406794480465068, -0.24845420652417666, -0.10500000000000009, -0.13000000000000017,
    -0.1000000000000002, -2.2204460492503131e-16, 0.099999999999999811, 0.1999999999999999,
    0.29999999999999999, 0.29000000000000015, 0.20500000000000018,
};

inline const p4a_operator_kind_t synthetic_pipeline_savgol_derivative_v1_operators[] = {
    P4A_OP_SAVGOL_DERIVATIVE,
};

inline const double synthetic_pipeline_savgol_derivative_v1_params[] = {
    5, 2, 1, 1,
};

inline const std::int32_t synthetic_pipeline_savgol_derivative_v1_n_params[] = {
    4,
};

inline const double synthetic_pipeline_asls_v1_x[] = {
    0.40000000000000002, 0.5319753086419754, 0.7701234567901234, 1.1944444444444446,
    1.0249382716049382, 0.9716049382716051, 1.0644444444444445, 1.1734567901234567,
    1.338641975308642, 1.4600000000000002, 0.29999999999999999, 0.37438271604938278,
    0.53308641975308646, 0.72611111111111115, 1.1834567901234567, 0.95512345679012345,
    0.80111111111111111, 0.8414197530864197, 0.9160493827160493, 1.0250000000000001,
    0.48999999999999999, 0.59037037037037043, 0.75814814814814813, 0.93333333333333335,
    1.1659259259259258, 1.5559259259259259, 1.5333333333333334, 1.488148148148148,
    1.6003703703703704, 1.7400000000000002,
};

inline const double synthetic_pipeline_asls_v1_y[] = {
    0, 0, 0,
};

inline const double synthetic_pipeline_asls_v1_transform_train[] = {
    -0.0041152240536706519, 0.017291480911322155, 0.14491176610061851, 0.4587846430842587,
    0.17893463225567174, 0.015340375543979823, -0.0020373785849225889, -0.0032394157026938153,
    0.051713648666998235, 0.0628332692159419, -0.0014311109311425652, -0.0046865580738422175,
    0.076393150435996926, 0.19186857952152392, 0.57179265479084396, 0.26619911499411075,
    0.035064519615998213, -0.0016611917703183154, -0.0040315860436254125, 0.02791621571581715,
    0.038846203032562143, -0.0046943458971168139, 0.019168627960308049, 0.050477714008914987,
    0.13927358478998553, 0.38559186507340115, 0.21945425227056359, 0.030843884606367222,
    -0.00027804511951945443, -0.0039534284958895327,
};

inline const p4a_operator_kind_t synthetic_pipeline_asls_v1_operators[] = {
    P4A_OP_ASLS_BASELINE,
};

inline const double synthetic_pipeline_asls_v1_params[] = {
    100, 0.01, 8,
};

inline const std::int32_t synthetic_pipeline_asls_v1_n_params[] = {
    3,
};

inline const double synthetic_pipeline_norris_williams_v1_x[] = {
    -0.41411200080598687, -0.42754631805511517, -0.37738476308781954, -0.25320390859672259,
    -0.056464247339503504, 0.20000000000000001, 0.49646424733950389, 0.81320390859672287,
    1.1373847630878196, 1.4675463180551152, 1.8141120008059868, -1.3624220254820714,
    -1.0923799283451934, -0.83764633682849898, -0.60749399359792511, -0.40984085089956729,
    -0.25, -0.12984085089956715, -0.04749399359792518, 0.0023536631715010067,
    0.027620071654806566, 0.037577974517928561, 1.8, 1.6119999999999999,
    1.3980000000000001, 1.278, 1.0920000000000001, 1,
    0.92199999999999993, 0.81799999999999995, 0.82799999999999996, 0.78200000000000003,
    0.80000000000000004,
};

inline const double synthetic_pipeline_norris_williams_v1_y[] = {
    0, 0, 0,
};

inline const double synthetic_pipeline_norris_williams_v1_transform_train[] = {
    0.4159942664690785, 0.65411200080598708, 0.93222973514289553, 1.239718711671645,
    1.5551956642892089, 1.7865435965406529, 1.9151956642892087, 1.9277187116716448,
    1.8122297351428953, 1.5741120008059868, 1.2959942664690782, 0.91545761903695988,
    1.0734880876830744, 1.1954576190369597, 1.2289413841204591, 1.18150184129333,
    1.0640000000000001, 0.89049815870666971, 0.67505861587954097, 0.48454238096304036,
    0.32651191231692556, 0.20454238096304034, -0.66200000000000025, -0.77800000000000014,
    -0.86800000000000055, -0.89240000000000019, -0.8520000000000002, -0.77200000000000024,
    -0.63400000000000023, -0.4795999999999998, -0.33799999999999975, -0.22199999999999986,
    -0.13199999999999978,
};

inline const p4a_operator_kind_t synthetic_pipeline_norris_williams_v1_operators[] = {
    P4A_OP_NORRIS_WILLIAMS,
};

inline const double synthetic_pipeline_norris_williams_v1_params[] = {
    5, 3, 1,
};

inline const std::int32_t synthetic_pipeline_norris_williams_v1_n_params[] = {
    3,
};

inline const double synthetic_pipeline_wavelet_haar_v1_x[] = {
    0.52000000000000002, 0.56999999999999995, 0.73999999999999999, 0.75,
    0.93000000000000005, 0.97999999999999998, 1.1100000000000001, 1.1900000000000002,
    1.3200000000000001, -0.01, 0.32743851458038087, 0.54509727294046217,
    0.83608110826069304, 0.92898461935558618, 1.0399655856782488, 0.92408578160969379,
    0.82578931325829696, 0.59847214410395655, 1, 0.98312500000000003,
    0.89249999999999996, 0.88812499999999994, 0.81000000000000005, 0.84812500000000002,
    0.80249999999999999, 0.83312499999999989, 0.78000000000000003,
};

inline const double synthetic_pipeline_wavelet_haar_v1_y[] = {
    0, 0, 0,
};

inline const double synthetic_pipeline_wavelet_haar_v1_transform_train[] = {
    0.55621320343559633, 0.56378679656440334, 0.72999999999999976, 0.72999999999999976,
    0.96621320343559636, 0.97378679656440337, 1.1162132034355963, 1.1537867965644033,
    1.3199999999999996, 0.026213203435596365, 0.32122531114478431, 0.55131047637605846,
    0.7998679048250964, 0.93519782279118224, 1.0037523822426522, 0.91787257817409718,
    0.86200251669389338, 0.59847214410395633, 0.97656249999999967, 0.97656249999999967,
    0.90531249999999952, 0.90531249999999952, 0.8234374999999996, 0.8234374999999996,
    0.8234374999999996, 0.8234374999999996, 0.77999999999999969,
};

inline const p4a_operator_kind_t synthetic_pipeline_wavelet_haar_v1_operators[] = {
    P4A_OP_WAVELET_DENOISE,
};

inline const double synthetic_pipeline_wavelet_haar_v1_params[] = {
    2, 0.029999999999999999,
};

inline const std::int32_t synthetic_pipeline_wavelet_haar_v1_n_params[] = {
    2,
};

inline const double synthetic_pipeline_osc_v1_x[] = {
    0.52000000000000002, -0.16500000000000004, -0.37, 0.54000000000000004,
    0.059999999999999956, -0.94999999999999996, 0.495, -0.48999999999999999,
    -0.54999999999999993, 0.42999999999999999, 0.32000000000000006, -0.15000000000000002,
    -0.0099999999999999707, 0.24999999999999994, -0.020000000000000018, -0.20999999999999991,
    0.084999999999999964, 0.020000000000000115, -0.16999999999999998, 0.0099999999999999343,
    0.82000000000000006, -0.41000000000000003, 0.46000000000000008, 0.44999999999999996,
    -0.39000000000000001, -0.46999999999999997, 0.14500000000000002, 0.37,
    -0.52000000000000002, -0.099999999999999964,
};

inline const double synthetic_pipeline_osc_v1_y[] = {
    -1, -0.59999999999999998, -0.19999999999999996, 0.20000000000000018,
    0.60000000000000009, 1,
};

inline const double synthetic_pipeline_osc_v1_transform_train[] = {
    0.048211658691724668, 0.055713016274572147, -0.48226291136903143, 0.21804981189137002,
    0.20651312390614834, 0.032181624363770478, 0.035513722230498423, -0.25628805972932323,
    0.1202445377176794, 0.12498504852894904, 0.01396632305866724, -0.0068306801901576175,
    -0.082821281371924863, 0.041161435686139475, 0.075038274800991164, -0.0081496878483439394,
    -0.009430038494413498, 0.068030656374541643, -0.032256566606107734, -0.052684295488866932,
    -0.016774259897375687, -0.018538501967358623, 0.26088801195414735, -0.12101799004055724,
    -0.1301410758013799, -0.039435658368442539, -0.056427517853140957, 0.47245358414159072,
    -0.2261812286485238, -0.2337110759458419,
};

inline const p4a_operator_kind_t synthetic_pipeline_osc_v1_operators[] = {
    P4A_OP_OSC,
};

inline const double synthetic_pipeline_osc_v1_params[] = {
    100, 9.9999999999999998e-13,
};

inline const std::int32_t synthetic_pipeline_osc_v1_n_params[] = {
    2,
};

inline const PipelineFixture kPipelineFixtures[] = {
    {
        "synthetic_pipeline_identity_v1",
        synthetic_pipeline_identity_v1_operators,
        sizeof(synthetic_pipeline_identity_v1_operators) / sizeof(p4a_operator_kind_t),
        synthetic_pipeline_identity_v1_params,
        synthetic_pipeline_identity_v1_n_params,
        MatrixRef{4, 3, synthetic_pipeline_identity_v1_x, sizeof(synthetic_pipeline_identity_v1_x) / sizeof(double), false},
        MatrixRef{4, 1, synthetic_pipeline_identity_v1_y, sizeof(synthetic_pipeline_identity_v1_y) / sizeof(double), false},
        MatrixRef{4, 3, synthetic_pipeline_identity_v1_transform_train, sizeof(synthetic_pipeline_identity_v1_transform_train) / sizeof(double), false}
    },
    {
        "synthetic_pipeline_center_v1",
        synthetic_pipeline_center_v1_operators,
        sizeof(synthetic_pipeline_center_v1_operators) / sizeof(p4a_operator_kind_t),
        synthetic_pipeline_center_v1_params,
        synthetic_pipeline_center_v1_n_params,
        MatrixRef{4, 3, synthetic_pipeline_center_v1_x, sizeof(synthetic_pipeline_center_v1_x) / sizeof(double), false},
        MatrixRef{4, 1, synthetic_pipeline_center_v1_y, sizeof(synthetic_pipeline_center_v1_y) / sizeof(double), false},
        MatrixRef{4, 3, synthetic_pipeline_center_v1_transform_train, sizeof(synthetic_pipeline_center_v1_transform_train) / sizeof(double), false}
    },
    {
        "synthetic_pipeline_autoscale_v1",
        synthetic_pipeline_autoscale_v1_operators,
        sizeof(synthetic_pipeline_autoscale_v1_operators) / sizeof(p4a_operator_kind_t),
        synthetic_pipeline_autoscale_v1_params,
        synthetic_pipeline_autoscale_v1_n_params,
        MatrixRef{4, 3, synthetic_pipeline_autoscale_v1_x, sizeof(synthetic_pipeline_autoscale_v1_x) / sizeof(double), false},
        MatrixRef{4, 1, synthetic_pipeline_autoscale_v1_y, sizeof(synthetic_pipeline_autoscale_v1_y) / sizeof(double), false},
        MatrixRef{4, 3, synthetic_pipeline_autoscale_v1_transform_train, sizeof(synthetic_pipeline_autoscale_v1_transform_train) / sizeof(double), false}
    },
    {
        "synthetic_pipeline_pareto_v1",
        synthetic_pipeline_pareto_v1_operators,
        sizeof(synthetic_pipeline_pareto_v1_operators) / sizeof(p4a_operator_kind_t),
        synthetic_pipeline_pareto_v1_params,
        synthetic_pipeline_pareto_v1_n_params,
        MatrixRef{4, 3, synthetic_pipeline_pareto_v1_x, sizeof(synthetic_pipeline_pareto_v1_x) / sizeof(double), false},
        MatrixRef{4, 1, synthetic_pipeline_pareto_v1_y, sizeof(synthetic_pipeline_pareto_v1_y) / sizeof(double), false},
        MatrixRef{4, 3, synthetic_pipeline_pareto_v1_transform_train, sizeof(synthetic_pipeline_pareto_v1_transform_train) / sizeof(double), false}
    },
    {
        "synthetic_pipeline_snv_v1",
        synthetic_pipeline_snv_v1_operators,
        sizeof(synthetic_pipeline_snv_v1_operators) / sizeof(p4a_operator_kind_t),
        synthetic_pipeline_snv_v1_params,
        synthetic_pipeline_snv_v1_n_params,
        MatrixRef{3, 3, synthetic_pipeline_snv_v1_x, sizeof(synthetic_pipeline_snv_v1_x) / sizeof(double), false},
        MatrixRef{3, 1, synthetic_pipeline_snv_v1_y, sizeof(synthetic_pipeline_snv_v1_y) / sizeof(double), false},
        MatrixRef{3, 3, synthetic_pipeline_snv_v1_transform_train, sizeof(synthetic_pipeline_snv_v1_transform_train) / sizeof(double), false}
    },
    {
        "synthetic_pipeline_msc_v1",
        synthetic_pipeline_msc_v1_operators,
        sizeof(synthetic_pipeline_msc_v1_operators) / sizeof(p4a_operator_kind_t),
        synthetic_pipeline_msc_v1_params,
        synthetic_pipeline_msc_v1_n_params,
        MatrixRef{4, 5, synthetic_pipeline_msc_v1_x, sizeof(synthetic_pipeline_msc_v1_x) / sizeof(double), false},
        MatrixRef{4, 1, synthetic_pipeline_msc_v1_y, sizeof(synthetic_pipeline_msc_v1_y) / sizeof(double), false},
        MatrixRef{4, 5, synthetic_pipeline_msc_v1_transform_train, sizeof(synthetic_pipeline_msc_v1_transform_train) / sizeof(double), false}
    },
    {
        "synthetic_pipeline_emsc_v1",
        synthetic_pipeline_emsc_v1_operators,
        sizeof(synthetic_pipeline_emsc_v1_operators) / sizeof(p4a_operator_kind_t),
        synthetic_pipeline_emsc_v1_params,
        synthetic_pipeline_emsc_v1_n_params,
        MatrixRef{4, 8, synthetic_pipeline_emsc_v1_x, sizeof(synthetic_pipeline_emsc_v1_x) / sizeof(double), false},
        MatrixRef{4, 1, synthetic_pipeline_emsc_v1_y, sizeof(synthetic_pipeline_emsc_v1_y) / sizeof(double), false},
        MatrixRef{4, 8, synthetic_pipeline_emsc_v1_transform_train, sizeof(synthetic_pipeline_emsc_v1_transform_train) / sizeof(double), false}
    },
    {
        "synthetic_pipeline_detrend_poly_v1",
        synthetic_pipeline_detrend_poly_v1_operators,
        sizeof(synthetic_pipeline_detrend_poly_v1_operators) / sizeof(p4a_operator_kind_t),
        synthetic_pipeline_detrend_poly_v1_params,
        synthetic_pipeline_detrend_poly_v1_n_params,
        MatrixRef{3, 6, synthetic_pipeline_detrend_poly_v1_x, sizeof(synthetic_pipeline_detrend_poly_v1_x) / sizeof(double), false},
        MatrixRef{3, 1, synthetic_pipeline_detrend_poly_v1_y, sizeof(synthetic_pipeline_detrend_poly_v1_y) / sizeof(double), false},
        MatrixRef{3, 6, synthetic_pipeline_detrend_poly_v1_transform_train, sizeof(synthetic_pipeline_detrend_poly_v1_transform_train) / sizeof(double), false}
    },
    {
        "synthetic_pipeline_savgol_smooth_v1",
        synthetic_pipeline_savgol_smooth_v1_operators,
        sizeof(synthetic_pipeline_savgol_smooth_v1_operators) / sizeof(p4a_operator_kind_t),
        synthetic_pipeline_savgol_smooth_v1_params,
        synthetic_pipeline_savgol_smooth_v1_n_params,
        MatrixRef{3, 9, synthetic_pipeline_savgol_smooth_v1_x, sizeof(synthetic_pipeline_savgol_smooth_v1_x) / sizeof(double), false},
        MatrixRef{3, 1, synthetic_pipeline_savgol_smooth_v1_y, sizeof(synthetic_pipeline_savgol_smooth_v1_y) / sizeof(double), false},
        MatrixRef{3, 9, synthetic_pipeline_savgol_smooth_v1_transform_train, sizeof(synthetic_pipeline_savgol_smooth_v1_transform_train) / sizeof(double), false}
    },
    {
        "synthetic_pipeline_savgol_derivative_v1",
        synthetic_pipeline_savgol_derivative_v1_operators,
        sizeof(synthetic_pipeline_savgol_derivative_v1_operators) / sizeof(p4a_operator_kind_t),
        synthetic_pipeline_savgol_derivative_v1_params,
        synthetic_pipeline_savgol_derivative_v1_n_params,
        MatrixRef{3, 9, synthetic_pipeline_savgol_derivative_v1_x, sizeof(synthetic_pipeline_savgol_derivative_v1_x) / sizeof(double), false},
        MatrixRef{3, 1, synthetic_pipeline_savgol_derivative_v1_y, sizeof(synthetic_pipeline_savgol_derivative_v1_y) / sizeof(double), false},
        MatrixRef{3, 9, synthetic_pipeline_savgol_derivative_v1_transform_train, sizeof(synthetic_pipeline_savgol_derivative_v1_transform_train) / sizeof(double), false}
    },
    {
        "synthetic_pipeline_asls_v1",
        synthetic_pipeline_asls_v1_operators,
        sizeof(synthetic_pipeline_asls_v1_operators) / sizeof(p4a_operator_kind_t),
        synthetic_pipeline_asls_v1_params,
        synthetic_pipeline_asls_v1_n_params,
        MatrixRef{3, 10, synthetic_pipeline_asls_v1_x, sizeof(synthetic_pipeline_asls_v1_x) / sizeof(double), false},
        MatrixRef{3, 1, synthetic_pipeline_asls_v1_y, sizeof(synthetic_pipeline_asls_v1_y) / sizeof(double), false},
        MatrixRef{3, 10, synthetic_pipeline_asls_v1_transform_train, sizeof(synthetic_pipeline_asls_v1_transform_train) / sizeof(double), false}
    },
    {
        "synthetic_pipeline_norris_williams_v1",
        synthetic_pipeline_norris_williams_v1_operators,
        sizeof(synthetic_pipeline_norris_williams_v1_operators) / sizeof(p4a_operator_kind_t),
        synthetic_pipeline_norris_williams_v1_params,
        synthetic_pipeline_norris_williams_v1_n_params,
        MatrixRef{3, 11, synthetic_pipeline_norris_williams_v1_x, sizeof(synthetic_pipeline_norris_williams_v1_x) / sizeof(double), false},
        MatrixRef{3, 1, synthetic_pipeline_norris_williams_v1_y, sizeof(synthetic_pipeline_norris_williams_v1_y) / sizeof(double), false},
        MatrixRef{3, 11, synthetic_pipeline_norris_williams_v1_transform_train, sizeof(synthetic_pipeline_norris_williams_v1_transform_train) / sizeof(double), false}
    },
    {
        "synthetic_pipeline_wavelet_haar_v1",
        synthetic_pipeline_wavelet_haar_v1_operators,
        sizeof(synthetic_pipeline_wavelet_haar_v1_operators) / sizeof(p4a_operator_kind_t),
        synthetic_pipeline_wavelet_haar_v1_params,
        synthetic_pipeline_wavelet_haar_v1_n_params,
        MatrixRef{3, 9, synthetic_pipeline_wavelet_haar_v1_x, sizeof(synthetic_pipeline_wavelet_haar_v1_x) / sizeof(double), false},
        MatrixRef{3, 1, synthetic_pipeline_wavelet_haar_v1_y, sizeof(synthetic_pipeline_wavelet_haar_v1_y) / sizeof(double), false},
        MatrixRef{3, 9, synthetic_pipeline_wavelet_haar_v1_transform_train, sizeof(synthetic_pipeline_wavelet_haar_v1_transform_train) / sizeof(double), false}
    },
    {
        "synthetic_pipeline_osc_v1",
        synthetic_pipeline_osc_v1_operators,
        sizeof(synthetic_pipeline_osc_v1_operators) / sizeof(p4a_operator_kind_t),
        synthetic_pipeline_osc_v1_params,
        synthetic_pipeline_osc_v1_n_params,
        MatrixRef{6, 5, synthetic_pipeline_osc_v1_x, sizeof(synthetic_pipeline_osc_v1_x) / sizeof(double), false},
        MatrixRef{6, 1, synthetic_pipeline_osc_v1_y, sizeof(synthetic_pipeline_osc_v1_y) / sizeof(double), false},
        MatrixRef{6, 5, synthetic_pipeline_osc_v1_transform_train, sizeof(synthetic_pipeline_osc_v1_transform_train) / sizeof(double), false}
    }
};

}  // namespace pls4all::test::fixtures
