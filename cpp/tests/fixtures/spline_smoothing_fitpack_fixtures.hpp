// SPDX-License-Identifier: CECILL-2.1
// Reference outputs of augmentation.splines.spline_smoothing captured from
// the vendored Fortran FITPACK build (SciPy 1.17.1 curfit/splev, gfortran
// 15.2 -O3) before its C translation replaced it.
#pragma once

#include <cstddef>

namespace n4m::test::fixtures {

inline constexpr double kSplineSmoothFitpackIn_4_2[] = {
    0x1.8afc072ad7184p-1, 0x1.6fbd34710b51cp-1, 0x1.fac7dbe19161ap-1,
    -0x1.1544210cd53c0p-1,
};

inline constexpr double kSplineSmoothFitpackOut_4_2[] = {
    0x1.8afc072ad7182p-1, 0x1.6fbd34710b51ap-1, 0x1.fac7dbe191619p-1,
    -0x1.1544210cd53c0p-1,
};

inline constexpr double kSplineSmoothFitpackIn_7_6[] = {
    0x1.e7a2ba0578e13p+1, 0x1.0b52e5fdce574p+6, 0x1.2055c45a5119dp+9,
    0x1.4fef189d90724p+7, 0x1.002987469d9a1p+9, 0x1.5e9fe326bc076p+8,
    0x1.964b258522260p+7,
};

inline constexpr double kSplineSmoothFitpackOut_7_6[] = {
    0x1.c620cc979c072p+1, 0x1.10cb1ebae4766p+6, 0x1.1ece3f71dee4ap+9,
    0x1.5786e5a59a2b9p+7, 0x1.fd8a41e0afae6p+8, 0x1.5fc5c06c20d87p+8,
    0x1.95e136445b0dcp+7,
};

inline constexpr double kSplineSmoothFitpackIn_12_6[] = {
    -0x1.2fd4ba2d259acp+2, 0x1.c84a4ae575e9dp+3, 0x1.9f2483cb04839p+6,
    0x1.25b9c9187d669p+9, 0x1.8ba1089d451b8p+8, 0x1.0eaf52735ab95p+7,
    0x1.f4eb60f425aa6p+7, 0x1.cea91daf13b63p+8, 0x1.0b5a376b87729p+9,
    0x1.75fbb872e2c20p+8, 0x1.d61008ccd53a0p+7, 0x1.986782c138399p+7,
};

inline constexpr double kSplineSmoothFitpackOut_12_6[] = {
    -0x1.2ee8dadb44ac8p+2, 0x1.c5e5f3d6d45a4p+3, 0x1.9fca36b0651f2p+6,
    0x1.25a20346d4b5bp+9, 0x1.8bbe37e61c304p+8, 0x1.0ea31ff024bb6p+7,
    0x1.f4e10ecd35e15p+7, 0x1.ceb109edc5848p+8, 0x1.0b55f23528a83p+9,
    0x1.76026530c6fa5p+8, 0x1.d60a1d183a2dfp+7, 0x1.98689bb8e7879p+7,
};

inline constexpr double kSplineSmoothFitpackIn_20_6[] = {
    -0x1.3029448d0988bp+2, 0x1.1d28387522f43p+3, 0x1.60c0e9193d86ep+4,
    0x1.da6969517c721p+5, 0x1.b281c106a1f57p+7, 0x1.0b0e5f4826fe7p+9,
    0x1.42bfe2fd6a373p+9, 0x1.6a4c24eeeba55p+8, 0x1.4108a9569ce97p+7,
    0x1.13c763096375bp+7, 0x1.a6ce9912e1ddfp+7, 0x1.459375521f753p+8,
    0x1.c606dce048675p+8, 0x1.0abebce7ef077p+9, 0x1.07121af6970d3p+9,
    0x1.b5b6cbadea8b0p+8, 0x1.50077107e523ap+8, 0x1.fc0196c0e94bfp+7,
    0x1.b298b5eba98fep+7, 0x1.9fe222a42f18ep+7,
};

inline constexpr double kSplineSmoothFitpackOut_20_6[] = {
    -0x1.302f512bcb385p+2, 0x1.1d256f749caffp+3, 0x1.60edbfba9df3dp+4,
    0x1.da271bfdf0d4bp+5, 0x1.b28f71ea5167ap+7, 0x1.0b13ad6faa6c4p+9,
    0x1.42b06658c156ep+9, 0x1.6a6ce6478f2f5p+8, 0x1.40e395ba47bf7p+7,
    0x1.13d4bde481a1ap+7, 0x1.a6c7b230731bfp+7, 0x1.4596ee8ba59dcp+8,
    0x1.c603aac713506p+8, 0x1.0ac05b95e50dep+9, 0x1.0710168fc58e0p+9,
    0x1.b5bb2800f41c8p+8, 0x1.5003cdd677fa0p+8, 0x1.fc064cc702b3ap+7,
    0x1.b296a70cf7ba6p+7, 0x1.9fe28d5554cb0p+7,
};

inline constexpr double kSplineSmoothFitpackIn_31_5[] = {
    0x0.0p+0, 0x0.0p+0, 0x0.0p+0,
    0x1.0000000000000p+0, 0x0.0p+0, 0x0.0p+0,
    0x0.0p+0, 0x0.0p+0, 0x0.0p+0,
    0x0.0p+0, 0x1.0000000000000p+0, 0x0.0p+0,
    0x0.0p+0, 0x0.0p+0, 0x0.0p+0,
    0x0.0p+0, 0x1.0000000000000p-1, 0x1.8000000000000p+0,
    0x1.0000000000000p-1, 0x1.0000000000000p-1, 0x1.0000000000000p-1,
    0x1.0000000000000p-1, 0x1.0000000000000p-1, 0x1.0000000000000p-1,
    0x1.8000000000000p+0, 0x1.0000000000000p-1, 0x1.0000000000000p-1,
    0x1.0000000000000p-1, 0x1.0000000000000p-1, 0x1.0000000000000p-1,
    0x1.0000000000000p-1,
};

inline constexpr double kSplineSmoothFitpackOut_31_5[] = {
    0x1.4c59a811b0bd6p-9, -0x1.c7b3d8a61f800p-7, 0x1.10e615cbd4be0p-5,
    0x1.e848c8b515a60p-1, 0x1.53f3d821d94e6p-5, -0x1.689ee54e85f52p-6,
    -0x1.5293cda5653d0p-9, 0x1.7b1ee9c825fb9p-6, -0x1.2cea1e973398ap-5,
    0x1.80544d5fddce0p-5, 0x1.e55e9cc3e4258p-1, 0x1.92922033acb7cp-5,
    -0x1.635b8f57d70cap-5, 0x1.1b534db8b3ff5p-5, -0x1.0f8066e611bd6p-6,
    -0x1.f86617792e630p-8, 0x1.0f37ae7edaeccp-1, 0x1.7525117d265c2p+0,
    0x1.15f91abe537fep-1, 0x1.db1ee8b763668p-2, 0x1.0c2d977aaa7bbp-1,
    0x1.fcae8d0f11f31p-2, 0x1.e983b92537d19p-2, 0x1.15d610fd003b3p-1,
    0x1.72ffae3ca7984p+0, 0x1.162f6fbb7217ep-1, 0x1.e4e9b50e55d9ep-2,
    0x1.05766cd17dc13p-1, 0x1.fdfd216131c76p-2, 0x1.ffa0b67b6b1d5p-2,
    0x1.001967fc255d4p-1,
};

inline constexpr double kSplineSmoothFitpackIn_33_9[] = {
    0x1.bfaa8cd1cea15p-12, 0x1.729959978ab35p-2, 0x1.58f7c3f08460bp-1,
    0x1.cad3ea0d63098p-1, 0x1.fdace3aa6c320p-1, 0x1.ec8080557aa28p-1,
    0x1.97ad59561dca2p-1, 0x1.0cc5a96557ed8p-1, 0x1.735f477899d6fp-3,
    -0x1.80ca172518b6cp-3, -0x1.0ed304eeb32f0p-1, -0x1.99d43f9ac42b2p-1,
    -0x1.ed7b2521330d1p-1, -0x1.fd65b835f79c2p-1, -0x1.c9789ee0e0435p-1,
    -0x1.56eb4c274f7d0p-1, -0x1.6b82e1ab3b4a8p-2, 0x1.f6e16b2416856p-8,
    0x1.77fafb28e3ee0p-2, 0x1.5bc310833389fp-1, 0x1.cc80ea4b2c7e1p-1,
    0x1.fdb9310a14465p-1, 0x1.eb98b08a202c6p-1, 0x1.96116e3c06140p-1,
    0x1.0984eac19bc9ep-1, 0x1.623344dc45475p-3, -0x1.8ef97a459f4c3p-3,
    -0x1.11f2b8764deeap-1, -0x1.9becce4270f5cp-1, -0x1.ee83670ca4b14p-1,
    -0x1.fce34edaf9851p-1, -0x1.c78a23aeea081p-1, -0x1.5420be3a2c44ep-1,
};

inline constexpr double kSplineSmoothFitpackOut_33_9[] = {
    -0x1.e9605eee665eap-5, 0x1.9ad2e4b89fd83p-2, 0x1.77269c27259d2p-1,
    0x1.dd2cdce2f0e90p-1, 0x1.fe079a01080b2p-1, 0x1.dace1fc02a00fp-1,
    0x1.7ec75594b8ceap-1, 0x1.ef8c1380fe73cp-2, 0x1.4e742a1520226p-3,
    -0x1.7919a9e3eb874p-3, -0x1.080d5a0dbb2b9p-1, -0x1.924659aa435d4p-1,
    -0x1.e57ffebfddff4p-1, -0x1.f03b287117569p-1, -0x1.b8c1dca582985p-1,
    -0x1.4b506af5f4b63p-1, -0x1.684645f685435p-2, -0x1.b9e87f15f7644p-9,
    0x1.68ba1fe329ff8p-2, 0x1.544ffd98ed4f8p-1, 0x1.c44d17b92203fp-1,
    0x1.f03491db8240ap-1, 0x1.daae90a5dd890p-1, 0x1.8c1535052377cp-1,
    0x1.0cc29fe643a7ep-1, 0x1.9e125f7c654f6p-3, -0x1.3c22175b472d2p-3,
    -0x1.ffa96cb210ba0p-2, -0x1.93cff6c51508ap-1, -0x1.f5134549366c1p-1,
    -0x1.092dd22b3d7e3p+0, -0x1.db8f176ea5256p-1, -0x1.4093a21377529p-1,
};

inline constexpr double kSplineSmoothFitpackIn_50_3[] = {
    0x1.0000000000000p+1, 0x1.f05397829cbc1p+0, 0x1.e0a72f0539783p+0,
    0x1.d0fac687d6344p+0, 0x1.c14e5e0a72f05p+0, 0x1.b1a1f58d0fac6p+0,
    0x1.a1f58d0fac688p+0, 0x1.9249249249249p+0, 0x1.829cbc14e5e0ap+0,
    0x1.72f05397829ccp+0, 0x1.6343eb1a1f58dp+0, 0x1.5397829cbc14ep+0,
    0x1.43eb1a1f58d10p+0, 0x1.343eb1a1f58d1p+0, 0x1.2492492492492p+0,
    0x1.14e5e0a72f054p+0, 0x1.05397829cbc15p+0, 0x1.eb1a1f58d0facp-1,
    0x1.cbc14e5e0a72ep-1, 0x1.ac687d6343eb2p-1, 0x1.8d0fac687d634p-1,
    0x1.6db6db6db6db8p-1, 0x1.4e5e0a72f053ap-1, 0x1.2f05397829cbcp-1,
    0x1.0fac687d63440p-1, 0x1.e0a72f0539780p-2, 0x1.a1f58d0fac688p-2,
    0x1.6343eb1a1f590p-2, 0x1.2492492492494p-2, 0x1.cbc14e5e0a730p-3,
    0x1.4e5e0a72f0538p-3, 0x1.a1f58d0fac680p-4, 0x1.4e5e0a72f0540p-5,
    -0x1.4e5e0a72f0500p-6, -0x1.4e5e0a72f0540p-4, -0x1.2492492492490p-3,
    -0x1.a1f58d0fac690p-3, -0x1.0fac687d63440p-2, -0x1.4e5e0a72f0538p-2,
    -0x1.8d0fac687d630p-2, -0x1.cbc14e5e0a730p-2, -0x1.05397829cbc18p-1,
    -0x1.2492492492490p-1, -0x1.43eb1a1f58d10p-1, -0x1.6343eb1a1f58cp-1,
    -0x1.829cbc14e5e0cp-1, -0x1.a1f58d0fac688p-1, -0x1.c14e5e0a72f04p-1,
    -0x1.e0a72f0539780p-1, -0x1.0000000000000p+0,
};

inline constexpr double kSplineSmoothFitpackOut_50_3[] = {
    0x1.ffffffffffff3p+0, 0x1.f05397829cbb7p+0, 0x1.e0a72f0539777p+0,
    0x1.d0fac687d633ap+0, 0x1.c14e5e0a72efdp+0, 0x1.b1a1f58d0fabfp+0,
    0x1.a1f58d0fac681p+0, 0x1.9249249249242p+0, 0x1.829cbc14e5e04p+0,
    0x1.72f05397829c5p+0, 0x1.6343eb1a1f588p+0, 0x1.5397829cbc14bp+0,
    0x1.43eb1a1f58d0cp+0, 0x1.343eb1a1f58cep+0, 0x1.2492492492490p+0,
    0x1.14e5e0a72f052p+0, 0x1.05397829cbc14p+0, 0x1.eb1a1f58d0fa9p-1,
    0x1.cbc14e5e0a72dp-1, 0x1.ac687d6343eaep-1, 0x1.8d0fac687d634p-1,
    0x1.6db6db6db6db5p-1, 0x1.4e5e0a72f0539p-1, 0x1.2f05397829cbcp-1,
    0x1.0fac687d6343fp-1, 0x1.e0a72f0539782p-2, 0x1.a1f58d0fac686p-2,
    0x1.6343eb1a1f58dp-2, 0x1.2492492492493p-2, 0x1.cbc14e5e0a72dp-3,
    0x1.4e5e0a72f053dp-3, 0x1.a1f58d0fac688p-4, 0x1.4e5e0a72f0540p-5,
    -0x1.4e5e0a72f0510p-6, -0x1.4e5e0a72f0534p-4, -0x1.249249249248ep-3,
    -0x1.a1f58d0fac686p-3, -0x1.0fac687d6343fp-2, -0x1.4e5e0a72f0539p-2,
    -0x1.8d0fac687d633p-2, -0x1.cbc14e5e0a72cp-2, -0x1.05397829cbc14p-1,
    -0x1.2492492492493p-1, -0x1.43eb1a1f58d0fp-1, -0x1.6343eb1a1f58ep-1,
    -0x1.829cbc14e5e0ap-1, -0x1.a1f58d0fac688p-1, -0x1.c14e5e0a72f04p-1,
    -0x1.e0a72f0539783p-1, -0x1.fffffffffffffp-1,
};

inline constexpr double kSplineSmoothFitpackIn_64_2[] = {
    0x1.72013b8c62adep-1, 0x1.3f777f814cfccp-1, -0x1.261d6b7ce1000p-11,
    0x1.64ff5a466d940p-5, -0x1.20cc700ff99c0p-4, -0x1.c2878b7f6806ep-1,
    -0x1.80a767802ab80p-2, 0x1.daf5ef17df4a6p-1, 0x1.e64dc05bce6ccp-2,
    -0x1.f2e3abae3f874p-1, -0x1.0052af1889dc0p-4, -0x1.49230a150f02ep-1,
    -0x1.4606be7fe82d8p-3, -0x1.829215d9d9312p-1, 0x1.b67551cba0b30p-1,
    0x1.38b66671656c8p-3, 0x1.a60bc103d705ep-1, 0x1.26cddff65f600p-9,
    0x1.9b7fef9194f66p-1, -0x1.003550b2dab8ep-1, 0x1.611a41d5a75a4p-2,
    -0x1.75db7a988b50cp-1, -0x1.0713f10da91d8p-1, 0x1.1a26230ddab18p-3,
    -0x1.4383a1a184f48p-3, -0x1.759d3db9bd164p-2, -0x1.600045f609300p-2,
    -0x1.36a63ba57cf04p-1, 0x1.69f68b8734048p-1, -0x1.c0c5122485900p-5,
    -0x1.c5436c46bf1b2p-1, 0x1.cbfbb7015b1e8p-1, -0x1.e081c0d64e6f8p-3,
    -0x1.8a319709e0722p-1, -0x1.fc4a9ca9e8184p-2, -0x1.b884e739e9810p-2,
    -0x1.fe74b1e9b14dcp-2, 0x1.834acab30e940p-4, -0x1.ec764b1fdd868p-2,
    0x1.8a08d1dba5220p-1, -0x1.336bdda52da04p-2, 0x1.5a76115fab8f0p-1,
    0x1.14c4f343e2230p-4, -0x1.1fd310093db12p-1, -0x1.46aa58de41a66p-1,
    0x1.3981eae9cd790p-3, 0x1.70fd8408633c8p-1, 0x1.9c7964dd0fb76p-1,
    0x1.d464e66667794p-2, 0x1.628691cacf6c8p-1, -0x1.54f947665057cp-2,
    0x1.7da4e53dc9198p-3, 0x1.f28ee02034046p-1, -0x1.fd9e6d95f3794p-1,
    -0x1.968644c735f24p-2, 0x1.eae24a5d51bd0p-3, -0x1.1c3b08bacc1dcp-2,
    0x1.fb809bbcb1590p-4, 0x1.2f1f30afc9ed8p-1, 0x1.70499cb2b2fa0p-3,
    0x1.7079a6b047a80p-1, -0x1.f6e084af019d8p-3, -0x1.f7cca2ccf9c80p-3,
    -0x1.bbec7a89652c4p-2,
};

inline constexpr double kSplineSmoothFitpackOut_64_2[] = {
    0x1.72077016825f6p-1, 0x1.3f60b68e47b7ap-1, -0x1.d569367342200p-12,
    0x1.6881ef84e292ep-5, -0x1.24decdd89e2f8p-4, -0x1.c21965a75fe88p-1,
    -0x1.8118a9b3851c8p-2, 0x1.db9bcec136154p-1, 0x1.e2707b869868ap-2,
    -0x1.ef874bf618fbep-1, -0x1.223781023ce40p-4, -0x1.447cccd15485ep-1,
    -0x1.5a292593a74bcp-3, -0x1.7d1dc30f54da8p-1, 0x1.b0cab3b2f8248p-1,
    0x1.4f36be9dfe670p-3, 0x1.a077992668deap-1, 0x1.b57735522f240p-7,
    0x1.95bbf7c2ca1e8p-1, -0x1.f57bd775b432ap-2, 0x1.57ee3a93299f4p-2,
    -0x1.7255b66a33a16p-1, -0x1.0b14899dc2db7p-1, 0x1.46b17efc7ac0ap-3,
    -0x1.ae1a364019981p-3, -0x1.2cc2dc84ef87fp-2, -0x1.9845f7875bd5dp-2,
    -0x1.29c504e7204f3p-1, 0x1.661936efd8f2fp-1, -0x1.c59682ee71ac4p-5,
    -0x1.c22d8b1546b43p-1, 0x1.c6e10a90ca3a8p-1, -0x1.c55ee1113558cp-3,
    -0x1.913efcf2b359dp-1, -0x1.f2162ba8fd70cp-2, -0x1.beaba4f0686c0p-2,
    -0x1.f8b9ecbae9ef2p-2, 0x1.64c6dd2cddbd2p-4, -0x1.e2db77f1de7bap-2,
    0x1.84cd3be150bcap-1, -0x1.29d16689c8b67p-2, 0x1.56d563a24485ap-1,
    0x1.26e01f6eb4c6ep-4, -0x1.211dd15a445b8p-1, -0x1.45d240bcdd597p-1,
    0x1.36a74de8e998ap-3, 0x1.71d16a156117dp-1, 0x1.9b3e822bb0e15p-1,
    0x1.d7c753bfede1cp-2, 0x1.60e189f5f415bp-1, -0x1.53c896e9e8829p-2,
    0x1.828f2102d8e11p-3, 0x1.efd4503dd6984p-1, -0x1.fabba4573fb09p-1,
    -0x1.9a8dac2b184efp-2, 0x1.f312e1c7221c3p-3, -0x1.24c9b58dc6b22p-2,
    0x1.17dc1365cb588p-3, 0x1.28afa45319f48p-1, 0x1.894eae4583695p-3,
    0x1.687ae3e642658p-1, -0x1.d648a3e40ba28p-3, -0x1.050aeea8db8e0p-2,
    -0x1.b9d7c3c851523p-2,
};

struct SplineSmoothFitpackCase {
    const char* name;
    std::size_t cols;
    const double* input;
    const double* expected;
};

inline constexpr SplineSmoothFitpackCase kSplineSmoothFitpackCases[] = {
    {"m4_kind2", 4, kSplineSmoothFitpackIn_4_2, kSplineSmoothFitpackOut_4_2},
    {"m7_kind6", 7, kSplineSmoothFitpackIn_7_6, kSplineSmoothFitpackOut_7_6},
    {"m12_kind6", 12, kSplineSmoothFitpackIn_12_6, kSplineSmoothFitpackOut_12_6},
    {"m20_kind6", 20, kSplineSmoothFitpackIn_20_6, kSplineSmoothFitpackOut_20_6},
    {"m31_kind5", 31, kSplineSmoothFitpackIn_31_5, kSplineSmoothFitpackOut_31_5},
    {"m33_kind9", 33, kSplineSmoothFitpackIn_33_9, kSplineSmoothFitpackOut_33_9},
    {"m50_kind3", 50, kSplineSmoothFitpackIn_50_3, kSplineSmoothFitpackOut_50_3},
    {"m64_kind2", 64, kSplineSmoothFitpackIn_64_2, kSplineSmoothFitpackOut_64_2},
};

}  // namespace n4m::test::fixtures
