// ---------------------------------------------------------------------
//
// Copyright (C) 2023 by the hpsint authors
//
// This file is part of the hpsint library.
//
// The hpsint library is free software; you can use it, redistribute
// it, and/or modify it under the terms of the GNU Lesser General
// Public License as published by the Free Software Foundation; either
// version 3.0 of the License, or (at your option) any later version.
// The full text of the license can be found in the file LICENSE.MD at
// the top level directory of hpsint.
//
// ---------------------------------------------------------------------

#include <deal.II/base/mpi.h>

#include <pf-applications/grain_tracker/tracking.h>

using namespace dealii;
using namespace GrainTracker;

int
main()
{
  /* A test to check the behavior of boost geometry for large scales. This test
   * also revealed the need to bring back the manual sorting since
   * bgi::nearest() does not always return a bounding box whose center is the
   * closest to the one used for the query. The test data is obtained from a
   * real simulation case, where the corresponding bug was observed. */

  constexpr unsigned int dim = 3;

  const auto add_grain = [](std::map<unsigned int, Grain<dim>> &grains,
                            const unsigned int                  grain_num,
                            const unsigned int                  grain_id,
                            const double                        x,
                            const double                        y,
                            const double                        z,
                            const double                        r) {
    grains.try_emplace(grain_num, grain_id, 0);
    grains.at(grain_num).add_segment(Point<dim>(x, y, z),
                                     r,
                                     4. / 3. * std::pow(r, 3) * M_PI,
                                     1.0);
  };

  std::map<unsigned int, Grain<dim>> old_grains;
  add_grain(old_grains, 0, 0, -69.1367, -69.144, -69.0989, 16.7985);
  add_grain(old_grains, 1, 1, -34.7463, -34.7509, -69.228, 17.1785);
  add_grain(old_grains, 2, 2, -34.7532, -69.2502, -34.7084, 17.2009);
  add_grain(old_grains, 3, 3, -69.2463, -34.7554, -34.6897, 17.4003);
  add_grain(old_grains, 4, 4, -52.1286, -52.1299, -52.0885, 7.04857);
  add_grain(old_grains, 5, 5, 0.00114182, -69.1792, -69.1801, 17.2009);
  add_grain(old_grains, 6, 6, 34.7431, -34.7453, -69.2737, 17.1675);
  add_grain(old_grains, 7, 7, 34.7394, -69.249, -34.7865, 17.1871);
  add_grain(old_grains, 8, 8, -17.4049, -52.155, -52.141, 6.5709);
  add_grain(old_grains, 9, 9, 0.000161063, -34.7527, -34.7528, 16.5456);
  add_grain(old_grains, 10, 10, 17.4045, -52.1613, -52.1705, 6.57986);
  add_grain(old_grains, 11, 11, -69.179, 0.000186061, -69.1414, 17.2028);
  add_grain(old_grains, 12, 12, -34.746, 34.7497, -69.2274, 17.1773);
  add_grain(old_grains, 13, 13, -52.1465, -17.4016, -52.1158, 6.78207);
  add_grain(old_grains, 14, 14, -34.7529, 5.1975e-05, -34.7127, 16.5847);
  add_grain(old_grains, 15, 15, -69.2465, 34.7547, -34.6885, 17.3992);
  add_grain(old_grains, 16, 16, -52.1473, 17.4012, -52.1154, 6.78202);
  add_grain(old_grains, 17, 17, 0.000853594, 0.000148451, -69.2919, 17.1572);
  add_grain(old_grains, 18, 18, 34.7441, 34.7442, -69.2735, 17.1666);
  add_grain(old_grains, 19, 19, -17.3957, -17.3953, -52.137, 6.55159);
  add_grain(old_grains, 20, 20, 17.3973, -17.4011, -52.1767, 6.56806);
  add_grain(old_grains, 21, 21, 34.7504, -4.19383e-05, -34.7861, 16.568);
  add_grain(old_grains, 22, 22, -17.3957, 17.3953, -52.137, 6.55159);
  add_grain(old_grains, 23, 23, -0.000179519, 34.752, -34.7529, 16.5462);
  add_grain(old_grains, 24, 24, 17.3978, 17.4007, -52.1763, 6.56802);
  add_grain(old_grains, 25, 25, -69.1743, -69.1786, 0.0686327, 17.2685);
  add_grain(old_grains, 26, 26, -52.153, -52.1524, -17.3602, 6.77849);
  add_grain(old_grains, 27, 27, -34.751, -34.7528, 0.0369621, 16.5866);
  add_grain(old_grains, 28, 28, -34.7393, -69.249, 34.785, 17.1875);
  add_grain(old_grains, 29, 29, -69.2483, -34.7373, 34.8014, 17.186);
  add_grain(old_grains, 30, 30, -52.1581, -52.1592, 17.452, 6.55754);
  add_grain(old_grains, 31, 31, 0.00019574, -69.2912, -7.4246e-05, 17.1566);
  add_grain(old_grains, 32, 32, -17.3957, -52.151, -17.3797, 6.555);
  add_grain(old_grains, 33, 33, 17.3997, -52.1584, -17.4183, 6.55409);
  add_grain(old_grains, 34, 34, 34.7516, -34.7521, -0.0364726, 16.5711);
  add_grain(old_grains, 35, 35, 34.754, -69.2485, 34.7077, 17.1858);
  add_grain(old_grains, 36, 36, -17.3981, -52.1576, 17.4185, 6.55463);
  add_grain(old_grains, 37, 37, 0.000364066, -34.7523, 34.7523, 16.5461);
  add_grain(old_grains, 38, 38, 17.3955, -52.1498, 17.3801, 6.55424);
  add_grain(old_grains, 39, 39, -69.2919, 0.000161823, 0.0879168, 17.2441);
  add_grain(old_grains, 40, 40, -52.1476, -17.3935, -17.3447, 6.55738);
  add_grain(old_grains, 41, 41, -52.1478, 17.3941, -17.3444, 6.55707);
  add_grain(old_grains, 42, 42, -34.7508, 34.7524, 0.0366686, 16.5718);
  add_grain(old_grains, 43, 43, -52.1542, -17.4019, 17.4399, 6.56162);
  add_grain(old_grains, 44, 44, -34.7497, 3.52282e-05, 34.7863, 16.5601);
  add_grain(old_grains, 45, 45, -69.248, 34.7376, 34.8019, 17.1856);
  add_grain(old_grains, 46, 46, -52.1542, 17.4019, 17.4399, 6.56162);
  add_grain(old_grains, 47, 47, -17.4032, -17.4042, -17.3915, 6.49058);
  add_grain(old_grains, 48, 48, 0.00171755, -2.85936e-15, -0.000124607, 16.509);
  add_grain(old_grains, 49, 49, 17.4058, -17.406, -17.4186, 6.49092);
  add_grain(old_grains, 50, 50, -17.4029, 17.4035, -17.3914, 6.49014);
  add_grain(old_grains, 51, 51, 17.4058, 17.406, -17.4186, 6.49092);
  add_grain(old_grains, 52, 52, 34.7525, 34.7509, -0.0369906, 16.5851);
  add_grain(old_grains, 53, 53, -17.4053, -17.4061, 17.418, 6.49096);
  add_grain(old_grains, 54, 54, 17.4032, -17.4042, 17.3915, 6.49058);
  add_grain(old_grains, 55, 55, 34.7532, -0.00039233, 34.7127, 16.5847);
  add_grain(old_grains, 56, 56, -17.4053, 17.4061, 17.418, 6.49096);
  add_grain(old_grains, 57, 57, 0.000473115, 34.752, 34.7522, 16.5464);
  add_grain(old_grains, 58, 58, 17.4029, 17.4035, 17.3914, 6.49014);
  add_grain(old_grains, 59, 59, 69.1314, -69.1363, -69.1845, 16.8096);
  add_grain(old_grains, 60, 60, 52.1214, -52.1225, -52.1682, 7.04278);
  add_grain(old_grains, 61, 61, 69.2491, -34.7368, -34.8016, 17.1864);
  add_grain(old_grains, 62, 62, 69.1726, 8.88009e-14, -69.2165, 17.1977);
  add_grain(old_grains, 63, 63, 52.1616, -17.4101, -52.1955, 6.56803);
  add_grain(old_grains, 64, 64, 52.1616, 17.4101, -52.1955, 6.56803);
  add_grain(old_grains, 65, 65, 69.2498, 34.736, -34.8015, 17.1871);
  add_grain(old_grains, 66, 66, 69.1757, -69.1769, -0.0676654, 17.2676);
  add_grain(old_grains, 67, 67, 52.1581, -52.1592, -17.452, 6.55754);
  add_grain(old_grains, 68, 68, 52.1533, -52.1518, 17.36, 6.77817);
  add_grain(old_grains, 69, 69, 69.247, -34.7544, 34.6887, 17.3994);
  add_grain(old_grains, 70, 70, 52.1557, -17.401, -17.4401, 6.56053);
  add_grain(old_grains, 71, 71, 69.2927, -0.000974504, -0.0883629, 17.2445);
  add_grain(old_grains, 72, 72, 52.1557, 17.401, -17.4401, 6.56053);
  add_grain(old_grains, 73, 73, 52.1478, -17.3941, 17.3444, 6.55707);
  add_grain(old_grains, 74, 74, 52.1476, 17.3935, 17.3447, 6.55738);
  add_grain(old_grains, 75, 75, 69.2489, 34.7526, 34.688, 17.3987);
  add_grain(old_grains, 76, 76, -69.1399, 69.1414, -69.0965, 16.7971);
  add_grain(old_grains, 77, 77, -52.1285, 52.1302, -52.0843, 7.05158);
  add_grain(old_grains, 78, 78, -34.7528, 69.2493, -34.7075, 17.1862);
  add_grain(old_grains, 79, 79, 0.000518957, 69.1778, -69.1771, 17.2006);
  add_grain(old_grains, 80, 80, -17.4053, 52.1538, -52.1405, 6.56983);
  add_grain(old_grains, 81, 81, 17.4043, 52.1607, -52.1707, 6.58019);
  add_grain(old_grains, 82, 82, 34.7397, 69.2482, -34.7859, 17.1871);
  add_grain(old_grains, 83, 83, -69.1738, 69.1785, 0.0683682, 17.2683);
  add_grain(old_grains, 84, 84, -52.1527, 52.1531, -17.359, 6.77797);
  add_grain(old_grains, 85, 85, -52.1579, 52.1599, 17.4521, 6.55778);
  add_grain(old_grains, 86, 86, -34.7391, 69.2492, 34.7866, 17.1874);
  add_grain(old_grains, 87, 87, -17.3956, 52.1504, -17.3796, 6.55459);
  add_grain(old_grains, 88, 88, -0.000821905, 69.2908, -0.000477616, 17.1573);
  add_grain(old_grains, 89, 89, 17.3996, 52.1574, -17.4171, 6.55353);
  add_grain(old_grains, 90, 90, -17.3997, 52.1584, 17.4183, 6.55409);
  add_grain(old_grains, 91, 91, 17.3956, 52.1504, 17.3796, 6.55459);
  add_grain(old_grains, 92, 92, 34.7524, 69.2487, 34.7074, 17.1721);
  add_grain(old_grains, 93, 93, 69.1323, 69.1326, -69.182, 16.8063);
  add_grain(old_grains, 94, 94, 52.1218, 52.1208, -52.1667, 7.042);
  add_grain(old_grains, 95, 95, 52.1586, 52.1588, -17.452, 6.55716);
  add_grain(old_grains, 96, 96, 69.1768, 69.177, -0.0674175, 17.2672);
  add_grain(old_grains, 97, 97, 52.1532, 52.1521, 17.3593, 6.77781);
  add_grain(old_grains, 98, 98, -69.1317, -69.1333, 69.1819, 16.8064);
  add_grain(old_grains, 99, 99, -52.1193, -52.1208, 52.1644, 7.04134);
  add_grain(old_grains, 100, 100, -34.7418, -34.7446, 69.273, 17.1666);
  add_grain(old_grains, 101, 101, -0.000138286, -69.1768, 69.1773, 17.2003);
  add_grain(old_grains, 102, 102, -17.4041, -52.1602, 52.1702, 6.58001);
  add_grain(old_grains, 103, 103, 17.4053, -52.1538, 52.1405, 6.56983);
  add_grain(old_grains, 104, 104, 34.7482, -34.7486, 69.2278, 17.1766);
  add_grain(old_grains, 105, 105, -69.1712, -5.97198e-05, 69.2161, 17.1979);
  add_grain(old_grains, 106, 106, -52.1594, -17.4096, 52.195, 6.56677);
  add_grain(old_grains, 107, 107, -52.1593, 17.4103, 52.1953, 6.56619);
  add_grain(old_grains, 108, 108, -34.7415, 34.7454, 69.2737, 17.1672);
  add_grain(old_grains, 109, 109, -17.3979, -17.3997, 52.1756, 6.56812);
  add_grain(old_grains, 110, 110, -0.000553406, -0.00114731, 69.2914, 17.1576);
  add_grain(old_grains, 111, 111, 17.3957, -17.3953, 52.137, 6.55159);
  add_grain(old_grains, 112, 112, -17.3946, 17.4008, 52.1762, 6.56807);
  add_grain(old_grains, 113, 113, 17.3957, 17.3953, 52.137, 6.55159);
  add_grain(old_grains, 114, 114, 34.7454, 34.7469, 69.2277, 17.1745);
  add_grain(old_grains, 115, 115, 69.1408, -69.1412, 69.0966, 16.7971);
  add_grain(old_grains, 116, 116, 52.1304, -52.1299, 52.0852, 7.05173);
  add_grain(old_grains, 117, 117, 52.1473, -17.4012, 52.1154, 6.78202);
  add_grain(old_grains, 118, 118, 69.1816, -0.000107541, 69.1421, 17.2025);
  add_grain(old_grains, 119, 119, 52.1473, 17.4012, 52.1154, 6.78202);
  add_grain(old_grains, 120, 120, -69.1306, 69.1365, 69.1844, 16.8097);
  add_grain(old_grains, 121, 121, -52.1152, 52.1263, 52.1704, 7.03674);
  add_grain(old_grains, 122, 122, -17.4043, 52.1607, 52.1707, 6.58019);
  add_grain(old_grains, 123, 123, -0.000737378, 69.1783, 69.1788, 17.2007);
  add_grain(old_grains, 124, 124, 17.4055, 52.1525, 52.14, 6.56885);
  add_grain(old_grains, 125, 125, 52.1304, 52.1299, 52.0852, 7.05173);
  add_grain(old_grains, 126, 126, 69.1409, 69.141, 69.0964, 16.797);

  // Shortcut for a better code formatting
  const auto x = numbers::invalid_unsigned_int;

  std::map<unsigned int, Grain<dim>> new_grains;
  add_grain(new_grains, 0, x, -69.272, -69.2802, -69.2429, 16.9399);
  add_grain(new_grains, 1, x, -34.7695, -34.7698, -69.3826, 17.1853);
  add_grain(new_grains, 2, x, -34.7688, -69.4011, -34.7434, 17.1774);
  add_grain(new_grains, 3, x, -69.3979, -34.7684, -34.726, 17.1935);
  add_grain(new_grains, 4, x, -52.1321, -52.1334, -52.0919, 6.50187);
  add_grain(new_grains, 5, x, 0.000671615, -69.3279, -69.3285, 17.129);
  add_grain(new_grains, 6, x, 34.7685, -34.77, -69.4269, 17.181);
  add_grain(new_grains, 7, x, 34.7706, -69.4036, -34.7963, 17.1894);
  add_grain(new_grains, 8, x, -17.4049, -52.155, -52.141, 6.5709);
  add_grain(new_grains, 9, x, 0.000244113, -34.7647, -34.7663, 16.5311);
  add_grain(new_grains, 10, x, 17.4045, -52.1613, -52.1705, 6.57986);
  add_grain(new_grains, 11, x, -69.3376, -0.000218129, -69.3036, 17.1419);
  add_grain(new_grains, 12, x, -34.7691, 34.7684, -69.3816, 17.1846);
  add_grain(new_grains, 13, x, -52.1492, -17.405, -52.1185, 6.56951);
  add_grain(new_grains, 14, x, -34.7598, -3.34166e-06, -34.7337, 16.5628);
  add_grain(new_grains, 15, x, -69.3981, 34.7675, -34.7261, 17.1933);
  add_grain(new_grains, 16, x, -52.1501, 17.4045, -52.1181, 6.57048);
  add_grain(new_grains, 17, x, 0.00121236, 0.000124048, -69.4506, 17.0881);
  add_grain(new_grains, 18, x, 34.7687, 34.77, -69.427, 17.1811);
  add_grain(new_grains, 19, x, -17.3957, -17.3953, -52.137, 6.55159);
  add_grain(new_grains, 20, x, 17.3973, -17.4011, -52.1767, 6.56806);
  add_grain(new_grains, 21, x, 34.7689, -4.1783e-05, -34.7907, 16.5553);
  add_grain(new_grains, 22, x, -17.3957, 17.3953, -52.137, 6.55159);
  add_grain(new_grains, 23, x, -0.00010868, 34.7652, -34.7654, 16.5308);
  add_grain(new_grains, 24, x, 17.3978, 17.4007, -52.1763, 6.56802);
  add_grain(new_grains, 25, x, -69.3261, -69.3298, 0.0639709, 17.1435);
  add_grain(new_grains, 26, x, -52.1558, -52.1552, -17.3635, 6.56104);
  add_grain(new_grains, 27, x, -34.7647, -34.7658, 0.0319895, 16.5738);
  add_grain(new_grains, 28, x, -34.7704, -69.404, 34.7953, 17.1889);
  add_grain(new_grains, 29, x, -69.4055, -34.7681, 34.8094, 17.1894);
  add_grain(new_grains, 30, x, -52.1581, -52.1592, 17.452, 6.55754);
  add_grain(new_grains, 31, x, -1.48879e-05, -69.4503, -0.000167075, 17.0885);
  add_grain(new_grains, 32, x, -17.3957, -52.151, -17.3797, 6.555);
  add_grain(new_grains, 33, x, 17.3997, -52.1584, -17.4183, 6.55409);
  add_grain(new_grains, 34, x, 34.7649, -34.7653, -0.0312666, 16.5558);
  add_grain(new_grains, 35, x, 34.7688, -69.4, 34.7425, 17.1774);
  add_grain(new_grains, 36, x, -17.3981, -52.1576, 17.4185, 6.55463);
  add_grain(new_grains, 37, x, 0.00022906, -34.7645, 34.7658, 16.5524);
  add_grain(new_grains, 38, x, 17.3955, -52.1498, 17.3801, 6.55424);
  add_grain(new_grains, 39, x, -69.4526, 0.000148814, 0.0590132, 17.0742);
  add_grain(new_grains, 40, x, -52.1476, -17.3935, -17.3447, 6.55738);
  add_grain(new_grains, 41, x, -52.1478, 17.3941, -17.3444, 6.55707);
  add_grain(new_grains, 42, x, -34.7645, 34.7652, 0.0315165, 16.5562);
  add_grain(new_grains, 43, x, -52.1542, -17.4019, 17.4399, 6.56162);
  add_grain(new_grains, 44, x, -34.7682, 3.50978e-05, 34.7909, 16.5552);
  add_grain(new_grains, 45, x, -69.4054, 34.7688, 34.8094, 17.19);
  add_grain(new_grains, 46, x, -52.1542, 17.4019, 17.4399, 6.56162);
  add_grain(new_grains, 47, x, -17.4032, -17.4042, -17.3915, 6.49058);
  add_grain(new_grains, 48, x, 5.00876e-15, 2.33229e-15, -8.55174e-15, 16.5088);
  add_grain(new_grains, 49, x, 17.4058, -17.406, -17.4186, 6.49092);
  add_grain(new_grains, 50, x, -17.4029, 17.4035, -17.3914, 6.49014);
  add_grain(new_grains, 51, x, 17.4058, 17.406, -17.4186, 6.49092);
  add_grain(new_grains, 52, x, 34.7656, 34.7643, -0.0317745, 16.5731);
  add_grain(new_grains, 53, x, -17.4053, -17.4061, 17.418, 6.49096);
  add_grain(new_grains, 54, x, 17.4032, -17.4042, 17.3915, 6.49058);
  add_grain(new_grains, 55, x, 34.7601, -0.000390946, 34.733, 16.5635);
  add_grain(new_grains, 56, x, -17.4053, 17.4061, 17.418, 6.49096);
  add_grain(new_grains, 57, x, 0.000371159, 34.7645, 34.7652, 16.5529);
  add_grain(new_grains, 58, x, 17.4029, 17.4035, 17.3914, 6.49014);
  add_grain(new_grains, 59, x, 69.2661, -69.2702, -69.3102, 16.9347);
  add_grain(new_grains, 60, x, 52.1248, -52.126, -52.1718, 6.50356);
  add_grain(new_grains, 61, x, 69.4057, -34.7674, -34.8095, 17.1887);
  add_grain(new_grains, 62, x, 69.3204, 1.38167e-13, -69.3586, 17.1422);
  add_grain(new_grains, 63, x, 52.1616, -17.4101, -52.1955, 6.56803);
  add_grain(new_grains, 64, x, 52.1616, 17.4101, -52.1955, 6.56803);
  add_grain(new_grains, 65, x, 69.4058, 34.7666, -34.8094, 17.1879);
  add_grain(new_grains, 66, x, 69.3273, -69.3282, -0.0630958, 17.1421);
  add_grain(new_grains, 67, x, 52.1581, -52.1592, -17.452, 6.55754);
  add_grain(new_grains, 68, x, 52.1561, -52.1546, 17.3632, 6.56147);
  add_grain(new_grains, 69, x, 69.3983, -34.7676, 34.7265, 17.1937);
  add_grain(new_grains, 70, x, 52.1557, -17.401, -17.4401, 6.56053);
  add_grain(new_grains, 71, x, 69.4535, -0.000807065, -0.0593358, 17.0747);
  add_grain(new_grains, 72, x, 52.1557, 17.401, -17.4401, 6.56053);
  add_grain(new_grains, 73, x, 52.1478, -17.3941, 17.3444, 6.55707);
  add_grain(new_grains, 74, x, 52.1476, 17.3935, 17.3447, 6.55738);
  add_grain(new_grains, 75, x, 69.3993, 34.7661, 34.7255, 17.1923);
  add_grain(new_grains, 76, x, -69.2753, 69.2769, -69.2418, 16.9377);
  add_grain(new_grains, 77, x, -52.132, 52.1337, -52.0876, 6.50525);
  add_grain(new_grains, 78, x, -34.7672, 69.3999, -34.7424, 17.1758);
  add_grain(new_grains, 79, x, -0.000566603, 69.3279, -69.3273, 17.1289);
  add_grain(new_grains, 80, x, -17.4053, 52.1538, -52.1405, 6.56983);
  add_grain(new_grains, 81, x, 17.4043, 52.1607, -52.1707, 6.58019);
  add_grain(new_grains, 82, x, 34.7702, 69.403, -34.7971, 17.1892);
  add_grain(new_grains, 83, x, -69.3249, 69.3294, 0.0634224, 17.143);
  add_grain(new_grains, 84, x, -52.1555, 52.1559, -17.3623, 6.56193);
  add_grain(new_grains, 85, x, -52.1579, 52.1599, 17.4521, 6.55778);
  add_grain(new_grains, 86, x, -34.7702, 69.4038, 34.7964, 17.189);
  add_grain(new_grains, 87, x, -17.3956, 52.1504, -17.3796, 6.55459);
  add_grain(new_grains, 88, x, -0.000726311, 69.4505, -0.000213426, 17.0876);
  add_grain(new_grains, 89, x, 17.3996, 52.1574, -17.4171, 6.55353);
  add_grain(new_grains, 90, x, -17.3997, 52.1584, 17.4183, 6.55409);
  add_grain(new_grains, 91, x, 17.3956, 52.1504, 17.3796, 6.55459);
  add_grain(new_grains, 92, x, 34.7683, 69.4001, 34.7427, 17.1769);
  add_grain(new_grains, 93, x, 69.267, 69.2662, -69.3083, 16.9322);
  add_grain(new_grains, 94, x, 52.1253, 52.1242, -52.1702, 6.5031);
  add_grain(new_grains, 95, x, 52.1586, 52.1588, -17.452, 6.55716);
  add_grain(new_grains, 96, x, 69.3277, 69.3289, -0.0628129, 17.1419);
  add_grain(new_grains, 97, x, 52.156, 52.1548, 17.3626, 6.5618);
  add_grain(new_grains, 98, x, -69.266, -69.2661, 69.3088, 16.9326);
  add_grain(new_grains, 99, x, -52.1227, -52.1243, 52.168, 6.50298);
  add_grain(new_grains, 100, x, -34.767, -34.7695, 69.4247, 17.1805);
  add_grain(new_grains, 101, x, 0.000806622, -69.3272, 69.3276, 17.1289);
  add_grain(new_grains, 102, x, -17.4041, -52.1602, 52.1702, 6.58001);
  add_grain(new_grains, 103, x, 17.4053, -52.1538, 52.1405, 6.56983);
  add_grain(new_grains, 104, x, 34.7702, -34.7689, 69.3823, 17.1856);
  add_grain(new_grains, 105, x, -69.3192, -5.83197e-05, 69.3582, 17.1421);
  add_grain(new_grains, 106, x, -52.1594, -17.4096, 52.195, 6.56677);
  add_grain(new_grains, 107, x, -52.1593, 17.4103, 52.1953, 6.56619);
  add_grain(new_grains, 108, x, -34.7675, 34.7702, 69.426, 17.1811);
  add_grain(new_grains, 109, x, -17.3979, -17.3997, 52.1756, 6.56812);
  add_grain(new_grains, 110, x, -0.00170076, -0.000827216, 69.4508, 17.0503);
  add_grain(new_grains, 111, x, 17.3957, -17.3953, 52.137, 6.55159);
  add_grain(new_grains, 112, x, -17.3946, 17.4008, 52.1762, 6.56807);
  add_grain(new_grains, 113, x, 17.3957, 17.3953, 52.137, 6.55159);
  add_grain(new_grains, 114, x, 34.7689, 34.7675, 69.382, 17.1842);
  add_grain(new_grains, 115, x, 69.2756, -69.2769, 69.2419, 16.9377);
  add_grain(new_grains, 116, x, 52.1338, -52.1334, 52.0886, 6.50437);
  add_grain(new_grains, 117, x, 52.1501, -17.4045, 52.1181, 6.57048);
  add_grain(new_grains, 118, x, 69.338, -7.1611e-05, 69.3039, 17.1417);
  add_grain(new_grains, 119, x, 52.1501, 17.4045, 52.1181, 6.57048);
  add_grain(new_grains, 120, x, -69.2651, 69.2702, 69.3108, 16.9355);
  add_grain(new_grains, 121, x, -52.1223, 52.127, 52.1712, 6.5048);
  add_grain(new_grains, 122, x, -17.4043, 52.1607, 52.1707, 6.58019);
  add_grain(new_grains, 123, x, 0.000194991, 69.3286, 69.3275, 17.1287);
  add_grain(new_grains, 124, x, 17.4055, 52.1525, 52.14, 6.56885);
  add_grain(new_grains, 125, x, 52.1338, 52.1334, 52.0886, 6.50437);
  add_grain(new_grains, 126, x, 69.276, 69.276, 69.2419, 16.9369);

  const unsigned int n_order_params = 1;

  const auto new_grains_to_old =
    transfer_grain_ids(new_grains, old_grains, n_order_params);

  std::cout << "# of old grains = " << old_grains.size() << std::endl;
  std::cout << "# of new grains = " << new_grains.size() << std::endl;

  std::cout << "Grains mapping (new_id -> old_id):" << std::endl;
  for (const auto &[new_id, old_id] : new_grains_to_old)
    std::cout << new_id << " -> " << old_id << std::endl;
}