// Leaf value model: expected net points of the hand for the player about to draw,
// fitted by least squares on 188550 self-play positions (tools/gen_value.mjs, tools/fit_value.py).
// Features: polynomial (degree 4) of e/20, o/20, s/10 + indicators [e<=10, o<=10, e<=0]
//   e = own heuristic evaluation, o = opponent deadwood, s = stock count
export const VALUE_W = [6.15588, -47.34051, 29.41742, 0.27825, 57.09412, 3.81004, -25.0417, -40.85034, 23.73888, -0.14317, -22.47936, 0.289, 2.93555, -2.29094, -0.68048, 8.56884, 15.67839, -3.26715, -6.58335, -0.28402, 2.15077, -0.10937, 1.63822, 0.11796, 0.49333, -1.80059, 0.11787, -0.05188, -0.10302, -0.96201, -1.23176, -1.30012, 1.29884, 0.97631, -0.03779, -4.13384, 2.23179, 4.05119];
