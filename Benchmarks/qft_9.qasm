OPENQASM 2.0;
include "qelib1.inc";
qreg q[9];
creg c0[1];
creg c1[1];
creg c2[1];
creg c3[1];
creg c4[1];
creg c5[1];
creg c6[1];
creg c7[1];
creg c8[1];
h q[0];
cu1(pi/2) q[0],q[1];
cu1(pi/4) q[0],q[2];
cu1(pi/8) q[0],q[3];
cu1(pi/16) q[0],q[4];
cu1(pi/32) q[0],q[5];
cu1(pi/64) q[0],q[6];
cu1(pi/128) q[0],q[7];
cu1(pi/256) q[0],q[8];
h q[1];
cu1(pi/2) q[1],q[2];
cu1(pi/4) q[1],q[3];
cu1(pi/8) q[1],q[4];
cu1(pi/16) q[1],q[5];
cu1(pi/32) q[1],q[6];
cu1(pi/64) q[1],q[7];
cu1(pi/128) q[1],q[8];
h q[2];
cu1(pi/2) q[2],q[3];
cu1(pi/4) q[2],q[4];
cu1(pi/8) q[2],q[5];
cu1(pi/16) q[2],q[6];
cu1(pi/32) q[2],q[7];
cu1(pi/64) q[2],q[8];
h q[3];
cu1(pi/2) q[3],q[4];
cu1(pi/4) q[3],q[5];
cu1(pi/8) q[3],q[6];
cu1(pi/16) q[3],q[7];
cu1(pi/32) q[3],q[8];
h q[4];
cu1(pi/2) q[4],q[5];
cu1(pi/4) q[4],q[6];
cu1(pi/8) q[4],q[7];
cu1(pi/16) q[4],q[8];
h q[5];
cu1(pi/2) q[5],q[6];
cu1(pi/4) q[5],q[7];
cu1(pi/8) q[5],q[8];
h q[6];
cu1(pi/2) q[6],q[7];
cu1(pi/4) q[6],q[8];
h q[7];
cu1(pi/2) q[7],q[8];
h q[8];
measure q[0] -> c0[0];
measure q[1] -> c1[0];
measure q[2] -> c2[0];
measure q[3] -> c3[0];
measure q[4] -> c4[0];
measure q[5] -> c5[0];
measure q[6] -> c6[0];
measure q[7] -> c7[0];
measure q[8] -> c8[0];
