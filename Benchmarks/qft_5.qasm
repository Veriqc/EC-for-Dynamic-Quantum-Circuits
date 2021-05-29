OPENQASM 2.0;
include "qelib1.inc";
qreg q[5];
creg c0[1];
creg c1[1];
creg c2[1];
creg c3[1];
creg c4[1];
h q[0];
cu1(pi/2) q[0],q[1];
cu1(pi/4) q[0],q[2];
cu1(pi/8) q[0],q[3];
cu1(pi/16) q[0],q[4];
h q[1];
cu1(pi/2) q[1],q[2];
cu1(pi/4) q[1],q[3];
cu1(pi/8) q[1],q[4];
h q[2];
cu1(pi/2) q[2],q[3];
cu1(pi/4) q[2],q[4];
h q[3];
cu1(pi/2) q[3],q[4];
h q[4];
measure q[0] -> c0[0];
measure q[1] -> c1[0];
measure q[2] -> c2[0];
measure q[3] -> c3[0];
measure q[4] -> c4[0];