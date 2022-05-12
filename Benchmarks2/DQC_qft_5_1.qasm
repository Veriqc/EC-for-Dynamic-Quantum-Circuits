OPENQASM 2.0;
include "qelib1.inc";
qreg q[5];
creg c0[1];
creg c1[1];
creg c2[1];
creg c3[1];
creg c4[1];
x q[0];
h q[0];
z q[0];
measure q[0] -> c0[0];
if(c0==1) u1(pi/4) q[1];
if(c0==1) u1(pi/4) q[1];
h q[1];
measure q[1] -> c1[0];
if(c0==1) u1(pi/4) q[2];
if(c1==1) u1(pi/2) q[2];
h q[2];
measure q[2] -> c2[0];
if(c0==1) u1(pi/8) q[3];
if(c1==1) u1(pi/4) q[3];
if(c2==1) u1(pi/2) q[3];
h q[3];
measure q[3] -> c3[0];
if(c0==1) u1(pi/16) q[4];
if(c1==1) u1(pi/8) q[4];
if(c2==1) u1(pi/4) q[4];
if(c3==1) u1(pi/2) q[4];
h q[4];
measure q[4] -> c4[0];