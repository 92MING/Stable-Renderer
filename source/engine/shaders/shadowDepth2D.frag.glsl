// TODO: light system is not yet finished.

#version 430 core

// Ouput data
layout(location = 0) out float fragmentdepth;

void main(){
	fragmentdepth = gl_FragCoord.z;
}