#version 430

in float depth;

void main(){
	gl_FragDepth = depth;
}