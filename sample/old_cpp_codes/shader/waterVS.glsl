#version 430

in layout(location = 0) vec3 position;
in layout(location = 1) vec2 uv;

uniform sampler2D currentWaterTex;
uniform sampler2D lastWaterTex;
uniform vec3 boatPos;
uniform float boatSpeed;

out float depth;

void main(){
	gl_Position = vec4(uv, 1.0, 1.0);
	depth = 0;
	if (length(boatPos-position) < 0.5){
		if (boatSpeed > 0.001){
			depth = 1.0 ;
		}
	}
	float texelSize = 1.0/textureSize(lastWaterTex,0).x;
	float x = texture2D(lastWaterTex, uv).r;
	float up = texture2D(currentWaterTex, vec2(uv.x, uv.y+texelSize)).r;
	float down = texture2D(currentWaterTex, vec2(uv.x, uv.y-texelSize)).r;
	float left = texture2D(currentWaterTex, vec2(uv.x-texelSize, uv.y)).r;
	float right = texture2D(currentWaterTex, vec2(uv.x+texelSize, uv.y)).r;
	depth += max((up+down+left+right)/2.0 -x,0) ;
	depth *= 0.99;
	return;
}