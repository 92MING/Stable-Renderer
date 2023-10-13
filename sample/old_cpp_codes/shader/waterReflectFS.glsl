#version 430

in vec4 pos;
//in vec2 uv;

uniform sampler2D waterRelectMap;
uniform mat4 camProjMatrix;
uniform mat4 camViewMatrix;

out vec4 color;

void main(){
	
	vec2 uv = (pos / pos.w).xy * 0.5 + 0.5;
	
	color = texture(waterRelectMap, uv);
	//color = vec4(1.0);
}