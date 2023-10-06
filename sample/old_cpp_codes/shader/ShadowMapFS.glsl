#version 430
//fragment shader of Directional Light & Spot light

in layout (location = 0) vec3 position;
in layout (location = 1) vec2 uv;
in layout (location = 2) vec3 normal;

uniform mat4 lightVP;
uniform mat4 modelMatrix;
uniform int hasDiffuseTex;
uniform sampler2D diffuseTex;

void main(){
	if (hasDiffuseTex == 1)
	{
		if (texture(diffuseTex, uv).a < 0.4) discard;
	}
}