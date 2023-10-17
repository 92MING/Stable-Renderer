// default VS for late shading
#version 330 core
layout (std140) uniform Matrices {
	mat4 model;
	mat4 view;
	mat4 projection;
	mat4 MVP;
	mat4 MVP_IT; // inverse transpose of MVP
};
layout (location = 0) in vec3 pos;
layout (location = 1) in vec3 normal;
layout (location = 2) in vec2 texcoord;

out vec4 globalPos;
out vec2 vertexUV;

void main()
{
	globalPos = model * vec4(pos, 1.0);
	vertexUV = texcoord;
	gl_Position = globalPos;
}