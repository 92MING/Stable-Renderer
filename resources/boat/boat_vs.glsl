#version 330 core
layout (std140) uniform Matrices {
	mat4 model;
	mat4 view;
	mat4 projection;
	mat4 MVP;
};
layout (location = 0) in vec3 pos;
layout (location = 1) in vec3 normal;
layout (location = 2) in vec2 texcoord;

out vec3 vertexPosition_modelspace;
out vec2 vertexUV;
out vec4 globalPos;

void main()
{
	vertexUV = texcoord;
	vertexPosition_modelspace = pos;
	vec4 global_pos =  MVP * vec4(pos, 1.0);
	gl_Position = global_pos;
	globalPos = global_pos;
}