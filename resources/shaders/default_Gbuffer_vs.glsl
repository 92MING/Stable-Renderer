// default VS for late shading
#version 330 core
layout (std140) uniform Matrices {
	mat4 model;
	mat4 view;
	mat4 projection;
	mat4 MVP;
	mat4 MVP_IT; // inverse transpose of MVP
	mat4 MV; // model-view matrix
	mat4 MV_IT; // inverse transpose of MV
	vec3 cameraPos;
	vec3 cameraDir;
};
layout (location = 0) in vec3 pos;
layout (location = 1) in vec3 normal;
layout (location = 2) in vec2 texcoord;

out vec4 worldPos;
out vec3 worldNormal;
out vec2 vertexUV;

void main()
{
	worldPos = (MV * vec4(pos, 1.0));
	worldNormal = normalize(MV_IT * vec4(normal, 0.0)).xyz;
	vertexUV = texcoord;

	gl_Position = projection * worldPos;
}