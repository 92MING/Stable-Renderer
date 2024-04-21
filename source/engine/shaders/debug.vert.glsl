// this is a debug vs for direct output to the screen
#version 430 core
#define RUNTIME_UBO_BINDING 0

layout (std140, binding=RUNTIME_UBO_BINDING) uniform Runtime {
	mat4 model;
	mat4 view;
	mat4 projection;
	mat4 MVP;
	mat4 MVP_IT; // inverse transpose of MVP
	mat4 MV; // model-view matrix
	mat4 MV_IT; // inverse transpose of MV
	vec3 cameraPos;
	vec3 cameraDir;
	vec2 cameraNearFar;	// x=near, y=far
	float cameraFov;
};

layout (location = 0) in vec3 pos;
layout (location = 1) in vec3 normal;
layout (location = 2) in vec2 texcoord;

out vec2 vertexUV;

void main() {
    gl_Position = MVP * vec4(pos, 1.0);
    vertexUV = texcoord;
}
