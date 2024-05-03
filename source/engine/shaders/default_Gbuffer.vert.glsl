// default VS for late shading
#version 430 core
#define MAX_LIGHTS_NUM 256  // this constant will be edit by python script
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
layout (location = 2) in vec3 tangent;
layout (location = 3) in vec3 bitangent;
layout (location = 4) in vec3 vertex_color;
layout (location = 5) in int vertex_id;
layout (location = 8) in vec2 texcoord;	// location >= 8 is reserved for texcoord

out vec3 modelPos;
out vec3 worldPos;
out vec3 modelNormal;	// not normal from normal map!
out vec3 worldNormal; 	// not normal from normal map!
out vec3 viewNormal;	// not normal from normal map!
out vec3 modelTangent;
out vec3 modelBitangent;
out vec2 uv;
flat out int vertexID; // vertex ID no need interpolation

void main()
{
	vertexID = vertex_id;
	
	modelPos = pos;
	worldPos = (MV * vec4(pos, 1.0)).xyz;

	modelNormal = normal;
	worldNormal = normalize(inverse(transpose(mat3(model))) * normal);
	viewNormal = normalize((MV_IT * vec4(normal, 0.0)).xyz);

	uv = texcoord;
	modelTangent = normalize(tangent);
	modelBitangent = normalize(bitangent);

	gl_Position = projection * vec4(worldPos, 1.0);
}