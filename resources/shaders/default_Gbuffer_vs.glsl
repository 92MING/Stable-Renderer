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
layout (location = 3) in vec3 tangent;

out vec3 modelPos;
out vec3 worldPos;
out vec3 modelNormal;	// not normal from normal map!
out vec3 worldNormal; 	// not normal from normal map!
out vec3 viewNormal;	// not normal from normal map!
out vec2 uv;
out vec3 modelTangent;

void main()
{
	modelPos = pos;
	worldPos = (MV * vec4(pos, 1.0)).xyz;

	modelNormal = normal;
	worldNormal = normalize(inverse(transpose(mat3(model))) * normal);
	viewNormal = normalize((MV_IT * vec4(normal, 0.0)).xyz);

	uv = texcoord;
	modelTangent = normalize(tangent);

	gl_Position = projection * vec4(worldPos, 1.0);
}