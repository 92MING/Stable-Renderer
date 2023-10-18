// default FS for late shading
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
in vec3 worldPos;
in vec3 worldNormal;
in vec2 vertexUV;

layout (location = 0) out vec3 outColor; // diffuse color
layout (location = 1) out vec3 outPos;
layout (location = 2) out vec3 outNormal;
layout (location = 3) out ivec3 outID;  // outID = (objID, uv_Xcoord, uv_Ycoord)

uniform sampler2D diffuseTex;
uniform sampler2D normalTex;
uniform int objID;

void main() {
    outColor = texture(diffuseTex, vertexUV).rgb;
    outPos = worldPos;
    outNormal = normalize((MV_IT * vec4(texture(normalTex, vertexUV).xyz, 0.0)).xyz);
	ivec2 uv = ivec2(vertexUV * ivec2(textureSize(diffuseTex, 0)));
	outID = ivec3(objID, uv.x, uv.y);
}
