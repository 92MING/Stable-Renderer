// default FS for late shading
#version 330 core
#define MAX_LIGHTS_NUM 256  // this constant will be edit by python script
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

layout (location = 0) out vec4 outColorAndDepth; // (r, g, b, depth)
layout (location = 1) out vec3 outPos;
layout (location = 2) out vec3 outNormal;
layout (location = 3) out ivec3 outID;  // outID = (objID, uv_Xcoord, uv_Ycoord)

uniform sampler2D diffuseTex;
uniform sampler2D normalTex;
uniform int objID;

in vec3 worldPos;
in vec3 worldNormal;
in vec2 vertexUV;

void main() {

	// get color & depth
	vec3 outColor = texture(diffuseTex, vertexUV).rgb;
	outColorAndDepth = vec4(outColor, 1.0 - gl_FragCoord.z);

	// get position
    outPos = worldPos;

	// get normal
	outNormal = texture(normalTex, vertexUV).xyz;

	// get id
	ivec2 uv = ivec2(vertexUV * ivec2(textureSize(diffuseTex, 0)));
	outID = ivec3(objID, uv.x, uv.y);
}
