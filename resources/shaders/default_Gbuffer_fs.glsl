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
struct LightInfo {
	int type; // 0: directional, 1: point, 2: spot
	// common
	vec3 position; // world position
	vec3 direction; // world direction
	vec3 color;
	float intensity;

	// attenuation
	float constant;
	float linear;
	float quadratic;

	// spot light
	float cutOff;
	float outerCutOff;

	// shadow
	int castShadow;
	mat4 lightSpaceMatrix;
	sampler2D shadowMap2D; // for directional light
	samplerCube shadowMapCube; // for point light, spot light
};
layout (std140) uniform Lights{
	vec3 ambient_color;
	float ambient_intensity;
	LightInfo lights[MAX_LIGHTS_NUM];
	int lightCount;
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
    outNormal = normalize((MV_IT * vec4(texture(normalTex, vertexUV).xyz, 0.0)).xyz);

	// get id
	ivec2 uv = ivec2(vertexUV * ivec2(textureSize(diffuseTex, 0)));
	outID = ivec3(objID, uv.x, uv.y);
}
