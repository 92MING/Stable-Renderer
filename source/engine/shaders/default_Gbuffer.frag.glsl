// default FS for late shading
#version 430 core
#define MAX_LIGHTS_NUM 256  // this constant will be edit by python script
#define RUNTIME_UBO_BINDING 0
#define PI 3.14159265359

// runtime datastruct contains all runtime-update values
layout (std140, binding=RUNTIME_UBO_BINDING) uniform Runtime {
	mat4 model;
	mat4 view;
	mat4 projection;
	mat4 MVP;
	mat4 MVP_IT; 	// inverse transpose of MVP
	mat4 MV; 		// model-view matrix
	mat4 MV_IT; 	// inverse transpose of MV
	vec3 cameraPos;
	vec3 cameraDir;
	vec2 cameraNearFar;	// x=near, y=far
	float cameraFov;
};

// outputs
layout (location = 0) out vec4 outColor; // rgba
layout (location = 1) out ivec4 outID;
// `outID` has various usages:
//		when normal render mode: outID = (objID, material id, uv_Xcoord, uv_Ycoord)
//		when baking mode: outID = (3D pixel index, material id, uv_Xcoord, uv_Ycoord)

layout (location = 2) out vec3 outPos; // global pos, output alpha for mask
layout (location = 3) out vec4 out_normal_and_depth;	//normal (in view space). depth is also outputed here.
layout (location = 4) out vec4 outNoise; // noise, in latent shape 

// textures
uniform sampler2D diffuseTex;	// 0
uniform int hasDiffuseTex;
uniform sampler2D normalTex;	// 1
uniform int hasNormalTex;
uniform sampler2D specularTex;	// 2
uniform int hasSpecularTex;
uniform sampler2D emissionTex;	// 3
uniform int hasEmissionTex;
uniform sampler2D occlusionTex;	// 4
uniform int hasOcclusionTex;
uniform sampler2D metallicTex;	// 5
uniform int hasMetallicTex;
uniform sampler2D roughnessTex;	// 6
uniform int hasRoughnessTex;
uniform sampler2D displacementTex;	// 7
uniform int hasDisplacementTex;	
uniform sampler2D alphaTex;	// 8
uniform int hasAlphaTex;
uniform sampler2D noiseTex;	// 9 (currently only support 1 channel noise texture)
uniform int hasNoiseTex;

// material
uniform int objID;
uniform int materialID;

//AI
uniform int isBaking;	// whether in baking mode
uniform int baking_k;	// constant k, 3D-pixel baking count = k^2

// from VS
in vec3 modelPos;
in vec3 worldPos;
in vec3 modelNormal;	// not normal from normal map, is from 3D mesh data
in vec3 worldNormal;	// not normal from normal map, is from 3D mesh data
in vec3 viewNormal;		// not normal from normal map, is from 3D mesh data
in vec2 uv;
in vec3 modelTangent;


void main() {

	// get color
	if (hasDiffuseTex == 0)
		outColor = vec4(1.0, 0.0, 1.0, 1.0); // pink color means no texture
	else
		outColor = texture(diffuseTex, uv).rgba;
	
	// get noise
	if (hasNoiseTex == 0)
		outNoise = vec4(0.0, 0.0, 0.0, 0.0); // no noise texture
	else
		outNoise = texture(noiseTex, uv).rgba;

	// get position
    outPos = worldPos;

	// get depth
	float depth = 1.0 - gl_FragCoord.z;	// reverse depth to make closer object as white

	// get normal
	if (hasNormalTex == 0){  // if no normal texture, use normal from mesh data
		out_normal_and_depth = vec4(normalize(viewNormal) * 0.5 + 0.5, depth);
	}
	else{
		vec3 bitangent = cross(modelNormal, modelTangent);
		mat3 TBN = mat3(modelTangent, bitangent, modelNormal);
		vec3 real_model_normal = normalize(TBN * normalize(texture(normalTex, uv).rgb * 2.0 - 1.0));
		vec3 real_view_normal = normalize(vec3(MV_IT * vec4(real_model_normal, 0.0)));
		out_normal_and_depth = vec4(real_view_normal, depth);
	}

	// get id
	ivec2 uvi = ivec2(uv * ivec2(textureSize(diffuseTex, 0)));
	if (isBaking == 1){
		// baking mode
		vec3 posToCamDir = normalize(worldPos - cameraPos);
		vec3 bitangent = cross(modelNormal, modelTangent);
		mat3 TBN_inverse = transpose(mat3(modelTangent, bitangent, modelNormal)); // transpose of orthonormal matrix is its inverse
		vec3 posToCamDirInTangetSpace = TBN_inverse * posToCamDir;
		float theta = dot(posToCamDirInTangetSpace, vec3(0.0, 1.0, 0.0));
		float phi = dot(posToCamDirInTangetSpace, vec3(1.0, 0.0, 0.0));
		
		float vertical_angle;
		if (posToCamDirInTangetSpace.x > 0.0) vertical_angle = PI/2 - theta;
		else vertical_angle = PI/2 + theta;

		float horizontal_angle;
		if (posToCamDirInTangetSpace.z > 0.0) horizontal_angle = PI/2 + phi;
		else horizontal_angle = PI/2 - phi;

		float angle_step = PI / float(baking_k);
		int x_index = clamp(int(horizontal_angle / angle_step), 0, baking_k-1);		// from left to right
		int y_index = clamp(int(vertical_angle / angle_step), 0, baking_k-1);		// from top to bottom
		int pixel_index = x_index + y_index * baking_k;

		outID = ivec4(pixel_index, materialID, uvi.x, uvi.y);
	}
	else{
		outID = ivec4(objID, materialID, uvi.x, uvi.y);
	}
}
