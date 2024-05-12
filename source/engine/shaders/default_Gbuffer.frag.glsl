// default FS for GBuffer submits
// this shader is available for both AI(Instance) or non-AI(normal mesh) rendering.

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
layout (location = 1) out uvec4 outID; 
// (spriteID, material id, map index, vertexID)
// explaination:
//	- spriteID: refers to `meshID` in `Mesh` class. Each mesh has a unique id to identify itself. 
// 	- materialID: refers to `materialID` in `Material` class. Each material has a unique id to identify itself.
// 				  One mesh can have multiple materials, so this id is used to identify which material is used in this fragment.
// 	- map index: different viewing angles resulting in different map index, e.g. when k=3, 3D pixel count=9, map index = 0~8,
//				 where the center is 4.
// 	- vertexID: Refers to the vertex index in the mesh. This ID could be created by giving a texture to it, or using the mesh's vertice/indices.
//				For objs with multiple & complicated texture mapping, it is recommended to use the vectex method to get a more accurate ID; when
//				the object has too less vertices, it is recommended to use the texture method to get a more accurate ID.
layout (location = 2) out vec3 outPos; // global pos, output alpha for mask
layout (location = 3) out vec4 out_normal_and_depth;	//normal (in view space). depth is also outputed here.
layout (location = 4) out vec4 outNoise; // latent noise 

// textures
uniform sampler2D currentColor;	// 0
uniform usampler2D currentIDs;	// 1
uniform sampler2D currentPos;	// 2
uniform sampler2D currentNormalDepth;	// 3
uniform sampler2D currentNoises;	// 4

uniform sampler2D diffuseTex;	// 5
uniform sampler2D normalTex;	// 6
uniform sampler2D specularTex;	// 7
uniform sampler2D emissionTex;	// 8
uniform sampler2D occlusionTex;	// 9
uniform sampler2D metallicTex;	// 10
uniform sampler2D roughnessTex;	// 11
uniform sampler2D displacementTex;	// 12
uniform sampler2D alphaTex;	// 13
uniform sampler2D noiseTex; // 14
uniform sampler2DArray correspond_map;  // 15

// flags
uniform int hasDiffuseTex;
uniform int hasNormalTex;
uniform int hasSpecularTex;
uniform int hasEmissionTex;
uniform int hasOcclusionTex;
uniform int hasMetallicTex;
uniform int hasRoughnessTex;
uniform int hasDisplacementTex;	
uniform int hasAlphaTex;
uniform int hasNoiseTex;
uniform int hasVertexColor;
uniform int hasCorrMap;

// for stable-rendering
uniform int spriteID;
uniform int materialID;
uniform int corrmap_k;	// constant k, 3D-pixel baking count = k^2
uniform int useTexcoordAsID; // it not, use texcoord(v*width+u) as ID
uniform int renderMode;
// 0: normal obj
// 1: AI obj, get color from corrmap
// 2: AI obj, baking

// from VS
in vec3 vertexColor;
in vec3 modelPos;
in vec3 worldPos;
in vec3 modelNormal;	// not normal from normal map, is from 3D mesh data
in vec3 worldNormal;	// not normal from normal map, is from 3D mesh data
in vec3 viewNormal;		// not normal from normal map, is from 3D mesh data
in vec2 uv;
in vec3 modelTangent;
in vec3 modelBitangent;
flat in int vertexID;	// interpolation is not needed, so use flat

void main() {

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
		mat3 TBN = mat3(modelTangent, modelBitangent, modelNormal); // TBN converts normal from tangent space to model space
		vec3 real_model_normal = normalize(TBN * normalize(texture(normalTex, uv).rgb * 2.0 - 1.0));
		vec3 real_view_normal = normalize(vec3(MV_IT * vec4(real_model_normal, 0.0)));
		out_normal_and_depth = vec4(real_view_normal, depth);
	}

	// get id
	int map_index;
	if (renderMode == 0){ // normal obj
		outID = ivec4(spriteID, materialID, 0, 0);
	} 
	else // AI obj
	{
		int real_vertex_id;
		ivec2 diffuseTexSize = textureSize(diffuseTex, 0);
		if (useTexcoordAsID == 0){
			real_vertex_id = vertexID;
		}
		else{ // use v * width + u as ID
			ivec2 uvi = ivec2(uv * ivec2(textureSize(diffuseTex, 0)));
			real_vertex_id = uvi.y * diffuseTexSize.x + uvi.x;
		}
		
		// get map index
		mat3 TBN_inverse = transpose(mat3(modelTangent, modelBitangent, modelNormal)); // transpose(TBN) = inverse(TBN)
		mat3 M_inverse = transpose(mat3(model));
		
		vec3 posToCamDirInWorldSpace = normalize(worldPos - cameraPos);
		vec3 posToCamDirInModelSpace = M_inverse * posToCamDirInWorldSpace;
		vec3 posToCamDirInTangetSpace = TBN_inverse * posToCamDirInModelSpace;

		float theta = dot(posToCamDirInTangetSpace, vec3(0.0, 1.0, 0.0));
		float phi = dot(posToCamDirInTangetSpace, vec3(1.0, 0.0, 0.0));
		
		float vertical_angle;
		if (posToCamDirInTangetSpace.x > 0.0) vertical_angle = PI/2 - theta;
		else vertical_angle = PI/2 + theta;

		float horizontal_angle;
		if (posToCamDirInTangetSpace.z > 0.0) horizontal_angle = PI/2 + phi;
		else horizontal_angle = PI/2 - phi;

		float angle_step = PI / float(corrmap_k);
		int x_index = clamp(int(horizontal_angle / angle_step), 0, corrmap_k-1);		// from left to right
		int y_index = clamp(int(vertical_angle / angle_step), 0, corrmap_k-1);		// from top to bottom
		map_index = x_index + y_index * corrmap_k;
		
		outID = ivec4(spriteID, materialID, map_index, real_vertex_id);
	}

	// get color
	if (renderMode == 0){
		if (hasDiffuseTex == 0){
			if (hasVertexColor == 1) {
				outColor = vec4(vertexColor, 1.0);
			}
			else {
				outColor = vec4(1.0, 0.0, 1.0, 1.0); // pink color means no texture
			}
		}
		else {
			outColor = texture(diffuseTex, uv).rgba;
		}
	}
	else{ // baked obj, get color from corresponding map
		if (renderMode == 2) // baking
			outColor = vec4(0.0, 0.0, 0.0, 0.0); // return no colors
		else{	// baked
			if (hasCorrMap == 1){
				vec3 corrmap_uv;
				if (useTexcoordAsID == 1){
					corrmap_uv = vec3(uv, float(map_index));
				}
				else{
					ivec3 corrmap_size = textureSize(correspond_map, 0);
					int corrmap_width = corrmap_size.x;
					int corrmap_height = corrmap_size.y;
					float u = float(vertexID % corrmap_width) / float(corrmap_width);
					float v = float(vertexID / corrmap_width) / float(corrmap_height);
					corrmap_uv = vec3(u, v, float(map_index));
				}
				outColor = texture(correspond_map, corrmap_uv).rgba;
			}
			else{
				if (hasDiffuseTex == 0){
					if (hasVertexColor == 1) {
						outColor = vec4(vertexColor, 1.0);
					}
					else {
						outColor = vec4(1.0, 0.0, 1.0, 1.0); // pink color means no texture
					}
				}
				else {
					outColor = texture(diffuseTex, uv).rgba;
				}
			}
		}
	}

	if (outColor.a < 1.0){
		vec2 current_pixel_uv = vec2(gl_FragCoord.x, gl_FragCoord.y);	// e.g. (511.5, 511.5)
		vec2 current_pixel_uv_in_ndc = current_pixel_uv / vec2(textureSize(currentColor, 0));	// normalize to [0, 1]
		
		vec4 current_color = texture(currentColor, current_pixel_uv).rgba;
		outColor = outColor * outColor.a + current_color * (1.0 - outColor.a); // one minus src alpha
		
		vec4 current_noises = texture(currentNoises, current_pixel_uv).rgba;
		if (current_noises.r + current_noises.g + current_noises.b + current_noises.a > 0.001) {	// if current pixel has noise, mix the 2 noises
			// mix the 2 latents(actually pixel spaces here) also.
			outNoise = outNoise * outColor.a + current_noises * (1.0 - outColor.a);	
		}

		if (renderMode != 0 && current_color.a == 0.0){
			// current fragment is transparent, keep the original data
			outID = texture(currentIDs, current_pixel_uv);
			outPos = texture(currentPos, current_pixel_uv).rgb;
			out_normal_and_depth = texture(currentNormalDepth, current_pixel_uv).rgba;
		}
	}
}
