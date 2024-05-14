// default FS for GBuffer submits
// this shader is available for both AI(Instance) or non-AI(normal mesh) rendering.

#version 430 core
#define MAX_LIGHTS_NUM 256  // this constant will be edit by python script
#define RUNTIME_UBO_BINDING 0
#define PI 3.14159265359
#define NON_AI_OBJ_MAP_INDEX 2048	// non-AI obj map index, since map size=k^2, k usually ~2-3, so 2048 is a safe number

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
layout (location = 5) out vec3 outCanny; // canny edge detection, should be changed to 1D texture(put in pos tex) in the future

// FBO textures
uniform sampler2D currentColor;	// 0
uniform isampler2D currentIDs;	// 1
uniform sampler2D currentPos;	// 2
uniform sampler2D currentNormalDepth;	// 3
uniform sampler2D currentNoises;	// 4
uniform sampler2D currentCanny;	// 5

// textures, starting from 6
uniform sampler2D diffuseTex;
uniform sampler2D normalTex;
uniform sampler2D specularTex;
uniform sampler2D emissionTex;
uniform sampler2D occlusionTex;
uniform sampler2D metallicTex;
uniform sampler2D roughnessTex;
uniform sampler2D displacementTex;
uniform sampler2D alphaTex;
uniform sampler2D noiseTex;
uniform sampler2DArray correspond_map;

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
	vec3 real_view_normal;
	if (hasNormalTex == 0){  // if no normal texture, use normal from mesh data
		real_view_normal = normalize(viewNormal);
	}
	else{
		mat3 TBN = mat3(modelTangent, modelBitangent, modelNormal); // TBN converts normal from tangent space to model space
		vec3 real_model_normal = normalize(TBN * normalize(texture(normalTex, uv).rgb * 2.0 - 1.0));
		real_view_normal = normalize(vec3(MV_IT * vec4(real_model_normal, 0.0)));
	}
	out_normal_and_depth = vec4(real_view_normal * 0.5 + 0.5, depth);

	// get id
	int map_index;
	int real_vertex_id;
	if (useTexcoordAsID == 0){
		real_vertex_id = vertexID;
	}
	else{ // use texcoord as ID
		if (hasDiffuseTex == 0){
			if (hasCorrMap == 1){
				ivec3 corrmap_size = textureSize(correspond_map, 0);
				int corrmap_width = corrmap_size.x;
				int corrmap_height = corrmap_size.y;
				real_vertex_id = int(uv.y * corrmap_height * corrmap_width + uv.x * corrmap_width);
			}
			else{ // both diffuse and corrmap are not available, use default size (512*512)
				real_vertex_id = int(uv.y * 512 * 512 + uv.x * 512);
			}
		}
		else{
			ivec2 diffuseTexSize = textureSize(diffuseTex, 0);
			real_vertex_id = int(uv.y * diffuseTexSize.y * diffuseTexSize.x + uv.x * diffuseTexSize.x);
		}
	}

	if (renderMode == 0){ // normal obj
		outID = uvec4(spriteID, materialID, NON_AI_OBJ_MAP_INDEX, real_vertex_id);
	} 
	else // AI obj
	{	
		// get map index
		float theta = dot(normalize(vec3(0, real_view_normal.y, real_view_normal.z)), vec3(0.0, 1.0, 0.0)); // vertical angle
		theta = PI/2 - theta; // [0, PI]
		float phi = dot(normalize(vec3(real_view_normal.x, 0, real_view_normal.z)), vec3(1.0, 0.0, 0.0)); // horizontal angle
		phi = PI/2 - phi;

		float angle_step = PI / corrmap_k;
		int x_index = clamp(int(theta / angle_step), 0, corrmap_k-1);		
		int y_index = clamp(int(phi / angle_step), 0, corrmap_k-1);
		map_index = x_index + (corrmap_k-1-y_index) * corrmap_k;	// from left to right, from top to bottom
		
		outID = uvec4(spriteID, materialID, map_index, real_vertex_id);
	}

	// get color
	if (renderMode == 0){	// normal obj
		if (hasDiffuseTex == 0){
			if (hasVertexColor == 1) {
				outColor = vec4(vertexColor, 1.0);
			}
			else {
				outColor = vec4(0.0, 0.0, 0.0, 0.0); // clear color
			}
		}
		else {
			outColor = texture(diffuseTex, uv).rgba;
		}
	}
	else{ // baked obj, get color from corresponding map
		if (renderMode == 2) // baking
			outColor = vec4(0.0, 0.0, 0.0, 0.0); // return no colors
		else if (renderMode==1) {	// baked
			if (hasCorrMap == 1){
				vec3 corrmap_uv;
				if (useTexcoordAsID == 1){
					corrmap_uv = vec3(uv, map_index);
				}
				else{
					// ivec3 corrmap_size = textureSize(correspond_map, 0);
					ivec3 corrmap_size = ivec3(512, 512, corrmap_k * corrmap_k);
					int corrmap_width = corrmap_size.x;
					int corrmap_height = corrmap_size.y;
					float u = float(vertexID % corrmap_width) / float(corrmap_width);
					float v = float(vertexID / corrmap_width) / float(corrmap_height);
					corrmap_uv = vec3(u, v, map_index);
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

	// get canny
	float canny_threshold = cos(PI * 4 / 9); // 80 degree
	float cur_view_cos = dot(real_view_normal, vec3(0.0, 0.0, 1.0));
	if (cur_view_cos < canny_threshold && cur_view_cos > 0) outCanny = vec3(1.0, 1.0, 1.0); // white edge
	else outCanny = vec3(0.0, 0.0, 0.0); // black, means no edge

	// blend
	vec2 current_pixel_uv_in_ndc =  gl_FragCoord.xy / vec2(textureSize(currentColor, 0));	// normalize to [0, 1]
	
	if (renderMode==2 || (outColor.a == 0.0 && renderMode==1)){
		// baked AI obj, current fragment is transparent, keep the original data
		// for baking mode, if it is correspond map, data is always overwrited
		outColor = texture(currentColor, current_pixel_uv_in_ndc).rgba;	// keep the original color
		if (renderMode==1)
			outID = texture(currentIDs, current_pixel_uv_in_ndc);	// only non-baking mode(render mode) will discard the corrmap id
		outPos = texture(currentPos, current_pixel_uv_in_ndc).rgb;
		out_normal_and_depth = texture(currentNormalDepth, current_pixel_uv_in_ndc).rgba;
		outCanny = texture(currentCanny, current_pixel_uv_in_ndc).rgb; 
	}
	else if (outColor.a < 1.0){	// normal object, do blending
		float latest_depth = texture(currentNormalDepth, current_pixel_uv_in_ndc).a;
		vec4 current_color = texture(currentColor, current_pixel_uv_in_ndc).rgba;
		vec4 current_noises = texture(currentNoises, current_pixel_uv_in_ndc).rgba;

		if (latest_depth < depth){ // depth is inversed, so this means overlapping
			outColor = vec4(outColor.rgb * outColor.a + current_color.rgb * (1.0 - outColor.a), outColor.a); // one minus src alpha
			if (current_noises.r + current_noises.g + current_noises.b + current_noises.a > 0.001) {	// if current pixel has noise, mix the 2 noises
				// mix the 2 latents(actually pixel spaces here) also.
				outNoise = outNoise * outColor.a + current_noises * (1.0 - outColor.a);	
			}
		}
		else{	// under the current pixel, but usually this should not happen
			outColor = vec4(current_color.rgb * current_color.a + outColor.rgb * (1.0 - current_color.a), current_color.a);
			if (current_noises.r + current_noises.g + current_noises.b + current_noises.a > 0.001) {	// if current pixel has noise, mix the 2 noises
				outNoise = current_noises * current_color.a + outNoise * (1.0 - current_color.a);
			}
			out_normal_and_depth.a = latest_depth;
		}
	}
}
