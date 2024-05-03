#version 430 core

uniform sampler2D gColor;   // vec4, rgba
uniform sampler2D gID;      
// ivec4
//	when normal render mode:  (objID, material id, vertexID, 0.0)
//  when baking mode: 		  (objID, material id, vertexID, 3D pixel index)  
uniform sampler2D gPos;     // vec3, world space position
uniform sampler2D gNormalAndDepth; // vec4, (vec3(view space normal), depth)
uniform sampler2D gNoise; // vec4, latent noise

in vec2 uv;  // screen space UV

out vec4 FragColor;

void main() {
    FragColor = vec4(texture(gColor, uv).rgb, 1.0);
}
